"""Retrieval evaluation harness.

Usage (from the backend/ folder):
    python3 -m retrieval_eval.run_eval                 # run everything, write results/
    python3 -m retrieval_eval.run_eval --quick         # skip the dense (LSA) model

What production does (api/chat_router.py): for each question it runs TWO retrievers and
gives the answer model everything they return:
    1. corpus.retrieve(text, limit=3)       hand-written keyword corpus (~142 entries)
    2. corpus_db.retrieve_db(text, limit=3) MongoDB $text + corroboration re-rank
This harness runs (1) for real and approximates only the MongoDB part of (2)
(see baseline.py for that limit). The proposed change replaces (2) only.

Metrics, over the questions in gold_set.json:
    hit@k            ANY expected section is among a retriever's top k results (k=1,3,5)
    MRR@5            mean of 1/rank of the first expected section (0 if not in top 5)
    context_hit      ANY expected section is in what the answer model would actually see:
                     the keyword-corpus hits (limit 3) plus that config's top 3
    answered         for out-of-scope questions: how often anything at all is returned
                     (the safe behaviour is to return nothing)

Configurations, fixed before looking at any numbers:
    production_approx   database path as production shapes it
    bm25                BM25 only
    dense_lsa           TF-IDF+SVD only (a stand-in for real embeddings, see hybrid_retrieval.py)
    rrf_bm25_dense      RRF(bm25, dense)
    rrf_prod_dense      RRF(production ranking, dense)
    rrf_prod_bm25       RRF(production ranking, bm25)
    hybrid_all          RRF(production, bm25, dense)                         <- proposed hybrid
    hybrid_all_gated    hybrid_all keeping only results found in the top 20 of at least
                        2 of the 3 rankers, so unrelated questions can still return nothing
                                                                             <- proposed for production
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from retrieval_eval.baseline import load_central_live_sections, production_ranking, python_corpus_refs
from hybrid_retrieval import LsaBackend, SectionIndex, bm25_scores, rrf_fuse, top_ranking
import retrieval_logic as rl

HERE = Path(__file__).resolve().parent
GOLD = HERE / "gold_set.json"
RESULTS_DIR = HERE / "results"
K = 5
PROD_LIMIT = 3          # retrieve_db(limit=3) in chat_router.py
POOL = 300
CONSENSUS_TOPN = 20

Ref = tuple[str, str]   # (act name text, section number)


# ── Gold set ──────────────────────────────────────────────────────────────────

def load_gold(path: Path = GOLD) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expected_pairs(gold: dict, q: dict) -> list[Ref]:
    """[(act needle, section number)] for a question."""
    out = []
    for ref in q["expected"]:
        alias, sec = ref.split(":", 1)
        out.append((gold["acts"][alias], sec))
    return out


def validate_gold(index: SectionIndex, gold: dict) -> None:
    """Raise if any expected section is not present (and live) in the corpus, so a typo
    or a repealed section can never silently make a question unanswerable."""
    present = {(d["act_name"], str(d["section_number"])) for d in index.docs}
    missing = []
    for q in gold["questions"]:
        for needle, sec in expected_pairs(gold, q):
            if not any(s == sec and needle in act for act, s in present):
                missing.append(f"{q['id']}:{needle}:{sec}")
    if missing:
        raise ValueError(f"Expected sections not found as live CENTRAL sections: {missing}")


def ref_matches(result: Ref, expected: Ref) -> bool:
    act, sec = result
    needle, esec = expected
    return sec == esec and needle.lower() in act.lower()


# ── Metrics ───────────────────────────────────────────────────────────────────

def first_hit_rank(results: list[Ref], expected: list[Ref], k: int = K) -> Optional[int]:
    for pos, res in enumerate(results[:k], start=1):
        if any(ref_matches(res, e) for e in expected):
            return pos
    return None


def summarize(ranks: list[Optional[int]]) -> dict:
    n = len(ranks)
    def hit(k): return sum(1 for r in ranks if r is not None and r <= k)
    mrr = sum((1.0 / r) for r in ranks if r is not None) / n if n else 0.0
    return {"n": n, "hit@1": hit(1), "hit@3": hit(3), "hit@5": hit(5), "mrr@5": round(mrr, 3)}


def bootstrap_ci(indicator: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(indicator, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_diff(a: list[int], b: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean(b) - mean(a) with a 95% paired-bootstrap interval."""
    rng = np.random.default_rng(seed)
    a_, b_ = np.asarray(a, float), np.asarray(b, float)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a_), len(a_))
        diffs.append(b_[idx].mean() - a_[idx].mean())
    return float(b_.mean() - a_.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ── Configurations ────────────────────────────────────────────────────────────

CONFIGS = [
    "production_approx", "bm25", "dense_lsa", "rrf_bm25_dense",
    "rrf_prod_dense", "rrf_prod_bm25", "hybrid_all", "hybrid_all_gated",
]


def run_configs(index: SectionIndex, dense: Optional[LsaBackend], raw_question: str) -> dict[str, list[int]]:
    """Top-K index rows for each database-path configuration for one question."""
    expanded = rl.expand_query(raw_question)
    prod = production_ranking(index, raw_question)
    bm25_rank = top_ranking(bm25_scores(index, expanded), POOL)
    out = {
        "production_approx": prod["final"][:K],
        "bm25": bm25_rank[:K],
    }
    if dense is None:
        return out
    dense_rank = top_ranking(dense.scores(expanded), POOL, min_score=-1.0)
    out["dense_lsa"] = dense_rank[:K]
    out["rrf_bm25_dense"] = [r for r, _ in rrf_fuse([bm25_rank, dense_rank])[:K]]
    out["rrf_prod_dense"] = [r for r, _ in rrf_fuse([prod["ranked"][:POOL], dense_rank])[:K]]
    out["rrf_prod_bm25"] = [r for r, _ in rrf_fuse([prod["ranked"][:POOL], bm25_rank])[:K]]
    fused = rrf_fuse([prod["ranked"][:POOL], bm25_rank, dense_rank])
    out["hybrid_all"] = [r for r, _ in fused[:K]]
    tops = [set(prod["ranked"][:CONSENSUS_TOPN]), set(bm25_rank[:CONSENSUS_TOPN]), set(dense_rank[:CONSENSUS_TOPN])]
    gated = [r for r, _ in fused if sum(r in t for t in tops) >= 2]
    out["hybrid_all_gated"] = gated[:K]
    return out


def rows_to_refs(index: SectionIndex, rows: list[int]) -> list[Ref]:
    return [(index.docs[r]["act_name"], str(index.docs[r]["section_number"])) for r in rows]


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(index: SectionIndex, dense: Optional[LsaBackend], gold: dict) -> dict:
    ranks: dict[str, list[Optional[int]]] = defaultdict(list)
    ctx_hit: dict[str, list[int]] = defaultdict(list)
    per_q: list[dict] = []
    t_total = []
    for q in gold["questions"]:
        exp = expected_pairs(gold, q)
        t0 = time.perf_counter()
        res = run_configs(index, dense, q["question"])
        t_total.append((time.perf_counter() - t0) * 1000)
        kw = python_corpus_refs(q["question"], limit=PROD_LIMIT)
        kw_rank = first_hit_rank(kw, exp)
        ranks["keyword_corpus_only"].append(kw_rank)
        ctx_hit["keyword_corpus_only"].append(1 if kw_rank else 0)
        row = {"id": q["id"], "question": q["question"], "expected": q["expected"],
               "ranks": {"keyword_corpus_only": kw_rank}, "top": {"keyword_corpus_only": [f"{a[:36]} s.{s}" for a, s in kw]}}
        for cfg, rows in res.items():
            refs = rows_to_refs(index, rows)
            r = first_hit_rank(refs, exp)
            ranks[cfg].append(r)
            union = kw + refs[:PROD_LIMIT]
            ctx_hit[cfg].append(1 if first_hit_rank(union, exp, k=len(union)) else 0)
            row["ranks"][cfg] = r
            row["top"][cfg] = [f"{a[:36]} s.{s}" for a, s in refs[:3]]
        per_q.append(row)

    answered: dict[str, int] = defaultdict(int)
    for q in gold.get("negatives", []):
        kw = python_corpus_refs(q["question"], limit=PROD_LIMIT)
        if kw:
            answered["keyword_corpus_only"] += 1
        res = run_configs(index, dense, q["question"])
        for cfg, rows in res.items():
            if rows or kw:           # context = keyword hits + this config's hits
                answered[cfg] += 1
        row_neg = {"id": q["id"], "question": q["question"], "answered_by": [c for c, rows in res.items() if rows]}
        per_q.append(row_neg)
    n_neg = len(gold.get("negatives", []))

    summary = {cfg: summarize(r) for cfg, r in ranks.items()}
    ind5 = {cfg: [1 if (r is not None and r <= 5) else 0 for r in rs] for cfg, rs in ranks.items()}
    for cfg in summary:
        lo, hi = bootstrap_ci(ind5[cfg])
        summary[cfg]["hit@5_ci95"] = [round(lo, 3), round(hi, 3)]
        summary[cfg]["context_hit"] = sum(ctx_hit[cfg])
        summary[cfg]["answered_out_of_scope"] = f"{answered.get(cfg, 0)}/{n_neg}"
    diffs = {}
    base = "production_approx"
    if base in ctx_hit:
        for cfg in summary:
            if cfg in (base, "keyword_corpus_only"):
                continue
            diffs[cfg] = {
                "hit@3_diff": [round(x, 3) for x in paired_bootstrap_diff(
                    [1 if (r is not None and r <= 3) else 0 for r in ranks[base]],
                    [1 if (r is not None and r <= 3) else 0 for r in ranks[cfg]])],
                "context_hit_diff": [round(x, 3) for x in paired_bootstrap_diff(ctx_hit[base], ctx_hit[cfg])],
            }
    return {
        "k": K,
        "prod_limit": PROD_LIMIT,
        "n_questions": len(gold["questions"]),
        "n_negatives": n_neg,
        "index_sections": len(index),
        "summary": summary,
        "diff_vs_production_approx": diffs,
        "mean_ms_per_question_all_configs": round(float(np.mean(t_total)), 1),
        "per_question": per_q,
    }


def to_markdown(rep: dict) -> str:
    n = rep["n_questions"]
    lines = [
        f"# Retrieval evaluation ({n} questions, {rep['n_negatives']} out-of-scope)",
        "",
        f"Database-path index: {rep['index_sections']:,} live CENTRAL sections. Production limit = {rep['prod_limit']} per retriever.",
        "",
        "| Config | hit@1 | hit@3 | hit@5 | hit@5 95% CI | MRR@5 | context hit (keyword corpus + top 3) | answered out-of-scope |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cfg in ["keyword_corpus_only"] + CONFIGS:
        s = rep["summary"].get(cfg)
        if not s:
            continue
        lines.append(
            f"| {cfg} | {s['hit@1']}/{n} | {s['hit@3']}/{n} | {s['hit@5']}/{n} | "
            f"{s['hit@5_ci95'][0]:.2f}-{s['hit@5_ci95'][1]:.2f} | {s['mrr@5']} | {s['context_hit']}/{n} | {s['answered_out_of_scope']} |"
        )
    lines += ["", "Paired difference vs production_approx, mean (95% bootstrap interval). An interval that includes 0 means the gap could be noise:", "",
              "| Config | hit@3 diff | context-hit diff |", "|---|---|---|"]
    for cfg, d in rep["diff_vs_production_approx"].items():
        a, b = d["hit@3_diff"], d["context_hit_diff"]
        lines.append(f"| {cfg} | {a[0]:+.2f} ({a[1]:+.2f} to {a[2]:+.2f}) | {b[0]:+.2f} ({b[1]:+.2f} to {b[2]:+.2f}) |")
    lines += ["", "## Questions the answer model would see NO expected section for (production_approx + keyword corpus)", ""]
    for q in rep["per_question"]:
        if "ranks" not in q:
            continue
        prod_rank = q["ranks"].get("production_approx")
        kw_rank = q["ranks"].get("keyword_corpus_only")
        if (prod_rank is None or prod_rank > rep["prod_limit"]) and not kw_rank:
            lines.append(f"- {q['id']}: {q['question']} (expected {', '.join(q['expected'])}; hybrid_all_gated rank: {q['ranks'].get('hybrid_all_gated')})")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the dense (LSA) model")
    ap.add_argument("--out", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)

    t0 = time.time()
    docs = load_central_live_sections()
    index = SectionIndex(docs)
    print(f"index built: {len(index):,} sections, {len(index.vocab):,} terms ({time.time()-t0:.0f}s)")
    dense = None
    if not args.quick:
        t1 = time.time()
        dense = LsaBackend().fit(index)
        print(f"LSA fitted ({time.time()-t1:.0f}s)")
    gold = load_gold()
    validate_gold(index, gold)
    rep = evaluate(index, dense, gold)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(rep)
    (out / "latest.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
