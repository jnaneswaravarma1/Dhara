"""Offline stand-in for the production keyword retrieval path (corpus_db.retrieve_db).

Production does:  MongoDB $text candidates (top 300)  ->  corroboration re-rank
                  ->  relative top cutoff.
Only the first step needs a live MongoDB. This module replaces ONLY that step with an
approximation of Mongo's text score and reuses the REAL second and third steps from
retrieval_logic.py (corroboration_score, relative_top_cutoff, expand_query) and the
REAL act-hint patterns and pool size from corpus_db.py.

IMPORTANT LIMITATION: the $text approximation (stemming, stopwords, TF damping, field
weights 10/1 taken from the corpus_db docstring) is NOT Mongo's exact algorithm, so the
baseline numbers are an estimate of production behaviour, not a measurement of it. To
get the true number, run the same questions through the real retrieve_db against the
production database (see README.md, "Plugging in the real retriever").
"""
from __future__ import annotations

import gzip
import json
import sys
import types
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _import_corpus_db_constants():
    """Import corpus_db for its constants without needing the `motor` package."""
    try:
        import motor.motor_asyncio  # noqa: F401
    except ImportError:
        motor = types.ModuleType("motor")
        motor_asyncio = types.ModuleType("motor.motor_asyncio")
        motor_asyncio.AsyncIOMotorDatabase = object
        motor.motor_asyncio = motor_asyncio
        sys.modules["motor"] = motor
        sys.modules["motor.motor_asyncio"] = motor_asyncio
    import corpus_db
    return corpus_db


corpus_db = _import_corpus_db_constants()
import retrieval_logic as rl  # noqa: E402
from hybrid_retrieval import SectionIndex, top_ranking  # noqa: E402

SEED = BACKEND_DIR / "corpus_seed" / "legal_sections.jsonl.gz"


def load_central_live_sections(path: Path = SEED) -> list[dict]:
    """CENTRAL-jurisdiction, non-dead sections (what retrieve_db searches when no
    state code is given)."""
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("jurisdiction") == "CENTRAL" and not d.get("is_dead_law"):
                out.append(d)
    return out


def mongo_text_approx(index: SectionIndex, question: str) -> np.ndarray:
    """Approximate MongoDB $text score: heading weight 10, body weight 1, damped TF,
    OR-semantics over stemmed non-stopword query terms."""
    scores = np.zeros(len(index), dtype=np.float32)
    Hc = index.heading_tf.tocsc()
    Tc = index.text_tf.tocsc()
    for j in index.query_term_ids(question):
        for mat, weight in ((Hc, 10.0), (Tc, 1.0)):
            col = mat[:, j]
            scores[col.indices] += weight * (1.0 + np.log(col.data))
    return scores


_CITATION_RE = __import__("re").compile(r"^(?P<act>.*?),?\s+(?:Section|Rule|Article)\s+(?P<sec>[0-9]+[A-Za-z]*)")


def python_corpus_refs(question: str, limit: int = 3) -> list[tuple[str, str]]:
    """Run the REAL hand-written keyword corpus (corpus.retrieve, ~142 entries) — the
    path production runs alongside the database search — and return (act text, section)
    pairs parsed from each hit's citation, e.g. ('Bharatiya Nagarik Suraksha Sanhita 2023', '173')."""
    import corpus
    refs = []
    for hit in corpus.retrieve(question, limit=limit):
        m = _CITATION_RE.match(hit.get("citation", ""))
        if m:
            refs.append((m.group("act"), m.group("sec")))
        else:
            refs.append((hit.get("citation", ""), ""))
    return refs


def production_ranking(index: SectionIndex, raw_question: str, pool: int | None = None) -> dict:
    """Run the production pipeline shape on `raw_question`.

    Returns {"ranked": [row,...]  (all candidates that clear the corroboration floor,
                                    best first, before the relative cutoff),
             "final": [row,...]   (after the relative top cutoff, limit=5)}.
    """
    pool = pool or corpus_db._CANDIDATE_POOL_SIZE
    act_hint = None
    for hint_name, pattern in corpus_db._ACT_HINTS.items():
        if pattern.search(raw_question):
            act_hint = hint_name
            break
    question = rl.expand_query(raw_question)
    question_sig = rl.significant_words(question)

    text_scores = mongo_text_approx(index, question)
    cand_rows = top_ranking(text_scores, pool)

    cands = []
    seen = set()
    for row in cand_rows:
        doc = index.docs[row]
        key = (doc.get("act_id", ""), doc.get("section_number", ""))
        if key in seen:
            continue
        corr = rl.corroboration_score(question_sig, doc, act_hint)
        if corr >= rl.MIN_COROBORATION_WORDS:
            seen.add(key)
            cands.append((corr, 0, float(text_scores[row]), row))
    cands.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

    final = rl.relative_top_cutoff(
        [(c[0], c[3]) for c in cands],
        limit=5,
        min_floor=rl.MIN_COROBORATION_WORDS,
        sentinel=9999,
    )
    return {"ranked": [c[3] for c in cands], "final": list(final)}
