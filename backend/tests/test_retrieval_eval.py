"""Offline tests for the retrieval test set, the eval metrics and hybrid_retrieval.

No database and no network. Builds an in-memory index from corpus_seed/ (a few seconds).
Run alone:  pytest tests/test_retrieval_eval.py --noconftest
"""
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from hybrid_retrieval import (  # noqa: E402
    EncoderBackend, HybridRetriever, SectionIndex, analyze, bm25_scores, rrf_fuse, top_ranking,
)
from retrieval_eval import run_eval  # noqa: E402
from retrieval_eval.baseline import load_central_live_sections  # noqa: E402


@pytest.fixture(scope="module")
def index():
    return SectionIndex(load_central_live_sections())


def test_gold_set_is_well_formed_and_every_expected_section_exists(index):
    gold = run_eval.load_gold()
    ids = [q["id"] for q in gold["questions"]]
    assert len(ids) == len(set(ids)) >= 50
    assert all(q["expected"] for q in gold["questions"])
    assert len(gold["negatives"]) >= 10
    run_eval.validate_gold(index, gold)  # raises if a section is missing or repealed


def test_validate_gold_rejects_a_missing_section(index):
    bad = {"acts": {"BNS": "Bharatiya Nyaya Sanhita"}, "questions": [{"id": "x", "question": "q", "expected": ["BNS:99999"]}]}
    with pytest.raises(ValueError):
        run_eval.validate_gold(index, bad)


def test_ref_matching_and_ranks():
    exp = [("Bharatiya Nyaya Sanhita", "303")]
    assert run_eval.ref_matches(("The Bharatiya Nyaya Sanhita, 2023", "303"), exp[0])
    assert not run_eval.ref_matches(("The Bharatiya Nyaya Sanhita, 2023", "304"), exp[0])
    assert not run_eval.ref_matches(("Some Other Act", "303"), exp[0])
    results = [("X Act", "1"), ("The Bharatiya Nyaya Sanhita, 2023", "303")]
    assert run_eval.first_hit_rank(results, exp) == 2
    assert run_eval.first_hit_rank(results, exp, k=1) is None


def test_summarize_counts_and_mrr():
    s = run_eval.summarize([1, 3, None, 5])
    assert (s["hit@1"], s["hit@3"], s["hit@5"]) == (1, 2, 3)
    assert s["mrr@5"] == round((1 + 1 / 3 + 0 + 1 / 5) / 4, 3)


def test_rrf_fuse_prefers_documents_ranked_well_by_several_rankers():
    fused = rrf_fuse([[10, 20, 30], [20, 10, 40], [20, 50]])
    assert [d for d, _ in fused][:2] == [20, 10]
    assert fused == rrf_fuse([[10, 20, 30], [20, 10, 40], [20, 50]])  # deterministic


def test_top_ranking_orders_and_drops_zero_scores():
    scores = np.array([0.0, 0.5, 0.9, 0.0, 0.7], dtype=np.float32)
    assert top_ranking(scores, pool=3) == [2, 4, 1]
    assert top_ranking(scores, pool=10) == [2, 4, 1]


def test_analyze_stems_and_drops_stopwords():
    assert analyze("The police were refusing to register complaints") == ["police", "refus", "register", "complaint"]


def test_bm25_finds_the_helmet_section(index):
    top = top_ranking(bm25_scores(index, "helmet"), 3)
    heads = [index.docs[r]["section_heading"] for r in top]
    assert any("headgear" in h.lower() for h in heads)


def test_encoder_backend_and_hybrid_with_a_fake_encoder(index):
    docs = index.docs[:500]
    small = SectionIndex(docs)

    def fake_encode(texts):  # hashing bag-of-words: deterministic, no model needed
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in analyze(t):
                out[i, hash(tok) % 64] += 1.0
        return out

    dense = EncoderBackend(fake_encode, batch=100).fit(small)
    assert dense._D.shape == (500, 64)
    scores = dense.scores("anything about courts")
    assert scores.shape == (500,)
    hybrid = HybridRetriever(small, dense, pool=50)
    res = hybrid.search("penalty for offences", limit=3)
    assert 1 <= len(res) <= 3 and all("_ranks" in r for r in res)
