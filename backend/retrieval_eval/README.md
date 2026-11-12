# Retrieval evaluation

Measures whether the sections DHARA retrieves actually contain the right law for a
citizen's question. Runs offline: no MongoDB, no LLM key, no network.

```
cd backend
python3 -m retrieval_eval.run_eval            # full run, writes results/latest.md + latest.json
python3 -m retrieval_eval.run_eval --quick    # skip the dense model
pytest tests/test_retrieval_eval.py tests/test_gateway_routing.py --noconftest
```

## Flow

```
gold_set.json  (50 citizen questions + the section(s) that answer each, + 10 out-of-scope questions)
      |
      v
for each question, run each retriever and take its top results
   keyword corpus (corpus.retrieve)            real code, ~142 hand-written entries
   production_approx                            Mongo $text (APPROXIMATED) + real corroboration re-rank + real cutoff
   bm25                                         BM25 over heading + act name + text
   dense_lsa                                    TF-IDF + SVD (stand-in for embeddings)
   RRF fusions of the above                     Reciprocal Rank Fusion, k=60 (not tuned)
      |
      v
score: hit@1 / hit@3 / hit@5, MRR@5, "context hit" (keyword corpus + top 3, i.e. what the
answer model would see), and how often an out-of-scope question still gets an answer
      |
      v
bootstrap 95% intervals + paired differences vs production_approx  ->  results/latest.md
```

## Reading the numbers

* `production_approx` is an ESTIMATE. Only the MongoDB `$text` step is approximated
  (`baseline.mongo_text_approx`); the corroboration re-rank, the relative cutoff, the synonym
  expansion, the act-hint patterns and the keyword corpus are the real production code.
  To get the true baseline, wrap `corpus_db.retrieve_db` (needs the database) in a function that
  returns `(act_name, section_number)` pairs and use it in place of `production_ranking`.
* `dense_lsa` is NOT a neural embedding. It is weak (it clusters by Act, not by meaning), and
  it makes fusions worse. It exists to exercise the plumbing. No claim about real embeddings
  can be made from it.
* 50 questions is small. Use the intervals: a difference whose interval includes 0 may be noise.
* The gold set was written by an AI assistant and has not been reviewed by a lawyer. Two labels
  (q23, q25) were widened after the first run because a second valid section had been missed;
  the numbers did not change. Add real user questions and have them reviewed before quoting accuracy.

## Abstain gate

`HybridRetriever.search(..., min_top_bm25=14.0)` returns nothing when the best BM25 score is below
the cutoff. On the 60 questions this cut out-of-scope answers from 10/10 to 4/10 (same as today) and
cost one context hit (21 -> 20 of 50). The cutoff was hand-set from the same 60 questions, so the
result is optimistic, and the score only partly separates the two groups (AUC about 0.83). The sweep
table in results/latest.md shows the trade-off: a higher cutoff refuses more out-of-scope questions
but also refuses answerable ones. A stronger gate probably needs the query parser's legal-issue output
or a real embedding model; validate on the advocate-checked questions before relying on it.

## Plugging in a real embedding model

```python
from sentence_transformers import SentenceTransformer     # not installed in the test sandbox
from hybrid_retrieval import EncoderBackend
model = SentenceTransformer("intfloat/multilingual-e5-small")
dense = EncoderBackend(lambda t: model.encode(t, normalize_embeddings=True),
                       query_prefix="query: ", doc_prefix="passage: ").fit(index)
```
Pass `dense` into `evaluate(index, dense, gold)` (see `run_eval.main`). Everything else is unchanged.
