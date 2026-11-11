"""hybrid_retrieval.py — keyword (BM25) + dense retrieval fused with Reciprocal Rank Fusion.

STATUS: prototype. Nothing in the live request path imports this module yet.
It is a self-contained, database-free retriever so it can be evaluated offline
(see retrieval_eval/) before anyone decides to wire it into corpus_db.retrieve_db.

Why this exists
---------------
Production retrieval today is MongoDB $text candidates (stemmed OR-match) followed by
the hand-written corroboration re-rank in retrieval_logic.py. That pipeline needs a
question to share >= 2 "significant" words with a section, so a paraphrase that uses
different words than the statute can fall through. A dense (embedding) ranker matches
meaning rather than words, and BM25 gives a better keyword ranking than $text.
Reciprocal Rank Fusion (RRF) combines any number of rankings without needing their
scores to be comparable, so no weights have to be tuned.

Pieces
------
* SectionIndex        - sparse term statistics over the sections (one-time build).
* bm25_scores         - BM25 ranking over heading + act name + body text.
* DenseBackend        - anything that maps text to vectors. Two implementations:
    - LsaBackend             : TF-IDF + truncated SVD. Needs only scikit-learn, runs
                               offline. A weak stand-in for real embeddings.
    - EncoderBackend         : wraps ANY encode function (sentence-transformers,
                               an ONNX model, an embeddings API). This is the one to
                               use in production; it has NOT been run in this repo's
                               test environment because no embedding model is installed.
* rrf_fuse            - Reciprocal Rank Fusion.
* HybridRetriever     - ties the above together.

Safety note: this module only RANKS. Everything downstream of ranking (dead-law
exclusion, judicial-invalidation flags, Law Status Guard, the answer-prompt locks)
stays exactly where it is. The index is built from non-dead sections only.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Iterable, Optional, Protocol, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD

_WORD_RE = re.compile(r"[a-z]{2,}")
_STOP = frozenset(ENGLISH_STOP_WORDS)


# ── Tokenisation ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    """Very small suffix stripper (enough to match 'stolen'/'steal' style variants
    only partially; it is NOT a full Porter/Snowball stemmer)."""
    w = word
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("sses"):
        return w[:-2]
    for suf in ("ings", "ing", "ed", "ly", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    return w


def analyze(text: str) -> list[str]:
    """Lowercase -> alphabetic tokens -> drop English stopwords -> stem."""
    return [stem(t) for t in _WORD_RE.findall((text or "").lower()) if t not in _STOP]


# ── Sparse index ──────────────────────────────────────────────────────────────

class SectionIndex:
    """Term statistics for a list of section dicts (needs act_name, section_heading,
    section_text). Keeps the dicts so callers can map a row number back to a section."""

    def __init__(self, docs: Sequence[dict], min_df: int = 1):
        self.docs = list(docs)
        self.vectorizer = CountVectorizer(analyzer=analyze, min_df=min_df)
        texts = [(d.get("section_text") or "")[:4000] for d in self.docs]
        self.text_tf = self.vectorizer.fit_transform(texts).tocsr().astype(np.float32)
        self.heading_tf = self.vectorizer.transform(
            [d.get("section_heading") or "" for d in self.docs]
        ).tocsr().astype(np.float32)
        self.act_tf = self.vectorizer.transform(
            [d.get("act_name") or "" for d in self.docs]
        ).tocsr().astype(np.float32)
        self.vocab = self.vectorizer.vocabulary_
        # Combined field matrix, heading counted 3x and act name 1x (BM25F-style).
        self.combined_tf = (self.text_tf + 3.0 * self.heading_tf + self.act_tf).tocsr()
        self.doc_len = np.asarray(self.combined_tf.sum(axis=1)).ravel()
        self.avg_len = float(self.doc_len.mean()) if len(self.doc_len) else 1.0
        df = np.asarray((self.combined_tf > 0).sum(axis=0)).ravel()
        n = self.combined_tf.shape[0]
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self._csc = self.combined_tf.tocsc()

    def __len__(self) -> int:
        return len(self.docs)

    def query_term_ids(self, query: str) -> list[int]:
        seen: dict[int, None] = {}
        for tok in analyze(query):
            j = self.vocab.get(tok)
            if j is not None:
                seen[j] = None
        return list(seen)


def bm25_scores(index: SectionIndex, query: str, k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """BM25 score for every section (0 where no query term matches)."""
    scores = np.zeros(len(index), dtype=np.float32)
    norm = k1 * (1.0 - b + b * index.doc_len / index.avg_len)
    for j in index.query_term_ids(query):
        col = index._csc[:, j]
        rows = col.indices
        tf = col.data
        scores[rows] += index.idf[j] * tf * (k1 + 1.0) / (tf + norm[rows])
    return scores


# ── Dense backends ────────────────────────────────────────────────────────────

class DenseBackend(Protocol):
    def fit(self, index: SectionIndex) -> "DenseBackend": ...
    def scores(self, query: str) -> np.ndarray: ...


class LsaBackend:
    """TF-IDF + truncated SVD ("latent semantic analysis").

    Offline and dependency-light, but it only captures word co-occurrence inside this
    corpus. It is a stand-in used to exercise the dense+fusion plumbing here; do not
    treat its accuracy as representative of a real multilingual embedding model.
    """

    def __init__(self, n_components: int = 256, seed: int = 0):
        self.n_components = n_components
        self.seed = seed

    def fit(self, index: SectionIndex) -> "LsaBackend":
        X = index.combined_tf
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        n = X.shape[0]
        keep = np.where((df >= 2) & (df <= 0.3 * n))[0]
        self._keep = keep
        self._vocab = index.vocab
        self._tfidf = TfidfTransformer(sublinear_tf=True)
        Xk = self._tfidf.fit_transform(X[:, keep])
        self._svd = TruncatedSVD(n_components=self.n_components, random_state=self.seed)
        D = self._svd.fit_transform(Xk)
        D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
        self._D = D.astype(np.float32)
        self._col_of = {int(j): i for i, j in enumerate(keep)}
        self._n_vocab = X.shape[1]
        return self

    def scores(self, query: str) -> np.ndarray:
        q = np.zeros((1, len(self._keep)), dtype=np.float32)
        for tok in analyze(query):
            j = self._vocab.get(tok)
            i = self._col_of.get(j) if j is not None else None
            if i is not None:
                q[0, i] += 1.0
        if not q.any():
            return np.zeros(self._D.shape[0], dtype=np.float32)
        qv = self._svd.transform(self._tfidf.transform(sparse.csr_matrix(q)))
        qv /= np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9
        return (self._D @ qv[0]).astype(np.float32)


def section_text_for_embedding(doc: dict, max_chars: int = 800) -> str:
    """The text a production embedding model should see for one section."""
    return f"{doc.get('act_name', '')}. {doc.get('section_heading', '')}. {(doc.get('section_text') or '')[:max_chars]}"


class EncoderBackend:
    """Dense retrieval with any text encoder.

    `encode(list_of_texts) -> 2-D float array`. Plug in, for example,
    SentenceTransformer("intfloat/multilingual-e5-small").encode, an ONNX session, or a
    call to an embeddings API. Vectors are L2-normalised here, so scoring is cosine.
    Pre-compute and cache the section vectors offline; only the query is encoded at
    request time.
    """

    def __init__(self, encode: Callable[[list[str]], np.ndarray], batch: int = 256,
                 query_prefix: str = "", doc_prefix: str = ""):
        self.encode = encode
        self.batch = batch
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def fit(self, index: SectionIndex) -> "EncoderBackend":
        texts = [self.doc_prefix + section_text_for_embedding(d) for d in index.docs]
        parts = [np.asarray(self.encode(texts[i:i + self.batch]), dtype=np.float32)
                 for i in range(0, len(texts), self.batch)]
        D = np.vstack(parts)
        D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
        self._D = D
        return self

    def scores(self, query: str) -> np.ndarray:
        q = np.asarray(self.encode([self.query_prefix + query]), dtype=np.float32)[0]
        q /= np.linalg.norm(q) + 1e-9
        return self._D @ q


# ── Fusion ────────────────────────────────────────────────────────────────────

def top_ranking(scores: np.ndarray, pool: int = 300, min_score: float = 0.0) -> list[int]:
    """Row numbers of the `pool` best scores, best first (scores <= min_score dropped)."""
    if pool >= len(scores):
        order = np.argsort(-scores, kind="stable")
    else:
        part = np.argpartition(-scores, pool)[:pool]
        order = part[np.argsort(-scores[part], kind="stable")]
    return [int(i) for i in order if scores[i] > min_score]


def rrf_fuse(rankings: Iterable[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank_d).
    k=60 is the value from the original paper and is NOT tuned here."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever:
    """BM25 + dense (+ optional extra rankings) fused with RRF."""

    def __init__(self, index: SectionIndex, dense: Optional[DenseBackend] = None,
                 pool: int = 300, rrf_k: int = 60):
        self.index = index
        self.dense = dense
        self.pool = pool
        self.rrf_k = rrf_k

    def rankings(self, query: str) -> dict[str, list[int]]:
        out = {"bm25": top_ranking(bm25_scores(self.index, query), self.pool)}
        if self.dense is not None:
            out["dense"] = top_ranking(self.dense.scores(query), self.pool, min_score=-1.0)
        return out

    def search(self, query: str, limit: int = 5,
               extra_rankings: Sequence[Sequence[int]] = ()) -> list[dict]:
        r = self.rankings(query)
        fused = rrf_fuse(list(r.values()) + [list(x) for x in extra_rankings], self.rrf_k)
        results = []
        for row, score in fused[:limit]:
            doc = self.index.docs[row]
            results.append({
                "act_name": doc.get("act_name"),
                "section_number": doc.get("section_number"),
                "section_heading": doc.get("section_heading"),
                "_row": row,
                "_rrf": score,
                "_ranks": {name: (lst.index(row) + 1 if row in lst else None) for name, lst in r.items()},
            })
        return results
