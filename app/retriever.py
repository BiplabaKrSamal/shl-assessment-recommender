"""
Hybrid Retriever: BM25 (lexical) + FAISS (semantic) with Reciprocal Rank Fusion.

WHY HYBRID?
- BM25 wins on exact keyword matches ("Java 8", "OPQ32", specific test names).
- Semantic wins on intent ("assess emotional resilience of a sales manager").
- RRF fusion captures both signals without needing to tune score scales.

Design decision: We use sentence-transformers/all-MiniLM-L6-v2 (22M params, ~80ms/query
on CPU) rather than a larger model. Tradeoff: slightly lower semantic quality vs. staying
well under the 30-second API timeout even on cold Render.com free tier.

Usage:
    retriever = CatalogRetriever("data/catalog.json")
    results = retriever.search("Java developer mid-level stakeholder communication", k=10)
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Lazy imports – only loaded once at module level after first use
# ---------------------------------------------------------------------------
_sentence_transformer = None
_faiss = None


def _get_encoder():
    global _sentence_transformer
    if _sentence_transformer is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="/tmp/st_cache",
        )
    return _sentence_transformer


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss as _faiss_lib
        _faiss = _faiss_lib
    return _faiss


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can i we you they he she it "
    "of in on at to for with by from as into through during before after "
    "above below between and or not but nor so yet also just only very "
    "more most some any all each every both few more no than then there "
    "their its our your my".split()
)


def _tokenize(text: str) -> list[str]:
    """Lower-case, remove punctuation, remove stopwords."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w and w not in _STOPWORDS]


def _build_document_text(assessment: dict) -> str:
    """
    Concatenate fields into a single rich text blob for indexing.
    Field weighting via repetition: name × 3, test_type_label × 2, description × 1.
    """
    name = assessment.get("name", "")
    description = assessment.get("description", "")
    test_type = assessment.get("test_type", "")
    test_types = assessment.get("test_types", [])

    type_labels = {
        "A": "ability aptitude reasoning numerical verbal inductive deductive",
        "B": "situational judgement biodata behavior scenarios",
        "C": "competencies leadership management",
        "D": "development 360 feedback",
        "E": "exercise simulation assessment center",
        "K": "knowledge skills technical programming coding",
        "M": "motivation engagement values",
        "P": "personality behavior traits occupational",
        "S": "simulation interactive realistic job preview",
    }

    type_text = " ".join(type_labels.get(t, "") for t in (test_types or [test_type]))

    return f"{name} {name} {name} {type_text} {type_text} {description}"


# ---------------------------------------------------------------------------
# BM25 (pure Python, no external state needed)
# ---------------------------------------------------------------------------
class BM25:
    """Okapi BM25 implementation over tokenized documents."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n = len(documents)
        self.avgdl = sum(len(d) for d in documents) / max(self.n, 1)

        # term → {doc_idx: freq}
        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = {}

        for doc in documents:
            freq: dict[str, int] = {}
            for token in doc:
                freq[token] = freq.get(token, 0) + 1
            self.tf.append(freq)
            for token in freq:
                df[token] = df.get(token, 0) + 1

        import math
        self.idf: dict[str, float] = {
            t: math.log((self.n - f + 0.5) / (f + 0.5) + 1)
            for t, f in df.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        result = np.zeros(self.n, dtype=np.float32)
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            for i, tf_doc in enumerate(self.tf):
                f = tf_doc.get(token, 0)
                if f == 0:
                    continue
                dl = sum(tf_doc.values())
                tf_norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                result[i] += idf * tf_norm
        return result


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------
class CatalogRetriever:
    """
    Hybrid retriever combining BM25 and dense FAISS search via RRF.

    Args:
        catalog_path: Path to catalog.json
        rrf_k: RRF constant (default 60, standard value from literature)
        bm25_weight: Weight for BM25 rank contribution (0-1)
        semantic_weight: Weight for semantic rank contribution (0-1)
    """

    def __init__(
        self,
        catalog_path: str = "data/catalog.json",
        rrf_k: int = 60,
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ):
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

        # Load catalog
        raw = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        self.catalog: list[dict] = raw
        self.n = len(self.catalog)

        # Build document text blobs
        self.doc_texts = [_build_document_text(a) for a in self.catalog]

        # Build BM25 index
        tokenized = [_tokenize(t) for t in self.doc_texts]
        self.bm25 = BM25(tokenized)
        self._tokenized_docs = tokenized

        # Build FAISS index (lazy – built on first search call)
        self._faiss_index = None
        self._embeddings: np.ndarray | None = None

    def _build_faiss(self) -> None:
        """Build FAISS flat L2 index. Called once on first semantic search."""
        faiss = _get_faiss()
        encoder = _get_encoder()

        embeddings = encoder.encode(
            self.doc_texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product == cosine similarity on normalized vecs
        index.add(embeddings)

        self._embeddings = embeddings
        self._faiss_index = index

    def _semantic_scores(self, query: str) -> np.ndarray:
        if self._faiss_index is None:
            self._build_faiss()

        encoder = _get_encoder()
        q_emb = encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        scores, indices = self._faiss_index.search(q_emb, self.n)
        result = np.zeros(self.n, dtype=np.float32)
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            result[idx] = score
        return result

    def _rrf_fusion(
        self, bm25_scores: np.ndarray, semantic_scores: np.ndarray
    ) -> np.ndarray:
        """
        Reciprocal Rank Fusion.
        For each document: RRF_score = w_b / (k + rank_bm25) + w_s / (k + rank_semantic)
        """
        k = self.rrf_k

        bm25_ranks = np.argsort(-bm25_scores)
        sem_ranks = np.argsort(-semantic_scores)

        bm25_rank_pos = np.empty(self.n, dtype=np.float32)
        sem_rank_pos = np.empty(self.n, dtype=np.float32)
        bm25_rank_pos[bm25_ranks] = np.arange(self.n)
        sem_rank_pos[sem_ranks] = np.arange(self.n)

        rrf = (
            self.bm25_weight / (k + bm25_rank_pos)
            + self.semantic_weight / (k + sem_rank_pos)
        )
        return rrf

    def search(
        self,
        query: str,
        k: int = 10,
        filter_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search catalog for assessments matching query.

        Args:
            query: Natural language query
            k: Number of results to return (max 10)
            filter_types: Optional list of test_type codes to restrict results

        Returns:
            List of assessment dicts with added 'score' field, sorted by relevance.
        """
        k = min(k, 10)

        query_tokens = _tokenize(query)
        bm25_scores = self.bm25.scores(query_tokens)
        semantic_scores = self._semantic_scores(query)
        rrf_scores = self._rrf_fusion(bm25_scores, semantic_scores)

        # Apply type filter before selecting top-k
        if filter_types:
            filter_set = set(t.upper() for t in filter_types)
            mask = np.array(
                [
                    bool(set(a.get("test_types", [a.get("test_type", "")])) & filter_set)
                    for a in self.catalog
                ],
                dtype=bool,
            )
            rrf_scores = np.where(mask, rrf_scores, -1.0)

        top_indices = np.argsort(-rrf_scores)[:k]

        results = []
        for idx in top_indices:
            if rrf_scores[idx] <= 0:
                continue
            item = {**self.catalog[idx], "score": float(rrf_scores[idx])}
            results.append(item)

        return results

    def get_by_name(self, name: str) -> dict | None:
        """Exact or fuzzy name lookup for comparison queries."""
        name_lower = name.lower().strip()
        # Exact match first
        for a in self.catalog:
            if a["name"].lower() == name_lower:
                return a
        # Substring match
        for a in self.catalog:
            if name_lower in a["name"].lower() or a["name"].lower() in name_lower:
                return a
        return None

    def get_all(self) -> list[dict]:
        return self.catalog
