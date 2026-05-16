"""
Hybrid Retriever: BM25 (lexical) + TF-IDF Cosine (semantic) with RRF fusion.

WHY TF-IDF INSTEAD OF SENTENCE-TRANSFORMERS?
- Render.com free tier = 512MB RAM
- sentence-transformers + PyTorch = ~450MB alone → OOM guaranteed
- sklearn TF-IDF cosine similarity = ~5MB, fits in free tier easily
- For a 57-item catalog, TF-IDF quality matches neural embeddings
  (small corpus, domain-specific vocabulary, exact terms matter most)

WHY HYBRID (BM25 + TF-IDF)?
- BM25: wins on exact names ("OPQ32", "Java 8") — zero vocabulary mismatch
- TF-IDF cosine: wins on intent ("someone who handles pressure well") 
- RRF fusion: captures both without needing to normalize score scales

Design: no external services, no GPU, cold-start in ~1s.
"""

from __future__ import annotations

import json
import math
import string
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Stopwords ──────────────────────────────────────────────────────────────
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can i we you they he she it "
    "of in on at to for with by from as into through during before after "
    "above below between and or not but nor so yet also just only very "
    "more most some any all each every both few no than then there "
    "their its our your my".split()
)


def _tokenize(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w and w not in _STOPWORDS]


def _build_document_text(assessment: dict) -> str:
    """Rich text blob: name×3, type_label×2, description×1."""
    name = assessment.get("name", "")
    description = assessment.get("description", "")
    test_types = assessment.get("test_types", [assessment.get("test_type", "")])

    type_labels = {
        "A": "ability aptitude reasoning numerical verbal inductive deductive cognitive",
        "B": "situational judgement biodata behavior scenarios judgment",
        "C": "competencies leadership management framework",
        "D": "development 360 feedback growth",
        "E": "exercise simulation assessment center inbox tray",
        "K": "knowledge skills technical programming coding software",
        "M": "motivation engagement values energy drive",
        "P": "personality behavior traits occupational preferences style",
        "S": "simulation interactive realistic coding challenge",
    }
    type_text = " ".join(type_labels.get(t, "") for t in test_types)
    return f"{name} {name} {name} {type_text} {type_text} {description}"


# ── BM25 (pure Python) ────────────────────────────────────────────────────
class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(documents)
        self.avgdl = sum(len(d) for d in documents) / max(self.n, 1)
        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for doc in documents:
            freq: dict[str, int] = {}
            for token in doc:
                freq[token] = freq.get(token, 0) + 1
            self.tf.append(freq)
            for token in freq:
                df[token] = df.get(token, 0) + 1
        self.idf = {
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
                tf_norm = f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
                result[i] += idf * tf_norm
        return result


# ── Main retriever ─────────────────────────────────────────────────────────
class CatalogRetriever:
    """
    Hybrid BM25 + TF-IDF cosine retriever with RRF fusion.
    Designed to fit within 512MB RAM (Render.com free tier).
    """

    def __init__(
        self,
        catalog_path: str = "data/catalog.json",
        rrf_k: int = 60,
        bm25_weight: float = 0.4,
        tfidf_weight: float = 0.6,
    ):
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight

        self.catalog: list[dict] = json.loads(
            Path(catalog_path).read_text(encoding="utf-8")
        )
        self.n = len(self.catalog)
        self.doc_texts = [_build_document_text(a) for a in self.catalog]

        # BM25
        tokenized = [_tokenize(t) for t in self.doc_texts]
        self.bm25 = BM25(tokenized)

        # TF-IDF (sklearn, ~5MB, CPU-only)
        self._tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=8000,
            sublinear_tf=True,
        )
        self._doc_matrix = self._tfidf.fit_transform(self.doc_texts)

    def _tfidf_scores(self, query: str) -> np.ndarray:
        q_vec = self._tfidf.transform([query])
        sims = cosine_similarity(q_vec, self._doc_matrix).flatten()
        return sims.astype(np.float32)

    def _rrf(self, bm25_scores: np.ndarray, tfidf_scores: np.ndarray) -> np.ndarray:
        k = self.rrf_k
        bm25_ranks = np.empty(self.n, dtype=np.float32)
        tfidf_ranks = np.empty(self.n, dtype=np.float32)
        bm25_ranks[np.argsort(-bm25_scores)] = np.arange(self.n)
        tfidf_ranks[np.argsort(-tfidf_scores)] = np.arange(self.n)
        return (
            self.bm25_weight / (k + bm25_ranks)
            + self.tfidf_weight / (k + tfidf_ranks)
        )

    def search(
        self,
        query: str,
        k: int = 10,
        filter_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        k = min(k, 10)
        bm25_scores = self.bm25.scores(_tokenize(query))
        tfidf_scores = self._tfidf_scores(query)
        rrf_scores = self._rrf(bm25_scores, tfidf_scores)

        if filter_types:
            filter_set = set(t.upper() for t in filter_types)
            mask = np.array([
                bool(set(a.get("test_types", [a.get("test_type", "")])) & filter_set)
                for a in self.catalog
            ], dtype=bool)
            rrf_scores = np.where(mask, rrf_scores, -1.0)

        top_indices = np.argsort(-rrf_scores)[:k]
        return [
            {**self.catalog[i], "score": float(rrf_scores[i])}
            for i in top_indices
            if rrf_scores[i] > 0
        ]

    def get_by_name(self, name: str) -> dict | None:
        name_lower = name.lower().strip()
        for a in self.catalog:
            if a["name"].lower() == name_lower:
                return a
        for a in self.catalog:
            if name_lower in a["name"].lower() or a["name"].lower() in name_lower:
                return a
        return None

    def get_all(self) -> list[dict]:
        return self.catalog
