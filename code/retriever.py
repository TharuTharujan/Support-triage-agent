from __future__ import annotations

import math
from typing import Any

from chunker import chunk_document, tokenize
from models import Chunk, Document, RetrievalHit


class HybridRetriever:
    def __init__(self, documents: list[Document]):
        self.chunks: list[Chunk] = [
            chunk for document in documents for chunk in chunk_document(document)
        ]
        self.backend_name = "lexical"
        self._vectorizer: Any = None
        self._matrix: Any = None
        self._faiss_index: Any = None
        self._embeddings: Any = None
        self._build_optional_backends()

    def retrieve(
        self, query: str, company: str = "", limit: int = 5
    ) -> list[RetrievalHit]:
        if self._faiss_index is not None:
            hits = self._retrieve_faiss(query, company, limit)
            if hits:
                return hits
        if self._vectorizer is not None and self._matrix is not None:
            hits = self._retrieve_tfidf(query, company, limit)
            if hits:
                return hits
        return self._retrieve_lexical(query, company, limit)

    def _build_optional_backends(self) -> None:
        self._build_tfidf()
        self._build_faiss()

    def _build_tfidf(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception:
            return
        texts = [f"{chunk.document.title} {chunk.text}" for chunk in self.chunks]
        if not texts:
            return
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
        )
        self._matrix = self._vectorizer.fit_transform(texts)
        self.backend_name = "tfidf"

    def _build_faiss(self) -> None:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except Exception:
            return
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [f"{chunk.document.title} {chunk.text}" for chunk in self.chunks]
            embeddings = model.encode(texts, normalize_embeddings=True)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
        except Exception:
            return
        self._embedding_model = model
        self._embeddings = embeddings
        self._faiss_index = index
        self.backend_name = "faiss"

    def _retrieve_tfidf(
        self, query: str, company: str = "", limit: int = 5
    ) -> list[RetrievalHit]:
        assert self._vectorizer is not None
        assert self._matrix is not None
        query_vector = self._vectorizer.transform([query])
        scores = (self._matrix @ query_vector.T).toarray().ravel()
        return self._hits_from_scores(scores, company, limit)

    def _retrieve_faiss(
        self, query: str, company: str = "", limit: int = 5
    ) -> list[RetrievalHit]:
        if self._faiss_index is None or not hasattr(self, "_embedding_model"):
            return []
        try:
            query_embedding = self._embedding_model.encode([query], normalize_embeddings=True)
            scores, indexes = self._faiss_index.search(query_embedding, min(len(self.chunks), limit * 8))
        except Exception:
            return []
        hits: list[RetrievalHit] = []
        normalized_company = company.lower()
        for score, index in zip(scores[0], indexes[0]):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            if normalized_company and normalized_company != "none":
                if chunk.document.company != normalized_company:
                    continue
            if score <= 0:
                continue
            hits.append(RetrievalHit(float(score), chunk))
            if len(hits) >= limit:
                break
        return hits

    def _hits_from_scores(
        self, scores: Any, company: str, limit: int
    ) -> list[RetrievalHit]:
        normalized_company = company.lower()
        hits: list[RetrievalHit] = []
        for index, score in enumerate(scores):
            if score <= 0:
                continue
            chunk = self.chunks[index]
            if normalized_company and normalized_company != "none":
                if chunk.document.company != normalized_company:
                    continue
            hits.append(RetrievalHit(float(score), chunk))
        return sorted(
            hits,
            key=lambda hit: (-hit.score, str(hit.chunk.document.path), hit.chunk.index),
        )[:limit]

    def _retrieve_lexical(
        self, query: str, company: str = "", limit: int = 5
    ) -> list[RetrievalHit]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []

        normalized_company = company.lower()
        hits: list[RetrievalHit] = []
        for chunk in self.chunks:
            if normalized_company and normalized_company != "none":
                if chunk.document.company != normalized_company:
                    continue
            overlap = query_tokens & chunk.tokens
            if not overlap:
                continue
            base = len(overlap) / math.sqrt(len(query_tokens) * max(len(chunk.tokens), 1))
            title_bonus = 0.04 * len(overlap & set(tokenize(chunk.document.title)))
            path_bonus = 0.02 * len(overlap & set(tokenize(str(chunk.document.path))))
            hits.append(RetrievalHit(base + title_bonus + path_bonus, chunk))
        return sorted(
            hits,
            key=lambda hit: (-hit.score, str(hit.chunk.document.path), hit.chunk.index),
        )[:limit]


LexicalRetriever = HybridRetriever
