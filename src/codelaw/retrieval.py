"""Hybrid retrieval: lexical entry points plus provider vectors and graph expansion."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    metadata: dict[str, str]


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class GraphExpander(Protocol):
    def expand(self, document_ids: list[str]) -> list[Document]: ...


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vector dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class DenseIndex:
    """Small dependency-free vector index for reproducible local benchmarks."""

    def __init__(self, documents: list[Document], vectors: list[list[float]]):
        if len(documents) != len(vectors):
            raise ValueError("Dense index documents and vectors must have equal length")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1:
            raise ValueError("Dense index vectors must have one dimension")
        self.documents = documents
        self.vectors = vectors

    def search(self, query_vector: list[float], limit: int = 10) -> list[tuple[Document, float]]:
        scored = [
            (document, _cosine(query_vector, vector))
            for document, vector in zip(self.documents, self.vectors)
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


class BM25Index:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.tokens = [_terms(item.text) for item in documents]
        self.avg_len = sum(map(len, self.tokens)) / max(1, len(self.tokens))
        self.df = Counter(term for tokens in self.tokens for term in set(tokens))

    def search(self, query: str, limit: int = 10) -> list[tuple[Document, float]]:
        terms = _terms(query)
        scored = []
        for document, tokens in zip(self.documents, self.tokens):
            tf = Counter(tokens)
            score = 0.0
            for term in terms:
                if not tf[term]:
                    continue
                idf = math.log(1 + (len(self.documents) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * tf[term] * 2.2 / (tf[term] + 1.2 * (1 - 0.75 + 0.75 * len(tokens) / max(1, self.avg_len)))
            if score:
                scored.append((document, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


class HybridRetriever:
    def __init__(
        self,
        lexical: BM25Index,
        graph: GraphExpander,
        dense: DenseIndex | None = None,
        embedder: Embedder | None = None,
    ):
        if dense is not None and embedder is None:
            raise ValueError("A dense index requires an embedder")
        self.lexical, self.graph, self.dense, self.embedder = lexical, graph, dense, embedder

    def retrieve(self, query: str, limit: int = 5) -> list[Document]:
        lexical_rows = self.lexical.search(query, limit)
        entries = [document for document, _ in lexical_rows]
        dense_rows = []
        if self.dense is not None and self.embedder is not None:
            dense_rows = self.dense.search(self.embedder.embed([query])[0], limit)
        seed_ids = list(dict.fromkeys([
            *(document.document_id for document in entries),
            *(document.document_id for document, _ in dense_rows),
        ]))
        related = self.graph.expand(seed_ids)
        if not dense_rows:
            deduped = {document.document_id: document for document in [*entries, *related]}
            return list(deduped.values())
        candidates = {
            document.document_id: document
            for document in [*entries, *(document for document, _ in dense_rows), *related]
        }
        scores: dict[str, float] = {document_id: 0.0 for document_id in candidates}
        for rank, (document, _) in enumerate(lexical_rows, 1):
            scores[document.document_id] += 1 / (60 + rank)
        for rank, (document, _) in enumerate(dense_rows, 1):
            scores[document.document_id] += 1 / (60 + rank)
        for rank, document in enumerate(related, 1):
            scores[document.document_id] += 1 / (60 + rank)
        return [
            candidates[document_id]
            for document_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        ]
