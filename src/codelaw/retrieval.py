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
    def __init__(self, lexical: BM25Index, graph: GraphExpander):
        self.lexical, self.graph = lexical, graph

    def retrieve(self, query: str, limit: int = 5) -> list[Document]:
        entries = [document for document, _ in self.lexical.search(query, limit)]
        related = self.graph.expand([document.document_id for document in entries])
        deduped = {document.document_id: document for document in [*entries, *related]}
        return list(deduped.values())
