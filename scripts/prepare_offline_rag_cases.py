#!/usr/bin/env python3
"""Attach common private-network vector evidence to selected benchmark cases."""

from __future__ import annotations

import argparse
import json
import sys
from array import array
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codelaw.benchmark import select_cases  # noqa: E402
from codelaw.config import load_settings  # noqa: E402
from codelaw.nvidia import embedding_client  # noqa: E402


def _retrieval_evidence(case: dict[str, Any], hits: list[dict[str, Any]], *, max_chars: int) -> list[dict[str, Any]]:
    existing = {str(item.get("evidence_id")) for item in case.get("evidence", [])}
    evidence: list[dict[str, Any]] = []
    for hit in hits:
        evidence_id = f"retrieval:{hit['document_id']}"
        if evidence_id in existing:
            continue
        metadata = hit.get("metadata", {})
        evidence.append({
            "evidence_id": evidence_id,
            "source_id": str(metadata.get("authority_id") or hit["document_id"]),
            "text": str(hit["text"])[:max_chars],
            "jurisdiction": str(metadata.get("jurisdiction") or case.get("jurisdiction") or "US"),
            "effective_on": metadata.get("issue_date"),
            "citation": metadata.get("citation", ""),
            "retrieval_score": round(float(hit["score"]), 6),
            "retrieval_source": str(hit["document_id"]).split(":", 1)[0],
        })
    return evidence


def _load_index(path: Path) -> tuple[list[dict[str, Any]], Any]:
    try:
        import numpy as np
    except ImportError as exc:  # explicit runtime prerequisite for the 1.76 GB index
        raise RuntimeError("offline RAG preparation requires numpy") from exc
    rows: list[dict[str, Any]] = []
    values = array("f")
    dimensions: int | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            vector = row.pop("vector")
            if dimensions is None:
                dimensions = len(vector)
            if not dimensions or len(vector) != dimensions:
                raise ValueError("embedding index has inconsistent vector dimensions")
            values.extend(float(value) for value in vector)
            rows.append(row)
    if not rows or not dimensions:
        raise ValueError("embedding index is empty")
    matrix = np.frombuffer(values, dtype=np.float32).reshape(len(rows), dimensions)
    norms = np.linalg.norm(matrix, axis=1)
    matrix /= np.maximum(norms, 1e-12)[:, None]
    return rows, matrix


def _top_hits(rows: list[dict[str, Any]], matrix: Any, queries: list[list[float]], *, per_source: int) -> list[list[dict[str, Any]]]:
    import numpy as np

    query_matrix = np.asarray(queries, dtype=np.float32)
    if query_matrix.ndim != 2 or query_matrix.shape[1] != matrix.shape[1]:
        raise ValueError("query and index vector dimensions do not match")
    query_matrix /= np.maximum(np.linalg.norm(query_matrix, axis=1), 1e-12)[:, None]
    scores = matrix @ query_matrix.T
    output: list[list[dict[str, Any]]] = []
    source_limit = max(1, per_source)
    for column in range(scores.shape[1]):
        candidate_count = min(len(rows), source_limit * 24)
        candidates = np.argpartition(scores[:, column], -candidate_count)[-candidate_count:]
        ranked = sorted((int(index) for index in candidates), key=lambda index: float(scores[index, column]), reverse=True)
        selected: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for index in ranked:
            row = rows[index]
            source = str(row["document_id"]).split(":", 1)[0]
            if counts.get(source, 0) >= source_limit:
                continue
            counts[source] = counts.get(source, 0) + 1
            selected.append({**row, "score": float(scores[index, column])})
        output.append(selected)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("data/processed/benchmark/cases.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/embedding-index/index.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--source", action="append")
    parser.add_argument("--per-source", type=int, default=2)
    parser.add_argument("--snippet-chars", type=int, default=800)
    args = parser.parse_args()
    if args.limit < 1 or args.per_source < 1 or args.snippet_chars < 1:
        parser.error("--limit, --per-source, and --snippet-chars must be positive")
    sources = [item.strip() for value in (args.source or []) for item in value.split(",") if item.strip()] or None
    cases = select_cases(args.cases, limit=args.limit, sources=sources)
    if not cases:
        print("offline_rag=blocked: no benchmark cases matched the selection")
        return 2
    if not args.index.exists():
        print(f"offline_rag=blocked: embedding index does not exist: {args.index}")
        return 2
    try:
        rows, matrix = _load_index(args.index)
        queries = embedding_client(load_settings()).embed([
            f"{case.get('task', '')}\n{case.get('prompt', '')}" for case in cases
        ])
        hits_by_case = _top_hits(rows, matrix, queries, per_source=args.per_source)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"offline_rag=failed: {exc}")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for case, hits in zip(cases, hits_by_case):
            augmented = dict(case)
            augmented["evidence"] = [
                *_retrieval_evidence(case, hits, max_chars=args.snippet_chars),
                *case.get("evidence", []),
            ]
            augmented["offline_rag"] = {
                "index": str(args.index),
                "per_source": args.per_source,
                "retrieved_evidence_ids": [item["evidence_id"] for item in augmented["evidence"] if str(item["evidence_id"]).startswith("retrieval:")],
            }
            output.write(json.dumps(augmented, ensure_ascii=False) + "\n")
    print(json.dumps({
        "offline_rag": "ready",
        "cases": len(cases),
        "index_records": len(rows),
        "per_source": args.per_source,
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
