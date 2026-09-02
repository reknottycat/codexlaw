#!/usr/bin/env python3
"""Query the real DGX embedding index and print provenance-bearing top-k rows."""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codelaw.config import SettingsError, load_settings  # noqa: E402
from codelaw.nvidia import embedding_client  # noqa: E402
from codelaw.retrieval import _cosine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=Path("data/processed/embedding-index/index.jsonl"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--snippet-chars", type=int, default=240)
    args = parser.parse_args()
    if args.limit < 1 or args.snippet_chars < 1:
        print("embedding_query=blocked: limit and snippet-chars must be positive")
        return 2
    if not args.index.exists():
        print(f"embedding_query=blocked: index does not exist: {args.index}")
        return 2
    try:
        settings = load_settings()
        query_vector = embedding_client(settings).embed([args.query])[0]
    except (SettingsError, OSError, ValueError, RuntimeError) as exc:
        print(f"embedding_query=failed: {exc}")
        return 1
    top: list[tuple[float, int, dict[str, object]]] = []
    with args.index.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            score = _cosine(query_vector, [float(value) for value in row["vector"]])
            item = (score, row_number, row)
            if len(top) < args.limit:
                heapq.heappush(top, item)
            elif score > top[0][0]:
                heapq.heapreplace(top, item)
    results = []
    for score, _, row in sorted(top, reverse=True):
        results.append({
            "score": round(score, 6),
            "document_id": row["document_id"],
            "metadata": row.get("metadata", {}),
            "text": str(row["text"])[:args.snippet_chars],
        })
    print(json.dumps({
        "query": args.query,
        "embedding_model": settings.embedding_model,
        "dimensions": len(query_vector),
        "results": results,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
