#!/usr/bin/env python3
"""Embed the downloaded authority and LegalBench-RAG corpus on the configured endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codelaw.config import SettingsError, load_settings  # noqa: E402
from codelaw.nvidia import embedding_client  # noqa: E402


def _chunks(text: str, *, max_chars: int, overlap: int) -> Iterator[str]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        chunk = normalized[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(normalized):
            break
        start = end - overlap


def _iter_documents(authority_path: Path, rag_root: Path, *, max_chars: int, overlap: int, limit: int | None) -> Iterator[dict[str, object]]:
    yielded = 0
    if authority_path.exists():
        with authority_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                metadata = {
                    "authority_id": str(row["authority_id"]),
                    "citation": str(row.get("citation", "")),
                    "source_url": str(row.get("source_url", "")),
                    "issue_date": str(row.get("issue_date", "")),
                    "jurisdiction": "US",
                }
                for chunk_index, chunk in enumerate(_chunks(str(row["text"]), max_chars=max_chars, overlap=overlap)):
                    if limit is not None and yielded >= limit:
                        return
                    yielded += 1
                    yield {
                        "document_id": f"{row['authority_id']}:chunk:{chunk_index}",
                        "text": chunk,
                        "metadata": metadata,
                    }
    if not rag_root.exists():
        return
    for path in sorted((rag_root / "corpus").rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(rag_root).as_posix()
        for chunk_index, chunk in enumerate(_chunks(text, max_chars=max_chars, overlap=overlap)):
            if limit is not None and yielded >= limit:
                return
            yielded += 1
            yield {
                "document_id": f"legalbench-rag:{relative}:{chunk_index}",
                "text": chunk,
                "metadata": {
                    "source_path": relative,
                    "jurisdiction": "US",
                },
            }


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, default=Path("data/processed/authority/authority.jsonl"))
    parser.add_argument("--rag-root", type=Path, default=Path("data/processed/legalbench-rag"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/embedding-index"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--limit", type=int, help="Optional smoke-test limit on indexed chunks")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_chars < 100 or not 0 <= args.overlap < args.max_chars:
        print("embedding_index=blocked: batch-size, max-chars, or overlap is invalid")
        return 2
    try:
        settings = load_settings()
        client = embedding_client(settings)
        rows = _iter_documents(
            args.authority,
            args.rag_root,
            max_chars=args.max_chars,
            overlap=args.overlap,
            limit=args.limit,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "index.jsonl"
        count = 0
        dimension = 0
        source_counts: dict[str, int] = {}
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            batch: list[dict[str, object]] = []

            def flush() -> None:
                nonlocal count, dimension, batch
                if not batch:
                    return
                vectors = client.embed([str(row["text"]) for row in batch], input_type='passage')
                if len(vectors) != len(batch):
                    raise RuntimeError("Embedding endpoint returned a different number of vectors")
                for row, vector in zip(batch, vectors):
                    dimension = len(vector)
                    row["vector"] = vector
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
                    source = str(row["document_id"]).split(":", 1)[0]
                    source_counts[source] = source_counts.get(source, 0) + 1
                if count % 512 == 0:
                    print(f"embedded={count}", file=sys.stderr, flush=True)
                batch = []

            for row in rows:
                batch.append(row)
                if len(batch) >= args.batch_size:
                    flush()
            flush()
    except (SettingsError, OSError, ValueError, RuntimeError) as exc:
        print(f"embedding_index=failed: {exc}")
        return 1
    manifest = {
        "schema_version": "1.0.0",
        "index_path": str(output_path),
        "records": count,
        "dimensions": dimension,
        "source_counts": source_counts,
        "embedding_model": settings.embedding_model,
        "embedding_format": {"document_prefix": "passage: ", "query_prefix": "query: "} if settings.embedding_model.endswith('Nemotron-3-Embed-1B-BF16') else {},
        "embedding_base_url": settings.embedding_base_url,
        "authority_sha256": _sha256(args.authority),
        "rag_path_map_sha256": _sha256(args.rag_root / "path-map.json"),
        "max_chars": args.max_chars,
        "overlap": args.overlap,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
