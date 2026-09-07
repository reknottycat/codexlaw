"""Re-encode existing document chunks without redownloading or altering v1."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from codelaw.config import load_settings
from codelaw.experiment import file_hash
from codelaw.nvidia import embedding_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'data/processed/embedding-index/index.jsonl')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/embedding-index-v2')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('batch-size and limit must be positive')
    if args.output_dir.resolve() == args.source.parent.resolve():
        parser.error('Use a new output directory; the original index must be preserved')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / 'index.jsonl'
    partial = args.output_dir / 'index.partial.jsonl'
    if target.exists() or partial.exists():
        parser.error('Output already exists; choose a new directory rather than overwrite evidence')
    settings = load_settings()
    client = embedding_client(settings)
    client.request_timeout_seconds = 120
    manifest = json.loads(args.source.with_name('manifest.json').read_text(encoding='utf-8'))
    source_sha = file_hash(args.source)
    count, dimensions = 0, None
    counts = {}
    start = time.monotonic()
    with args.source.open(encoding='utf-8') as source, partial.open('x', encoding='utf-8', newline='\n') as output:
        batch = []
        def flush():
            nonlocal count, batch, dimensions
            if not batch:
                return
            vectors = client.embed([str(row['text']) for row in batch], input_type='passage')
            for row, vector in zip(batch, vectors):
                if dimensions is None:
                    dimensions = len(vector)
                if len(vector) != dimensions or not vector or not any(vector) or not all(math.isfinite(v) for v in vector):
                    raise ValueError('Endpoint returned an invalid vector; partial output retained, not promoted')
                output.write(json.dumps(dict(row, vector=vector), ensure_ascii=False) + '\n')
                count += 1
                name = row['document_id'].split(':', 1)[0]
                counts[name] = counts.get(name, 0) + 1
            output.flush()
            if count % 512 == 0:
                print(f'reembedded={count} elapsed_seconds={time.monotonic()-start:.1f}', flush=True)
            batch = []
        for line in source:
            if not line.strip():
                continue
            if args.limit is not None and count + len(batch) >= args.limit:
                break
            row = json.loads(line)
            row.pop('vector')
            batch.append(row)
            if len(batch) == args.batch_size:
                flush()
        flush()
    partial.rename(target)
    manifest.update(schema_version='2.0.0', index_path=str(target), records=count, dimensions=dimensions,
                    source_counts=counts, embedding_model=settings.embedding_model, embedding_base_url=settings.embedding_base_url,
                    embedding_format=dict(document_prefix='passage: ', query_prefix='query: '),
                    rebuilt_from=dict(path=str(args.source), sha256=source_sha), index_sha256=file_hash(target),
                    created_at=datetime.now(timezone.utc).isoformat(), duration_seconds=round(time.monotonic()-start, 3))
    target.with_name('manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['records','dimensions','index_sha256','duration_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
