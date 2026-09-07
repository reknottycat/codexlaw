"""Query the live graph after vector retrieval; never silently fall back."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from codelaw.experiment import file_hash
from codelaw.graph import AuthorityGraph, connection, normalise_graph_uri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--neighbours', type=int, default=2)
    parser.add_argument('--snippet-chars', type=int, default=800)
    args = parser.parse_args()
    if args.neighbours < 1 or args.snippet_chars < 1:
        parser.error('neighbours and snippet-chars must be positive')
    if args.output.exists():
        parser.error('Output exists; choose a new path to preserve previous evidence')
    manifest_path = ROOT / 'data/processed/authority/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    uri, user, password = connection(ROOT)
    safe_uri = normalise_graph_uri(uri)
    graph = AuthorityGraph(uri, user, password, allowed_hashes={s['sha256'] for s in manifest['sources']})
    cases = [json.loads(line) for line in args.cases.read_text(encoding='utf-8').splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.partial.jsonl')
    added = query_count = 0
    try:
        counts = graph.counts()
        with partial.open('x', encoding='utf-8', newline='\n') as output:
            for case in cases:
                seeds = sorted({e['source_id'] for e in case.get('evidence', []) if str(e.get('source_id', '')).startswith('ecfr:')})
                hits = graph.expand(seeds, limit=args.neighbours)
                query_count += bool(seeds)
                new_evidence = []
                for hit in hits:
                    a = hit['authority']
                    new_evidence.append(dict(evidence_id='graph:'+a['authority_id'], source_id=a['authority_id'],
                        text=a['text'][:args.snippet_chars], jurisdiction='US', effective_on=a['issue_date'], citation=a['citation'],
                        source_url=a['source_url'], source_sha256=a['sha256'], graph_relation=hit['relation'], graph_via=hit['via']))
                case['evidence'] = [*case.get('evidence', []), *new_evidence]
                case['graph_rag'] = dict(uri=safe_uri, seeds=seeds, evidence_ids=[e['evidence_id'] for e in new_evidence],
                    relations=[dict(authority_id=h['authority']['authority_id'], relation=h['relation'], via=h['via']) for h in hits],
                    checked_at=datetime.now(timezone.utc).isoformat(), provenance_verified=True)
                added += len(new_evidence)
                output.write(json.dumps(case, ensure_ascii=False)+'\n')
        partial.rename(args.output)
    finally:
        graph.close()
    record = dict(schema_version='1.0', retrieval_mode='vector_graph', graph_uri=safe_uri, graph_counts=counts,
        input_path=str(args.cases.resolve()), input_sha256=file_hash(args.cases), output_path=str(args.output.resolve()),
        output_sha256=file_hash(args.output), authority_manifest_sha256=file_hash(manifest_path),
        graph_queries=query_count, cases=len(cases), graph_evidence_added=added, neighbours=args.neighbours, snippet_chars=args.snippet_chars,
        created_at=datetime.now(timezone.utc).isoformat(), method='live authority lookup + REFERENCES/SAME_PART one-hop expansion; not semantic entailment')
    args.output.with_suffix('.manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
