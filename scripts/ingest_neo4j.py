#!/usr/bin/env python3
"""Load normalized authority JSONL into Neo4j without discarding original text."""

from __future__ import annotations

import argparse
import json
import re
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from codelaw.graph import connection, normalise_graph_uri


def _sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_jsonl")
    parser.add_argument('--uri')
    parser.add_argument('--user', default='neo4j')
    parser.add_argument('--password', help='Prefer NEO4J_PASSWORD to avoid process-list exposure')
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--manifest", help="Optional prepare_authority.py manifest for provenance validation")
    parser.add_argument("--replace-source", action="store_true", help="Remove Authority nodes from source versions absent from the supplied manifest")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.uri:
        uri, user, password = normalise_graph_uri(args.uri), args.user, args.password or os.environ.get('NEO4J_PASSWORD')
        if not password:
            parser.error('NEO4J_PASSWORD is required with --uri')
    else:
        uri, user, password = connection(ROOT)
    from neo4j import GraphDatabase
    records_path = args.records_jsonl
    with open(records_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise SystemExit("records_jsonl is empty")
    if any(not _sha256(str(row.get("sha256", ""))) for row in rows):
        raise SystemExit("authority records must carry a 64-character source sha256")
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected = {str(source["sha256"]) for source in manifest.get("sources", [])}
        actual = {str(row["sha256"]) for row in rows}
        if not actual.issubset(expected):
            raise SystemExit("authority record sha256 does not match the supplied manifest")
    else:
        expected = set()
    if args.replace_source and not expected:
        raise SystemExit("--replace-source requires --manifest with source hashes")
    for row in rows:
        _, title, section = row['authority_id'].split(':', 2)
        section = section.lstrip('§ ').strip()
        row['title'] = int(title)
        row['part_id'] = f'ecfr:{title}:{section.split(".", 1)[0]}'
    ids = {r['authority_id'] for r in rows}
    references = []
    for row in rows:
        # Only explicit local-title section references, and only existing nodes.
        for explicit_title, section in set(re.findall(r'(?:(\d+)\s+CFR\s*)?§{1,2}\s*(\d+\.\d+[A-Za-z]?)', row['text'])):
            target = f'ecfr:{explicit_title or row["title"]}:{section}'
            if target in ids and target != row['authority_id']:
                references.append({'source':row['authority_id'], 'target':target})
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.execute_query(
            "CREATE CONSTRAINT authority_id_unique IF NOT EXISTS "
            "FOR (a:Authority) REQUIRE a.authority_id IS UNIQUE",
            database_=args.database,
        )
        for label, key in [('Part', 'part_id'), ('SourceVersion', 'sha256')]:
            driver.execute_query(f'CREATE CONSTRAINT {label.lower()}_unique IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE', database_=args.database)
        if args.replace_source:
            driver.execute_query(
                "MATCH (a:Authority) "
                "WHERE a.source_url IN $source_urls AND NOT a.sha256 IN $sha256s "
                "DETACH DELETE a",
                source_urls=[str(source['source_url']) for source in manifest.get('sources', [])],
                sha256s=list(expected),
                database_=args.database,
            )
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            driver.execute_query(
                'UNWIND $rows AS row MERGE (a:Authority {authority_id: row.authority_id}) SET a += row '
                'MERGE (p:Part {part_id: row.part_id}) SET p.title=row.title '
                'MERGE (a)-[:IN_PART]->(p) '
                'MERGE (s:SourceVersion {sha256:row.sha256}) SET s.issue_date=row.issue_date, s.source_url=row.source_url '
                'MERGE (a)-[:FROM_SOURCE]->(s)',
                rows=batch,
                database_=args.database,
            )
        for start in range(0, len(references), args.batch_size):
            driver.execute_query('UNWIND $rows AS row MATCH (a:Authority {authority_id:row.source}), (b:Authority {authority_id:row.target}) MERGE (a)-[:REFERENCES]->(b)',
                                 rows=references[start:start+args.batch_size], database_=args.database)
    print(f'ingested={len(rows)} references={len(references)} uri={uri}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
