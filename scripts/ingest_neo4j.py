#!/usr/bin/env python3
"""Load normalized authority JSONL into Neo4j without discarding original text."""

from __future__ import annotations

import argparse
import json
import re


def _sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_jsonl")
    parser.add_argument("--uri", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--manifest", help="Optional prepare_authority.py manifest for provenance validation")
    parser.add_argument("--replace-source", action="store_true", help="Remove Authority nodes from source versions absent from the supplied manifest")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
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
    with GraphDatabase.driver(args.uri, auth=(args.user, args.password)) as driver:
        driver.execute_query(
            "CREATE CONSTRAINT authority_id_unique IF NOT EXISTS "
            "FOR (a:Authority) REQUIRE a.authority_id IS UNIQUE",
            database=args.database,
        )
        if args.replace_source:
            driver.execute_query(
                "MATCH (a:Authority) WHERE NOT a.sha256 IN $sha256s DETACH DELETE a",
                sha256s=list(expected),
                database=args.database,
            )
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            driver.execute_query(
                "UNWIND $rows AS row MERGE (a:Authority {authority_id: row.authority_id}) SET a += row",
                rows=batch,
                database=args.database,
            )
    print(f"ingested={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
