#!/usr/bin/env python3
"""Load normalized authority JSONL into Neo4j without discarding original text."""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_jsonl")
    parser.add_argument("--uri", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()
    from neo4j import GraphDatabase
    rows = [json.loads(line) for line in open(args.records_jsonl, encoding="utf-8") if line.strip()]
    with GraphDatabase.driver(args.uri, auth=(args.user, args.password)) as driver:
        driver.execute_query("UNWIND $rows AS row MERGE (a:Authority {authority_id: row.authority_id}) SET a += row", rows=rows)
    print(f"ingested={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
