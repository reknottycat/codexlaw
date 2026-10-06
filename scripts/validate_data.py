#!/usr/bin/env python3
"""Validate every normalized benchmark and authority row without model calls."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def _validate_cases(path: Path) -> dict[str, object]:
    required = {"case_id", "source", "task", "prompt", "expected_answer", "answer_type", "evidence"}
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    invalid = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - row.keys()
            case_id = str(row.get("case_id", ""))
            evidence = row.get("evidence")
            if missing or not case_id or case_id in seen or not isinstance(evidence, list) or not evidence:
                invalid += 1
                continue
            if any(not str(item.get("text", "")).strip() or not item.get("evidence_id") for item in evidence):
                invalid += 1
                continue
            seen.add(case_id)
            counts[str(row["source"])] += 1
    return {"path": str(path), "rows": len(seen), "source_counts": dict(counts), "invalid_rows": invalid}


def _validate_authority(path: Path) -> dict[str, object]:
    seen: set[str] = set()
    versions: set[str] = set()
    invalid = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            authority_id = str(row.get("authority_id", ""))
            digest = str(row.get("sha256", ""))
            if (
                not authority_id
                or authority_id in seen
                or not str(row.get("text", "")).strip()
                or not str(row.get("source_url", "")).startswith("https://")
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                invalid += 1
                continue
            seen.add(authority_id)
            versions.add(digest)
    return {"path": str(path), "rows": len(seen), "source_versions": len(versions), "invalid_rows": invalid}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("data/processed/benchmark/cases.jsonl"))
    parser.add_argument("--authority", type=Path, default=Path("data/processed/authority/authority.jsonl"))
    args = parser.parse_args()
    try:
        cases = _validate_cases(args.cases)
        authority = _validate_authority(args.authority)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"data_validation=failed: {exc}")
        return 1
    report = {"cases": cases, "authority": authority}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not cases["invalid_rows"] and not authority["invalid_rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
