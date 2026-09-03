#!/usr/bin/env python3
"""Recompute A/B quality metrics from immutable live-run JSONL rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codelaw.benchmark import answer_matches, summarize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path, help="Live A/B JSONL result file")
    args = parser.parse_args()
    if not args.rows.exists():
        print(f"rescore=blocked: rows file does not exist: {args.rows}")
        return 2
    try:
        rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        print(f"rescore=blocked: invalid JSONL in {args.rows}: {exc}")
        return 2
    if not rows:
        print(f"rescore=blocked: rows file is empty: {args.rows}")
        return 2
    for row in rows:
        row["answer_correct"] = answer_matches(
            answer=str(row.get("answer", "")),
            expected=str(row.get("expected_answer", "")),
            answer_type=str(row.get("answer_type", "classification")),
        )
        row["success"] = bool(row["answer_correct"] and row.get("citation_valid") and row.get("workflow_compliant"))
    print(json.dumps({"rows_path": str(args.rows), "metrics": summarize(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
