#!/usr/bin/env python3
"""Run the real-data A/B gate with one serial NVIDIA chat client."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codelaw.benchmark import run_ab, select_cases, summarize  # noqa: E402
from codelaw.config import SettingsError, load_settings  # noqa: E402
from codelaw.lawgent_adapter import LawgentAdapter  # noqa: E402
from codelaw.live import NvidiaChatClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("data/processed/benchmark/cases.jsonl"))
    parser.add_argument("--limit", type=int, default=3, help="Number of cases; source-balanced when --source is used")
    parser.add_argument("--source", action="append", help="Repeat or use comma-separated sources, e.g. legalbench-rag,casehold")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--max-evidence-chars", type=int, default=12000)
    args = parser.parse_args()
    if args.limit < 1 or args.max_evidence_chars < 1:
        print("live_benchmark=blocked: limit and max-evidence-chars must be positive")
        return 2
    if not args.cases.exists():
        print(f"live_benchmark=blocked: cases file does not exist: {args.cases}")
        return 2
    sources = [
        source.strip()
        for value in (args.source or [])
        for source in value.split(",")
        if source.strip()
    ] or None
    try:
        settings = load_settings()
        settings.require_live_provider()
    except SettingsError as exc:
        print(f"live_benchmark=blocked: {exc}")
        return 2
    cases = select_cases(args.cases, limit=args.limit, sources=sources)
    if not cases:
        print("live_benchmark=blocked: no benchmark cases matched the selection")
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Include microseconds and entropy so concurrent runs cannot overwrite
    # result files created in the same second.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + '-' + uuid4().hex[:8]
    rows_path = args.output_dir / f"ab-{run_id}.jsonl"
    summary_path = args.output_dir / f"ab-{run_id}.summary.json"
    client = NvidiaChatClient(settings)
    lawgent_root = PROJECT_ROOT / "vendor" / "lawgent"
    if lawgent_root.exists():
        sys.path.insert(0, str(lawgent_root))
    def record_row(row: dict[str, object]) -> None:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        outcome = "ok" if row["success"] else "failed"
        print(
            f"live_benchmark=row case={row['case_id']} architecture={row['architecture']} "
            f"outcome={outcome} elapsed_seconds={row['elapsed_seconds']}"
        )

    with rows_path.open("w", encoding="utf-8", newline="\n") as output:
        try:
            rows = run_ab(
                cases,
                ask=client.ask,
                interval_seconds=settings.interval_seconds,
                lawgent=LawgentAdapter(),
                max_evidence_chars=args.max_evidence_chars,
                on_row=record_row,
            )
        except (TimeoutError, SettingsError) as exc:
            print(f"live_benchmark=failed: {exc}")
            return 1
    summary = {
        "run_id": run_id,
        "cases": [case["case_id"] for case in cases],
        "case_sources": [case["source"] for case in cases],
        "model": settings.chat_model,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "reasoning_effort": settings.reasoning_effort,
        "stream": settings.stream,
        "request_timeout_seconds": settings.request_timeout_seconds,
        "retries": settings.retries,
        "interval_seconds": settings.interval_seconds,
        "metrics": summarize(rows),
        "rows_path": str(rows_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
