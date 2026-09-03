#!/usr/bin/env python3
"""Compare Lawgent's native workflow with CodexLaw's Codex orchestration."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codelaw.benchmark import build_prompt, parse_decision, run_case, select_cases  # noqa: E402
from codelaw.minimax import MiniMaxChatClient, MiniMaxRuntime  # noqa: E402


def _project_row(case: dict[str, Any], *, project: str, response: str, error: str | None, elapsed_seconds: float) -> dict[str, Any]:
    row = run_case(case, architecture="B", ask=lambda _: response)
    row["project"] = project
    row["architecture"] = project
    row["provider_error"] = error
    row["elapsed_seconds"] = round(elapsed_seconds, 3)
    row["common_success"] = bool(row["answer_correct"] and row["citation_valid"] and not error)
    # This common comparison deliberately scores response quality and supplied
    # evidence only. A CodexLaw workflow-node count would not describe Lawgent.
    row.pop("workflow_compliant", None)
    row.pop("completed_nodes", None)
    row.pop("lawgent_categories", None)
    return row


def _lawgent_response(case_id: str, *, cases_path: Path, timeout_seconds: float) -> tuple[str, str | None, float]:
    started = time.monotonic()
    command = [str(PROJECT_ROOT / ".runtime" / "lawgent-venv" / "Scripts" / "python.exe"), "-u", str(PROJECT_ROOT / "scripts" / "run_lawgent_minimax.py"), "--cases", str(cases_path.resolve()), "--case-id", case_id]
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_seconds, env=environment, check=False)
    except subprocess.TimeoutExpired:
        return "", f"Lawgent runner exceeded {int(timeout_seconds)} seconds", time.monotonic() - started
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return "", f"Lawgent runner exited with status {completed.returncode} without a machine-readable result", time.monotonic() - started
    return str(payload.get("response", "")), payload.get("error") or (None if completed.returncode == 0 else f"Lawgent runner exited with status {completed.returncode}"), time.monotonic() - started


def _normalise_lawgent_response(case: dict[str, Any], response: str) -> str:
    """Extract Lawgent's explicit Markdown answer into the shared score schema."""
    if parse_decision(response).answer:
        return response
    answer_match = re.search(
        r"(?im)^\s*(?:[-*]\s*)?(?:\*{1,2})?(?:answer|答案)(?:\*{1,2})?\s*[:：]\s*(.+?)\s*$",
        response,
    )
    if not answer_match:
        return response
    answer = answer_match.group(1).strip().strip("`* ")
    if case.get("answer_type") == "multiple_choice":
        choice_match = re.search(r"\b([a-eA-E]|[0-4])\b", answer)
        if not choice_match:
            return response
        answer = choice_match.group(1).upper()
    elif case.get("answer_type") == "classification":
        answer = answer.split()[0].strip(".,:;`* ")
    else:
        answer = answer.split("\n", 1)[0].strip()
    citation_ids = [
        str(item["evidence_id"])
        for item in case.get("evidence", [])
        if str(item["evidence_id"]) in response
    ]
    return json.dumps({"answer": answer, "citation_ids": citation_ids, "confidence": None})


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"total_rows": len(rows), "projects": {}}
    for project in sorted({str(row["project"]) for row in rows}):
        group = [row for row in rows if row["project"] == project]
        total = len(group)
        completed = [row for row in group if not row.get("provider_error")]
        output["projects"][project] = {
            "rows": total,
            "execution_completion_rate": len(completed) / max(1, total),
            "answer_accuracy_when_completed": sum(bool(row["answer_correct"]) for row in completed) / max(1, len(completed)),
            "citation_validity_when_completed": sum(bool(row["citation_valid"]) for row in completed) / max(1, len(completed)),
            "common_success_rate": sum(bool(row["common_success"]) for row in group) / max(1, total),
            "provider_errors": sum(bool(row.get("provider_error")) for row in group),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("data/processed/benchmark/cases.jsonl"))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--source", action="append")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    sources = [item.strip() for value in (args.source or []) for item in value.split(",") if item.strip()] or None
    cases = select_cases(args.cases, limit=args.limit, sources=sources)
    if not cases:
        print("project_benchmark=blocked: no benchmark cases matched the selection")
        return 2
    runtime = MiniMaxRuntime(timeout_seconds=args.timeout)
    codex = MiniMaxChatClient(runtime)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows_path = args.output_dir / f"project-ab-{run_id}.jsonl"
    rows: list[dict[str, Any]] = []
    with rows_path.open("w", encoding="utf-8", newline="\n") as output:
        for case in cases:
            started = time.monotonic()
            try:
                codex_raw = codex.ask(build_prompt(case, architecture="B", categories=[], max_evidence_chars=12000))
                codex_error = None
            except Exception as exc:  # keep paired provider errors inspectable
                codex_raw, codex_error = "", str(exc)
            codex_row = _project_row(case, project="CodexLaw", response=codex_raw, error=codex_error, elapsed_seconds=time.monotonic() - started)
            rows.append(codex_row)
            output.write(json.dumps(codex_row, ensure_ascii=False) + "\n")
            output.flush()
            print(f"project_benchmark=row project=CodexLaw case={case['case_id']} outcome={'ok' if codex_row['common_success'] else 'failed'}")

            lawgent_raw, lawgent_error, lawgent_elapsed = _lawgent_response(str(case["case_id"]), cases_path=args.cases, timeout_seconds=args.timeout)
            lawgent_row = _project_row(case, project="Lawgent", response=_normalise_lawgent_response(case, lawgent_raw), error=lawgent_error, elapsed_seconds=lawgent_elapsed)
            rows.append(lawgent_row)
            output.write(json.dumps(lawgent_row, ensure_ascii=False) + "\n")
            output.flush()
            print(f"project_benchmark=row project=Lawgent case={case['case_id']} outcome={'ok' if lawgent_row['common_success'] else 'failed'}")
    summary = {"run_id": run_id, "model": runtime.model, "max_tokens": runtime.max_tokens, "temperature": runtime.temperature, "timeout_seconds": runtime.timeout_seconds, "cases": [case["case_id"] for case in cases], "metrics": _metrics(rows), "rows_path": str(rows_path)}
    summary_path = args.output_dir / f"project-ab-{run_id}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
