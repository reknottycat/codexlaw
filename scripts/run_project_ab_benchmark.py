#!/usr/bin/env python3
"""Compare project harnesses under one recorded, symmetric input contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codelaw.benchmark import run_case, select_cases  # noqa: E402
from codelaw.minimax import MiniMaxRuntime  # noqa: E402
from codelaw.experiment import file_hash, json_hash, normalise_response, prepare_case, shared_task, source_snapshot
from codelaw.diagnostics import diagnose, telemetry
from codelaw.process import run_process


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


def _project_response(case: dict[str, Any], *, project: str, runtime: MiniMaxRuntime, codex_engine: str = 'python') -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    interpreter = str(PROJECT_ROOT / '.runtime/lawgent-venv/Scripts/python.exe') if project == 'Lawgent' else sys.executable
    command = [interpreter, '-u', str(PROJECT_ROOT / 'scripts/run_project_case.py'), '--project', project,
               '--timeout', str(runtime.timeout_seconds), '--max-tokens', str(runtime.max_tokens),
               '--temperature', str(runtime.temperature), '--model', runtime.model, '--codex-engine', codex_engine]
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = run_process(command, capture_output=True, input=json.dumps(case, ensure_ascii=False), text=True,
                                encoding='utf-8', errors='replace', timeout=runtime.timeout_seconds, env=environment, check=False)
    except subprocess.TimeoutExpired:
        return dict(response='', error=f'{project} exceeded the common {runtime.timeout_seconds:g}s case deadline; process tree stopped', calls=[]), time.monotonic() - started
    except OSError as exc:
        return dict(response='', error=f'{project} runner could not start: {exc}', calls=[]), time.monotonic() - started
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return dict(response='', error=f'{project} exited with status {completed.returncode} without a machine-readable result', calls=[]), time.monotonic() - started
    if completed.returncode and not payload.get('error'):
        payload['error'] = f'{project} runner exited with status {completed.returncode}'
    return payload, time.monotonic() - started


def _normalise_lawgent_response(case: dict[str, Any], response: str) -> str:
    """Compatibility entry point; the current normalizer is applied to both sides."""
    return normalise_response(case, response)


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
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--max-evidence-chars', type=int, default=0, help='Shared evidence budget; 0 keeps all evidence')
    parser.add_argument('--codex-engine', choices=['python','cli'], default='python', help='cli runs the actual Codex executable using the bounded MiniMax adapter')
    args = parser.parse_args()
    if args.limit < 1 or args.timeout <= 0 or args.max_tokens < 1 or not 0 < args.temperature <= 1 or args.max_evidence_chars < 0:
        parser.error('limit, timeout and max-tokens must be positive; temperature in (0,1]; evidence budget >= 0')
    sources = [item.strip() for value in (args.source or []) for item in value.split(",") if item.strip()] or None
    cases = select_cases(args.cases, limit=args.limit, sources=sources)
    if not cases:
        print("project_benchmark=blocked: no benchmark cases matched the selection")
        return 2
    runtime = MiniMaxRuntime(timeout_seconds=args.timeout, max_tokens=args.max_tokens, temperature=args.temperature)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Include microseconds and entropy so concurrent runs cannot overwrite
    # snapshots, rows, traces, or metadata created in the same second.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + '-' + uuid4().hex[:8]
    rows_path = args.output_dir / f"project-ab-{run_id}.jsonl"
    snapshot_path = args.output_dir / f'project-ab-{run_id}.inputs.jsonl'
    trace_path = args.output_dir / f'project-ab-{run_id}.trace.jsonl'
    metadata_path = args.output_dir / f'project-ab-{run_id}.metadata.json'
    prepared = [prepare_case(c, args.max_evidence_chars) for c in cases]
    snapshot_path.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in prepared), encoding='utf-8')
    metadata = dict(schema_version='2.0', run_id=run_id, created_at=datetime.now(timezone.utc).isoformat(),
        input_path=str(args.cases.resolve()), input_sha256=file_hash(args.cases), snapshot_path=str(snapshot_path.resolve()),
        snapshot_sha256=file_hash(snapshot_path), source=source_snapshot(PROJECT_ROOT),
        retrieval_mode='vector_graph' if any(c.get('graph_rag') for c in cases) else 'vector' if any(c.get('offline_rag') for c in cases) else 'supplied_evidence',
        engines={'Lawgent':'WorkflowExecutor with serial MiniMax text provider', 'CodexLaw':'Codex CLI + Responses-to-MiniMax adapter' if args.codex_engine=='cli' else 'direct MiniMax + local Python validation (not Codex CLI)'},
        codex_engine=args.codex_engine,
        controls=dict(shared_task=True, shared_evidence=True, shared_normalizer=True, max_evidence_chars=args.max_evidence_chars,
                      model=runtime.model, temperature=runtime.temperature, max_tokens_per_call=runtime.max_tokens,
                      case_deadline_seconds=runtime.timeout_seconds, provider_timeout_seconds=runtime.timeout_seconds,
                      order='alternating A/B by case index', external_tools=False, provider_requests='serial',
                      total_model_calls='workflow-dependent; recorded, not forced equal'))
    if args.codex_engine == 'cli':
        metadata['codex_cli_version'] = subprocess.run(['codex.cmd','--version'],capture_output=True,text=True,check=True).stdout.strip()
    metadata['python_version'] = sys.version
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    rows: list[dict[str, Any]] = []
    with rows_path.open('w', encoding='utf-8', newline='\n') as output, trace_path.open('w', encoding='utf-8', newline='\n') as traces:
        for index, case in enumerate(prepared):
            order = ['Lawgent', 'CodexLaw'] if index % 2 == 0 else ['CodexLaw', 'Lawgent']
            for project in order:
                payload, elapsed = _project_response(case, project=project, runtime=runtime, codex_engine=args.codex_engine)
                raw = str(payload.get('response', ''))
                normalised = normalise_response(case, raw)
                row = _project_row(case, project=project, response=normalised, error=payload.get('error'), elapsed_seconds=elapsed)
                row['model_response'] = raw
                row['normalised_response'] = normalised
                row['telemetry'] = telemetry(payload.get('calls', []))
                row['input_sha256'] = json_hash(case)
                row['task_sha256'] = json_hash(shared_task(case))
                row['execution_order'] = order.index(project) + 1
                row['engine'] = payload.get('engine') or ('lawgent-workflow' if project=='Lawgent' else 'python')
                row['diagnostics'] = diagnose(row, cases[index], evidence_limit=args.max_evidence_chars)
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
                traces.write(json.dumps(dict(case_id=case['case_id'], project=project, **payload), ensure_ascii=False, default=str) + '\n')
                traces.flush()
                print(f"project_benchmark=row project={project} case={case['case_id']} outcome={'ok' if row['common_success'] else 'failed'} reason={row['diagnostics']['category']}", flush=True)
    summary = {"run_id": run_id, "model": runtime.model, "max_tokens": runtime.max_tokens, "temperature": runtime.temperature, "timeout_seconds": runtime.timeout_seconds, "cases": [case["case_id"] for case in cases], "metrics": _metrics(rows), "rows_path": str(rows_path),
               'schema_version':'2.0', 'metadata_path':str(metadata_path), 'metadata_sha256':file_hash(metadata_path),
               'rows_sha256':file_hash(rows_path), 'trace_path':str(trace_path), 'trace_sha256':file_hash(trace_path),
               'retrieval_mode':metadata['retrieval_mode'], 'fairness_controls':metadata['controls'], 'codex_engine':args.codex_engine}
    summary_path = args.output_dir / f"project-ab-{run_id}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
