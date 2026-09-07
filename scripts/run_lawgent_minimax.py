#!/usr/bin/env python3
"""Run Lawgent's native workflow over supplied benchmark evidence using MiniMax."""

from __future__ import annotations

import json
import sys
import types
import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codelaw.minimax import MiniMaxChatClient, MiniMaxRuntime  # noqa: E402
from codelaw.experiment import shared_task  # noqa: E402


def _load_lawgent() -> tuple[Any, Any, Any, Any, Any]:
    # Lawgent imports its PDF writer while loading the workflow. PDF export is
    # outside this text-only benchmark and needs GTK DLLs unavailable on this
    # Windows host, so supply the import symbol without changing Lawgent itself.
    pdf_stub = types.ModuleType("weasyprint")
    pdf_stub.HTML = type("HTML", (), {})
    sys.modules.setdefault("weasyprint", pdf_stub)
    import legal_helper.config as config
    from legal_helper.config import Settings
    from legal_helper.providers.base import RunResult, StreamEvent
    from legal_helper.workflow import WorkflowExecutor

    return config, Settings, RunResult, StreamEvent, WorkflowExecutor


def _content(message: dict[str, Any]) -> str:
    value = message.get("content", "")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(value)


class MiniMaxLawgentProvider:
    """Provider protocol adapter that preserves Lawgent's own workflow layer."""

    name = "minimax"
    supports_hosted_web_search = False
    supports_file_search = False
    supports_structured_outputs = False

    def __init__(self, runtime: MiniMaxRuntime, run_result: Any, stream_event: Any) -> None:
        self.model = runtime.model
        self.client = MiniMaxChatClient(runtime)
        self._run_result = run_result
        self._stream_event = stream_event

    def _answer(self, system: str, messages: list[dict[str, Any]]) -> str:
        return self.client.ask("\n\n".join(_content(message) for message in messages), system=system)

    def run(self, system: str, messages: list[dict[str, Any]], tools: Any = (), *, max_iterations: int | None = None, structured_output: Any = None) -> Any:
        return self._run_result(text=self._answer(system, messages), provider=self.name, model=self.model, iterations=1)

    def tool_runner(self, system: str, messages: list[dict[str, Any]], tools: Any, *, max_iterations: int, structured_output: Any = None) -> Any:
        return self.run(system, messages, tools, max_iterations=max_iterations, structured_output=structured_output)

    def stream(self, system: str, messages: list[dict[str, Any]], tools: Any = (), *, max_iterations: int | None = None) -> Iterator[Any]:
        answer = self._answer(system, messages)
        yield self._stream_event("delta", {"text": answer})
        yield self._stream_event("done", {"text": answer, "usage": {}})


@contextmanager
def _specialist_provider_override(executor_type: Any, provider: MiniMaxLawgentProvider) -> Iterator[None]:
    """Keep Lawgent's native specialist workflow on the benchmark provider."""
    workflow_module = sys.modules[executor_type.__module__]
    original_build_provider = workflow_module.build_provider
    workflow_module.build_provider = lambda *_args, **_kwargs: provider
    try:
        yield
    finally:
        workflow_module.build_provider = original_build_provider


def _task(case: dict[str, Any]) -> str:
    return shared_task(case)


def execute_case(case: dict[str, Any], runtime: MiniMaxRuntime) -> dict[str, Any]:
    config, settings_type, run_result, stream_event, executor_type = _load_lawgent()
    settings = settings_type(
        provider='openai', default_jurisdiction='US', default_language='en', max_tokens=runtime.max_tokens,
        parent_max_iterations=8, enable_web_search=False, enable_web_fetch=False, enable_cite_check=False,
        outputs_dir=PROJECT_ROOT / '.runtime/lawgent-benchmark/outputs',
        state_dir=PROJECT_ROOT / '.runtime/lawgent-benchmark/state',
        logs_dir=PROJECT_ROOT / '.runtime/lawgent-benchmark/logs',
    )
    provider = MiniMaxLawgentProvider(runtime, run_result, stream_event)
    final_text = ''
    error = None
    events = []
    try:
        with config.override_current_settings(settings):
            executor = executor_type(settings=settings, provider=provider)
            with _specialist_provider_override(executor_type, provider):
                for event in executor.stream(_task(case)):
                    events.append({'kind': event.kind, 'data': event.data})
                    if event.kind == 'delta':
                        final_text += str(event.data.get('text', ''))
                    if event.kind == 'error':
                        error = str(event.data.get('message', 'Lawgent workflow failed'))
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    return dict(response=final_text, error=error, events=events, calls=provider.client.calls)


def _load_case(args: argparse.Namespace) -> dict[str, Any]:
    if args.case_id:
        with args.cases.open(encoding="utf-8") as source:
            for line in source:
                case = json.loads(line)
                if case.get("case_id") == args.case_id:
                    return case
        raise ValueError(f"case ID not found in {args.cases}: {args.case_id}")
    return json.loads(sys.stdin.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "data" / "processed" / "benchmark" / "cases.jsonl")
    parser.add_argument("--case-id", help="Read one case directly from --cases instead of stdin")
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.1)
    args = parser.parse_args()
    try:
        case = _load_case(args)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"lawgent_runner=blocked: could not load benchmark case: {exc}")
        return 2
    payload = execute_case(case, MiniMaxRuntime(timeout_seconds=args.timeout, max_tokens=args.max_tokens, temperature=args.temperature))
    print(json.dumps(payload, ensure_ascii=False, default=str))
    return 0 if not payload['error'] else 1


if __name__ == "__main__":
    raise SystemExit(main())
