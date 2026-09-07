"""Run one side in an isolated process under the parent's common deadline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from codelaw.experiment import shared_task
from codelaw.minimax import MiniMaxChatClient, MiniMaxRuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', choices=['Lawgent', 'CodexLaw'], required=True)
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('--max-tokens', type=int, required=True)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--codex-engine', choices=['python', 'cli'], default='python')
    args = parser.parse_args()
    case = json.load(sys.stdin)
    runtime = MiniMaxRuntime(model=args.model, timeout_seconds=args.timeout, max_tokens=args.max_tokens, temperature=args.temperature)
    if args.project == 'Lawgent':
        from scripts.run_lawgent_minimax import execute_case
        result = execute_case(case, runtime)
    elif args.codex_engine == 'cli':
        import tempfile
        from codelaw.codex_runtime import run_codex
        with tempfile.TemporaryDirectory(prefix='codex-case-', dir=ROOT / '.runtime') as directory:
            result = run_codex(shared_task(case), runtime, Path(directory))
    else:
        client = MiniMaxChatClient(runtime)
        response, error = '', None
        try:
            response = client.ask(shared_task(case))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        result = dict(response=response, error=error, calls=client.calls, events=[])
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if not result['error'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
