#!/usr/bin/env python3
"""Run the K3 quality gate serially; requires an environment-provided key."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codelaw.benchmark import build_prompt, select_cases
from codelaw.config import SettingsError, load_settings
from codelaw.live import LiveTask, NvidiaChatClient, run_serial


def real_tasks(settings):
    cases_path = Path(os.environ.get("LEGALBENCH_CASES_PATH", "data/processed/benchmark/cases.jsonl"))
    cases = select_cases(cases_path, limit=settings.live_cases, sources=["legalbench-rag", "legalbench", "casehold"])
    if not cases:
        raise SettingsError(f"No real benchmark cases found at {cases_path}; run scripts/prepare_benchmark.py first")
    return [
        LiveTask(
            str(case["case_id"]),
            build_prompt(case, architecture="B", categories=[], max_evidence_chars=12000),
        )
        for case in cases
    ]


def main() -> int:
    settings = load_settings()
    client = NvidiaChatClient(settings)
    try:
        rows = run_serial(settings, real_tasks(settings), client.ask)
    except SettingsError as exc:
        print(f"live_gate=blocked: {exc}")
        return 2
    except TimeoutError:
        print("live_gate=failed: NVIDIA K3 request timed out before a response was received; verify endpoint availability and retry later")
        return 1
    except HTTPError as exc:
        exc.close()
        print(f"live_gate=failed: NVIDIA K3 endpoint returned HTTP {exc.code}; verify model, key, payload, and endpoint, then retry")
        return 1
    except URLError as exc:
        print(f"live_gate=failed: NVIDIA K3 request could not reach the provider ({exc.reason}); verify the endpoint and network, then retry")
        return 1
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
