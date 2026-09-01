#!/usr/bin/env python3
"""Run the K3 quality gate serially; requires an environment-provided key."""

from __future__ import annotations

import json

from codelaw.config import SettingsError, load_settings
from codelaw.live import LiveTask, NvidiaChatClient, run_serial

TASKS = [
    LiveTask("sensitive-1", "Classify the supplied contract clause. Return JSON only."),
    LiveTask("sensitive-2", "Identify whether the supplied clause creates an assignment restriction. Return JSON only."),
    LiveTask("sensitive-3", "Identify whether the supplied clause limits liability. Return JSON only."),
]


def main() -> int:
    settings = load_settings()
    client = NvidiaChatClient(settings)
    try:
        rows = run_serial(settings, TASKS[:settings.live_cases], client.ask)
    except SettingsError as exc:
        print(f"live_gate=blocked: {exc}")
        return 2
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
