#!/usr/bin/env python3
"""Validate non-secret runtime configuration before any live run."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codelaw.config import SettingsError, load_settings


def main() -> int:
    settings = load_settings()
    print(f"chat_model={settings.chat_model}")
    print(f"embedding_model={settings.embedding_model}")
    print(f"embedding_base_url={settings.embedding_base_url}")
    print(f"embedding_requires_api_key={str(settings.embedding_requires_api_key).lower()}")
    print(f"serial_interval_seconds={settings.interval_seconds}")
    print(f"max_tokens={settings.max_tokens}")
    print(f"temperature={settings.temperature}")
    print(f"reasoning_effort={settings.reasoning_effort}")
    print(f"stream={str(settings.stream).lower()}")
    print(f"request_timeout_seconds={settings.request_timeout_seconds}")
    print(f"retries={settings.retries}")
    try:
        settings.require_live_provider()
    except SettingsError as exc:
        print(f"live_provider=blocked: {exc}")
        return 2
    print("live_provider=ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
