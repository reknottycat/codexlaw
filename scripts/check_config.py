#!/usr/bin/env python3
"""Validate non-secret runtime configuration before any live run."""

from codelaw.config import SettingsError, load_settings


def main() -> int:
    settings = load_settings()
    print(f"chat_model={settings.chat_model}")
    print(f"embedding_model={settings.embedding_model}")
    print(f"serial_interval_seconds={settings.interval_seconds}")
    try:
        settings.require_live_provider()
    except SettingsError as exc:
        print(f"live_provider=blocked: {exc}")
        return 2
    print("live_provider=ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
