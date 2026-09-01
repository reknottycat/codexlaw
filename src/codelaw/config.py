"""Runtime configuration. Secrets are never read from repository files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


class SettingsError(ValueError):
    """Raised when configuration would make a live run unsafe or ambiguous."""


@dataclass(frozen=True)
class Settings:
    nvidia_base_url: str
    chat_model: str
    embedding_model: str
    interval_seconds: float
    live_cases: int
    live_confirmed: bool
    nvidia_api_key: str | None

    def require_live_provider(self) -> None:
        if not self.live_confirmed:
            raise SettingsError("Set LEGALBENCH_LIVE_CONFIRM=true before a live provider request")
        if not self.nvidia_api_key:
            raise SettingsError("NVIDIA_API_KEY is required for a live provider request")
        if self.interval_seconds < 30:
            raise SettingsError("LIVE_LEGALBENCH_INTERVAL_SECONDS must be at least 30 for serial K3 runs")


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    values = os.environ if env is None else env
    try:
        interval = float(values.get("LIVE_LEGALBENCH_INTERVAL_SECONDS", "30"))
        cases = int(values.get("LIVE_LEGALBENCH_CASES", "3"))
    except ValueError as exc:
        raise SettingsError("LIVE_LEGALBENCH interval and case count must be numeric") from exc
    if interval <= 0 or cases < 1:
        raise SettingsError("LIVE_LEGALBENCH interval and case count must be positive")
    return Settings(
        nvidia_base_url=values.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/"),
        chat_model=values.get("NVIDIA_CHAT_MODEL", "moonshotai/kimi-k3"),
        embedding_model=values.get("NVIDIA_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2"),
        interval_seconds=interval,
        live_cases=cases,
        live_confirmed=_bool(values.get("LEGALBENCH_LIVE_CONFIRM", "false")),
        nvidia_api_key=values.get("NVIDIA_API_KEY") or None,
    )
