"""Runtime configuration. Secrets are never read from repository files."""

from __future__ import annotations

import math
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
    max_tokens: int = 65536
    embedding_base_url: str = "http://192.168.1.6:8002/v1"
    embedding_api_key: str | None = None
    embedding_requires_api_key: bool = False
    temperature: float = 1.0
    reasoning_effort: str = "max"
    stream: bool = True
    request_timeout_seconds: float = 900.0
    retries: int = 2

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
        max_tokens = int(values.get("LIVE_LEGALBENCH_MAX_TOKENS", "65536"))
        temperature = float(values.get("NVIDIA_TEMPERATURE", "1.0"))
        request_timeout_seconds = float(values.get("NVIDIA_REQUEST_TIMEOUT_SECONDS", "900"))
        retries = int(values.get("NVIDIA_RETRIES", "2"))
    except ValueError as exc:
        raise SettingsError("NVIDIA and LIVE_LEGALBENCH numeric settings must be numeric") from exc
    if not math.isfinite(interval) or interval <= 0 or cases < 1:
        raise SettingsError("LIVE_LEGALBENCH interval and case count must be positive")
    if not 1 <= max_tokens <= 65536:
        raise SettingsError("LIVE_LEGALBENCH_MAX_TOKENS must be between 1 and 65536")
    if not math.isfinite(temperature) or not 0 <= temperature <= 1:
        raise SettingsError("NVIDIA_TEMPERATURE must be between 0 and 1")
    if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
        raise SettingsError("NVIDIA_REQUEST_TIMEOUT_SECONDS must be positive")
    if not 0 <= retries <= 5:
        raise SettingsError("NVIDIA_RETRIES must be between 0 and 5")
    reasoning_effort = values.get("NVIDIA_REASONING_EFFORT", "max").strip().lower()
    if reasoning_effort not in {"low", "high", "max"}:
        raise SettingsError("NVIDIA_REASONING_EFFORT must be low, high, or max")
    embedding_base_url = values.get("NVIDIA_EMBEDDING_BASE_URL", "http://192.168.1.6:8002/v1").rstrip("/")
    default_embedding_key_required = "integrate.api.nvidia.com" in embedding_base_url
    embedding_requires_api_key = _bool(values.get("NVIDIA_EMBEDDING_REQUIRE_API_KEY", str(default_embedding_key_required).lower()))
    embedding_api_key = values.get("NVIDIA_EMBEDDING_API_KEY") or (values.get("NVIDIA_API_KEY") if embedding_requires_api_key else None)
    return Settings(
        nvidia_base_url=values.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/"),
        chat_model=values.get("NVIDIA_CHAT_MODEL", "moonshotai/kimi-k3"),
        embedding_model=values.get("NVIDIA_EMBEDDING_MODEL", "nvidia/Nemotron-3-Embed-1B-BF16"),
        interval_seconds=interval,
        live_cases=cases,
        live_confirmed=_bool(values.get("LEGALBENCH_LIVE_CONFIRM", "false")),
        nvidia_api_key=values.get("NVIDIA_API_KEY") or None,
        max_tokens=max_tokens,
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        embedding_requires_api_key=embedding_requires_api_key,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        stream=_bool(values.get("NVIDIA_STREAM", "true")),
        request_timeout_seconds=request_timeout_seconds,
        retries=retries,
    )
