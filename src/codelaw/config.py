"""Runtime configuration. Secrets are never read from repository files."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit


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
    allow_custom_endpoint: bool = False

    def require_live_provider(self) -> None:
        if not self.live_confirmed:
            raise SettingsError("Set LEGALBENCH_LIVE_CONFIRM=true before a live provider request")
        if not self.nvidia_api_key:
            raise SettingsError("NVIDIA_API_KEY is required for a live provider request")
        if self.interval_seconds < 30:
            raise SettingsError("LIVE_LEGALBENCH_INTERVAL_SECONDS must be at least 30 for serial K3 runs")
        endpoint = normalise_endpoint(self.nvidia_base_url, "NVIDIA_BASE_URL")
        if endpoint != self.nvidia_base_url:
            raise SettingsError("NVIDIA_BASE_URL must be a normalized URL")
        if not is_official_nvidia_endpoint(endpoint) and not self.allow_custom_endpoint:
            raise SettingsError("Set NVIDIA_ALLOW_CUSTOM_ENDPOINT=true before sending a key to a non-NVIDIA endpoint")


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


_OFFICIAL_NVIDIA_HOSTS = {"integrate.api.nvidia.com"}


def normalise_endpoint(value: str, name: str) -> str:
    """Validate an HTTP endpoint without allowing embedded credentials/secrets."""
    raw = str(value).strip().rstrip("/")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SettingsError(f"{name} must be an http(s) URL with a hostname")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SettingsError(f"{name} must not contain userinfo, query, or fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise SettingsError(f"{name} contains an invalid port") from exc
    host = parsed.hostname.lower()
    if host in _OFFICIAL_NVIDIA_HOSTS and parsed.scheme != "https":
        raise SettingsError(f"{name} must use https for the NVIDIA hosted endpoint")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme.lower()}://{netloc}{path}"


def is_official_nvidia_endpoint(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.hostname is not None and parsed.hostname.lower() in _OFFICIAL_NVIDIA_HOSTS and parsed.scheme == "https"


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
    nvidia_base_url = normalise_endpoint(values.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"), "NVIDIA_BASE_URL")
    embedding_base_url = normalise_endpoint(values.get("NVIDIA_EMBEDDING_BASE_URL", "http://192.168.1.6:8002/v1"), "NVIDIA_EMBEDDING_BASE_URL")
    allow_custom_endpoint = _bool(values.get("NVIDIA_ALLOW_CUSTOM_ENDPOINT", "false"))
    default_embedding_key_required = is_official_nvidia_endpoint(embedding_base_url)
    embedding_requires_api_key = _bool(values.get("NVIDIA_EMBEDDING_REQUIRE_API_KEY", str(default_embedding_key_required).lower()))
    embedding_api_key = values.get("NVIDIA_EMBEDDING_API_KEY") or (values.get("NVIDIA_API_KEY") if embedding_requires_api_key else None)
    return Settings(
        nvidia_base_url=nvidia_base_url,
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
        allow_custom_endpoint=allow_custom_endpoint,
    )
