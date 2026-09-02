"""Minimal NVIDIA OpenAI-compatible embedding client; no local fake vectors."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

from .config import SettingsError


class NvidiaEmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        require_api_key: bool = True,
        request_timeout_seconds: float = 120.0,
    ):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.require_api_key = require_api_key
        self.request_timeout_seconds = request_timeout_seconds

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.require_api_key and not self.api_key:
            raise SettingsError("NVIDIA_API_KEY is required for NVIDIA embeddings")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": texts, "encoding_format": "float"}).encode(),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.request_timeout_seconds) as response:
            body = json.loads(response.read())
        return [item["embedding"] for item in sorted(body["data"], key=lambda item: item["index"])]


def embedding_client(settings: object) -> NvidiaEmbeddingClient:
    """Build the configured client, including anonymous private DGX endpoints."""
    return NvidiaEmbeddingClient(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        require_api_key=settings.embedding_requires_api_key,
        request_timeout_seconds=settings.request_timeout_seconds,
    )
