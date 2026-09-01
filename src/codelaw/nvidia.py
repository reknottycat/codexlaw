"""Minimal NVIDIA OpenAI-compatible embedding client; no local fake vectors."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

from .config import SettingsError


class NvidiaEmbeddingClient:
    def __init__(self, *, base_url: str, api_key: str | None, model: str):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise SettingsError("NVIDIA_API_KEY is required for NVIDIA embeddings")
        request = Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": texts, "encoding_format": "float"}).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read())
        return [item["embedding"] for item in sorted(body["data"], key=lambda item: item["index"])]
