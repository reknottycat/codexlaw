"""Minimal NVIDIA OpenAI-compatible embedding client; no local fake vectors."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

from .config import SettingsError, is_official_nvidia_endpoint, normalise_endpoint


class NvidiaEmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        require_api_key: bool = True,
        allow_custom_endpoint: bool = False,
        request_timeout_seconds: float = 120.0,
    ):
        self.base_url = normalise_endpoint(base_url, "NVIDIA_EMBEDDING_BASE_URL")
        self.api_key = api_key if require_api_key else None
        if self.api_key and not is_official_nvidia_endpoint(self.base_url) and not allow_custom_endpoint:
            raise SettingsError("Set NVIDIA_ALLOW_CUSTOM_ENDPOINT=true before sending an embedding key to a non-NVIDIA endpoint")
        self.model = model
        self.require_api_key = require_api_key
        self.request_timeout_seconds = request_timeout_seconds

    def embed(self, texts: list[str], *, input_type: str = 'query') -> list[list[float]]:
        if input_type not in ('query', 'passage'):
            raise ValueError('input_type must be query or passage')
        if self.require_api_key and not self.api_key:
            raise SettingsError("NVIDIA_API_KEY is required for NVIDIA embeddings")
        headers = {"Content-Type": "application/json"}
        if self.require_api_key and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # The model's bundled README explicitly requires these prefixes for
        # /v1/embeddings. /v2/embed applies them itself; this client uses v1.
        inputs = [f'{input_type}: {text}' for text in texts] if self.model.split('/')[-1] == 'Nemotron-3-Embed-1B-BF16' else texts
        request = Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": inputs, "encoding_format": "float"}).encode(),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.request_timeout_seconds) as response:
            body = json.loads(response.read())
        vectors = [item['embedding'] for item in sorted(body['data'], key=lambda item: item['index'])]
        if len(vectors) != len(texts):
            raise ValueError('Embedding endpoint returned a different number of vectors')
        return vectors


def embedding_client(settings: object) -> NvidiaEmbeddingClient:
    """Build the configured client, including anonymous private DGX endpoints."""
    return NvidiaEmbeddingClient(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        require_api_key=settings.embedding_requires_api_key,
        allow_custom_endpoint=getattr(settings, "allow_custom_endpoint", False),
        request_timeout_seconds=settings.request_timeout_seconds,
    )
