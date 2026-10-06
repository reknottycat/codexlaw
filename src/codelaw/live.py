"""Low-frequency, strictly serial NVIDIA K3 quality gate."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from http.client import RemoteDisconnected
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Settings


@dataclass(frozen=True)
class LiveTask:
    task_id: str
    prompt: str


class NvidiaChatClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def ask(self, prompt: str) -> str:
        self.settings.require_live_provider()
        payload = {
            "model": self.settings.chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.settings.max_tokens,
            "seed": 0,
            "stream": self.settings.stream,
            "temperature": self.settings.temperature,
            "reasoning_effort": self.settings.reasoning_effort,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.nvidia_api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if self.settings.stream else "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            try:
                return self._request(payload, headers)
            except HTTPError as exc:
                if not _retryable_http_status(exc.code):
                    exc.close()
                    raise
                last_error = exc
                exc.close()
                if attempt >= self.settings.retries:
                    raise
                time.sleep(min(30.0, 2.0 ** attempt))
            except (TimeoutError, URLError, RemoteDisconnected) as exc:
                last_error = exc
                if attempt >= self.settings.retries:
                    raise
                time.sleep(min(30.0, 2.0 ** attempt))
        assert last_error is not None
        raise last_error

    def _request(self, payload: dict[str, Any], headers: dict[str, str]) -> str:
        request = Request(
            f"{self.settings.nvidia_base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.settings.request_timeout_seconds) as response:
            status = getattr(response, "status", 200)
            if status == 202:
                body = json.loads(response.read())
                return self._poll(body)
            if self.settings.stream:
                return _read_sse_content(response)
            body = json.loads(response.read())
        return _content_from_json(body)

    def _poll(self, pending: dict[str, Any]) -> str:
        request_id = pending.get("requestId") or pending.get("request_id") or pending.get("id")
        if not request_id:
            raise ValueError("NVIDIA returned HTTP 202 without a request ID")
        deadline = time.monotonic() + self.settings.request_timeout_seconds
        while time.monotonic() < deadline:
            request = Request(
                f"{self.settings.nvidia_base_url}/status/{request_id}",
                headers={"Authorization": f"Bearer {self.settings.nvidia_api_key}", "Accept": "application/json"},
                method="GET",
            )
            with urlopen(request, timeout=min(60.0, self.settings.request_timeout_seconds)) as response:
                status = getattr(response, "status", 200)
                body = json.loads(response.read())
            if status == 200:
                return _content_from_json(body)
            if status != 202:
                raise RuntimeError(f"NVIDIA status polling returned HTTP {status}")
            time.sleep(2.0)
        raise TimeoutError("NVIDIA K3 status polling exceeded the configured timeout")


def _retryable_http_status(status: int) -> bool:
    return status in {408, 425, 429} or 500 <= status <= 599


def _content_from_json(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("NVIDIA chat response did not contain choices")
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def _read_sse_content(response: Any) -> str:
    content: list[str] = []
    while True:
        line = response.readline()
        if not line:
            break
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece:
                content.append(piece)
    return "".join(content)


def run_serial(settings: Settings, tasks: list[LiveTask], ask: Callable[[str], str], sleep: Callable[[float], None] = time.sleep) -> list[dict[str, str]]:
    settings.require_live_provider()
    results = []
    for index, task in enumerate(tasks):
        if index:
            sleep(settings.interval_seconds)
        results.append({"task_id": task.task_id, "response": ask(task.prompt)})
    return results
