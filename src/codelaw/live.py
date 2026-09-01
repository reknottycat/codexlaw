"""Low-frequency, strictly serial NVIDIA K3 quality gate."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
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
        request = Request(
            f"{self.settings.nvidia_base_url}/chat/completions",
            data=json.dumps({"model": self.settings.chat_model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"Authorization": f"Bearer {self.settings.nvidia_api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            body = json.loads(response.read())
        return body["choices"][0]["message"].get("content", "")


def run_serial(settings: Settings, tasks: list[LiveTask], ask: Callable[[str], str], sleep: Callable[[float], None] = time.sleep) -> list[dict[str, str]]:
    settings.require_live_provider()
    results = []
    for index, task in enumerate(tasks):
        if index:
            sleep(settings.interval_seconds)
        results.append({"task_id": task.task_id, "response": ask(task.prompt)})
    return results
