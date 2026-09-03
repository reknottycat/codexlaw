"""Small, keyless wrapper around the locally configured MiniMax CLI."""

from __future__ import annotations

import subprocess
import json
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class MiniMaxRuntime:
    model: str = "MiniMax-M3"
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout_seconds: float = 300.0
    # `mmx` resolves to a PowerShell shim on Windows; child processes need the
    # accompanying command wrapper instead.
    command: str = "mmx.cmd"


class MiniMaxChatClient:
    """Invoke `mmx` without reading, printing, or persisting its API key."""

    def __init__(
        self,
        runtime: MiniMaxRuntime,
        *,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.runtime = runtime
        self._run = run

    def ask(self, prompt: str, *, system: str = "") -> str:
        prompt = _command_text(prompt)
        system = _command_text(system)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        command = [
            self.runtime.command,
            "text",
            "chat",
            "--model",
            self.runtime.model,
            "--messages-file",
            "-",
            "--max-tokens",
            str(self.runtime.max_tokens),
            "--temperature",
            str(self.runtime.temperature),
            "--output",
            "text",
            "--quiet",
            "--non-interactive",
            "--timeout",
            str(int(self.runtime.timeout_seconds)),
        ]
        completed = self._run(
            command,
            capture_output=True,
            text=True,
            input=json.dumps(messages, ensure_ascii=False),
            timeout=self.runtime.timeout_seconds + 15,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"MiniMax CLI exited with status {completed.returncode}")
        response = _response_text(completed.stdout.strip())
        if not response:
            raise RuntimeError("MiniMax CLI returned an empty response")
        return response


def _command_text(value: str) -> str:
    """Replace lone surrogates that Windows cannot pass to a child process."""
    return value.encode("utf-8", errors="replace").decode("utf-8")


def _response_text(response: str) -> str:
    """Accept both CLI text mode and its raw MiniMax API response fallback."""
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return response
    content = payload.get("content") if isinstance(payload, dict) else None
    if not isinstance(content, list):
        return response
    text = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return text.strip() or response
