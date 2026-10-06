"""Small, keyless wrapper around the locally configured MiniMax CLI."""

from __future__ import annotations

import subprocess
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .process import run_process


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
        run: Callable[..., subprocess.CompletedProcess[str]] = run_process,
    ) -> None:
        self.runtime = runtime
        self._run = run
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def ask(self, prompt: str, *, system: str = "") -> str:
        # Lawgent may fan specialists out; serialize actual provider requests.
        return self.ask_with_trace(prompt, system=system)[0]

    def ask_with_trace(self, prompt: str, *, system: str = "") -> tuple[str, dict]:
        """Return the answer and its trace while holding the serialization lock."""
        with self._lock:
            response = self._ask(prompt, system=system)
            return response, dict(self.calls[-1])

    def _ask(self, prompt: str, *, system: str = "") -> str:
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
            "json",
            "--non-interactive",
            "--timeout",
            str(int(self.runtime.timeout_seconds)),
        ]
        # mmx --quiet overrides --output json and drops usage/stop_reason.
        started = time.monotonic()
        trace = {'messages': messages, 'requested_model': self.runtime.model,
                 'max_tokens': self.runtime.max_tokens, 'temperature': self.runtime.temperature,
                 'timeout_seconds': self.runtime.timeout_seconds}
        self.calls.append(trace)
        try:
            completed = self._run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                  input=json.dumps(messages, ensure_ascii=False),
                                  timeout=self.runtime.timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            trace.update(error='provider_timeout', elapsed_seconds=round(time.monotonic() - started, 3))
            raise
        trace.update(elapsed_seconds=round(time.monotonic() - started, 3), exit_code=completed.returncode)
        if completed.returncode:
            trace['error'] = f'cli_exit_{completed.returncode}'
            raise RuntimeError(f"MiniMax CLI exited with status {completed.returncode}")
        try:
            envelope = json.loads(completed.stdout)
        except json.JSONDecodeError:
            envelope = None
        if isinstance(envelope, dict) and ('content' in envelope or 'choices' in envelope):
            trace['provider_response'] = envelope
            trace['usage'] = envelope.get('usage')
            trace['model'] = envelope.get('model')
            trace['stop_reason'] = envelope.get('stop_reason') or next((c.get('finish_reason') for c in envelope.get('choices', [])), None)
        else:
            trace.update(usage=None, stop_reason=None, response_format='plain_text_without_telemetry')
        response = _response_text(completed.stdout.strip())
        if not response:
            trace['error'] = 'empty_response'
            raise RuntimeError("MiniMax CLI returned an empty response")
        trace['response_text'] = response
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
