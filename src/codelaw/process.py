"""Bound subprocess wall time and clean up its descendants on timeout."""
from __future__ import annotations

import os
import signal
import subprocess
from typing import Any


def run_process(command: list[str], *, input: str | None = None, timeout: float | None = None,
                capture_output: bool = True, check: bool = False, **kwargs: Any) -> subprocess.CompletedProcess:
    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.name != 'nt':
        kwargs['start_new_session'] = True
    with subprocess.Popen(command, stdin=subprocess.PIPE if input is not None else None, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input=input, timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, check=False)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.kill()
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
        completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if check:
            completed.check_returncode()
        return completed
