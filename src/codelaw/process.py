"""Bound subprocess wall time and clean up its descendants on timeout."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from typing import Any
from pathlib import Path

import psutil


def _kill_process_tree(pid: int) -> None:
    """Freeze parents before discovering children, including separate sessions."""
    try:
        pending = [psutil.Process(pid)]
    except psutil.NoSuchProcess:
        return
    stopped = []
    try:
        while pending:
            process = pending.pop()
            try:
                # A stopped parent cannot fork another child after enumeration.
                process.suspend()
                stopped.append(process)
                pending.extend(process.children())
            except psutil.NoSuchProcess:
                continue
    finally:
        # Kill leaves first, then their stopped parents. psutil checks identity
        # before signalling, so a reused PID is not mistaken for a descendant.
        for process in reversed(stopped):
            with suppress(psutil.NoSuchProcess):
                process.kill()
        # Signals can be pending when kill() returns. Wait for execution to
        # stop, without reaping the Popen-owned process or waiting on zombies.
        deadline = time.monotonic() + 1
        while stopped and time.monotonic() < deadline:
            alive = []
            for process in stopped:
                with suppress(psutil.NoSuchProcess):
                    if process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                        alive.append(process)
            stopped = alive
            if stopped:
                time.sleep(0.01)


def run_process(command: list[str], *, input: str | None = None, timeout: float | None = None,
                capture_output: bool = True, check: bool = False, **kwargs: Any) -> subprocess.CompletedProcess:
    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    owns_group = os.name != 'nt' and os.environ.get('CODELAW_PROCESS_GROUP') != str(os.getpgrp())
    launched_command = command
    if os.name != 'nt':
        # Nested calls must remain in the case's group. Otherwise, once an
        # intermediate parent exits, its detached CLI cannot be discovered.
        kwargs['start_new_session'] = owns_group
        if owns_group:
            launched_command = [sys.executable, str(Path(__file__).with_name('_process_exec.py')), *command]
    process = subprocess.Popen(launched_command, stdin=subprocess.PIPE if input is not None else None, **kwargs)
    try:
        with process:
            try:
                stdout, stderr = process.communicate(input=input, timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    _kill_process_tree(process.pid)
                finally:
                    if owns_group:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                    process.kill()
                try:
                    stdout, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired as exc:
                    # Never turn a deadline into an unbounded pipe-draining wait.
                    stdout, stderr = exc.output, exc.stderr
                    for pipe in (process.stdout, process.stderr):
                        if pipe is not None:
                            pipe.close()
                raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
            completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            if check:
                completed.check_returncode()
            return completed
    finally:
        if owns_group:
            # Also clean up reparented helper descendants after a normal exit.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
