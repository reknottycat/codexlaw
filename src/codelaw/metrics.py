"""Prometheus text exposition with bounded labels."""

from __future__ import annotations

from collections import Counter


class Metrics:
    def __init__(self):
        self.values: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()

    def inc(self, name: str, **labels: str) -> None:
        allowed = {"architecture", "task_type", "jurisdiction", "difficulty", "result", "node"}
        extra = set(labels) - allowed
        if extra:
            raise ValueError(f"High-cardinality or unsupported labels: {sorted(extra)}")
        self.values[(name, tuple(sorted(labels.items())))] += 1

    def render(self) -> str:
        lines = []
        for (name, labels), value in sorted(self.values.items()):
            suffix = "{" + ",".join(f'{key}="{val}"' for key, val in labels) + "}" if labels else ""
            lines.append(f"{name}{suffix} {value}")
        return "\n".join(lines) + ("\n" if lines else "")
