"""External evidence state that survives agent-context compaction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_id: str
    text: str
    jurisdiction: str
    effective_on: str | None = None


@dataclass
class EvidenceLedger:
    by_id: dict[str, Evidence] = field(default_factory=dict)

    def add(self, evidence: Evidence) -> None:
        if not evidence.text.strip():
            raise ValueError("Evidence text must be retained outside model context")
        self.by_id[evidence.evidence_id] = evidence

    def require(self, *ids: str) -> list[Evidence]:
        missing = [item for item in ids if item not in self.by_id]
        if missing:
            raise KeyError(f"Required evidence missing: {', '.join(missing)}")
        return [self.by_id[item] for item in ids]
