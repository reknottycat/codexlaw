"""Deterministic legal workflow gates around an agent-controlled retrieval loop."""

from __future__ import annotations

from dataclasses import dataclass, field

from .citations import CitationVerifier
from .evidence import EvidenceLedger

REQUIRED_NODES = ("CONTRACT_INTAKE", "JURISDICTION", "EFFECTIVE_DATE", "EXCEPTION_CHECK", "CITATION_VERIFICATION", "FINAL_REVIEW")


@dataclass
class WorkflowState:
    completed: list[str] = field(default_factory=list)
    citation_id: str | None = None

    def complete(self, node: str) -> None:
        if node not in REQUIRED_NODES:
            raise ValueError(f"Unknown deterministic node: {node}")
        if node not in self.completed:
            self.completed.append(node)


class LegalWorkflow:
    def __init__(self, ledger: EvidenceLedger, verifier: CitationVerifier | None = None):
        self.ledger = ledger
        self.verifier = verifier or CitationVerifier()

    def final_review(self, state: WorkflowState, *, jurisdiction: str, effective_on: str, claim_terms: tuple[str, ...]) -> None:
        missing = [node for node in REQUIRED_NODES[:-1] if node not in state.completed]
        if missing:
            raise RuntimeError(f"Workflow violation: required nodes not complete: {', '.join(missing)}")
        if not state.citation_id:
            raise RuntimeError("Workflow violation: no citation selected")
        result = self.verifier.verify(self.ledger, state.citation_id, jurisdiction=jurisdiction, effective_on=effective_on, claim_terms=claim_terms)
        if not result.citation_valid:
            raise RuntimeError("Citation verification failed; return to retrieval/reasoning")
        state.complete("FINAL_REVIEW")
