"""Independent result evaluator; the tested harness never grades itself."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Evaluation:
    answer_correct: bool
    citation_valid: bool
    workflow_compliant: bool

    @property
    def success(self) -> bool:
        return self.answer_correct and self.citation_valid and self.workflow_compliant


def evaluate(*, answer: str, expected_answer: str, citation_valid: bool, completed_nodes: list[str]) -> Evaluation:
    required = {"CONTRACT_INTAKE", "JURISDICTION", "EFFECTIVE_DATE", "EXCEPTION_CHECK", "CITATION_VERIFICATION", "FINAL_REVIEW"}
    return Evaluation(answer.strip().lower() == expected_answer.strip().lower(), citation_valid, required.issubset(completed_nodes))
