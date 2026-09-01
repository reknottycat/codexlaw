"""Architecture B: Codex model/session boundary plus external legal workflow state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .workflow import LegalWorkflow, WorkflowState


@dataclass(frozen=True)
class AssistantTurn:
    content: str
    finish_reason: str | None = None
    reasoning_content: str | None = None


class CodexLegalAdapter:
    def __init__(self, ask: Callable[[str], AssistantTurn], workflow: LegalWorkflow):
        self.ask, self.workflow = ask, workflow
        self.turns: list[AssistantTurn] = []

    def analyze(self, prompt: str, state: WorkflowState, *, jurisdiction: str, effective_on: str, claim_terms: tuple[str, ...]) -> str:
        turn = self.ask(prompt)
        self.turns.append(turn)
        self.workflow.final_review(state, jurisdiction=jurisdiction, effective_on=effective_on, claim_terms=claim_terms)
        return turn.content
