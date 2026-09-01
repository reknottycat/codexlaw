import unittest

from codelaw.codex_legal_adapter import AssistantTurn, CodexLegalAdapter
from codelaw.evidence import Evidence, EvidenceLedger
from codelaw.workflow import LegalWorkflow, WorkflowState


class CodexLegalAdapterTest(unittest.TestCase):
    def test_adapter_preserves_turn_but_uses_external_workflow(self):
        ledger = EvidenceLedger()
        ledger.add(Evidence("e", "source", "assignment needs consent", "US", "2025-01-01"))
        state = WorkflowState(citation_id="e")
        for node in ("CONTRACT_INTAKE", "JURISDICTION", "EFFECTIVE_DATE", "EXCEPTION_CHECK", "CITATION_VERIFICATION"):
            state.complete(node)
        adapter = CodexLegalAdapter(lambda _: AssistantTurn('{"answer":"Yes"}', "stop", "internal"), LegalWorkflow(ledger))
        self.assertEqual(adapter.analyze("classify", state, jurisdiction="US", effective_on="2026-01-01", claim_terms=("assignment", "consent")), '{"answer":"Yes"}')
        self.assertEqual(adapter.turns[0].reasoning_content, "internal")
