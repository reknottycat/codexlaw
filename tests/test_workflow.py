import unittest

from codelaw.evidence import Evidence, EvidenceLedger
from codelaw.workflow import LegalWorkflow, WorkflowState


class WorkflowTest(unittest.TestCase):
    def ready_state(self, citation_id="e-1"):
        state = WorkflowState(citation_id=citation_id)
        for node in ("CONTRACT_INTAKE", "JURISDICTION", "EFFECTIVE_DATE", "EXCEPTION_CHECK", "CITATION_VERIFICATION"):
            state.complete(node)
        return state

    def test_final_review_requires_all_deterministic_nodes(self):
        with self.assertRaisesRegex(RuntimeError, "required nodes"):
            LegalWorkflow(EvidenceLedger()).final_review(WorkflowState(), jurisdiction="US", effective_on="2026-01-01", claim_terms=("assignment",))

    def test_valid_evidence_reaches_final_review(self):
        ledger = EvidenceLedger()
        ledger.add(Evidence("e-1", "16-CFR-1", "assignment requires written consent", "US", "2025-01-01"))
        state = self.ready_state()
        LegalWorkflow(ledger).final_review(state, jurisdiction="US", effective_on="2026-01-01", claim_terms=("assignment", "consent"))
        self.assertIn("FINAL_REVIEW", state.completed)

    def test_missing_evidence_blocks_final_review(self):
        with self.assertRaisesRegex(RuntimeError, "Citation verification failed"):
            LegalWorkflow(EvidenceLedger()).final_review(self.ready_state(), jurisdiction="US", effective_on="2026-01-01", claim_terms=("assignment",))


if __name__ == "__main__":
    unittest.main()
