import unittest

from codelaw.evaluator import evaluate
from codelaw.metrics import Metrics


class EvaluatorMetricsTest(unittest.TestCase):
    def test_evaluator_requires_answer_citation_and_workflow(self):
        result = evaluate(answer="Yes", expected_answer="Yes", citation_valid=True, completed_nodes=["CONTRACT_INTAKE", "JURISDICTION", "EFFECTIVE_DATE", "EXCEPTION_CHECK", "CITATION_VERIFICATION", "FINAL_REVIEW"])
        self.assertTrue(result.success)

    def test_metrics_refuse_contract_id_as_label(self):
        with self.assertRaisesRegex(ValueError, "High-cardinality"):
            Metrics().inc("legal_agent_requests_total", contract_id="secret")
