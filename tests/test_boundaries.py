import json
import unittest
from unittest.mock import patch

from codelaw.citations import CitationVerifier
from codelaw.config import SettingsError, load_settings
from codelaw.evidence import Evidence, EvidenceLedger
from codelaw.evaluator import evaluate
from codelaw.live import NvidiaChatClient
from codelaw.metrics import Metrics
from codelaw.nvidia import NvidiaEmbeddingClient


class CitationVerifierTest(unittest.TestCase):
    def setUp(self):
        self.ledger = EvidenceLedger()
        self.ledger.add(Evidence("e-1", "16-CFR-1", "Assignment requires written consent", "US", "2025-01-01"))
        self.verifier = CitationVerifier()

    def test_rejects_unknown_evidence(self):
        result = self.verifier.verify(self.ledger, "missing", jurisdiction="US", effective_on="2026-01-01", claim_terms=("assignment",))
        self.assertFalse(result.citation_valid)
        self.assertFalse(result.source_exists)
        self.assertEqual(result.confidence, 0.0)

    def test_rejects_wrong_jurisdiction(self):
        result = self.verifier.verify(self.ledger, "e-1", jurisdiction="CA", effective_on="2026-01-01", claim_terms=("assignment",))
        self.assertFalse(result.citation_valid)
        self.assertFalse(result.jurisdiction_valid)

    def test_rejects_evidence_published_after_requested_effective_date(self):
        result = self.verifier.verify(self.ledger, "e-1", jurisdiction="US", effective_on="2024-12-31", claim_terms=("assignment",))
        self.assertFalse(result.citation_valid)
        self.assertFalse(result.effective_date_valid)

    def test_rejects_evidence_that_does_not_support_every_claim_term(self):
        result = self.verifier.verify(self.ledger, "e-1", jurisdiction="US", effective_on="2026-01-01", claim_terms=("assignment", "liability"))
        self.assertFalse(result.citation_valid)
        self.assertFalse(result.supports_claim)


class EvidenceLedgerTest(unittest.TestCase):
    def test_rejects_blank_text(self):
        with self.assertRaisesRegex(ValueError, "must be retained"):
            EvidenceLedger().add(Evidence("e-1", "source", "  ", "US"))

    def test_require_preserves_requested_order(self):
        ledger = EvidenceLedger()
        ledger.add(Evidence("first", "source-1", "first", "US"))
        ledger.add(Evidence("second", "source-2", "second", "US"))
        self.assertEqual([row.evidence_id for row in ledger.require("second", "first")], ["second", "first"])

    def test_require_reports_all_missing_ids(self):
        with self.assertRaisesRegex(KeyError, "missing-1, missing-2"):
            EvidenceLedger().require("missing-1", "missing-2")


class EvaluatorTest(unittest.TestCase):
    def test_success_requires_correct_answer_citation_and_all_workflow_nodes(self):
        required = ["CONTRACT_INTAKE", "JURISDICTION", "EFFECTIVE_DATE", "EXCEPTION_CHECK", "CITATION_VERIFICATION", "FINAL_REVIEW"]
        self.assertFalse(evaluate(answer="No", expected_answer="Yes", citation_valid=True, completed_nodes=required).success)
        self.assertFalse(evaluate(answer="Yes", expected_answer="Yes", citation_valid=False, completed_nodes=required).success)
        self.assertFalse(evaluate(answer="Yes", expected_answer="Yes", citation_valid=True, completed_nodes=required[:-1]).success)


class MetricsTest(unittest.TestCase):
    def test_render_increments_and_sorts_labels(self):
        metrics = Metrics()
        metrics.inc("requests_total", task_type="intake", architecture="codex")
        metrics.inc("requests_total", task_type="intake", architecture="codex")
        self.assertEqual(metrics.render(), 'requests_total{architecture="codex",task_type="intake"} 2\n')

    def test_render_escapes_prometheus_label_values(self):
        metrics = Metrics()
        metrics.inc("requests_total", jurisdiction='US"\\\n')
        self.assertEqual(metrics.render(), 'requests_total{jurisdiction="US\\"\\\\\\n"} 1\n')


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


class NvidiaClientTest(unittest.TestCase):
    def test_embedding_request_and_result_order(self):
        response = _Response({"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]})
        client = NvidiaEmbeddingClient(base_url="https://nvidia.test/v1/", api_key="test-key", model="embed-model")
        with patch("codelaw.nvidia.urlopen", return_value=response) as open_url:
            self.assertEqual(client.embed(["first", "second"]), [[1.0], [2.0]])
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "https://nvidia.test/v1/embeddings")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(request.data), {"model": "embed-model", "input": ["first", "second"], "encoding_format": "float"})

    def test_chat_client_rejects_missing_key_before_network_request(self):
        settings = load_settings({"LEGALBENCH_LIVE_CONFIRM": "true"})
        with patch("codelaw.live.urlopen") as open_url:
            with self.assertRaisesRegex(SettingsError, "NVIDIA_API_KEY"):
                NvidiaChatClient(settings).ask("hello")
        open_url.assert_not_called()

    def test_chat_request_uses_configured_model_and_returns_message_content(self):
        settings = load_settings({
            "NVIDIA_API_KEY": "test-key",
            "NVIDIA_BASE_URL": "https://nvidia.test/v1/",
            "NVIDIA_CHAT_MODEL": "chat-model",
            "LIVE_LEGALBENCH_MAX_TOKENS": "256",
            "LEGALBENCH_LIVE_CONFIRM": "true",
        })
        response = _Response({"choices": [{"message": {"content": "ok"}}]})
        with patch("codelaw.live.urlopen", return_value=response) as open_url:
            self.assertEqual(NvidiaChatClient(settings).ask("hello"), "ok")
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "https://nvidia.test/v1/chat/completions")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(request.data), {
            "model": "chat-model",
            "temperature": 0,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hello"}],
        })


if __name__ == "__main__":
    unittest.main()
