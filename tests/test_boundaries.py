import json
import unittest
from http.client import RemoteDisconnected
from urllib.error import HTTPError
from unittest.mock import patch

from codelaw.citations import CitationVerifier
from codelaw.config import SettingsError, load_settings
from codelaw.evidence import Evidence, EvidenceLedger
from codelaw.evaluator import evaluate
from codelaw.live import NvidiaChatClient
from codelaw.metrics import Metrics
from codelaw.nvidia import NvidiaEmbeddingClient, embedding_client


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

    def test_private_embedding_endpoint_can_be_anonymous(self):
        response = _Response({"data": [{"index": 0, "embedding": [1.0]}]})
        client = NvidiaEmbeddingClient(base_url="http://dgx.test/v1", api_key=None, model="embed-model", require_api_key=False)
        with patch("codelaw.nvidia.urlopen", return_value=response) as open_url:
            self.assertEqual(client.embed(["text"]), [[1.0]])
        request = open_url.call_args.args[0]
        self.assertIsNone(request.get_header("Authorization"))

    def test_embedding_client_uses_dgx_defaults_without_a_secret(self):
        settings = load_settings({})
        client = embedding_client(settings)
        self.assertEqual(client.base_url, "http://192.168.1.6:8002/v1")
        self.assertEqual(client.model, "nvidia/Nemotron-3-Embed-1B-BF16")
        self.assertFalse(client.require_api_key)

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
            "NVIDIA_STREAM": "false",
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
            "temperature": 1,
            "max_tokens": 256,
            "seed": 0,
            "stream": False,
            "reasoning_effort": "max",
            "messages": [{"role": "user", "content": "hello"}],
        })

    def test_chat_client_collects_streamed_answer_content(self):
        class StreamResponse(_Response):
            def __init__(self):
                self.lines = iter([
                    b'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"first"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":" second"}}]}\n',
                    b'data: [DONE]\n',
                ])

            def readline(self):
                return next(self.lines, b"")

        settings = load_settings({
            "NVIDIA_API_KEY": "test-key",
            "NVIDIA_BASE_URL": "https://nvidia.test/v1/",
            "LEGALBENCH_LIVE_CONFIRM": "true",
        })
        with patch("codelaw.live.urlopen", return_value=StreamResponse()):
            self.assertEqual(NvidiaChatClient(settings).ask("hello"), "first second")

    def test_chat_client_does_not_retry_non_transient_http_errors(self):
        settings = load_settings({
            "NVIDIA_API_KEY": "test-key",
            "LEGALBENCH_LIVE_CONFIRM": "true",
            "NVIDIA_RETRIES": "2",
        })
        error = HTTPError("https://nvidia.test/v1/chat/completions", 400, "bad request", {}, None)
        with patch("codelaw.live.urlopen", side_effect=error) as open_url:
            with self.assertRaises(HTTPError):
                NvidiaChatClient(settings).ask("hello")
        self.assertEqual(open_url.call_count, 1)

    def test_chat_client_polls_an_accepted_request(self):
        class PendingResponse(_Response):
            status = 202

        class CompletedResponse(_Response):
            status = 200

        settings = load_settings({
            "NVIDIA_API_KEY": "test-key",
            "LEGALBENCH_LIVE_CONFIRM": "true",
            "NVIDIA_STREAM": "false",
        })
        with patch("codelaw.live.urlopen", side_effect=[
            PendingResponse({"requestId": "request-1"}),
            CompletedResponse({"choices": [{"message": {"content": "done"}}]}),
        ]) as open_url:
            self.assertEqual(NvidiaChatClient(settings).ask("hello"), "done")
        self.assertEqual(open_url.call_count, 2)
        self.assertEqual(open_url.call_args_list[1].args[0].full_url, "https://integrate.api.nvidia.com/v1/status/request-1")

    def test_chat_client_retries_two_transient_timeouts_when_configured(self):
        settings = load_settings({
            "NVIDIA_API_KEY": "test-key",
            "LEGALBENCH_LIVE_CONFIRM": "true",
            "NVIDIA_STREAM": "false",
            "NVIDIA_RETRIES": "2",
        })
        with patch("codelaw.live.urlopen", side_effect=[TimeoutError(), TimeoutError(), _Response({"choices": [{"message": {"content": "done"}}]})]) as open_url, patch("codelaw.live.time.sleep") as sleep:
            self.assertEqual(NvidiaChatClient(settings).ask("hello"), "done")
        self.assertEqual(open_url.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    def test_chat_client_retries_remote_disconnect_when_configured(self):
        settings = load_settings({
            "NVIDIA_API_KEY": "test-key",
            "LEGALBENCH_LIVE_CONFIRM": "true",
            "NVIDIA_STREAM": "false",
            "NVIDIA_RETRIES": "2",
        })
        with patch("codelaw.live.urlopen", side_effect=[RemoteDisconnected("closed"), _Response({"choices": [{"message": {"content": "done"}}]})]) as open_url, patch("codelaw.live.time.sleep") as sleep:
            self.assertEqual(NvidiaChatClient(settings).ask("hello"), "done")
        self.assertEqual(open_url.call_count, 2)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1])


if __name__ == "__main__":
    unittest.main()
