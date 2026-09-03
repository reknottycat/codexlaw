import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from codelaw.config import load_settings
from scripts.check_config import main
from scripts import rescore_ab_results, run_k3_serial, run_project_ab_benchmark


class CheckConfigScriptTest(unittest.TestCase):
    def test_reports_effective_token_budget_without_exposing_a_key(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(output):
            exit_code = main()
        self.assertEqual(exit_code, 2)
        self.assertIn("max_tokens=65536", output.getvalue())
        self.assertIn("live_provider=blocked", output.getvalue())
        self.assertNotIn("NVIDIA_API_KEY", output.getvalue())


class RunK3SerialScriptTest(unittest.TestCase):
    def test_reports_provider_timeout_without_traceback(self):
        settings = load_settings({"NVIDIA_API_KEY": "test-key", "LEGALBENCH_LIVE_CONFIRM": "true"})
        output = io.StringIO()
        with patch.object(run_k3_serial, "load_settings", return_value=settings), patch.object(run_k3_serial, "real_tasks", return_value=[run_k3_serial.LiveTask("fixture", "real")]), patch.object(run_k3_serial, "run_serial", side_effect=TimeoutError), redirect_stdout(output):
            exit_code = run_k3_serial.main()
        self.assertEqual(exit_code, 1)
        self.assertIn("NVIDIA K3 request timed out", output.getvalue())
        self.assertIn("retry later", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())


class RescoreABResultsScriptTest(unittest.TestCase):
    def test_rescores_a_concise_grounded_evidence_answer_without_rewriting_source(self):
        row = {
            "architecture": "A",
            "answer": "end of the current calendar year automatically renewed",
            "expected_answer": "The agreement remains in effect until the end of the current calendar year and is automatically renewed.",
            "answer_type": "evidence_span",
            "citation_valid": True,
            "workflow_compliant": True,
            "success": False,
            "error": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            source = json.dumps(row)
            path.write_text(source + "\n", encoding="utf-8")
            output = io.StringIO()
            with patch.object(sys, "argv", ["rescore_ab_results.py", str(path)]), redirect_stdout(output):
                exit_code = rescore_ab_results.main()
            self.assertEqual(exit_code, 0)
            self.assertEqual(path.read_text(encoding="utf-8"), source + "\n")
        self.assertIn('"success_rate": 1.0', output.getvalue())


class ProjectBenchmarkScriptTest(unittest.TestCase):
    def test_reports_project_quality_and_availability_separately(self):
        metrics = run_project_ab_benchmark._metrics([
            {"project": "CodexLaw", "provider_error": None, "answer_correct": True, "citation_valid": True, "common_success": True},
            {"project": "Lawgent", "provider_error": "timeout", "answer_correct": False, "citation_valid": False, "common_success": False},
        ])
        self.assertEqual(metrics["projects"]["CodexLaw"]["common_success_rate"], 1.0)
        self.assertEqual(metrics["projects"]["Lawgent"]["execution_completion_rate"], 0.0)

    def test_reports_provider_http_error_without_traceback(self):
        settings = load_settings({"NVIDIA_API_KEY": "test-key", "LEGALBENCH_LIVE_CONFIRM": "true"})
        output = io.StringIO()
        error = HTTPError("https://nvidia.test/v1/chat/completions", 400, "bad request", {}, None)
        with patch.object(run_k3_serial, "load_settings", return_value=settings), patch.object(run_k3_serial, "real_tasks", return_value=[run_k3_serial.LiveTask("fixture", "real")]), patch.object(run_k3_serial, "run_serial", side_effect=error), redirect_stdout(output):
            exit_code = run_k3_serial.main()
        self.assertEqual(exit_code, 1)
        self.assertIn("endpoint returned HTTP 400", output.getvalue())
        self.assertIn("verify model, key, payload, and endpoint", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())


if __name__ == "__main__":
    unittest.main()
