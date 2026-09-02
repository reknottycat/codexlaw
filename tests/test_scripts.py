import io
import os
import unittest
from contextlib import redirect_stdout
from urllib.error import HTTPError
from unittest.mock import patch

from codelaw.config import load_settings
from scripts.check_config import main
from scripts import run_k3_serial


class CheckConfigScriptTest(unittest.TestCase):
    def test_reports_effective_token_budget_without_exposing_a_key(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(output):
            exit_code = main()
        self.assertEqual(exit_code, 2)
        self.assertIn("max_tokens=20000", output.getvalue())
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
