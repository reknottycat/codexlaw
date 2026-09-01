import unittest

from codelaw.config import load_settings
from codelaw.live import LiveTask, run_serial


class LiveGateTest(unittest.TestCase):
    def test_calls_are_strictly_serial_with_30_second_spacing(self):
        settings = load_settings({"NVIDIA_API_KEY": "test", "LEGALBENCH_LIVE_CONFIRM": "true", "LIVE_LEGALBENCH_INTERVAL_SECONDS": "30"})
        waits, calls = [], []
        rows = run_serial(settings, [LiveTask("1", "a"), LiveTask("2", "b")], lambda prompt: calls.append(prompt) or "ok", waits.append)
        self.assertEqual(calls, ["a", "b"])
        self.assertEqual(waits, [30])
        self.assertEqual(len(rows), 2)
