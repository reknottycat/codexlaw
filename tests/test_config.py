import unittest

from codelaw.config import SettingsError, load_settings


class SettingsTest(unittest.TestCase):
    def test_defaults_are_serial_k3_settings(self):
        settings = load_settings({})
        self.assertEqual(settings.chat_model, "moonshotai/kimi-k3")
        self.assertEqual(settings.interval_seconds, 30)
        self.assertEqual(settings.live_cases, 3)

    def test_live_run_needs_explicit_confirmation_and_key(self):
        settings = load_settings({"NVIDIA_API_KEY": "test", "LEGALBENCH_LIVE_CONFIRM": "false"})
        with self.assertRaisesRegex(SettingsError, "LEGALBENCH_LIVE_CONFIRM"):
            settings.require_live_provider()

    def test_live_run_rejects_short_interval(self):
        settings = load_settings({"NVIDIA_API_KEY": "test", "LEGALBENCH_LIVE_CONFIRM": "true", "LIVE_LEGALBENCH_INTERVAL_SECONDS": "29"})
        with self.assertRaisesRegex(SettingsError, "at least 30"):
            settings.require_live_provider()


if __name__ == "__main__":
    unittest.main()
