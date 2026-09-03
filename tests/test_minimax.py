import json
import subprocess
import unittest

from codelaw.minimax import MiniMaxChatClient, MiniMaxRuntime


class MiniMaxChatClientTest(unittest.TestCase):
    def test_uses_argument_list_and_never_requires_a_key_argument(self):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, stdout="answer\n", stderr="")

        answer = MiniMaxChatClient(MiniMaxRuntime(model="MiniMax-M3", max_tokens=512), run=run).ask(
            "question", system="system"
        )
        self.assertEqual(answer, "answer")
        self.assertEqual(calls[0][0][:4], ["mmx.cmd", "text", "chat", "--model"])
        self.assertNotIn("--api-key", calls[0][0])
        self.assertIn("--non-interactive", calls[0][0])
        self.assertIn("--messages-file", calls[0][0])
        self.assertEqual(json.loads(calls[0][1]["input"])[0]["content"], "system")
        self.assertEqual(calls[0][1]["encoding"], "utf-8")
        self.assertEqual(calls[0][1]["errors"], "replace")

    def test_reports_empty_and_failed_cli_responses(self):
        failed = MiniMaxChatClient(
            MiniMaxRuntime(),
            run=lambda command, **kwargs: subprocess.CompletedProcess(command, 2, stdout="", stderr="failed"),
        )
        with self.assertRaisesRegex(RuntimeError, "status 2"):
            failed.ask("question")
        empty = MiniMaxChatClient(
            MiniMaxRuntime(),
            run=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
        )
        with self.assertRaisesRegex(RuntimeError, "empty response"):
            empty.ask("question")

    def test_replaces_lone_surrogates_before_invoking_windows_cli(self):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        MiniMaxChatClient(MiniMaxRuntime(), run=run).ask("bad\udc9d evidence")
        self.assertNotIn("\udc9d", calls[0])

    def test_extracts_text_from_raw_minimax_api_response(self):
        raw = '{"content":[{"type":"text","text":"normalized"}]}'
        client = MiniMaxChatClient(
            MiniMaxRuntime(),
            run=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=raw, stderr=""),
        )
        self.assertEqual(client.ask("question"), "normalized")
