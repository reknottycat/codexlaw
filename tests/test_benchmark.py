import json
import tempfile
import unittest
from pathlib import Path

from codelaw.benchmark import answer_matches, parse_decision, run_ab, run_case, select_cases, summarize
from codelaw.lawgent_adapter import LawgentAdapter


class BenchmarkParsingTest(unittest.TestCase):
    def test_parser_accepts_json_surrounded_by_markdown(self):
        decision = parse_decision('```json\n{"answer":"B","citation_ids":["e-1"],"confidence":0.8}\n```')
        self.assertEqual(decision.answer, "B")
        self.assertEqual(decision.citation_ids, ["e-1"])
        self.assertIsNone(decision.parse_error)

    def test_multiple_choice_accepts_letter_or_index(self):
        self.assertTrue(answer_matches(answer="option C", expected="2", answer_type="multiple_choice"))
        self.assertTrue(answer_matches(answer="2", expected="2", answer_type="multiple_choice"))
        self.assertFalse(answer_matches(answer="A", expected="2", answer_type="multiple_choice"))

    def test_evidence_span_uses_token_overlap_for_faithful_answer(self):
        self.assertTrue(answer_matches(
            answer="written consent is required",
            expected="Written consent is required by this agreement",
            answer_type="evidence_span",
        ))


class BenchmarkRunnerTest(unittest.TestCase):
    def setUp(self):
        self.case = {
            "case_id": "fixture:1",
            "source": "fixture",
            "task": "fixture",
            "prompt": "Does assignment require consent?",
            "expected_answer": "Yes",
            "answer_type": "classification",
            "jurisdiction": "US",
            "effective_on": None,
            "evidence": [{
                "evidence_id": "fixture:1:evidence",
                "source_id": "fixture-source",
                "text": "Assignment requires written consent.",
                "jurisdiction": "US",
                "effective_on": None,
            }],
        }

    def test_a_and_b_share_evaluator_and_workflow_gates(self):
        ask = lambda _: '{"answer":"Yes","citation_ids":["fixture:1:evidence"],"confidence":0.9}'
        lawgent = LawgentAdapter(lambda _: '[{"categories":["assignment_novation"]}]')
        a = run_case(self.case, architecture="A", ask=ask, lawgent=lawgent)
        b = run_case(self.case, architecture="B", ask=ask, lawgent=lawgent)
        self.assertTrue(a["success"])
        self.assertTrue(b["success"])
        self.assertEqual(a["completed_nodes"], b["completed_nodes"])
        self.assertEqual(a["lawgent_categories"], ["assignment_novation"])
        self.assertEqual(b["lawgent_categories"], [])

    def test_serial_runner_waits_between_every_model_request(self):
        waits = []
        calls = []
        ask = lambda prompt: calls.append(prompt) or '{"answer":"Yes","citation_ids":["fixture:1:evidence"]}'
        rows = run_ab(
            [self.case],
            ask=ask,
            interval_seconds=30,
            lawgent=LawgentAdapter(lambda _: "[]"),
            sleep=waits.append,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(waits, [30])

    def test_invalid_citation_does_not_count_as_completed_verification(self):
        row = run_case(
            self.case,
            architecture="B",
            ask=lambda _: '{"answer":"Yes","citation_ids":["missing"]}',
            lawgent=LawgentAdapter(lambda _: "[]"),
        )
        self.assertFalse(row["citation_valid"])
        self.assertNotIn("CITATION_VERIFICATION", row["completed_nodes"])
        self.assertFalse(row["workflow_compliant"])

    def test_select_cases_balances_requested_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            rows = []
            for source in ("a", "b", "c"):
                for index in range(3):
                    rows.append({"source": source, "case_id": f"{source}-{index}"})
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            selected = select_cases(path, limit=3, sources=["a", "b", "c"])
            self.assertEqual([row["source"] for row in selected], ["a", "b", "c"])

    def test_summary_reports_architecture_rates(self):
        summary = summarize([
            {"architecture": "A", "answer_correct": True, "citation_valid": True, "workflow_compliant": True, "success": True, "error": None},
            {"architecture": "B", "answer_correct": False, "citation_valid": True, "workflow_compliant": True, "success": False, "error": "wrong"},
        ])
        self.assertEqual(summary["architectures"]["A"]["success_rate"], 1.0)
        self.assertEqual(summary["architectures"]["B"]["errors"], 1)


if __name__ == "__main__":
    unittest.main()
