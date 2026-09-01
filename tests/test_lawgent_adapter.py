import unittest

from codelaw.lawgent_adapter import LawgentAdapter


class LawgentAdapterTest(unittest.TestCase):
    def test_adapter_uses_lawgent_extractor_result(self):
        adapter = LawgentAdapter(lambda _: '[{"categories":["anti-assignment","consent"]}]')
        result = adapter.contract_intake("No assignment without consent")
        self.assertEqual(result["categories"], ["anti-assignment", "consent"])


if __name__ == "__main__":
    unittest.main()
