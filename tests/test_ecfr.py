import unittest

from codelaw.ecfr import parse_title_xml


class EcfrParserTest(unittest.TestCase):
    def test_parser_keeps_original_text_and_provenance(self):
        xml = "<ROOT><DIV8 TYPE='SECTION'><SECTNO>§1.1</SECTNO><SUBJECT>Assignment</SUBJECT><P>Written consent is required.</P></DIV8></ROOT>".encode()
        rows = parse_title_xml(xml, title=16, source_url="https://example.gov/title-16.xml", issue_date="2026-08-26")
        self.assertEqual(rows[0].citation, "16 CFR §1.1")
        self.assertIn("Written consent", rows[0].text)
        self.assertTrue(rows[0].sha256)
