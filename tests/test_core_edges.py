import unittest

from codelaw.config import SettingsError, load_settings
from codelaw.ecfr import parse_title_xml
from codelaw.retrieval import BM25Index, DenseIndex, Document, HybridRetriever
from codelaw.workflow import REQUIRED_NODES, WorkflowState


class ConfigEdgeTest(unittest.TestCase):
    def test_load_settings_normalizes_url_and_boolean_values(self):
        settings = load_settings({
            "NVIDIA_BASE_URL": "https://nvidia.test/v1///",
            "LEGALBENCH_LIVE_CONFIRM": " YES ",
        })
        self.assertEqual(settings.nvidia_base_url, "https://nvidia.test/v1")
        self.assertTrue(settings.live_confirmed)

    def test_load_settings_rejects_non_numeric_values(self):
        with self.assertRaisesRegex(SettingsError, "must be numeric"):
            load_settings({"LIVE_LEGALBENCH_INTERVAL_SECONDS": "slow"})

    def test_load_settings_rejects_non_positive_case_count(self):
        with self.assertRaisesRegex(SettingsError, "must be positive"):
            load_settings({"LIVE_LEGALBENCH_CASES": "0"})

    def test_load_settings_rejects_non_finite_interval(self):
        with self.assertRaisesRegex(SettingsError, "must be positive"):
            load_settings({"LIVE_LEGALBENCH_INTERVAL_SECONDS": "nan"})

    def test_load_settings_reads_output_token_budget(self):
        settings = load_settings({"LIVE_LEGALBENCH_MAX_TOKENS": "256"})
        self.assertEqual(settings.max_tokens, 256)

    def test_default_output_token_budget_allows_reasoning(self):
        self.assertEqual(load_settings({}).max_tokens, 20000)


class WorkflowStateEdgeTest(unittest.TestCase):
    def test_complete_rejects_unknown_nodes(self):
        with self.assertRaisesRegex(ValueError, "Unknown deterministic node"):
            WorkflowState().complete("MODEL_DECISION")

    def test_complete_is_idempotent(self):
        state = WorkflowState()
        state.complete(REQUIRED_NODES[0])
        state.complete(REQUIRED_NODES[0])
        self.assertEqual(state.completed, [REQUIRED_NODES[0]])


class EcfrParserEdgeTest(unittest.TestCase):
    def test_parser_skips_sections_without_number(self):
        xml = """
          <ROOT>
            <DIV8 TYPE="SECTION"><SECTNO>§1.1</SECTNO><SUBJECT>Assignment</SUBJECT><P>Consent.</P></DIV8>
            <DIV8 TYPE="SECTION"><SUBJECT>No number</SUBJECT><P>Ignored.</P></DIV8>
          </ROOT>
        """.encode()
        rows = parse_title_xml(xml, title=16, source_url="https://example.gov/title-16.xml", issue_date="2026-08-26")
        self.assertEqual([row.citation for row in rows], ["16 CFR §1.1"])


class RetrievalEdgeTest(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document("assignment", "Assignment requires written consent", {"jurisdiction": "US"}),
            Document("liability", "Liability is limited by this rule", {"jurisdiction": "US"}),
        ]

    def test_bm25_query_is_case_insensitive_and_honors_limit(self):
        rows = BM25Index(self.documents).search("ASSIGNMENT", limit=1)
        self.assertEqual([document.document_id for document, _ in rows], ["assignment"])

    def test_bm25_returns_no_rows_for_an_unmatched_query(self):
        self.assertEqual(BM25Index(self.documents).search("jurisdiction"), [])

    def test_hybrid_retriever_deduplicates_graph_documents(self):
        documents = self.documents

        class Graph:
            def expand(self, document_ids):
                return [documents[0], Document("exception", "A related exception", {})]

        rows = HybridRetriever(BM25Index(documents), Graph()).retrieve("assignment")
        self.assertEqual([document.document_id for document in rows], ["assignment", "exception"])

    def test_dense_index_ranks_by_cosine_similarity(self):
        rows = DenseIndex(self.documents, [[1.0, 0.0], [0.0, 1.0]]).search([0.9, 0.1], limit=1)
        self.assertEqual(rows[0][0].document_id, "assignment")

    def test_dense_index_rejects_mismatched_vectors(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            DenseIndex(self.documents, [[1.0, 0.0]])

    def test_hybrid_retriever_can_fuse_dense_and_lexical_results(self):
        class Embedder:
            def embed(self, texts):
                return [[0.0, 1.0] for _ in texts]

        class Graph:
            def expand(self, document_ids):
                return []

        retriever = HybridRetriever(
            BM25Index(self.documents),
            Graph(),
            dense=DenseIndex(self.documents, [[1.0, 0.0], [0.0, 1.0]]),
            embedder=Embedder(),
        )
        self.assertEqual(retriever.retrieve("unmatched", limit=1)[0].document_id, "liability")

    def test_hybrid_retriever_requires_embedder_for_dense_index(self):
        with self.assertRaisesRegex(ValueError, "requires an embedder"):
            HybridRetriever(BM25Index(self.documents), lambda: None, dense=DenseIndex(self.documents, [[1.0, 0.0], [0.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
