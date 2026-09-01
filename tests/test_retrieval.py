import unittest

from codelaw.nvidia import NvidiaEmbeddingClient
from codelaw.retrieval import BM25Index, Document, HybridRetriever


class Graph:
    def expand(self, document_ids):
        return [Document("exception", "assignment exception requires written consent", {})] if "rule" in document_ids else []


class RetrievalTest(unittest.TestCase):
    def test_bm25_finds_legal_entry_then_expands_graph(self):
        retriever = HybridRetriever(BM25Index([Document("rule", "assignment is prohibited", {"jurisdiction": "US"})]), Graph())
        self.assertEqual({row.document_id for row in retriever.retrieve("assignment")}, {"rule", "exception"})

    def test_embedding_client_never_sends_request_without_key(self):
        with self.assertRaisesRegex(Exception, "NVIDIA_API_KEY"):
            NvidiaEmbeddingClient(base_url="https://integrate.api.nvidia.com/v1", api_key=None, model="nvidia/llama-nemotron-embed-vl-1b-v2").embed(["x"])
