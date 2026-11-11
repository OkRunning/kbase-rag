"""Evidence Fusion Step 8 单元测试。"""

import unittest
from unittest.mock import patch

from app.query_process.agent.nodes import node_rerank as rerank_module
from app.query_process.agent.nodes import node_rrf as rrf_module
from app.query_process.agent.state import create_query_default_state


class EvidenceFusionStep8Test(unittest.TestCase):
    """EvidenceDocument 标准化、去重和 Rerank 输入测试用例。"""

    def test_rrf_merges_duplicate_local_and_hyde_hits(self):
        """验证 local 与 HyDE 命中同一 chunk 时只保留一条证据。

        Returns:
            None.
        """
        local_hit = {
            "id": "milvus_1",
            "distance": 0.9,
            "entity": {
                "chunk_id": "chunk_1",
                "content": "同一段证据内容",
                "title": "操作说明",
                "item_name": "HAK 180",
            },
        }
        hyde_hit = {
            "id": "milvus_1_hyde",
            "distance": 0.8,
            "entity": {
                "chunk_id": "chunk_1",
                "content": "同一段证据内容",
                "title": "操作说明",
                "item_name": "HAK 180",
            },
        }
        state = create_query_default_state(
            session_id="evidence_fusion_step8",
            embedding_chunks=[local_hit],
            hyde_embedding_chunks=[hyde_hit],
        )

        with (
            patch.object(rrf_module, "add_running_task", return_value=None),
            patch.object(rrf_module, "add_done_task", return_value=None),
            patch.object(rrf_module, "record_retrieval_docs", return_value=None),
        ):
            result = rrf_module.node_rrf(state)

        self.assertEqual(len(result["rrf_chunks"]), 1)
        doc = result["rrf_chunks"][0]
        self.assertEqual(doc["evidence_id"], "chunk:chunk_1")
        self.assertGreater(doc["rrf_score"], 0)
        self.assertEqual({hit["source"] for hit in doc["source_hits"]}, {"local_vector", "hyde"})

    def test_rrf_includes_web_as_evidence_document(self):
        """验证 Web 结果会进入统一 EvidenceDocument。

        Returns:
            None.
        """
        state = create_query_default_state(
            session_id="evidence_fusion_step8",
            web_search_docs=[
                {
                    "snippet": "Web 证据内容",
                    "title": "Web 标题",
                    "url": "https://example.com/a",
                    "domain": "example.com",
                    "trust_level": "external",
                }
            ],
        )

        with (
            patch.object(rrf_module, "add_running_task", return_value=None),
            patch.object(rrf_module, "add_done_task", return_value=None),
            patch.object(rrf_module, "record_retrieval_docs", return_value=None),
        ):
            result = rrf_module.node_rrf(state)

        self.assertEqual(len(result["rrf_chunks"]), 1)
        doc = result["rrf_chunks"][0]
        self.assertEqual(doc["source_type"], "web")
        self.assertEqual(doc["url"], "https://example.com/a")
        self.assertEqual(doc["text"], "Web 证据内容")

    def test_rerank_uses_evidence_documents_without_duplicate_web_merge(self):
        """验证 Rerank 直接消费 EvidenceDocument，不重复合并 Web。

        Returns:
            None.
        """
        rrf_chunks = [
            {
                "evidence_id": "web:https://example.com/a",
                "source_type": "web",
                "source": "web",
                "text": "Web 证据内容",
                "title": "Web 标题",
                "chunk_id": "",
                "url": "https://example.com/a",
                "rrf_score": 0.01,
                "source_hits": [{"source": "web", "rank": 1, "score": 0.0}],
            }
        ]
        web_search_docs = [
            {
                "snippet": "Web 证据内容",
                "title": "Web 标题",
                "url": "https://example.com/a",
            }
        ]

        merged_docs = rerank_module.step_1_merge_rrf_web_search_results(rrf_chunks, web_search_docs)

        self.assertEqual(len(merged_docs), 1)
        self.assertEqual(merged_docs[0]["evidence_id"], "web:https://example.com/a")
        self.assertEqual(merged_docs[0]["source"], "web")


if __name__ == "__main__":
    unittest.main()
