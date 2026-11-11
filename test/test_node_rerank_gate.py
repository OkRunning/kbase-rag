"""Rerank Gate 单元测试。"""

import unittest
from unittest.mock import patch

from app.query_process.agent.nodes import node_rerank_gate as gate_module
from app.query_process.agent.state import create_query_default_state


class NodeRerankGateTest(unittest.TestCase):
    """Rerank 前置跳过门控测试用例。"""

    def _state(self, query_type="fact", rrf_chunks=None, web_search_docs=None):
        """构造 Rerank Gate 测试状态。

        Args:
            query_type: 查询类型。
            rrf_chunks: Evidence Fusion 输出候选。
            web_search_docs: Web Search 原始候选。

        Returns:
            查询流程状态。
        """
        return create_query_default_state(
            session_id="rerank_gate",
            rewritten_query="HAK 180 额定功率是多少？",
            query_type=query_type,
            rrf_chunks=rrf_chunks or [],
            web_search_docs=web_search_docs or [],
        )

    def _local_doc(self, index=1, score=0.9, source_hits=None):
        """构造本地 EvidenceDocument。

        Args:
            index: 文档序号。
            score: 原始召回分数。
            source_hits: 多源命中列表。

        Returns:
            EvidenceDocument 字典。
        """
        return {
            "evidence_id": f"chunk:local_{index}",
            "source_type": "local",
            "source": "local_vector",
            "text": f"本地证据 {index}",
            "chunk_id": f"local_{index}",
            "title": f"本地文档 {index}",
            "score": score,
            "rrf_score": 0.01,
            "source_hits": source_hits or [{"source": "local_vector", "rank": index, "score": score}],
        }

    def _run_gate(self, state):
        """执行 Rerank Gate 并隔离任务与指标副作用。

        Args:
            state: 查询流程状态。

        Returns:
            Rerank Gate 执行后的状态。
        """
        with (
            patch.object(gate_module, "add_running_task", return_value=None),
            patch.object(gate_module, "add_done_task", return_value=None),
            patch.object(gate_module, "record_rerank_docs", return_value=None),
        ):
            return gate_module.node_rerank_gate(state)

    def test_no_candidates_skips_rerank_as_insufficient(self):
        """验证没有候选时跳过 Rerank 并标记资料不足。

        Returns:
            None.
        """
        result = self._run_gate(self._state(rrf_chunks=[]))

        self.assertFalse(result["rerank_gate"]["should_rerank"])
        self.assertEqual(result["rerank_gate"]["reason"], "no_candidates")
        self.assertEqual(result["reranked_docs"], [])
        self.assertEqual(result["rerank_quality"]["answerability"], "insufficient")

    def test_single_candidate_skips_rerank(self):
        """验证单候选无需排序，直接跳过 Rerank。

        Returns:
            None.
        """
        result = self._run_gate(self._state(rrf_chunks=[self._local_doc(score=0.86)]))

        self.assertFalse(result["rerank_gate"]["should_rerank"])
        self.assertEqual(result["rerank_gate"]["reason"], "single_candidate")
        self.assertEqual(len(result["reranked_docs"]), 1)
        self.assertTrue(result["reranked_docs"][0]["rerank_skipped"])
        self.assertEqual(result["rerank_quality"]["answerability"], "answerable")

    def test_high_confidence_simple_local_query_skips_rerank(self):
        """验证简单事实问题本地高置信小候选集可跳过 Rerank。

        Returns:
            None.
        """
        docs = [self._local_doc(index=1, score=0.88), self._local_doc(index=2, score=0.82)]
        result = self._run_gate(self._state(query_type="fact", rrf_chunks=docs))

        self.assertFalse(result["rerank_gate"]["should_rerank"])
        self.assertEqual(result["rerank_gate"]["reason"], "high_confidence_local_simple_query")
        self.assertEqual(len(result["reranked_docs"]), 2)

    def test_hyde_candidates_keep_rerank(self):
        """验证 HyDE 参与时继续执行 Rerank。

        Returns:
            None.
        """
        docs = [
            self._local_doc(
                score=0.9,
                source_hits=[
                    {"source": "local_vector", "rank": 1, "score": 0.9},
                    {"source": "hyde", "rank": 1, "score": 0.8},
                ],
            )
        ]
        result = self._run_gate(self._state(query_type="fact", rrf_chunks=docs))

        self.assertTrue(result["rerank_gate"]["should_rerank"])
        self.assertEqual(result["rerank_gate"]["reason"], "hyde_candidates")
        self.assertEqual(result["reranked_docs"], [])

    def test_web_candidates_keep_rerank(self):
        """验证 Web 候选参与时继续执行 Rerank。

        Returns:
            None.
        """
        docs = [
            {
                "evidence_id": "web:https://example.com/a",
                "source_type": "web",
                "source": "web",
                "text": "Web 证据",
                "url": "https://example.com/a",
                "score": 0.8,
                "rrf_score": 0.01,
                "source_hits": [{"source": "web", "rank": 1, "score": 0.8}],
            }
        ]
        result = self._run_gate(self._state(query_type="fact", rrf_chunks=docs))

        self.assertTrue(result["rerank_gate"]["should_rerank"])
        self.assertEqual(result["rerank_gate"]["reason"], "web_candidates")

    def test_complex_query_keeps_rerank(self):
        """验证复杂查询类型继续执行 Rerank。

        Returns:
            None.
        """
        docs = [self._local_doc(index=1, score=0.9), self._local_doc(index=2, score=0.86)]
        result = self._run_gate(self._state(query_type="comparison", rrf_chunks=docs))

        self.assertTrue(result["rerank_gate"]["should_rerank"])
        self.assertEqual(result["rerank_gate"]["reason"], "complex_query_type")


if __name__ == "__main__":
    unittest.main()
