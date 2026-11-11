"""Rerank 节点 Step 4 单元测试。"""

import unittest
from unittest.mock import Mock, patch

from app.query_process.agent.nodes import node_rerank as rerank_module
from app.query_process.agent.state import create_query_default_state


class NodeRerankStep4Test(unittest.TestCase):
    """Rerank 输入限制、质量诊断和降级测试用例。"""

    def setUp(self):
        """重置 Reranker 熔断状态。

        Returns:
            None.
        """
        rerank_module._RERANK_CIRCUIT_OPEN_UNTIL = 0.0

    def _state(self, local_count=3, web_count=2):
        """构造 Rerank 节点测试状态。

        Args:
            local_count: 本地 RRF 候选数量。
            web_count: Web 候选数量。

        Returns:
            查询流程状态。
        """
        rrf_chunks = [
            {
                "entity": {
                    "chunk_id": f"local_{index}",
                    "content": f"本地候选内容 {index}",
                    "title": f"本地文档 {index}",
                }
            }
            for index in range(local_count)
        ]
        web_docs = [
            {
                "snippet": f"Web 候选内容 {index}",
                "title": f"Web 文档 {index}",
                "url": f"https://example.com/{index}",
            }
            for index in range(web_count)
        ]
        return create_query_default_state(
            session_id="rerank_step4",
            rewritten_query="HAK 180 操作步骤",
            rrf_chunks=rrf_chunks,
            web_search_docs=web_docs,
        )

    def test_rerank_limits_input_candidates_and_outputs_quality(self):
        """验证 Rerank 输入数量被限制并输出质量诊断。

        Returns:
            None.
        """
        captured_pairs = []
        model = Mock()

        def fake_compute_score(pairs, normalize=True):
            """记录模型输入并返回递减分数。

            Args:
                pairs: Reranker 输入问答对。
                normalize: 是否归一化分数。

            Returns:
                模拟 Rerank 分数列表。
            """
            captured_pairs.extend(pairs)
            return [0.9 - index * 0.01 for index in range(len(pairs))]

        model.compute_score.side_effect = fake_compute_score
        with (
            patch.object(rerank_module, "add_running_task", return_value=None),
            patch.object(rerank_module, "add_done_task", return_value=None),
            patch.object(rerank_module, "record_rerank", return_value=None),
            patch.object(rerank_module, "record_rerank_docs", return_value=None),
            patch.object(rerank_module, "get_reranker_model", return_value=model),
        ):
            result = rerank_module.node_rerank(self._state(local_count=20, web_count=8))

        self.assertEqual(len(captured_pairs), rerank_module.RERANK_TOTAL_CANDIDATE_LIMIT)
        quality = result["rerank_quality"]
        self.assertEqual(quality["status"], "ok")
        self.assertEqual(quality["input_count"], 15)
        self.assertEqual(quality["local_doc_count"], 12)
        self.assertEqual(quality["web_doc_count"], 3)
        self.assertEqual(quality["answerability"], "answerable")
        self.assertGreater(quality["top1_score"], 0.5)

    def test_rerank_failure_falls_back_to_original_order(self):
        """验证 Reranker 超时时降级为原始召回排序。

        Returns:
            None.
        """
        model = Mock()
        model.compute_score.side_effect = TimeoutError("rerank timeout")
        with (
            patch.object(rerank_module, "add_running_task", return_value=None),
            patch.object(rerank_module, "add_done_task", return_value=None),
            patch.object(rerank_module, "record_rerank", return_value=None),
            patch.object(rerank_module, "record_rerank_docs", return_value=None),
            patch.object(rerank_module, "get_reranker_model", return_value=model),
        ):
            result = rerank_module.node_rerank(self._state(local_count=2, web_count=1))

        self.assertEqual(len(result["reranked_docs"]), 3)
        self.assertEqual(result["reranked_docs"][0]["chunk_id"], "local_0")
        self.assertTrue(result["reranked_docs"][0]["rerank_fallback"])
        quality = result["rerank_quality"]
        self.assertEqual(quality["status"], "fallback")
        self.assertEqual(quality["error_type"], "timeout")
        self.assertEqual(quality["answerability"], "insufficient")

    def test_rerank_oom_opens_circuit_and_next_call_skips_model(self):
        """验证 OOM 会打开短时熔断并让后续调用跳过模型。

        Returns:
            None.
        """
        model = Mock()
        model.compute_score.side_effect = MemoryError("CUDA out of memory")
        with (
            patch.object(rerank_module, "add_running_task", return_value=None),
            patch.object(rerank_module, "add_done_task", return_value=None),
            patch.object(rerank_module, "record_rerank", return_value=None),
            patch.object(rerank_module, "record_rerank_docs", return_value=None),
            patch.object(rerank_module, "get_reranker_model", return_value=model) as mock_get_model,
        ):
            first = rerank_module.node_rerank(self._state(local_count=1, web_count=0))
            second = rerank_module.node_rerank(self._state(local_count=1, web_count=0))

        self.assertEqual(first["rerank_quality"]["error_type"], "oom")
        self.assertEqual(second["rerank_quality"]["error_type"], "circuit_open")
        self.assertEqual(mock_get_model.call_count, 1)


if __name__ == "__main__":
    unittest.main()
