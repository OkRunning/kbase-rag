"""HyDE 节点 Step 2 单元测试。"""

import unittest
from unittest.mock import Mock, patch

from app.query_process.agent.nodes import node_search_embedding_hyde as hyde_module
from app.query_process.agent.state import create_query_default_state


class NodeSearchEmbeddingHydeStep2Test(unittest.TestCase):
    """HyDE 节点按检索计划跳过和降级的测试用例。"""

    def _state(self, use_hyde=True):
        """构造 HyDE 节点测试状态。

        Args:
            use_hyde: retrieval_plan.use_hyde 的值。

        Returns:
            查询流程状态。
        """
        return create_query_default_state(
            session_id="hyde_step2",
            original_query="HAK 180 不能加热怎么修？",
            rewritten_query="HAK 180 不能加热故障排查",
            item_names=["HAK 180"],
            retrieval_plan={"use_hyde": use_hyde},
        )

    def test_hyde_skips_when_disabled_by_plan(self):
        """验证 use_hyde=false 时不调用 LLM。

        Returns:
            None.
        """
        with (
            patch.object(hyde_module, "add_running_task", return_value=None),
            patch.object(hyde_module, "add_done_task", return_value=None),
            patch.object(hyde_module, "record_retrieval_docs", return_value=None),
            patch.object(hyde_module, "get_llm_client", side_effect=AssertionError("LLM should not be called")),
        ):
            result = hyde_module.node_search_embedding_hyde(self._state(use_hyde=False))

        self.assertEqual(result["hyde_embedding_chunks"], [])
        diagnostics = result["retrieval_diagnostics"]["hyde"]
        self.assertEqual(diagnostics["status"], "skipped")
        self.assertEqual(diagnostics["skip_reason"], "disabled_by_plan")
        self.assertEqual(diagnostics["output_count"], 0)

    def test_hyde_llm_failure_returns_empty_result(self):
        """验证 HyDE LLM 失败时返回空结果并记录错误。

        Returns:
            None.
        """
        with (
            patch.object(hyde_module, "add_running_task", return_value=None),
            patch.object(hyde_module, "add_done_task", return_value=None),
            patch.object(hyde_module, "record_retrieval_docs", return_value=None),
            patch.object(hyde_module, "load_prompt", return_value="hyde prompt"),
            patch.object(hyde_module, "get_llm_client") as mock_get_llm_client,
        ):
            mock_get_llm_client.return_value.invoke.side_effect = RuntimeError("llm timeout")
            result = hyde_module.node_search_embedding_hyde(self._state(use_hyde=True))

        self.assertEqual(result["hyde_embedding_chunks"], [])
        diagnostics = result["retrieval_diagnostics"]["hyde"]
        self.assertEqual(diagnostics["status"], "error")
        self.assertEqual(diagnostics["skip_reason"], "hyde_failed")
        self.assertEqual(diagnostics["error_type"], "RuntimeError")

    def test_hyde_hypothesis_answer_is_limited_before_embedding(self):
        """验证 HyDE 假设答案在向量化前被截断。

        Returns:
            None.
        """
        captured_texts = []
        long_hypothesis = "测" * 260
        fake_hit = {"id": "1", "distance": 0.9, "entity": {"chunk_id": "c1", "content": "content"}}

        def fake_generate_embeddings(texts):
            """记录向量化输入并返回模拟向量。

            Args:
                texts: 待向量化文本列表。

            Returns:
                模拟 embedding 结果。
            """
            captured_texts.extend(texts)
            return {"dense": [[0.1]], "sparse": [[0.2]]}

        with (
            patch.object(hyde_module, "add_running_task", return_value=None),
            patch.object(hyde_module, "add_done_task", return_value=None),
            patch.object(hyde_module, "record_retrieval_docs", return_value=None),
            patch.object(hyde_module, "load_prompt", return_value="hyde prompt"),
            patch.object(hyde_module, "get_llm_client") as mock_get_llm_client,
            patch.object(hyde_module, "generate_embeddings", side_effect=fake_generate_embeddings),
            patch.object(hyde_module, "get_milvus_client", return_value=Mock()),
            patch.object(hyde_module, "build_active_document_version_filter", return_value=""),
            patch.object(hyde_module, "build_active_vector_version_filter", return_value=""),
            patch.object(hyde_module, "create_hybrid_search_requests", return_value=["req"]),
            patch.object(hyde_module, "hybrid_search", return_value=[[fake_hit]]),
            patch.object(hyde_module, "normalize_milvus_hits", return_value=[fake_hit]),
        ):
            mock_get_llm_client.return_value.invoke.return_value.content = long_hypothesis
            result = hyde_module.node_search_embedding_hyde(self._state(use_hyde=True))

        self.assertEqual(len(captured_texts), 1)
        self.assertEqual(len(captured_texts[0]), len("HAK 180 不能加热故障排查") + 200)
        self.assertEqual(result["retrieval_diagnostics"]["hyde"]["hypothesis_answer_chars"], 200)
        self.assertEqual(len(result["hyde_embedding_chunks"]), 1)


if __name__ == "__main__":
    unittest.main()
