"""Retrieval Gate Step 7 单元测试。"""

import unittest
from unittest.mock import patch

from app.query_process.agent.nodes import node_retrieval_gate as gate_module
from app.query_process.agent.state import create_query_default_state


class NodeRetrievalGateStep7Test(unittest.TestCase):
    """本地召回质量二次门控测试用例。"""

    def _state(self, query_type="fact", use_hyde=True, use_web=False, scores=None):
        """构造 Retrieval Gate 测试状态。

        Args:
            query_type: 查询类型。
            use_hyde: 初始计划是否启用 HyDE。
            use_web: 初始计划是否启用 Web。
            scores: 本地向量召回分数列表。

        Returns:
            查询流程状态。
        """
        embedding_chunks = [
            {"id": f"c{index}", "distance": score, "entity": {"chunk_id": f"c{index}", "content": "text"}}
            for index, score in enumerate(scores or [], start=1)
        ]
        return create_query_default_state(
            session_id="retrieval_gate_step7",
            original_query="HAK 180 怎么操作？",
            rewritten_query="HAK 180 操作步骤",
            query_type=query_type,
            embedding_chunks=embedding_chunks,
            retrieval_plan={
                "use_local_vector": True,
                "use_hyde": use_hyde,
                "use_web": use_web,
            },
        )

    def test_high_confidence_local_skips_supplemental_retrieval(self):
        """验证本地高置信时关闭 HyDE 和 Web 补充。

        Returns:
            None.
        """
        with (
            patch.object(gate_module, "add_running_task", return_value=None),
            patch.object(gate_module, "add_done_task", return_value=None),
        ):
            result = gate_module.node_retrieval_gate(
                self._state(query_type="troubleshooting", use_hyde=True, use_web=True, scores=[0.9, 0.8, 0.7])
            )

        self.assertIs(result["retrieval_plan"]["use_hyde"], False)
        self.assertIs(result["retrieval_plan"]["use_web"], False)
        diagnostics = result["retrieval_diagnostics"]["retrieval_gate"]
        self.assertTrue(diagnostics["local_high_confidence"])
        self.assertEqual(diagnostics["skip_reason"], "local_high_confidence")

    def test_low_confidence_local_triggers_hyde_for_non_realtime_query(self):
        """验证本地低置信时为非实时问题触发 HyDE。

        Returns:
            None.
        """
        with (
            patch.object(gate_module, "add_running_task", return_value=None),
            patch.object(gate_module, "add_done_task", return_value=None),
        ):
            result = gate_module.node_retrieval_gate(
                self._state(query_type="procedure", use_hyde=False, use_web=False, scores=[0.2])
            )

        self.assertIs(result["retrieval_plan"]["use_hyde"], True)
        self.assertIs(result["retrieval_plan"]["use_web"], False)
        diagnostics = result["retrieval_diagnostics"]["retrieval_gate"]
        self.assertTrue(diagnostics["local_low_confidence"])
        self.assertEqual(diagnostics["skip_reason"], "local_low_confidence")

    def test_low_confidence_unknown_query_can_trigger_web_when_enabled(self):
        """验证未知问题低置信时可按配置触发 Web 补充。

        Returns:
            None.
        """
        with (
            patch.object(gate_module, "add_running_task", return_value=None),
            patch.object(gate_module, "add_done_task", return_value=None),
            patch.object(gate_module.query_optimization_config, "web_search_on_low_confidence", True),
        ):
            result = gate_module.node_retrieval_gate(
                self._state(query_type="unknown", use_hyde=False, use_web=False, scores=[])
            )

        self.assertIs(result["retrieval_plan"]["use_hyde"], True)
        self.assertIs(result["retrieval_plan"]["use_web"], True)
        diagnostics = result["retrieval_diagnostics"]["retrieval_gate"]
        self.assertEqual(diagnostics["local_doc_count"], 0)
        self.assertTrue(diagnostics["local_low_confidence"])


if __name__ == "__main__":
    unittest.main()
