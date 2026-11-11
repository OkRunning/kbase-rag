"""Rerank Gate 图路由测试。"""

import os
import unittest
from unittest.mock import patch

os.environ["LANGGRAPH_CHECKPOINT_BACKEND"] = "memory"
os.environ["MONGO_URL"] = "mongodb://localhost:1/?serverSelectionTimeoutMS=50&connectTimeoutMS=50"
os.environ["MONGO_DB_NAME"] = "dataset_rag_test"

from app.query_process.agent import main_graph as graph_module
from app.query_process.agent.state import create_query_default_state


class MainGraphRerankGateTest(unittest.TestCase):
    """查询图中 Evidence Fusion 后 Rerank Gate 路由测试用例。"""

    def test_route_after_rrf_uses_rerank_gate_when_enabled(self):
        """验证 Rerank Gate 开关打开时 RRF 后进入 Rerank Gate。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="graph_rerank_gate")

        with patch.object(graph_module.query_optimization_config, "enable_rerank_gate", True):
            self.assertEqual(graph_module.route_after_rrf(state), "node_rerank_gate")

    def test_route_after_rrf_keeps_rerank_when_disabled(self):
        """验证 Rerank Gate 开关关闭时保持旧 Rerank 路径。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="graph_rerank_gate")

        with patch.object(graph_module.query_optimization_config, "enable_rerank_gate", False):
            self.assertEqual(graph_module.route_after_rrf(state), "node_rerank")

    def test_route_after_rerank_gate_runs_rerank_when_needed(self):
        """验证 Gate 判定需要精排时进入 Rerank。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="graph_rerank_gate", rerank_gate={"should_rerank": True})

        self.assertEqual(graph_module.route_after_rerank_gate(state), "node_rerank")

    def test_route_after_rerank_gate_skips_to_answer_when_evidence_gate_disabled(self):
        """验证跳过 Rerank 且 Evidence Gate 关闭时进入答案生成。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="graph_rerank_gate", rerank_gate={"should_rerank": False})

        with patch.object(graph_module.query_optimization_config, "enable_evidence_gate", False):
            self.assertEqual(graph_module.route_after_rerank_gate(state), "node_answer_output")

    def test_route_after_rerank_gate_skips_to_evidence_gate_when_enabled(self):
        """验证跳过 Rerank 且 Evidence Gate 打开时进入 Evidence Gate。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="graph_rerank_gate", rerank_gate={"should_rerank": False})

        with patch.object(graph_module.query_optimization_config, "enable_evidence_gate", True):
            self.assertEqual(graph_module.route_after_rerank_gate(state), "node_evidence_gate")


if __name__ == "__main__":
    unittest.main()
