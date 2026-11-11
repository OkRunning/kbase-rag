"""Evidence Gate Step 9 图路由测试。"""

import os
import unittest
from unittest.mock import patch

os.environ["LANGGRAPH_CHECKPOINT_BACKEND"] = "memory"
os.environ["MONGO_URL"] = "mongodb://localhost:1/?serverSelectionTimeoutMS=50&connectTimeoutMS=50"
os.environ["MONGO_DB_NAME"] = "dataset_rag_test"

from app.query_process.agent import main_graph as graph_module
from app.query_process.agent.state import create_query_default_state


class MainGraphEvidenceGateStep9Test(unittest.TestCase):
    """查询图中 Rerank 后 Evidence Gate 路由测试用例。"""

    def test_route_after_rerank_uses_evidence_gate_when_enabled(self):
        """验证证据门控开关打开时 Rerank 后进入 Evidence Gate。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="graph_evidence_gate_step9")

        with patch.object(graph_module.query_optimization_config, "enable_evidence_gate", True):
            self.assertEqual(graph_module.route_after_rerank(state), "node_evidence_gate")

    def test_route_after_rerank_skips_evidence_gate_when_disabled(self):
        """验证证据门控开关关闭时保持旧生成路径。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="graph_evidence_gate_step9")

        with patch.object(graph_module.query_optimization_config, "enable_evidence_gate", False):
            self.assertEqual(graph_module.route_after_rerank(state), "node_answer_output")


if __name__ == "__main__":
    unittest.main()
