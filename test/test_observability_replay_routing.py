"""查询流程回放路由测试：检索计划与三级门控节点接入回放。"""

import os
import unittest
from unittest.mock import patch

os.environ["LANGGRAPH_CHECKPOINT_BACKEND"] = "memory"
os.environ["MONGO_URL"] = "mongodb://localhost:1/?serverSelectionTimeoutMS=50&connectTimeoutMS=50"
os.environ["MONGO_DB_NAME"] = "dataset_rag_test"
os.environ["MINIO_ENDPOINT"] = "localhost:1"

from app.observability import replay as replay_module
from app.query_process.agent.state import create_query_default_state


def patch_query_config(
    use_retrieval_planner=True, enable_rerank_gate=False, enable_evidence_gate=False
):
    """固定灰度开关，保证测试不依赖 .env 配置。

    Args:
        use_retrieval_planner: 是否启用检索计划。
        enable_rerank_gate: 是否启用 Rerank 前置门控。
        enable_evidence_gate: 是否启用证据门控。

    Returns:
        上下文管理器。
    """
    return patch.multiple(
        replay_module.query_optimization_config,
        use_retrieval_planner=use_retrieval_planner,
        enable_rerank_gate=enable_rerank_gate,
        enable_evidence_gate=enable_evidence_gate,
    )


class ReplayQueryRoutingTest(unittest.TestCase):
    """查询流程回放节点链路由测试用例。"""

    def test_pre_retrieval_routes_to_planner_when_enabled(self):
        """验证检索计划开关打开时前置理解后进入检索计划节点。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="replay_routing")

        with patch_query_config(use_retrieval_planner=True):
            self.assertEqual(
                replay_module._query_next_nodes("node_item_name_confirm", state),
                [
                    "node_retrieval_planner",
                    "node_search_embedding",
                    "node_retrieval_gate",
                    "node_rrf",
                    "node_rerank",
                    "node_answer_output",
                ],
            )

    def test_pre_retrieval_keeps_legacy_parallel_when_disabled(self):
        """验证检索计划开关关闭时保持旧三路并行行为。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="replay_routing")

        with patch_query_config(use_retrieval_planner=False):
            self.assertEqual(
                replay_module._query_next_nodes("node_clarification_decision", state),
                [
                    "node_search_embedding",
                    "node_search_embedding_hyde",
                    "node_web_search_mcp",
                    "node_rrf",
                    "node_rerank",
                    "node_answer_output",
                ],
            )

    def test_pre_retrieval_terminates_when_answer_set(self):
        """验证已生成答案时直接进入答案节点。

        Returns:
            None.
        """
        state = create_query_default_state(
            session_id="replay_routing",
            answer="已给出结论",
        )

        self.assertEqual(
            replay_module._query_next_nodes("node_item_name_confirm", state),
            ["node_answer_output"],
        )

    def test_planner_follows_plan_without_local_vector(self):
        """验证检索计划关闭本地向量时按计划补充 HyDE 和 Web。

        Returns:
            None.
        """
        state = create_query_default_state(
            session_id="replay_routing",
            retrieval_plan={
                "use_local_vector": False,
                "use_hyde": True,
                "use_web": True,
            },
        )

        with patch_query_config():
            self.assertEqual(
                replay_module._query_next_nodes("node_retrieval_planner", state),
                [
                    "node_search_embedding_hyde",
                    "node_web_search_mcp",
                    "node_rrf",
                    "node_rerank",
                    "node_answer_output",
                ],
            )

    def test_planner_empty_plan_goes_answer(self):
        """验证检索计划无任何召回源时直接进入答案节点。

        Returns:
            None.
        """
        state = create_query_default_state(
            session_id="replay_routing",
            retrieval_plan={
                "use_local_vector": False,
                "use_hyde": False,
                "use_web": False,
            },
        )

        with patch_query_config():
            self.assertEqual(
                replay_module._query_next_nodes("node_retrieval_planner", state),
                ["node_answer_output"],
            )

    def test_search_embedding_uses_retrieval_gate_when_enabled(self):
        """验证本地召回后进入二次检索门控。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="replay_routing")

        with patch_query_config(use_retrieval_planner=True):
            self.assertEqual(
                replay_module._query_next_nodes("node_search_embedding", state),
                [
                    "node_retrieval_gate",
                    "node_rrf",
                    "node_rerank",
                    "node_answer_output",
                ],
            )

    def test_retrieval_gate_follows_updated_plan(self):
        """验证二次门控更新计划后按 use_hyde/use_web 补充召回。

        Returns:
            None.
        """
        state = create_query_default_state(
            session_id="replay_routing",
            retrieval_plan={"use_hyde": True, "use_web": False},
        )

        with patch_query_config():
            self.assertEqual(
                replay_module._query_next_nodes("node_retrieval_gate", state),
                [
                    "node_search_embedding_hyde",
                    "node_rrf",
                    "node_rerank",
                    "node_answer_output",
                ],
            )

    def test_rrf_uses_rerank_gate_when_enabled(self):
        """验证 Rerank Gate 开关打开时融合后进入 Rerank Gate。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="replay_routing")

        with patch_query_config(enable_rerank_gate=True):
            self.assertEqual(
                replay_module._query_next_nodes("node_rrf", state),
                ["node_rerank_gate", "node_rerank", "node_answer_output"],
            )

    def test_rrf_keeps_rerank_when_disabled(self):
        """验证 Rerank Gate 开关关闭时保持旧 Rerank 路径。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="replay_routing")

        with patch_query_config(enable_rerank_gate=False):
            self.assertEqual(
                replay_module._query_next_nodes("node_rrf", state),
                ["node_rerank", "node_answer_output"],
            )

    def test_rerank_gate_runs_rerank_when_needed(self):
        """验证 Gate 判定需要精排时进入 Rerank。

        Returns:
            None.
        """
        state = create_query_default_state(
            session_id="replay_routing",
            rerank_gate={"should_rerank": True},
        )

        with patch_query_config():
            self.assertEqual(
                replay_module._query_next_nodes("node_rerank_gate", state),
                ["node_rerank", "node_answer_output"],
            )

    def test_rerank_gate_skips_to_evidence_when_enabled(self):
        """验证跳过 Rerank 且 Evidence Gate 打开时进入证据门控。

        Returns:
            None.
        """
        state = create_query_default_state(
            session_id="replay_routing",
            rerank_gate={"should_rerank": False},
        )

        with patch_query_config(enable_evidence_gate=True):
            self.assertEqual(
                replay_module._query_next_nodes("node_rerank_gate", state),
                ["node_evidence_gate", "node_answer_output"],
            )

    def test_rerank_uses_evidence_gate_when_enabled(self):
        """验证 Evidence Gate 开关打开时精排后进入证据门控。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="replay_routing")

        with patch_query_config(enable_evidence_gate=True):
            self.assertEqual(
                replay_module._query_next_nodes("node_rerank", state),
                ["node_evidence_gate", "node_answer_output"],
            )

    def test_evidence_gate_goes_answer(self):
        """验证证据门控后进入答案生成节点。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="replay_routing")

        with patch_query_config():
            self.assertEqual(
                replay_module._query_next_nodes("node_evidence_gate", state),
                ["node_answer_output"],
            )

    def test_workflow_plan_supports_new_nodes(self):
        """验证全流程回放计划支持检索计划与三级门控节点作为起点。

        Returns:
            None.
        """
        state = create_query_default_state(session_id="replay_routing")
        for start_node in (
            "node_retrieval_planner",
            "node_retrieval_gate",
            "node_rerank_gate",
            "node_evidence_gate",
        ):
            planned = replay_module._workflow_plan("query", start_node, state)
            self.assertEqual(planned[0], start_node)
            self.assertIn(start_node, replay_module._NODE_REGISTRY["query"])


if __name__ == "__main__":
    unittest.main()
