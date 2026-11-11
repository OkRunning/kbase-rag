"""查询检索计划 Step 1 单元测试。"""

import unittest
from unittest.mock import patch

from app.query_process.agent.nodes.query_understanding_utils import build_query_intelligence
from app.query_process.agent.nodes.node_retrieval_planner import node_retrieval_planner
from app.query_process.agent.state import create_query_default_state


class QueryRetrievalPlannerStep1Test(unittest.TestCase):
    """查询检索计划 Step 1 测试用例。"""

    def test_fact_plan_uses_local_only(self):
        """验证事实类问题只启用本地召回。

        Returns:
            None.
        """
        result = build_query_intelligence(
            original_query="HAK 180 的额定功率参数是什么？",
            rewritten_query="HAK 180 额定功率参数",
            item_names=["HAK 180"],
        )

        self.assertEqual(result["query_type"], "fact")
        self.assertIs(result["retrieval_plan"]["use_local_vector"], True)
        self.assertIs(result["retrieval_plan"]["use_hyde"], False)
        self.assertIs(result["retrieval_plan"]["use_web"], False)

    def test_troubleshooting_plan_enables_hyde(self):
        """验证故障排查类问题启用 HyDE。

        Returns:
            None.
        """
        result = build_query_intelligence(
            original_query="HAK 180 不能加热怎么修？",
            rewritten_query="HAK 180 不能加热故障排查",
            item_names=["HAK 180"],
        )

        self.assertEqual(result["query_type"], "troubleshooting")
        self.assertIs(result["retrieval_plan"]["use_local_vector"], True)
        self.assertIs(result["retrieval_plan"]["use_hyde"], True)
        self.assertIs(result["retrieval_plan"]["use_web"], False)

    def test_realtime_plan_enables_web(self):
        """验证实时类问题启用 Web Search。

        Returns:
            None.
        """
        result = build_query_intelligence(
            original_query="今天 HAK 180 的最新价格是多少？",
            rewritten_query="HAK 180 最新价格",
            item_names=["HAK 180"],
        )

        self.assertEqual(result["query_type"], "realtime")
        self.assertIs(result["retrieval_plan"]["use_local_vector"], True)
        self.assertIs(result["retrieval_plan"]["use_hyde"], False)
        self.assertIs(result["retrieval_plan"]["use_web"], True)

    def test_greeting_with_business_question_is_not_chitchat(self):
        """验证寒暄加业务问题时优先识别业务意图。

        Returns:
            None.
        """
        result = build_query_intelligence(
            original_query="你好，烫金机如何使用",
            rewritten_query="烫金机如何使用",
            item_names=["烫金机"],
        )

        self.assertEqual(result["query_type"], "procedure")
        self.assertIs(result["retrieval_plan"]["use_local_vector"], True)
        self.assertIs(result["retrieval_plan"]["use_hyde"], False)

    def test_greeting_phrase_combo_is_chitchat(self):
        """验证多个寒暄短语组合仍识别为闲聊。

        Returns:
            None.
        """
        result = build_query_intelligence(
            original_query="你好，在吗",
            rewritten_query="你好，在吗",
            item_names=[""],
        )

        self.assertEqual(result["query_type"], "chitchat")
        self.assertIs(result["retrieval_plan"]["use_local_vector"], False)
        self.assertIs(result["retrieval_plan"]["use_hyde"], False)
        self.assertIs(result["retrieval_plan"]["use_web"], False)

    def test_pure_greeting_is_chitchat(self):
        """验证纯寒暄问题不进入检索。

        Returns:
            None.
        """
        result = build_query_intelligence(original_query="你好")

        self.assertEqual(result["query_type"], "chitchat")
        self.assertIs(result["retrieval_plan"]["use_local_vector"], False)
        self.assertIs(result["retrieval_plan"]["use_hyde"], False)
        self.assertIs(result["retrieval_plan"]["use_web"], False)

    def test_node_retrieval_planner_writes_query_type_plan_and_empty_defaults(self):
        """验证独立检索计划节点写入 query_type、plan 和空召回默认值。

        Returns:
            None.
        """
        state = create_query_default_state(
            session_id="planner_step1",
            original_query="HAK 180 不能加热怎么修？",
            rewritten_query="HAK 180 不能加热故障排查",
            item_names=["HAK 180"],
        )

        with (
            patch("app.query_process.agent.nodes.node_retrieval_planner.add_running_task", return_value=None),
            patch("app.query_process.agent.nodes.node_retrieval_planner.add_done_task", return_value=None),
            patch("app.query_process.agent.nodes.node_retrieval_planner.emit_event", return_value={}),
        ):
            result = node_retrieval_planner(state)

        self.assertEqual(result["query_type"], "troubleshooting")
        self.assertIs(result["retrieval_plan"]["use_hyde"], True)
        self.assertTrue(result["retrieval_plan"]["reason"])
        self.assertTrue(result["retrieval_diagnostics"]["retrieval_plan_reason"])
        self.assertEqual(result["embedding_chunks"], [])
        self.assertEqual(result["hyde_embedding_chunks"], [])
        self.assertEqual(result["web_search_docs"], [])


if __name__ == "__main__":
    unittest.main()
