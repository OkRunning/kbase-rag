"""Web Search 节点 Step 3 单元测试。"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.query_process.agent.nodes import node_web_search_mcp as web_module
from app.query_process.agent.state import create_query_default_state


class NodeWebSearchMcpStep3Test(unittest.TestCase):
    """Web Search 节点按检索计划跳过和降级的测试用例。"""

    def _state(self, use_web=True):
        """构造 Web Search 节点测试状态。

        Args:
            use_web: retrieval_plan.use_web 的值。

        Returns:
            查询流程状态。
        """
        return create_query_default_state(
            session_id="web_step3",
            original_query="今天 HAK 180 最新价格是多少？",
            rewritten_query="HAK 180 最新价格",
            retrieval_plan={"use_web": use_web},
        )

    def test_web_skips_when_disabled_by_plan(self):
        """验证 use_web=false 时不调用 MCP。

        Returns:
            None.
        """
        with (
            patch.object(web_module, "add_running_task", return_value=None),
            patch.object(web_module, "add_done_task", return_value=None),
            patch.object(web_module, "record_retrieval_docs", return_value=None),
            patch.object(web_module, "mcp_tool_call", side_effect=AssertionError("MCP should not be called")),
        ):
            result = web_module.node_web_search_mcp(self._state(use_web=False))

        self.assertEqual(result["web_search_docs"], [])
        diagnostics = result["retrieval_diagnostics"]["web"]
        self.assertEqual(diagnostics["status"], "skipped")
        self.assertEqual(diagnostics["skip_reason"], "disabled_by_plan")
        self.assertEqual(diagnostics["output_count"], 0)

    def test_web_failure_returns_empty_result(self):
        """验证 Web Search 失败时返回空结果并记录错误。

        Returns:
            None.
        """

        async def fake_mcp_tool_call(query):
            """模拟 MCP 调用超时。

            Args:
                query: Web 搜索查询。

            Raises:
                TimeoutError: 模拟工具调用超时。
            """
            raise TimeoutError("web timeout")

        with (
            patch.object(web_module, "add_running_task", return_value=None),
            patch.object(web_module, "add_done_task", return_value=None),
            patch.object(web_module, "record_retrieval_docs", return_value=None),
            patch.object(web_module, "mcp_tool_call", side_effect=fake_mcp_tool_call),
        ):
            result = web_module.node_web_search_mcp(self._state(use_web=True))

        self.assertEqual(result["web_search_docs"], [])
        diagnostics = result["retrieval_diagnostics"]["web"]
        self.assertEqual(diagnostics["status"], "error")
        self.assertEqual(diagnostics["skip_reason"], "web_search_failed")
        self.assertEqual(diagnostics["error_type"], "TimeoutError")

    def test_web_documents_are_normalized(self):
        """验证 Web Search 结果补齐来源、URL、域名和可信度字段。

        Returns:
            None.
        """
        pages = [
            {
                "snippet": "官方页面说明 HAK 180 的最新维护通知。",
                "title": "HAK 180 维护通知",
                "url": "https://example.com/docs/hak180",
            }
        ]
        fake_result = SimpleNamespace(
            isError=False,
            content=[SimpleNamespace(text=json.dumps({"pages": pages}, ensure_ascii=False))],
        )

        async def fake_mcp_tool_call(query):
            """返回模拟 MCP Web Search 结果。

            Args:
                query: Web 搜索查询。

            Returns:
                模拟 MCP 工具响应。
            """
            return fake_result

        with (
            patch.object(web_module, "add_running_task", return_value=None),
            patch.object(web_module, "add_done_task", return_value=None),
            patch.object(web_module, "record_retrieval_docs", return_value=None),
            patch.object(web_module, "mcp_tool_call", side_effect=fake_mcp_tool_call),
        ):
            result = web_module.node_web_search_mcp(self._state(use_web=True))

        self.assertEqual(len(result["web_search_docs"]), 1)
        doc = result["web_search_docs"][0]
        self.assertEqual(doc["source_type"], "web")
        self.assertEqual(doc["source"], "web")
        self.assertEqual(doc["url"], "https://example.com/docs/hak180")
        self.assertEqual(doc["domain"], "example.com")
        self.assertEqual(doc["trust_level"], "external")
        diagnostics = result["retrieval_diagnostics"]["web"]
        self.assertEqual(diagnostics["status"], "ok")
        self.assertEqual(diagnostics["output_count"], 1)
        self.assertEqual(diagnostics["tool_call_timeout_seconds"], web_module.WEB_TOOL_CALL_TIMEOUT_SECONDS)
        self.assertEqual(diagnostics["max_retry_attempts"], web_module.WEB_MAX_RETRY_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
