"""Citation 和 sources Step 10 单元测试。"""

import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ["LANGGRAPH_CHECKPOINT_BACKEND"] = "memory"
os.environ["MONGO_URL"] = "mongodb://localhost:1/?serverSelectionTimeoutMS=50&connectTimeoutMS=50"
os.environ["MONGO_DB_NAME"] = "dataset_rag_test"

from app.query_process.agent.nodes import node_answer_output as answer_module
from app.query_process.agent.state import create_query_default_state


class AnswerSourcesStep10Test(unittest.TestCase):
    """答案来源、Prompt 编号和 API sources 返回测试用例。"""

    def _state(self):
        """构造带重排序证据的查询状态。

        Returns:
            查询流程状态。
        """
        return create_query_default_state(
            session_id="answer_sources_step10",
            original_query="HAK 180 怎么操作？",
            rewritten_query="HAK 180 操作步骤",
            eval_mode=True,
            reranked_docs=[
                {
                    "source": "local",
                    "source_type": "local",
                    "file_title": "HAK 180 操作手册.pdf",
                    "section": "操作步骤",
                    "page": 3,
                    "score": 0.92,
                    "chunk_id": "chunk_1",
                    "evidence_id": "chunk:chunk_1",
                    "item_name": "HAK 180",
                    "text": "开启电源后，先预热再设置温度。",
                },
                {
                    "source": "web",
                    "source_type": "web",
                    "title": "HAK 180 FAQ",
                    "url": "https://example.com/faq",
                    "domain": "example.com",
                    "score": 0.81,
                    "evidence_id": "web:https://example.com/faq",
                    "text": "如果机器无法加热，请检查保险丝。",
                },
            ],
        )

    def test_build_answer_sources_maps_reranked_docs(self):
        """验证 reranked_docs 会转换为前端可展示 sources。

        Returns:
            None.
        """
        sources = answer_module.build_answer_sources(self._state())

        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["source_id"], "[1]")
        self.assertEqual(sources[0]["file_title"], "HAK 180 操作手册.pdf")
        self.assertEqual(sources[0]["page"], 3)
        self.assertEqual(sources[1]["source_type"], "web")
        self.assertEqual(sources[1]["url"], "https://example.com/faq")

    def test_prompt_context_uses_numbered_source_blocks(self):
        """验证 Prompt context 使用 Step 10 要求的编号资料格式。

        Returns:
            None.
        """
        captured = {}

        def fake_load_prompt(name, **kwargs):
            """捕获 Prompt 入参。

            Args:
                name: Prompt 名称。
                **kwargs: Prompt 变量。

            Returns:
                模拟 Prompt。
            """
            captured.update(kwargs)
            return "prompt"

        with patch.object(answer_module, "load_prompt", side_effect=fake_load_prompt):
            prompt = answer_module.step_2_get_prompt(self._state())

        self.assertEqual(prompt, "prompt")
        context = captured["context"]
        self.assertIn("[1]\nsource_type: local", context)
        self.assertIn("file_title: HAK 180 操作手册.pdf", context)
        self.assertIn("page: 3", context)
        self.assertIn("text: 开启电源后，先预热再设置温度。", context)

    def test_stream_final_returns_sources(self):
        """验证流式 FINAL 事件携带 sources。

        Returns:
            None.
        """
        state = self._state()
        state["is_stream"] = True
        llm = Mock()
        llm.stream.return_value = [Mock(content="答案 [1]")]
        pushed_events = []
        task_results = {}

        def fake_set_task_result(session_id, key, value):
            """记录任务结果写入。

            Args:
                session_id: 会话 ID。
                key: 结果字段。
                value: 结果值。

            Returns:
                None.
            """
            task_results[key] = value

        with (
            patch.object(answer_module, "add_running_task", return_value=None),
            patch.object(answer_module, "add_done_task", return_value=None),
            patch.object(answer_module, "set_task_result", side_effect=fake_set_task_result),
            patch.object(answer_module, "push_to_session", side_effect=lambda *args: pushed_events.append(args)),
            patch.object(answer_module, "get_llm_client", return_value=llm),
            patch.object(answer_module, "load_prompt", return_value="prompt"),
        ):
            result = answer_module.node_answer_output(state)

        final_events = [event for event in pushed_events if event[1] == answer_module.SSEEvent.FINAL]
        self.assertEqual(result["answer"], "答案 [1]")
        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0][2]["sources"][0]["source_id"], "[1]")
        self.assertEqual(json.loads(task_results["sources"])[0]["source_id"], "[1]")

    def test_query_server_sources_helper_parses_task_result(self):
        """验证 API 层能把任务结果中的 sources JSON 解析为列表。

        Returns:
            None.
        """
        from app.query_process.api import query__server as query_server

        with patch.object(
            query_server,
            "get_task_result",
            return_value=json.dumps([{"source_id": "[1]"}], ensure_ascii=False),
        ):
            self.assertEqual(query_server._get_task_sources("answer_sources_step10"), [{"source_id": "[1]"}])


if __name__ == "__main__":
    unittest.main()
