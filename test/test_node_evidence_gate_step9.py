"""Evidence Gate Step 9 单元测试。"""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

from app.query_process.agent.nodes import node_evidence_gate as gate_module
from app.query_process.agent.state import create_query_default_state


class NodeEvidenceGateStep9Test(unittest.TestCase):
    """答案生成前证据充分性门控测试用例。"""

    def _state(self, docs=None, **overrides):
        """构造 Evidence Gate 测试状态。

        Args:
            docs: 重排序后的证据列表。
            **overrides: 需要覆盖的状态字段。

        Returns:
            查询流程状态。
        """
        state = create_query_default_state(
            session_id="evidence_gate_step9",
            original_query="HAK 180 怎么操作？",
            rewritten_query="HAK 180 操作步骤",
            reranked_docs=docs or [],
            rerank_quality={"answerability": "answerable"},
        )
        state.update(overrides)
        return state

    def _skipped_doc(self, index=1, score=0.016, text="本地证据", item_name=None):
        """构造 Rerank Gate 跳过精排后的证据文档。

        跳过精排时写入的分数是 Milvus distance 或 rrf_score，量纲与 Rerank 归一化分不同。

        Args:
            index: 文档序号。
            score: 跳过精排时写入的原始分数。
            text: 证据文本。
            item_name: 商品名称。

        Returns:
            标记了 rerank_skipped 的证据文档字典。
        """
        doc = {
            "source": "local",
            "source_type": "local",
            "score": score,
            "rrf_score": score,
            "rerank_skipped": True,
            "text": f"{text} {index}",
        }
        if item_name:
            doc["item_name"] = item_name
        return doc

    def _run_gate(self, state, **config_overrides):
        """执行 Evidence Gate 并隔离任务与事件副作用。

        Args:
            state: 查询流程状态。
            **config_overrides: 需要临时覆盖的配置字段。

        Returns:
            Evidence Gate 执行后的状态。
        """
        patches = [
            patch.object(gate_module, "add_running_task", return_value=None),
            patch.object(gate_module, "add_done_task", return_value=None),
            patch.object(gate_module, "emit_event", return_value=None),
        ]
        for key, value in config_overrides.items():
            patches.append(patch.object(gate_module.query_optimization_config, key, value))

        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            return gate_module.node_evidence_gate(state)

    def test_empty_reranked_docs_is_insufficient(self):
        """验证没有重排序证据时直接判定资料不足。

        Returns:
            None.
        """
        result = self._run_gate(self._state())

        self.assertEqual(result["evidence_status"], "insufficient")
        self.assertEqual(result["rerank_quality"]["answerability"], "insufficient")
        self.assertEqual(result["evidence_gate"]["doc_count"], 0)

    def test_low_top_score_is_insufficient(self):
        """验证 top rerank 分数低于阈值时判定资料不足。

        Returns:
            None.
        """
        result = self._run_gate(
            self._state(docs=[{"source": "local", "source_type": "local", "score": 0.3, "text": "操作步骤"}]),
            evidence_min_rerank_score=0.45,
        )

        self.assertEqual(result["evidence_status"], "insufficient")
        self.assertIn("top rerank score", result["evidence_gate"]["reason"])

    def test_web_only_docs_are_external_only_when_disallowed(self):
        """验证仅命中 Web 且不允许 external_only 时阻断普通生成。

        Returns:
            None.
        """
        result = self._run_gate(
            self._state(
                docs=[
                    {
                        "source": "web",
                        "source_type": "web",
                        "score": 0.82,
                        "text": "Web 上的操作说明",
                        "url": "https://example.com/manual",
                    }
                ]
            ),
            allow_external_only=False,
        )

        self.assertEqual(result["evidence_status"], "external_only")
        self.assertEqual(result["rerank_quality"]["answerability"], "insufficient")

    def test_comparison_requires_each_entity_evidence(self):
        """验证对比问题缺少任一实体证据时判定资料不足。

        Returns:
            None.
        """
        result = self._run_gate(
            self._state(
                docs=[
                    {
                        "source": "local",
                        "source_type": "local",
                        "score": 0.86,
                        "item_name": "HAK 180",
                        "text": "HAK 180 支持恒温操作。",
                    }
                ],
                query_type="comparison",
                item_names=["HAK 180", "HAK 200"],
            )
        )

        self.assertEqual(result["evidence_status"], "insufficient")
        self.assertEqual(result["evidence_gate"]["missing_info"], ["HAK 200"])

    def test_conflicting_high_score_docs_block_generation(self):
        """验证高分证据出现显性冲突时判定 conflicting。

        Returns:
            None.
        """
        result = self._run_gate(
            self._state(
                docs=[
                    {"source": "local", "source_type": "local", "score": 0.91, "text": "HAK 180 支持自动预热。"},
                    {"source": "local", "source_type": "local", "score": 0.88, "text": "HAK 180 不支持自动预热。"},
                ]
            )
        )

        self.assertEqual(result["evidence_status"], "conflicting")
        self.assertEqual(result["rerank_quality"]["answerability"], "insufficient")

    def test_sufficient_docs_are_answerable(self):
        """验证本地高分证据充足时允许普通生成。

        Returns:
            None.
        """
        result = self._run_gate(
            self._state(
                docs=[
                    {"source": "local", "source_type": "local", "score": 0.82, "text": "HAK 180 操作前需要预热。"}
                ]
            )
        )

        self.assertEqual(result["evidence_status"], "sufficient")
        self.assertEqual(result["rerank_quality"]["answerability"], "answerable")

    def test_skipped_rerank_low_score_is_not_insufficient(self):
        """验证跳过精排后 rrf_score≈0.016 不再因分数阈值判 insufficient。

        分数域错配修复：跳过精排的文档分数不是 Rerank 归一化分，只按数量和实体覆盖判定。

        Returns:
            None.
        """
        result = self._run_gate(
            self._state(docs=[self._skipped_doc()]),
            evidence_min_rerank_score=0.45,
        )

        self.assertEqual(result["evidence_status"], "sufficient")
        self.assertEqual(result["rerank_quality"]["answerability"], "answerable")
        self.assertNotIn("top rerank score", result["evidence_gate"]["reason"])

    def test_skipped_rerank_below_min_docs_is_insufficient(self):
        """验证跳过精排后数量不足仍然判 insufficient。

        Returns:
            None.
        """
        result = self._run_gate(self._state(docs=[self._skipped_doc()]), evidence_min_docs=2)

        self.assertEqual(result["evidence_status"], "insufficient")
        self.assertIn("数量低于", result["evidence_gate"]["reason"])

    def test_skipped_rerank_comparison_missing_entity_is_insufficient(self):
        """验证跳过精排后对比问题仍按实体覆盖判 insufficient。

        Returns:
            None.
        """
        result = self._run_gate(
            self._state(
                docs=[self._skipped_doc(item_name="HAK 180", text="HAK 180 支持恒温操作。")],
                query_type="comparison",
                item_names=["HAK 180", "HAK 200"],
            )
        )

        self.assertEqual(result["evidence_status"], "insufficient")
        self.assertEqual(result["evidence_gate"]["missing_info"], ["HAK 200"])

    def test_rerank_score_below_min_score_still_insufficient_when_not_skipped(self):
        """验证正常 Rerank 路径分数低于阈值仍然判 insufficient。

        只有真正经过 Rerank 打分的证据才应用 min_score 分数阈值。

        Returns:
            None.
        """
        result = self._run_gate(
            self._state(docs=[{"source": "local", "source_type": "local", "score": 0.44, "text": "操作步骤"}]),
            evidence_min_rerank_score=0.45,
        )

        self.assertEqual(result["evidence_status"], "insufficient")
        self.assertIn("top rerank score", result["evidence_gate"]["reason"])


if __name__ == "__main__":
    unittest.main()
