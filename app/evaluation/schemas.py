"""RAG 离线评测的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EvalCase:
    """表示一条 RAG 评测样本。

    Attributes:
        case_id: 评测样本唯一标识。
        question: 用户问题。
        tenant_id: 租户标识。
        kb_id: 知识库标识。
        query_type: 问题类型。
        expected_answer: 参考答案。
        expected_doc_ids: 期望命中的文档 ID 列表。
        expected_chunk_ids: 期望命中的 chunk ID 列表。
        required_keywords: 答案中必须覆盖的关键词。
        forbidden_keywords: 答案中不应出现的关键词。
        difficulty: 样本难度。
        tags: 样本标签。
        metadata: 额外元数据。
    """

    case_id: str
    question: str
    tenant_id: str = "default"
    kb_id: str = "default"
    query_type: str = "unknown"
    expected_answer: str = ""
    expected_doc_ids: List[str] = field(default_factory=list)
    expected_chunk_ids: List[str] = field(default_factory=list)
    required_keywords: List[str] = field(default_factory=list)
    forbidden_keywords: List[str] = field(default_factory=list)
    difficulty: str = "unknown"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalCase":
        """从字典创建评测样本。

        Args:
            data: JSONL 中读取的一条样本数据。

        Returns:
            评测样本对象。

        Raises:
            ValueError: 缺少 `case_id` 或 `question` 时抛出。
        """
        case_id = str(data.get("case_id") or "").strip()
        question = str(data.get("question") or "").strip()
        if not case_id:
            raise ValueError("eval case missing required field: case_id")
        if not question:
            raise ValueError(f"eval case {case_id} missing required field: question")
        return cls(
            case_id=case_id,
            question=question,
            tenant_id=str(data.get("tenant_id") or "default"),
            kb_id=str(data.get("kb_id") or "default"),
            query_type=str(data.get("query_type") or "unknown"),
            expected_answer=str(data.get("expected_answer") or ""),
            expected_doc_ids=[str(value) for value in data.get("expected_doc_ids", [])],
            expected_chunk_ids=[str(value) for value in data.get("expected_chunk_ids", [])],
            required_keywords=[str(value) for value in data.get("required_keywords", [])],
            forbidden_keywords=[str(value) for value in data.get("forbidden_keywords", [])],
            difficulty=str(data.get("difficulty") or "unknown"),
            tags=[str(value) for value in data.get("tags", [])],
            metadata=dict(data.get("metadata") or {}),
        )

    def to_langfuse_input(self) -> Dict[str, Any]:
        """转换为 Langfuse dataset item input。

        Returns:
            可写入 Langfuse 的输入字段。
        """
        return {
            "question": self.question,
            "tenant_id": self.tenant_id,
            "kb_id": self.kb_id,
        }

    def to_expected_output(self) -> Dict[str, Any]:
        """转换为评测期望输出结构。

        Returns:
            包含答案、证据和规则关键词的期望输出。
        """
        return {
            "expected_answer": self.expected_answer,
            "expected_doc_ids": self.expected_doc_ids,
            "expected_chunk_ids": self.expected_chunk_ids,
            "required_keywords": self.required_keywords,
            "forbidden_keywords": self.forbidden_keywords,
        }


@dataclass
class EvalResult:
    """表示单条评测样本的运行结果。

    Attributes:
        case: 评测样本。
        status: 运行状态，例如 ok 或 error。
        session_id: 本次评测使用的会话 ID。
        answer: 模型答案。
        retrieved_chunk_ids: 检索阶段命中的 chunk ID。
        reranked_chunk_ids: Rerank 后命中的 chunk ID。
        metrics: 指标名称到分值的映射。
        trace_id: Langfuse trace ID。
        trace_url: Langfuse trace URL。
        error: 错误信息。
    """

    case: EvalCase
    status: str
    session_id: str
    answer: str = ""
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    reranked_chunk_ids: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    trace_id: str = ""
    trace_url: str = ""
    error: str = ""

    def passed(self, threshold: float = 1.0) -> bool:
        """判断该样本是否通过基础规则。

        Args:
            threshold: 布尔型规则指标的通过阈值。

        Returns:
            全部关键规则满足时返回 True。
        """
        if self.status != "ok":
            return False
        hit_score = self.metrics.get("hit_at_5")
        if hit_score is None:
            hit_score = next(
                (value for name, value in self.metrics.items() if name.startswith("hit_at_")),
                0.0,
            )
        required_scores = [
            hit_score,
            self.metrics.get("required_keywords_pass", 0.0),
            self.metrics.get("forbidden_keywords_pass", 0.0),
            self.metrics.get("has_answer", 0.0),
        ]
        return all(value >= threshold for value in required_scores)
