"""RAG 评测指标计算。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from app.evaluation.schemas import EvalCase


def extract_chunk_id(document: Any) -> str:
    """从不同检索结果结构中提取 chunk ID。

    Args:
        document: 检索或重排序节点返回的一条文档。

    Returns:
        提取到的 chunk ID，无法提取时返回空字符串。
    """
    if not isinstance(document, dict):
        return ""
    entity = document.get("entity")
    candidates = [
        document.get("chunk_id"),
        entity.get("chunk_id") if isinstance(entity, dict) else None,
        document.get("id"),
        entity.get("id") if isinstance(entity, dict) else None,
    ]
    for value in candidates:
        if value is not None and str(value).strip():
            return str(value)
    return ""


def extract_chunk_ids(documents: Iterable[Any]) -> List[str]:
    """批量提取 chunk ID 并保持顺序去重。

    Args:
        documents: 检索结果列表。

    Returns:
        chunk ID 列表。
    """
    ids: List[str] = []
    seen = set()
    for document in documents or []:
        chunk_id = extract_chunk_id(document)
        if chunk_id and chunk_id not in seen:
            ids.append(chunk_id)
            seen.add(chunk_id)
    return ids


def recall_at_k(actual_ids: Sequence[str], expected_ids: Sequence[str], k: int) -> float:
    """计算 Recall@K。

    Args:
        actual_ids: 实际排序后的 chunk ID。
        expected_ids: 期望命中的 chunk ID。
        k: 截断位置。

    Returns:
        Recall@K 分值；没有期望证据时返回 1.0。
    """
    expected = {str(value) for value in expected_ids if str(value)}
    if not expected:
        return 1.0
    actual = set(str(value) for value in actual_ids[:k])
    return len(actual & expected) / len(expected)


def hit_at_k(actual_ids: Sequence[str], expected_ids: Sequence[str], k: int) -> float:
    """计算 Hit@K。

    Args:
        actual_ids: 实际排序后的 chunk ID。
        expected_ids: 期望命中的 chunk ID。
        k: 截断位置。

    Returns:
        命中任一目标证据时返回 1.0，否则返回 0.0。
    """
    expected = {str(value) for value in expected_ids if str(value)}
    if not expected:
        return 1.0
    actual = set(str(value) for value in actual_ids[:k])
    return 1.0 if actual & expected else 0.0


def mrr_at_k(actual_ids: Sequence[str], expected_ids: Sequence[str], k: int) -> float:
    """计算 MRR@K。

    Args:
        actual_ids: 实际排序后的 chunk ID。
        expected_ids: 期望命中的 chunk ID。
        k: 截断位置。

    Returns:
        第一个目标证据排名的倒数；没有命中时返回 0.0。
    """
    expected = {str(value) for value in expected_ids if str(value)}
    if not expected:
        return 1.0
    for rank, chunk_id in enumerate(actual_ids[:k], start=1):
        if str(chunk_id) in expected:
            return 1.0 / rank
    return 0.0


def keyword_pass(answer: str, required_keywords: Sequence[str], forbidden_keywords: Sequence[str]) -> Dict[str, float]:
    """计算答案关键词规则。

    Args:
        answer: 模型答案。
        required_keywords: 必须出现的关键词。
        forbidden_keywords: 禁止出现的关键词。

    Returns:
        关键词规则指标。
    """
    answer_text = answer or ""
    required = [keyword for keyword in required_keywords if keyword]
    forbidden = [keyword for keyword in forbidden_keywords if keyword]
    required_pass = all(keyword in answer_text for keyword in required) if required else True
    forbidden_pass = all(keyword not in answer_text for keyword in forbidden) if forbidden else True
    coverage = 1.0
    if required:
        coverage = sum(1 for keyword in required if keyword in answer_text) / len(required)
    return {
        "required_keywords_pass": 1.0 if required_pass else 0.0,
        "required_keywords_coverage": coverage,
        "forbidden_keywords_pass": 1.0 if forbidden_pass else 0.0,
    }


def evaluate_case_result(
    case: EvalCase,
    answer: str,
    retrieved_chunk_ids: Sequence[str],
    reranked_chunk_ids: Sequence[str],
    k: int = 5,
) -> Dict[str, float]:
    """计算单条样本的基础评测指标。

    Args:
        case: 评测样本。
        answer: 模型答案。
        retrieved_chunk_ids: 多路召回和 RRF 后的候选 chunk ID。
        reranked_chunk_ids: Rerank 后的最终 chunk ID。
        k: TopK 截断位置。

    Returns:
        指标名称到分值的映射。
    """
    expected = case.expected_chunk_ids
    metrics = {
        f"retrieval_recall_at_{k}": recall_at_k(retrieved_chunk_ids, expected, k),
        f"retrieval_hit_at_{k}": hit_at_k(retrieved_chunk_ids, expected, k),
        f"retrieval_mrr_at_{k}": mrr_at_k(retrieved_chunk_ids, expected, k),
        f"recall_at_{k}": recall_at_k(reranked_chunk_ids, expected, k),
        f"hit_at_{k}": hit_at_k(reranked_chunk_ids, expected, k),
        f"mrr_at_{k}": mrr_at_k(reranked_chunk_ids, expected, k),
        "has_answer": 1.0 if (answer or "").strip() else 0.0,
        "has_local_citation": 1.0 if reranked_chunk_ids else 0.0,
    }
    metrics.update(keyword_pass(answer, case.required_keywords, case.forbidden_keywords))
    return metrics

