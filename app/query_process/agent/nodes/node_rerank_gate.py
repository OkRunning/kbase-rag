"""Rerank 前置跳过门控节点。"""

import sys
from typing import Any, Dict, List, Tuple

from app.core.logger import logger
from app.observability.metrics import record_rerank_docs
from app.observability.sanitizers import sanitize_for_log
from app.query_process.agent.nodes.node_rerank import (
    RERANK_MAX_TOPK,
    _limit_rerank_candidates,
    step_1_merge_rrf_web_search_results,
)
from app.query_process.agent.nodes.query_understanding_utils import normalize_query_type
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task

SIMPLE_QUERY_TYPES = {"fact", "safety"}
COMPLEX_QUERY_TYPES = {"comparison", "troubleshooting", "procedure", "summary"}
HIGH_CONFIDENCE_LOCAL_SCORE = 0.75
SMALL_LOCAL_CANDIDATE_LIMIT = 3


def _safe_float(value: Any) -> float:
    """安全转换浮点值。

    Args:
        value: 原始数值。

    Returns:
        转换后的浮点值。
    """
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _source_names(doc: Dict[str, Any]) -> set[str]:
    """提取候选证据涉及的召回源名称。

    Args:
        doc: Rerank 候选文档。

    Returns:
        来源名称集合。
    """
    names = {
        str(doc.get("source") or "").strip().lower(),
        str(doc.get("source_type") or "").strip().lower(),
    }
    for hit in doc.get("source_hits") or []:
        if isinstance(hit, dict):
            names.add(str(hit.get("source") or "").strip().lower())
    return {name for name in names if name}


def _has_source(docs: List[Dict[str, Any]], source_names: set[str]) -> bool:
    """判断候选集合是否包含指定来源。

    Args:
        docs: Rerank 候选文档列表。
        source_names: 需要检测的来源名称集合。

    Returns:
        命中任一来源时返回 True。
    """
    for doc in docs:
        if _source_names(doc) & source_names:
            return True
    return False


def _score_for_skipped_doc(doc: Dict[str, Any]) -> float:
    """选择跳过 Rerank 时用于后续 Evidence Gate 的分数。

    Args:
        doc: Rerank 候选文档。

    Returns:
        优先使用原始召回分，其次使用 rrf_score。
    """
    score = _safe_float(doc.get("score"))
    if score > 0:
        return score
    return _safe_float(doc.get("rrf_score"))


def _build_skipped_docs(candidate_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """构建跳过 Rerank 后的 reranked_docs。

    Args:
        candidate_docs: 已裁剪的 Rerank 候选文档。

    Returns:
        可被 Answer 和 Evidence Gate 消费的文档列表。
    """
    docs = []
    for doc in candidate_docs[:RERANK_MAX_TOPK]:
        skipped_doc = dict(doc)
        skipped_doc["score"] = _score_for_skipped_doc(skipped_doc)
        skipped_doc["rerank_skipped"] = True
        docs.append(skipped_doc)
    return docs


def _build_rerank_quality_for_skip(final_docs: List[Dict[str, Any]], input_docs: List[Dict[str, Any]], reason: str) -> Dict[str, Any]:
    """构建跳过 Rerank 时的质量诊断。

    Args:
        final_docs: 输出给答案节点的文档。
        input_docs: Gate 输入候选文档。
        reason: 跳过原因。

    Returns:
        Rerank 质量诊断字典。
    """
    scores = [_safe_float(doc.get("score")) for doc in final_docs]
    top1_score = scores[0] if scores else 0.0
    top3_scores = scores[:3]
    top3_avg_score = sum(top3_scores) / len(top3_scores) if top3_scores else 0.0
    if not final_docs:
        answerability = "insufficient"
    elif top1_score < 0.35:
        answerability = "insufficient"
    elif top1_score < 0.5:
        answerability = "low_confidence"
    else:
        answerability = "answerable"
    return {
        "status": "skipped",
        "skip_reason": reason,
        "error_type": "",
        "top1_score": top1_score,
        "top3_avg_score": top3_avg_score,
        "score_gap": scores[0] - scores[1] if len(scores) > 1 else top1_score,
        "answerability": answerability,
        "local_doc_count": sum(1 for doc in input_docs if doc.get("source") in {"local", "local_vector"}),
        "web_doc_count": sum(1 for doc in input_docs if doc.get("source") == "web"),
        "input_count": len(input_docs),
        "output_count": len(final_docs),
    }


def _decide_rerank(state: QueryGraphState, candidate_docs: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """判断当前候选是否需要执行 Rerank。

    Args:
        state: 查询流程状态。
        candidate_docs: 已裁剪候选文档。

    Returns:
        二元组，第一个值表示是否需要 Rerank，第二个值为原因。
    """
    if not candidate_docs:
        return False, "no_candidates"

    query_type = normalize_query_type(state.get("query_type", "unknown"))
    if query_type in COMPLEX_QUERY_TYPES:
        return True, "complex_query_type"

    # Web/HyDE 召回通常代表证据来源更复杂，哪怕只有单条也保留精排，避免低成本路径误伤答案质量。
    if _has_source(candidate_docs, {"web", "external_web"}):
        return True, "web_candidates"
    if _has_source(candidate_docs, {"hyde"}):
        return True, "hyde_candidates"
    if len(candidate_docs) == 1:
        return False, "single_candidate"

    top_score = _score_for_skipped_doc(candidate_docs[0])
    if (
        query_type in SIMPLE_QUERY_TYPES
        and len(candidate_docs) <= SMALL_LOCAL_CANDIDATE_LIMIT
        and top_score >= HIGH_CONFIDENCE_LOCAL_SCORE
    ):
        return False, "high_confidence_local_simple_query"

    return True, "needs_precision_ranking"


def node_rerank_gate(state: QueryGraphState):
    """在 Evidence Fusion 后决定是否跳过 Rerank 模型。

    Args:
        state: 查询流程状态，包含 rrf_chunks、web_search_docs 和 query_type。

    Returns:
        写入 rerank_gate；跳过时同时写入 reranked_docs 和 rerank_quality。
    """
    func_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    add_running_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点开始执行，全局状态为：{sanitize_for_log(state)}")

    candidate_docs = step_1_merge_rrf_web_search_results(state.get("rrf_chunks") or [], state.get("web_search_docs") or [])
    limited_docs = _limit_rerank_candidates(candidate_docs)
    should_rerank, reason = _decide_rerank(state, limited_docs)
    gate_result = {
        "status": "ok",
        "should_rerank": should_rerank,
        "reason": reason,
        "input_count": len(candidate_docs),
        "candidate_count": len(limited_docs),
    }
    state["rerank_gate"] = gate_result

    if should_rerank:
        logger.info(f"【{func_name}】决定执行 Rerank：{sanitize_for_log(gate_result)}")
        add_done_task(session_id, func_name, is_stream)
        return state

    final_docs = _build_skipped_docs(limited_docs)
    state["reranked_docs"] = final_docs
    state["rerank_quality"] = _build_rerank_quality_for_skip(final_docs, limited_docs, reason)
    record_rerank_docs("input", len(limited_docs))
    record_rerank_docs("scored", 0)
    record_rerank_docs("output", len(final_docs))

    add_done_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】跳过 Rerank，诊断信息：{sanitize_for_log({'gate': gate_result, 'quality': state['rerank_quality']})}")
    return state
