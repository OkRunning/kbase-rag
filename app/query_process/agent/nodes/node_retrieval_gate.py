"""本地召回质量二次门控节点。"""

import sys
from typing import Any, Dict, List

from app.conf.query_optimization_config import query_optimization_config
from app.core.logger import logger
from app.observability.sanitizers import sanitize_for_log
from app.query_process.agent.nodes.query_understanding_utils import normalize_query_type
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task

LOCAL_HIGH_CONFIDENCE_TOP1 = 0.75
LOCAL_HIGH_CONFIDENCE_TOP3_AVG = 0.65
LOCAL_LOW_CONFIDENCE_TOP1 = 0.45


def _extract_local_scores(embedding_chunks: List[Dict[str, Any]]) -> List[float]:
    """提取本地向量召回分数。

    Args:
        embedding_chunks: 普通向量检索返回的 chunk 列表。

    Returns:
        按分数倒序排列的召回分数列表。
    """
    scores: List[float] = []
    for chunk in embedding_chunks or []:
        score = chunk.get("distance", chunk.get("score", 0.0))
        try:
            scores.append(float(score or 0.0))
        except (TypeError, ValueError):
            continue
    scores.sort(reverse=True)
    return scores


def _allows_low_confidence_web_fallback(query_type: str, retrieval_plan: Dict[str, Any]) -> bool:
    """判断低置信本地召回是否允许 Web 补充。

    Args:
        query_type: 标准化后的查询类型。
        retrieval_plan: 当前检索计划。

    Returns:
        允许低置信 Web 补充时返回 True。
    """
    if retrieval_plan.get("use_web"):
        return True
    if not query_optimization_config.web_search_on_low_confidence:
        return False
    return query_type in {"realtime", "unknown"}


def node_retrieval_gate(state: QueryGraphState):
    """根据本地向量召回质量决定是否补充 HyDE 或 Web。

    Args:
        state: 查询流程状态，包含本地向量召回结果和检索计划。

    Returns:
        写入二次门控诊断和有效检索计划后的查询状态。
    """
    func_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    add_running_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点开始执行，全局状态为：{sanitize_for_log(state)}")

    retrieval_plan = dict(state.get("retrieval_plan") or {})
    query_type = normalize_query_type(state.get("query_type", "unknown"))
    scores = _extract_local_scores(state.get("embedding_chunks") or [])
    top1_score = scores[0] if scores else 0.0
    top3_scores = scores[:3]
    top3_avg_score = sum(top3_scores) / len(top3_scores) if top3_scores else 0.0
    local_empty = not scores
    local_high_confidence = top1_score >= LOCAL_HIGH_CONFIDENCE_TOP1 and top3_avg_score >= LOCAL_HIGH_CONFIDENCE_TOP3_AVG
    local_low_confidence = local_empty or top1_score < LOCAL_LOW_CONFIDENCE_TOP1

    plan_wants_hyde = bool(retrieval_plan.get("use_hyde"))
    plan_wants_web = bool(retrieval_plan.get("use_web"))
    should_use_hyde = False
    should_use_web = False
    skip_reason = ""

    # 二次门控只在本地召回后执行：本地足够好时关闭补充召回，本地很差时再打开补充召回。
    if local_high_confidence:
        skip_reason = "local_high_confidence"
    else:
        should_use_hyde = plan_wants_hyde or (local_low_confidence and query_type not in {"chitchat", "realtime"})
        should_use_web = plan_wants_web or (local_low_confidence and _allows_low_confidence_web_fallback(query_type, retrieval_plan))
        if local_low_confidence:
            skip_reason = "local_low_confidence"
        else:
            skip_reason = "follow_initial_plan"

    retrieval_plan["use_hyde"] = should_use_hyde
    retrieval_plan["use_web"] = should_use_web
    state["retrieval_plan"] = retrieval_plan

    retrieval_diagnostics = dict(state.get("retrieval_diagnostics") or {})
    retrieval_diagnostics["retrieval_gate"] = {
        "status": "ok",
        "skip_reason": skip_reason,
        "error_type": "",
        "local_doc_count": len(state.get("embedding_chunks") or []),
        "local_top1_score": top1_score,
        "local_top3_avg_score": top3_avg_score,
        "local_high_confidence": local_high_confidence,
        "local_low_confidence": local_low_confidence,
        "use_hyde": should_use_hyde,
        "use_web": should_use_web,
    }
    state["retrieval_diagnostics"] = retrieval_diagnostics

    add_done_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点执行完成，诊断信息：{sanitize_for_log(retrieval_diagnostics['retrieval_gate'])}")
    return state
