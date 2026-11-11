"""答案生成前的证据充分性门控节点。"""

import sys
from typing import Any, Dict, List, Tuple

from app.conf.query_optimization_config import query_optimization_config
from app.core.logger import logger
from app.observability.events import emit_event
from app.observability.sanitizers import sanitize_for_log
from app.query_process.agent.nodes.query_understanding_utils import normalize_string_list
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task

LOW_CONFIDENCE_MARGIN = 0.1
CONFLICT_SCORE_FLOOR = 0.65
_WEB_SOURCES = {"web", "external_web"}
_LOCAL_SOURCES = {"local", "local_vector", "hyde"}
_CONFLICT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("支持", "不支持"),
    ("可以", "不可以"),
    ("允许", "不允许"),
    ("需要", "不需要"),
    ("开启", "关闭"),
    ("打开", "关闭"),
    ("有", "没有"),
    ("是", "不是"),
    ("true", "false"),
    ("yes", "no"),
)


def _safe_score(doc: Dict[str, Any]) -> float:
    """读取证据文档的 rerank 分数。

    Args:
        doc: 重排序后的证据文档。

    Returns:
        可比较的浮点分数。
    """
    try:
        return float(doc.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _source_type(doc: Dict[str, Any]) -> str:
    """读取证据来源类型。

    Args:
        doc: 重排序后的证据文档。

    Returns:
        标准化后的来源类型。
    """
    return str(doc.get("source_type") or doc.get("source") or "").strip().lower()


def _has_rerank_scores(docs: List[Dict[str, Any]]) -> bool:
    """判断证据分数是否全部来自 BGE-Reranker 归一化分。

    Rerank Gate 跳过精排时写入的分数是 Milvus distance 或 rrf_score，
    量纲与 Rerank 归一化分不同，不能与 evidence_min_rerank_score 直接比较。

    Args:
        docs: 重排序后的证据文档。

    Returns:
        True 表示所有证据分数均为真实的 Rerank 打分。
    """
    return not any(bool(doc.get("rerank_skipped")) for doc in docs)


def _extract_compare_entities(state: QueryGraphState) -> List[str]:
    """提取对比问题需要覆盖的实体名称。

    Args:
        state: 查询流程状态。

    Returns:
        去重后的实体名称列表。
    """
    entity_resolution = state.get("entity_resolution") or {}
    entities = state.get("entities") or {}
    candidates: List[str] = []
    for values in (
        entity_resolution.get("confirmed_item_names"),
        state.get("item_names"),
        entities.get("item_names"),
        entities.get("model_numbers"),
    ):
        candidates.extend(normalize_string_list(values))
    return normalize_string_list(candidates)


def _find_missing_entities(docs: List[Dict[str, Any]], entities: List[str]) -> List[str]:
    """判断对比问题是否缺少某些实体的证据。

    Args:
        docs: 重排序后的证据文档列表。
        entities: 需要覆盖的实体名称列表。

    Returns:
        未被任何证据命中的实体名称列表。
    """
    missing: List[str] = []
    for entity in entities:
        entity_lower = entity.lower()
        found = False
        for doc in docs:
            # 对比问题需要每个实体至少有一条证据命中；同时检查 item_name 和正文，兼容旧格式。
            searchable = " ".join(
                str(doc.get(key, "") or "")
                for key in ("item_name", "title", "file_title", "section", "text")
            ).lower()
            if entity_lower in searchable:
                found = True
                break
        if not found:
            missing.append(entity)
    return missing


def _has_conflicting_high_score_docs(docs: List[Dict[str, Any]]) -> bool:
    """使用关键词对检测高分证据之间的显性冲突。

    Args:
        docs: 重排序后的证据文档列表。

    Returns:
        存在高分证据结论冲突时返回 True。
    """
    high_score_texts = [
        str(doc.get("text") or "").lower()
        for doc in docs
        if _safe_score(doc) >= CONFLICT_SCORE_FLOOR
    ]
    if len(high_score_texts) < 2:
        return False

    for positive, negative in _CONFLICT_PAIRS:
        positive_lower = positive.lower()
        negative_lower = negative.lower()
        has_positive = any(positive_lower in text and negative_lower not in text for text in high_score_texts)
        has_negative = any(negative_lower in text for text in high_score_texts)
        if has_positive and has_negative:
            return True
    return False


def _build_gate_result(state: QueryGraphState) -> Dict[str, Any]:
    """根据规则构建 Evidence Gate 诊断结果。

    Args:
        state: 查询流程状态。

    Returns:
        包含 evidence_status、confidence、reason 和 missing_info 的诊断字典。
    """
    docs = list(state.get("reranked_docs") or [])
    min_docs = max(int(query_optimization_config.evidence_min_docs), 1)
    min_score = float(query_optimization_config.evidence_min_rerank_score)
    # 分数域错配修复：跳过精排的文档分数是 Milvus distance 或 rrf_score，
    # 量纲与 Rerank 归一化分不同，此时不再用 min_score 做分数阈值比较。
    rerank_scores = _has_rerank_scores(docs)
    scores = [_safe_score(doc) for doc in docs]
    top_score = scores[0] if scores else 0.0
    source_types = [_source_type(doc) for doc in docs]
    local_doc_count = sum(1 for source_type in source_types if source_type in _LOCAL_SOURCES)
    web_doc_count = sum(1 for source_type in source_types if source_type in _WEB_SOURCES)
    missing_info: List[str] = []

    if len(docs) < min_docs:
        status = "insufficient"
        reason = "reranked_docs 数量低于 Evidence Gate 最低证据数量"
    elif rerank_scores and top_score < min_score:
        status = "insufficient"
        reason = "top rerank score 低于 Evidence Gate 阈值"
    elif web_doc_count > 0 and local_doc_count == 0:
        status = "external_only"
        if query_optimization_config.allow_external_only:
            reason = "仅命中外部 Web 证据，当前配置允许 external_only 回答"
        else:
            reason = "仅命中外部 Web 证据，当前配置不允许 external_only 回答"
    elif state.get("query_type") == "comparison" and len(_extract_compare_entities(state)) >= 2:
        entities = _extract_compare_entities(state)
        missing_info = _find_missing_entities(docs, entities)
        if missing_info:
            status = "insufficient"
            reason = "对比问题缺少部分实体的证据"
        elif _has_conflicting_high_score_docs(docs):
            status = "conflicting"
            reason = "多个高分证据存在显性冲突"
        else:
            status = "sufficient" if not rerank_scores or top_score >= min_score + LOW_CONFIDENCE_MARGIN else "low_confidence"
            reason = "对比问题实体证据覆盖完整"
    elif _has_conflicting_high_score_docs(docs):
        status = "conflicting"
        reason = "多个高分证据存在显性冲突"
    else:
        status = "sufficient" if not rerank_scores or top_score >= min_score + LOW_CONFIDENCE_MARGIN else "low_confidence"
        reason = "证据通过规则门控"

    return {
        "evidence_status": status,
        "confidence": top_score,
        "reason": reason,
        "missing_info": missing_info,
        "top_score": top_score,
        "min_score": min_score,
        "min_docs": min_docs,
        "doc_count": len(docs),
        "local_doc_count": local_doc_count,
        "web_doc_count": web_doc_count,
        "allow_external_only": query_optimization_config.allow_external_only,
    }


def _answerability_from_evidence_gate(gate_result: Dict[str, Any]) -> str:
    """将 Evidence Gate 状态映射为答案生成策略。

    Args:
        gate_result: Evidence Gate 输出的诊断结果。

    Returns:
        node_answer_output 可识别的 answerability。
    """
    status = str(gate_result.get("evidence_status") or "")
    if status == "sufficient":
        return "answerable"
    if status == "low_confidence":
        return "low_confidence"
    if status == "external_only" and gate_result.get("allow_external_only"):
        return "low_confidence"
    return "insufficient"


def node_evidence_gate(state: QueryGraphState):
    """在答案生成前统一判断证据是否足够。

    Args:
        state: 查询流程状态，包含 reranked_docs、query_type 和实体信息。

    Returns:
        写入 evidence_status、evidence_gate 和回答策略后的查询流程状态。
    """
    func_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    add_running_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点开始执行，全局状态为：{sanitize_for_log(state)}")

    gate_result = _build_gate_result(state)
    evidence_status = gate_result["evidence_status"]
    state["evidence_status"] = evidence_status
    state["evidence_gate"] = gate_result

    # Evidence Gate 是最终生成前的统一裁判，需要同步覆盖 rerank_quality.answerability。
    rerank_quality = dict(state.get("rerank_quality") or {})
    rerank_quality["answerability"] = _answerability_from_evidence_gate(gate_result)
    rerank_quality["evidence_status"] = evidence_status
    state["rerank_quality"] = rerank_quality

    emit_event(
        "query.evidence_gate.completed",
        graph="query",
        node_name=func_name,
        session_id=session_id,
        tenant_id=state.get("tenant_id", "default"),
        kb_id=state.get("kb_id", "default"),
        status=evidence_status,
        summary=gate_result,
        tags=["query", func_name, "evidence_gate", evidence_status],
    )

    add_done_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点执行完成，门控结果：{sanitize_for_log(gate_result)}")
    return state
