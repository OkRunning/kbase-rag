import sys
from typing import Any, Dict, List, Tuple

from app.core.logger import logger
from app.observability.metrics import record_retrieval_docs
from app.observability.sanitizers import sanitize_for_log
from app.query_process.agent.evidence import (
    EvidenceDocument,
    hyde_hit_to_evidence,
    local_hit_to_evidence,
    web_doc_to_evidence,
)
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task


def _merge_source_hits(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """合并多源命中信息。

    Args:
        left: 已有命中列表。
        right: 新增命中列表。

    Returns:
        按 source 去重后的命中列表。
    """
    merged: List[Dict[str, Any]] = []
    seen = set()
    for hit in [*(left or []), *(right or [])]:
        source = hit.get("source")
        if source in seen:
            continue
        seen.add(source)
        merged.append(hit)
    return merged


def _get_source_weight(source_weights: Dict[str, Any], name: str, default: float) -> float:
    """读取召回源权重。

    Args:
        source_weights: 检索计划中的来源权重。
        name: 来源名称。
        default: 默认权重。

    Returns:
        可用于 RRF 的浮点权重。
    """
    try:
        return float(source_weights.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_rrf_top_k(retrieval_plan: Dict[str, Any], default: int = 5) -> int:
    """读取 RRF 输出数量上限。

    Args:
        retrieval_plan: 当前检索计划。
        default: 默认 TopK。

    Returns:
        大于 0 的 TopK。
    """
    try:
        top_k = int(retrieval_plan.get("rrf_top_k", default) or default)
    except (TypeError, ValueError):
        return default
    return max(1, top_k)


def _normalize_sources(
    embedding_chunks: list,
    hyde_embedding_chunks: list,
    web_search_docs: list,
    source_weights: Dict[str, float] | None = None,
) -> List[Tuple[List[EvidenceDocument], float]]:
    """将各召回源标准化为 EvidenceDocument。

    Args:
        embedding_chunks: 普通向量召回结果。
        hyde_embedding_chunks: HyDE 向量召回结果。
        web_search_docs: Web Search 召回结果。
        source_weights: 检索计划中的来源权重。

    Returns:
        带来源权重的标准证据列表。
    """
    source_with_weight: List[Tuple[List[EvidenceDocument], float]] = []
    weights = source_weights or {}
    local_docs = [
        doc
        for rank, chunk in enumerate(embedding_chunks or [], start=1)
        if (doc := local_hit_to_evidence(chunk, source="local_vector", rank=rank)) is not None
    ]
    hyde_docs = [
        doc
        for rank, chunk in enumerate(hyde_embedding_chunks or [], start=1)
        if (doc := hyde_hit_to_evidence(chunk, rank=rank)) is not None
    ]
    web_docs = [
        doc
        for rank, web_doc in enumerate(web_search_docs or [], start=1)
        if (doc := web_doc_to_evidence(web_doc, rank=rank)) is not None
    ]
    if local_docs:
        source_with_weight.append((local_docs, _get_source_weight(weights, "local_vector", 1.0)))
    if hyde_docs:
        source_with_weight.append((hyde_docs, _get_source_weight(weights, "hyde", 0.7)))
    if web_docs:
        source_with_weight.append((web_docs, _get_source_weight(weights, "web", 0.5)))
    return source_with_weight


def _ensure_item_group_coverage(
    ranked_docs: List[EvidenceDocument],
    all_docs: List[EvidenceDocument],
    top_k: int,
) -> List[EvidenceDocument]:
    """为多商品/实体查询保留每个 item 的至少一条证据。

    Args:
        ranked_docs: 已排序并截断的证据列表。
        all_docs: 全量候选证据列表。
        top_k: 返回数量上限。

    Returns:
        补齐 item 覆盖后的证据列表。
    """
    selected_ids = {doc.get("evidence_id") for doc in ranked_docs}
    selected_items = {doc.get("item_name") for doc in ranked_docs if doc.get("item_name")}
    all_items = {doc.get("item_name") for doc in all_docs if doc.get("item_name")}
    if len(all_items) <= 1:
        return ranked_docs

    # 多实体对比时，避免某一实体高分文档挤掉另一实体的全部证据。
    for item_name in sorted(all_items - selected_items):
        candidate = next(
            (doc for doc in all_docs if doc.get("item_name") == item_name and doc.get("evidence_id") not in selected_ids),
            None,
        )
        if not candidate:
            continue
        if len(ranked_docs) < top_k:
            ranked_docs.append(candidate)
        else:
            ranked_docs[-1] = candidate
        selected_ids.add(candidate.get("evidence_id"))
    ranked_docs.sort(key=lambda doc: float(doc.get("rrf_score", 0.0) or 0.0), reverse=True)
    return ranked_docs[:top_k]


def step_3_reciprocal_rank_fusion(source_with_weight, top_k: int = 5):
    """使用 RRF 算法对标准证据进行融合排序。

    Args:
        source_with_weight: 标准证据列表和权重列表。
        top_k: 返回的最大证据数。

    Returns:
        排序后的 EvidenceDocument 列表。
    """
    evidence_dict: Dict[str, EvidenceDocument] = {}
    score_dict: Dict[str, float] = {}
    all_docs: List[EvidenceDocument] = []
    active_sources = sum(1 for docs, _ in source_with_weight if docs)

    """
    {
        "evidence_id":"",
        "source_type": "local",
        "source": source,
        "text": text,
        "chunk_id": str(hit.get("chunk_id") or entity.get("chunk_id") or hit.get("id") or ""),
        "title": str(hit.get("title") or entity.get("title") or ""),
        "file_title": str(hit.get("file_title") or entity.get("file_title") or ""),
        "section": str(hit.get("section") or entity.get("section") or entity.get("title") or ""),
        "parent_title": str(hit.get("parent_title") or entity.get("parent_title") or ""),
        "item_name": str(hit.get("item_name") or entity.get("item_name") or ""),
        "doc_id": str(hit.get("doc_id") or entity.get("doc_id") or ""),
        "url": "",
        "domain": "",
        "score": score,
        "rrf_score": 0.0,
        "source_hits": [{"source": source, "rank": rank, "score": score}],
        "metadata": {"raw_id": hit.get("id")},
    }
    """

    for docs, weight in source_with_weight:
        for rank, doc in enumerate(docs, start=1):
            evidence_id = doc.get("evidence_id")
            if not evidence_id:
                logger.warning(f"跳过缺少 evidence_id 的召回结果：{sanitize_for_log(doc)}")
                continue
            all_docs.append(doc)
            rrf_score = (1 / (60 + rank)) * weight
            if evidence_id not in evidence_dict:
                evidence_dict[evidence_id] = dict(doc)
                evidence_dict[evidence_id]["source_hits"] = list(doc.get("source_hits") or [])
            else:
                existing_doc = evidence_dict[evidence_id]
                existing_doc["source_hits"] = _merge_source_hits(
                    existing_doc.get("source_hits") or [],
                    doc.get("source_hits") or [],
                )
                # 同一 chunk 被多路召回时，保留更高的原始召回分数，方便诊断。
                existing_doc["score"] = max(float(existing_doc.get("score", 0.0) or 0.0), float(doc.get("score", 0.0) or 0.0))
            score_dict[evidence_id] = score_dict.get(evidence_id, 0.0) + rrf_score

    if active_sources <= 1:
        # 单源少量候选不必计算融合排名，但仍完成标准化、去重和 source_hits 保留。
        ranked_docs = list(evidence_dict.values())[:top_k]
        for index, doc in enumerate(ranked_docs, start=1):
            doc["rrf_score"] = 1 / (60 + index)
        logger.info(f"证据标准化去重后结果为：{sanitize_for_log(ranked_docs)}")
        return ranked_docs

    for evidence_id, score in score_dict.items():
        evidence_dict[evidence_id]["rrf_score"] = score

    ranked_docs = sorted(evidence_dict.values(), key=lambda doc: float(doc.get("rrf_score", 0.0) or 0.0), reverse=True)
    ranked_docs = _ensure_item_group_coverage(ranked_docs[:top_k], ranked_docs, top_k)
    logger.info(f"Evidence Fusion 排序后结果为：{sanitize_for_log(ranked_docs)}")
    return ranked_docs


def node_rrf(state: QueryGraphState):
    """执行证据融合排序节点。

    Args:
        state: 查询流程状态，包含 local、HyDE 和 Web 召回结果。

    Returns:
        更新 `rrf_chunks` 为标准 EvidenceDocument 列表后的查询流程状态。
    """
    func_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    add_running_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点开始执行")

    embedding_chunks = state["embedding_chunks"]
    hyde_embedding_chunks = state["hyde_embedding_chunks"]
    web_search_docs = state.get("web_search_docs", [])
    retrieval_plan = state.get("retrieval_plan") or {}
    rrf_top_k = _get_rrf_top_k(retrieval_plan)
    source_weights = retrieval_plan.get("source_weights") or {}

    source_with_weight = _normalize_sources(embedding_chunks, hyde_embedding_chunks, web_search_docs, source_weights)
    rrf_response = step_3_reciprocal_rank_fusion(source_with_weight, top_k=rrf_top_k)
    record_retrieval_docs("rrf", len(rrf_response))
    state["rrf_chunks"] = rrf_response

    add_done_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点执行完成")
    return state

# ================================
# 本地测试入口
# ================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rrf 本地测试")
    print("=" * 50)

    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"]
    }

    try:
        from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
        from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde

        emb_res = node_search_embedding(mock_state)
        hyde_res = node_search_embedding_hyde(mock_state)
        mock_state['embedding_chunks'] = emb_res.get("embedding_chunks") or []
        mock_state['hyde_embedding_chunks'] = hyde_res.get("hyde_embedding_chunks") or []

        result = node_rrf(mock_state)
        rrf_chunks = result.get("rrf_chunks", [])

        emb_cnt = len(mock_state.get("embedding_chunks") or [])
        hyde_cnt = len(mock_state.get("hyde_embedding_chunks") or [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={emb_cnt}, HyDE={hyde_cnt}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            doc_id = doc.get("chunk_id") or doc.get("id")
            content = (doc.get("content") or "")[:20]
            print(f"Rank {i}: ID={doc_id}, Content={content}...")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
