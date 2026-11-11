"""查询流程 Rerank 节点。"""

import sys
import time
from typing import Any

from app.conf.query_optimization_config import query_optimization_config
from app.conf.reranker_config import reranker_config
from app.lm.reranker_utils import get_reranker_model
from app.observability.metrics import record_rerank, record_rerank_docs
from app.observability.sanitizers import sanitize_for_log
from app.query_process.agent.evidence import is_evidence_document, web_doc_to_evidence
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def _normalize_rerank_text(value: Any) -> str:
    """将检索文档文本规范化为 reranker 可接受的字符串。

    Args:
        value: 原始文本字段，可能是字符串，也可能是快照压缩后的字典。

    Returns:
        可传入 reranker 的字符串；无法提取文本时返回空字符串。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # safe/debug 快照会把长文本压缩为 {preview, chars, sha256}，回放时要还原为可打分文本。
        for key in ("text", "content", "snippet", "preview"):
            text = value.get(key)
            if text:
                return str(text)
        return ""
    return str(value)


def step_1_merge_rrf_web_search_results(rrf_chunks, web_search_docs):
    """合并标准证据或旧格式 RRF 结果和 Web 搜索结果。

    Args:
        rrf_chunks: RRF 融合后的证据列表。
        web_search_docs: Web 搜索结果列表。

    Returns:
        可传入 reranker 的统一文档列表。
    """
    merged_docs = []
    evidence_mode = any(is_evidence_document(doc) for doc in rrf_chunks or [])

    for rrf_chunk in rrf_chunks:
        if is_evidence_document(rrf_chunk):
            text = _normalize_rerank_text(rrf_chunk.get("text"))
            if not text:
                logger.warning(f"跳过缺少可重排文本的证据文档：{sanitize_for_log(rrf_chunk)}")
                continue
            merged_docs.append({
                "text": text,
                "chunk_id": rrf_chunk.get("chunk_id", ""),
                "title": rrf_chunk.get("title") or rrf_chunk.get("file_title") or "",
                "file_title": rrf_chunk.get("file_title", ""),
                "section": rrf_chunk.get("section", ""),
                "item_name": rrf_chunk.get("item_name", ""),
                "source": rrf_chunk.get("source_type") or rrf_chunk.get("source") or "local",
                "source_type": rrf_chunk.get("source_type") or rrf_chunk.get("source") or "local",
                "url": rrf_chunk.get("url", ""),
                "domain": rrf_chunk.get("domain", ""),
                "evidence_id": rrf_chunk.get("evidence_id", ""),
                "score": rrf_chunk.get("score", 0.0),
                "rrf_score": rrf_chunk.get("rrf_score", 0.0),
                "source_hits": rrf_chunk.get("source_hits", []),
                "metadata": rrf_chunk.get("metadata", {}),
            })
            continue

        entity = rrf_chunk.get("entity")
        if entity:
            text = _normalize_rerank_text(entity.get("content"))
            if not text:
                logger.warning(f"跳过缺少可重排文本的 RRF 文档：{sanitize_for_log(entity)}")
                continue
            merged_docs.append({
                "text": text,
                "chunk_id": entity.get("chunk_id"),
                "title": entity.get("title"),
                "source": "local",
                "url": ""
            })

    if evidence_mode:
        logger.info(f"EvidenceDocument 转换后的 Rerank 输入：{sanitize_for_log(merged_docs)}")
        return merged_docs

    # 兼容旧格式：当 RRF 尚未输出标准证据时，继续在 Rerank 前合并 Web。
    for doc in web_search_docs:
        evidence_doc = web_doc_to_evidence(doc)
        text = _normalize_rerank_text(evidence_doc.get("text") if evidence_doc else "")
        if not text:
            logger.warning(f"跳过缺少可重排文本的 Web 文档：{sanitize_for_log(doc)}")
            continue
        merged_docs.append({
            "text": text,
            "title": evidence_doc.get("title", ""),
            "source": "web",
            "source_type": "web",
            "url": evidence_doc.get("url", ""),
            "domain": evidence_doc.get("domain", ""),
            "chunk_id": "",
            "evidence_id": evidence_doc.get("evidence_id", ""),
            "source_hits": evidence_doc.get("source_hits", []),
        })

    logger.info(f"rrf_chunks 和 mcp查询结果 合并后的列表：{sanitize_for_log(merged_docs)}")
    return merged_docs


def step_2_rerank(query, merged_docs):
    """调用 reranker 对候选文档打分排序。

    Args:
        query: 用户问题或改写后的问题。
        merged_docs: 已统一结构的候选文档列表。

    Returns:
        按 rerank 分数倒序排列后的文档列表。
    """
    if not merged_docs:
        return []

    start = time.perf_counter()
    status = "ok"
    # 1. 获取重排序模型
    model = get_reranker_model()

    # 2. 计算重排序分数前，先合并问题以及答案，格式[['query', 'passage'],['query', 'passage'],...]
    questions_pairs = []
    query_text = _normalize_rerank_text(query)
    for doc in merged_docs:
        text = _normalize_rerank_text(doc.get("text"))
        if text:
            questions_pairs.append([query_text, text])

    if not questions_pairs:
        return []

    # 3. 调用模型获取分数
    # normalize=True 默认False 分的范围不确定 7  8 9    0  - 1  -3  -8 分差特别大
    #                   True  分 缩放到 0 - 1 分
    # 返回值： [0.93,0.91,0.90,0.39,0.6,0.5,0.4,0.3,0.2,0.1] 成对问题对应的分！ 顺序也是相同
    try:
        scores = model.compute_score(questions_pairs, normalize=True)
    except Exception:
        status = "error"
        raise
    finally:
        record_rerank(
            reranker_config.bge_reranker_large or "unknown",
            status,
            time.perf_counter() - start,
        )
    if not isinstance(scores, list):
        scores = [scores]

    # 4. 将模型打分结果添加到文档中
    for index, doc in enumerate(merged_docs):
        doc["score"] = scores[index]

    # 5. 排序
    merged_docs.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"已经完成排序和打分！最终结果为：{merged_docs}")
    return merged_docs


# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5  # 最大间断分值
RERANK_LOCAL_CANDIDATE_LIMIT: int = 12
RERANK_WEB_CANDIDATE_LIMIT: int = 3
RERANK_TOTAL_CANDIDATE_LIMIT: int = 15
RERANK_INSUFFICIENT_THRESHOLD: float = 0.35
RERANK_LOW_CONFIDENCE_THRESHOLD: float = 0.5
RERANK_CIRCUIT_BREAK_SECONDS: int = 60
_RERANK_CIRCUIT_OPEN_UNTIL: float = 0.0


def _limit_rerank_candidates(merged_docs):
    """限制 Rerank 输入候选数量。

    Args:
        merged_docs: 已合并的本地和 Web 候选文档。

    Returns:
        按来源和总量裁剪后的候选文档列表。
    """
    limited_docs = []
    local_count = 0
    web_count = 0
    for doc in merged_docs:
        source = doc.get("source")
        if source == "web":
            if web_count >= RERANK_WEB_CANDIDATE_LIMIT:
                continue
            web_count += 1
        else:
            if local_count >= RERANK_LOCAL_CANDIDATE_LIMIT:
                continue
            local_count += 1
        limited_docs.append(doc)
        if len(limited_docs) >= RERANK_TOTAL_CANDIDATE_LIMIT:
            break
    return limited_docs


def _classify_rerank_error(exc: Exception) -> str:
    """识别 Reranker 错误类型。

    Args:
        exc: Reranker 调用过程中抛出的异常。

    Returns:
        标准化错误类型。
    """
    message = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timeout" in message or "timed out" in message:
        return "timeout"
    if isinstance(exc, MemoryError) or "out of memory" in message or "oom" in message:
        return "oom"
    if "model" in message and ("load" in message or "init" in message):
        return "model_load_error"
    return type(exc).__name__


def _should_open_rerank_circuit(error_type: str) -> bool:
    """判断 Reranker 异常是否需要打开短时熔断。

    Args:
        error_type: 标准化错误类型。

    Returns:
        需要打开熔断时返回 True。
    """
    return error_type in {"oom", "model_load_error"}


def _fallback_rerank_docs(candidate_docs, reason: str):
    """构建 Reranker 降级结果。

    Args:
        candidate_docs: 裁剪后的候选文档。
        reason: 降级原因。

    Returns:
        带兜底分数和降级原因的文档列表。
    """
    fallback_docs = []
    # RRF 和 Web 原始顺序已经代表召回阶段排序；降级时保留该排序，并补齐 score 供后续答案链路使用。
    for doc in candidate_docs:
        fallback_doc = dict(doc)
        fallback_doc["score"] = float(fallback_doc.get("score", 0.0) or 0.0)
        fallback_doc["rerank_fallback"] = True
        fallback_doc["fallback_reason"] = reason
        fallback_docs.append(fallback_doc)
    return fallback_docs[:RERANK_MAX_TOPK]


def _build_rerank_quality(final_docs, input_docs, status: str = "ok", error_type: str = ""):
    """构建 Rerank 质量诊断信息。

    Args:
        final_docs: 最终输出给答案节点的文档。
        input_docs: 传入 reranker 或兜底排序的候选文档。
        status: Rerank 执行状态。
        error_type: Rerank 异常类型。

    Returns:
        Rerank 质量诊断字典。
    """
    scores = [float(doc.get("score", 0.0) or 0.0) for doc in final_docs]
    top1_score = scores[0] if scores else 0.0
    top3_scores = scores[:3]
    top3_avg_score = sum(top3_scores) / len(top3_scores) if top3_scores else 0.0
    score_gap = scores[0] - scores[1] if len(scores) > 1 else top1_score
    if not final_docs or top1_score < RERANK_INSUFFICIENT_THRESHOLD:
        answerability = "insufficient"
    elif top1_score < RERANK_LOW_CONFIDENCE_THRESHOLD:
        answerability = "low_confidence"
    else:
        answerability = "answerable"
    return {
        "status": status,
        "error_type": error_type,
        "top1_score": top1_score,
        "top3_avg_score": top3_avg_score,
        "score_gap": score_gap,
        "answerability": answerability,
        "local_doc_count": sum(1 for doc in input_docs if doc.get("source") == "local"),
        "web_doc_count": sum(1 for doc in input_docs if doc.get("source") == "web"),
        "input_count": len(input_docs),
        "output_count": len(final_docs),
    }


def step_3_topk_and_gap(rerank_score_list):
    """根据 TopK 和分数断崖截取 Rerank 结果。

    Args:
        rerank_score_list: 已按 Rerank 分数倒序排列的候选文档。

    Returns:
        截断后的候选文档列表。
    """
    """
       [
         {
               text:内容 snippet content,
               chunk_id: chunk_id rrf有  mcp None,
               title: title ,
               url : rrfNone mcp url ,
               source: web -> mcp  || local -> rrf ,
               score: rerank打的分
         }
       ]
    """
    # 1. 判断topK
    top_k = min(len(rerank_score_list), RERANK_MAX_TOPK)
    # 2.循环处理数据列表，进行双指针处理和比较
    if top_k > RERANK_MIN_TOPK:
        for i in range(RERANK_MIN_TOPK - 1, top_k - 1):
            score1 = rerank_score_list[i].get("score", 0.0)
            score2 = rerank_score_list[i+1].get("score", 0.0)

            gap = abs(score1 - score2)
            # 加一个1e-6 避免分值为0
            gap_relative = gap / (score1 + 1e-6)
            if gap_relative > RERANK_GAP_RATIO or gap > RERANK_GAP_ABS:
                top_k = i + 1
                logger.info(f"数据集合{i}和{i + 1}的位置发生了断崖，结束循环！！")
                break

    topk_doc_list = rerank_score_list[:top_k]
    # 4.打印日志处理
    logger.info(f"最终截取的长度：{top_k},截取的内容:{topk_doc_list}")
    # 5.返回结果
    return topk_doc_list


def node_rerank(state: QueryGraphState):
    """使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。

    Args:
        state: 查询流程状态，包含 RRF 结果、Web 结果和改写问题。

    Returns:
        更新 `reranked_docs` 后的查询流程状态。

    Raises:
        Exception: Reranker 失败且降级开关关闭时抛出。

    rrf_chunks 格式:[
        {"id":"",
        "distance":"",
        entity:{
                "chunk_id":"",
                "content":"",
            }
        },
        ...
    ]
    mcp结果格式：[
                   {"snippet":"内容","title":"标题","url":"本次返回的参考url"},
                   {"snippet":"内容","title":"标题","url":"本次返回的参考url"},
                ...
              ]
    """
    func_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    add_running_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点开始执行")

    # 1. 获取向量查询结果 和 mcp 查询结果
    rrf_chunks = state["rrf_chunks"]
    web_search_docs = state["web_search_docs"]

    # 2. 对两个结果进行合并
    merged_docs = step_1_merge_rrf_web_search_results(rrf_chunks, web_search_docs)
    limited_docs = _limit_rerank_candidates(merged_docs)
    record_rerank_docs("input", len(limited_docs))

    # 3. 使用rerank模型，进行打分 获取结果 并进行排序
    """
        [
          {
                text:内容 snippet content,
                chunk_id: chunk_id rrf有  mcp None,
                title: title ,
                url : rrfNone mcp url ,
                source: web -> mcp  || local -> rrf ,
                score: rerank打的分 
          }
        ]
        """
    query = state["rewritten_query"]
    global _RERANK_CIRCUIT_OPEN_UNTIL
    status = "ok"
    error_type = ""
    if time.time() < _RERANK_CIRCUIT_OPEN_UNTIL:
        status = "fallback"
        error_type = "circuit_open"
        rerank_docs_with_score = _fallback_rerank_docs(limited_docs, error_type)
        logger.warning(f"Reranker 熔断中，跳过模型打分并使用原始召回排序：{error_type}")
    else:
        try:
            rerank_docs_with_score = step_2_rerank(query, limited_docs)
        except Exception as exc:
            if not query_optimization_config.rerank_fallback_enabled:
                raise
            status = "fallback"
            error_type = _classify_rerank_error(exc)
            if _should_open_rerank_circuit(error_type):
                # OOM 或模型加载失败通常短时间内会连续失败，打开短时熔断保护查询链路。
                _RERANK_CIRCUIT_OPEN_UNTIL = time.time() + RERANK_CIRCUIT_BREAK_SECONDS
            rerank_docs_with_score = _fallback_rerank_docs(limited_docs, error_type)
            logger.exception(f"Reranker 执行失败，已降级为原始召回排序：{exc}")

    # 4.  启动算法进行放断崖以及topk处理  0.9  0.89  0.35
    final_doc_list = step_3_topk_and_gap(rerank_docs_with_score)
    record_rerank_docs("scored", len(rerank_docs_with_score))
    record_rerank_docs("output", len(final_doc_list))

    # 5. 将最终重排序结果和质量诊断保存到 State，供后续低置信拒答与证据门控使用。
    state["reranked_docs"] = final_doc_list
    state["rerank_quality"] = _build_rerank_quality(final_doc_list, limited_docs, status, error_type)

    add_done_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点执行完成，质量诊断：{state['rerank_quality']}")
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)

    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [
        {"entity": {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9}},
        {"entity": {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8}},
        {"entity": {"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1}}  # 预期低分
    ]

    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"}  # 预期低分
    ]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",  # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")

        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
