"""HyDE 向量检索节点。"""

import sys
from typing import Any, Dict

from langchain_core.messages import HumanMessage

from app.clients.document_versioning_utils import build_active_document_version_filter, build_active_vector_version_filter
from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
    normalize_milvus_hits,
)
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.core.tenant_scope import build_tenant_kb_filter
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.observability.metrics import record_retrieval_docs
from app.observability.sanitizers import sanitize_for_log
from app.query_process.agent.state import QueryGraphState
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_done_task, add_running_task

HYDE_MAX_CHINESE_CHARS = 200
HYDE_MAX_ENGLISH_WORDS = 120


def _merge_hyde_diagnostics(state: QueryGraphState, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """合并 HyDE 检索诊断信息。

    Args:
        state: 查询流程状态。
        diagnostics: 当前 HyDE 节点诊断信息。

    Returns:
        合并后的检索诊断信息。
    """
    retrieval_diagnostics = dict(state.get("retrieval_diagnostics") or {})
    retrieval_diagnostics["hyde"] = diagnostics
    return retrieval_diagnostics


def _empty_hyde_result(state: QueryGraphState, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """构建空 HyDE 召回结果。

    Args:
        state: 查询流程状态。
        diagnostics: 当前 HyDE 节点诊断信息。

    Returns:
        包含空 HyDE 结果和诊断信息的状态更新。
    """
    return {
        "hyde_embedding_chunks": [],
        "retrieval_diagnostics": _merge_hyde_diagnostics(state, diagnostics),
    }


def _limit_hypothesis_answer(text: str) -> str:
    """限制 HyDE 假设答案长度。

    Args:
        text: LLM 生成的 HyDE 假设答案。

    Returns:
        截断后的 HyDE 假设答案。
    """
    value = str(text or "").strip()
    if not value:
        return ""
    has_cjk = any("\u4e00" <= char <= "\u9fff" for char in value)
    if has_cjk:
        return value[:HYDE_MAX_CHINESE_CHARS]
    words = value.split()
    if len(words) <= HYDE_MAX_ENGLISH_WORDS:
        return value
    return " ".join(words[:HYDE_MAX_ENGLISH_WORDS])


def node_search_embedding_hyde(state: QueryGraphState):
    """按检索计划执行 HyDE 假设答案向量检索节点。

    Args:
        state: 查询流程状态，包含重写问题、商品名、会话信息和检索计划。

    Returns:
        包含 `hyde_embedding_chunks` 和 HyDE 诊断信息的状态更新。
    """
    func_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    rewritten_query = state["rewritten_query"]
    item_names = state["item_names"]
    tenant_id = state["tenant_id"]
    kb_id = state["kb_id"]

    logger.info(f"{func_name}节点开始执行，全局状态为{sanitize_for_log(state)}")
    add_running_task(session_id, func_name, is_stream)
    retrieval_plan = state.get("retrieval_plan") or {}
    use_hyde = retrieval_plan.get("use_hyde", True)
    if not use_hyde:
        # LangGraph 旧链路仍可能调用 HyDE 节点；这里按 retrieval_plan 做节点内二次门控。
        diagnostics = {
            "status": "skipped",
            "skip_reason": "disabled_by_plan",
            "error_type": "",
            "input_count": 0,
            "output_count": 0,
        }
        record_retrieval_docs("hyde", 0)
        add_done_task(session_id, func_name, is_stream)
        logger.info(f"{func_name}按检索计划跳过，诊断信息：{diagnostics}")
        return _empty_hyde_result(state, diagnostics)

    try:
        """
        1. 调用大模型先获取一个假设答案
        2. 将问题和假设答案向量化
        3. 调用milvus进行向量检索
        """

        # 1. 调用大模型先获取一个假设答案
        # 1.1 获取提示词
        prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
        messages = [
            HumanMessage(content=prompt)
        ]
        # 1.2 调用大模型
        llm_resp = get_llm_client().invoke(messages)
        hypothesis_answer = _limit_hypothesis_answer(llm_resp.content)
        logger.info(f"用户问题【{rewritten_query}】获取的假设答案为：{hypothesis_answer}")

        # 2. 将问题和假设答案向量化
        embeddings = generate_embeddings([rewritten_query + hypothesis_answer])
        dense_vector = embeddings['dense'][0]
        sparse_vector = embeddings['sparse'][0]

        # 3. 调用milvus进行向量检索
        # 3.1 获取milvus客户端
        milvus_client = get_milvus_client()
        # 3.2 创建多个 AnnSearchRequest 实例
        item_names_expr = ",".join(f'"{escape_milvus_string(name)}"' for name in item_names)
        extra_filters = [f"item_name in [{item_names_expr}]"]
        active_version_filter = build_active_document_version_filter(tenant_id, kb_id, item_names)
        active_vector_filter = build_active_vector_version_filter(tenant_id, kb_id)
        if active_version_filter:
            extra_filters.extend([
                'doc_status == "active"',
                'chunk_status == "active"',
                'vector_status == "active"',
                active_version_filter,
            ])
            if active_vector_filter:
                extra_filters.append(active_vector_filter)
        # HyDE 检索与普通检索共享同一版本过滤逻辑，确保生成答案时不混入历史版本。
        reqs = create_hybrid_search_requests(dense_vector,
                                             sparse_vector,
                                             expr=build_tenant_kb_filter(
                                                 tenant_id,
                                                 kb_id,
                                                 " and ".join(extra_filters),
                                             ))

        # 3.3 执行混合搜索
        response = hybrid_search(client=milvus_client,
                                 collection_name=milvus_config.chunks_collection,
                                 reqs=reqs,
                                 ranker_weights=(0.9, 0.1),
                                 norm_score=True,
                                 output_fields=[
                                     "chunk_id",
                                     "content",
                                     "file_title",
                                     "title",
                                     "parent_title",
                                     "item_name",
                                     "tenant_id",
                                     "kb_id",
                                     "doc_id",
                                     "doc_version",
                                     "doc_status",
                                     "chunk_status",
                                     "vector_status",
                                     "vector_version",
                                     "embedding_model",
                                     "embedding_version",
                                 ])

        # Milvus 原始 hit 可能同时存在顶层 id 和 entity.chunk_id，先统一结构再进入快照/RRF。
        hyde_embedding_chunks = normalize_milvus_hits(response[0]) if response else []
        hyde_embedding_chunks.sort(key=lambda x: x["distance"], reverse=True)
        record_retrieval_docs("hyde", len(hyde_embedding_chunks))
        logger.info(f"HyDE 检索结果为：{sanitize_for_log(hyde_embedding_chunks)}")
        diagnostics = {
            "status": "ok",
            "skip_reason": "",
            "error_type": "",
            "input_count": 1,
            "output_count": len(hyde_embedding_chunks),
            "hypothesis_answer_chars": len(hypothesis_answer),
        }
        return {
            "hyde_embedding_chunks": hyde_embedding_chunks,
            "retrieval_diagnostics": _merge_hyde_diagnostics(state, diagnostics),
        }
    except Exception as exc:
        # HyDE 是补充召回，失败时必须降级为空结果，避免中断本地向量主链路。
        diagnostics = {
            "status": "error",
            "skip_reason": "hyde_failed",
            "error_type": type(exc).__name__,
            "input_count": 1,
            "output_count": 0,
        }
        record_retrieval_docs("hyde", 0)
        logger.exception(f"{func_name}执行失败，已降级为空 HyDE 结果：{exc}")
        return _empty_hyde_result(state, diagnostics)
    finally:
        add_done_task(session_id, func_name, is_stream)
        logger.info(f"{func_name}节点执行完毕")


if __name__ == "__main__":
    # 本地测试代码
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_embedding_hyde 本地测试")
    print("=" * 50)

    # 模拟输入状态
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_search_embedding_hyde(mock_state)

        print(result)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
