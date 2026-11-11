"""标准实体解析与消歧节点。"""

import sys

from app.clients.mongo_history_utils import save_chat_message
from app.core.logger import logger
from app.observability.events import emit_event
from app.observability.sanitizers import sanitize_for_log
from app.query_process.agent.nodes.query_understanding_utils import (
    normalize_string_list,
    step_2_embedding_search_item_names,
    step_3_classify_item_names,
)
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task


def _build_entity_resolution_result(
    model_item_names: list[str],
    resolved_item_names: list[str],
    options_item_names: list[str],
) -> dict:
    """组装实体解析结果。

    Args:
        model_item_names: 模型原始抽取的商品名。
        resolved_item_names: 最终确认的商品名。
        options_item_names: 可供追问的候选商品名。

    Returns:
        结构化实体解析结果。
    """
    return {
        "model_item_names": model_item_names,
        "confirmed_item_names": resolved_item_names,
        "options_item_names": options_item_names,
    }


def node_entity_resolution(state: QueryGraphState):
    """执行商品实体解析和消歧。

    Args:
        state: 查询流程状态，包含 query_understanding 阶段输出。

    Returns:
        补充 item_names 和 entity_resolution 的查询状态。

    Raises:
        Exception: 商品实体向量检索失败时抛出。
    """
    func_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    tenant_id = state.get("tenant_id", "default")
    kb_id = state.get("kb_id", "default")
    original_query = state.get("original_query", "")
    # rewritten_query = state.get("rewritten_query") or original_query
    entities = state.get("entities") or {}
    model_item_names = normalize_string_list(entities.get("item_names") or state.get("item_names") or [])
    need_item_name = bool(state.get("need_item_name", False))

    add_running_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点开始执行，全局状态为：{sanitize_for_log(state)}")

    confirmed_item_names: list[str] = []
    options_item_names: list[str] = []

    # 1. 只有当查询理解明确依赖商品名时，才执行实体检索和消歧。
    #    这样泛知识库问题不会被商品名入口卡住，也不会误把空实体写成阻断条件。
    if need_item_name and model_item_names:
        embedding_result = step_2_embedding_search_item_names(model_item_names, tenant_id, kb_id)
        final_item_names = step_3_classify_item_names(embedding_result)
        confirmed_item_names = final_item_names.get("confirmed_item_names", [])
        options_item_names = final_item_names.get("options_item_names", [])
        state["item_names"] = confirmed_item_names
    elif not need_item_name:
        state["item_names"] = model_item_names
    else:
        state["item_names"] = []

    state["entity_resolution"] = _build_entity_resolution_result(
        model_item_names=model_item_names,
        resolved_item_names=state.get("item_names", []),
        options_item_names=options_item_names,
    )

    logger.info(
        "实体解析完成："
        f"{sanitize_for_log({'item_names': state.get('item_names', []), 'query_type': state.get('query_type', 'unknown')})}"
    )
    emit_event(
        "query.entity_resolution.completed",
        graph="query",
        node_name=func_name,
        session_id=session_id,
        tenant_id=tenant_id,
        kb_id=kb_id,
        status="ok",
        summary={
            "need_item_name": need_item_name,
            "model_item_names": model_item_names,
            "confirmed_item_names": confirmed_item_names,
            "options_item_names": options_item_names,
            "query_type": state["query_type"],
        },
        tags=["query", func_name, "entity_resolution"],
    )

    # 2. 统一保存原始用户问题，保证后续流式/非流式路径的历史记录一致。
    save_chat_message(
        session_id=session_id,
        role="user",
        text=original_query,
        tenant_id=tenant_id,
        kb_id=kb_id,
        user_id=state.get("user_id", "anonymous"),
    )

    add_done_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点结束执行，全局状态为：{sanitize_for_log(state)}")
    return state
