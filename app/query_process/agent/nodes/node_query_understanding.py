"""标准查询理解入口节点。"""

import sys

from app.clients.mongo_history_utils import get_recent_messages
from app.core.logger import logger
from app.observability.sanitizers import sanitize_for_log
from app.query_process.agent.nodes.query_understanding_utils import (
    extract_entity_payload,
    normalize_query_type,
    step_1_query_understanding,
)
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task


def node_query_understanding(state: QueryGraphState):
    """执行标准查询理解。

    Args:
        state: 查询流程状态，包含原始问题和会话范围信息。

    Returns:
        补充历史、意图、实体、query_type 和改写问题后的查询状态。

    Raises:
        Exception: 大模型调用失败或结果解析失败时抛出。
    """
    func_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    tenant_id = state.get("tenant_id", "default")
    kb_id = state.get("kb_id", "default")
    original_query = state["original_query"]

    add_running_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点开始执行，全局状态为：{sanitize_for_log(state)}")

    # 1. 读取历史消息，保证改写和指代消解能拿到最近的上下文。
    history = get_recent_messages(session_id, tenant_id=tenant_id, kb_id=kb_id)
    state["history"] = history

    # 2. 使用统一的查询理解 Prompt 识别意图、实体和改写问题。
    model_result = step_1_query_understanding(history, original_query)
    entities = extract_entity_payload(model_result)
    rewritten_query = model_result.get("rewritten_query", original_query)
    state["intent"] = str(model_result.get("intent", "") or "")
    state["query_type"] = normalize_query_type(model_result.get("query_type", ""))
    state["rewritten_query"] = rewritten_query
    state["need_item_name"] = bool(model_result.get("need_item_name", False))
    state["clarification_hint"] = str(model_result.get("clarification_hint", "") or "")
    state["entities"] = entities
    state["item_names"] = entities.get("item_names", [])

    logger.info(
        "标准查询理解完成："
        f"{sanitize_for_log({'query_type': state['query_type'], 'intent': state['intent'], 'need_item_name': state['need_item_name']})}"
    )
    add_done_task(session_id, func_name, is_stream)
    return state
