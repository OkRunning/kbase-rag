"""标准追问决策节点。"""

import sys

from app.core.logger import logger
from app.observability.sanitizers import sanitize_for_log
from app.query_process.agent.nodes.query_understanding_utils import build_clarification_question
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task


def node_clarification_decision(state: QueryGraphState):
    """决定当前问题是否需要追问用户。

    Args:
        state: 查询流程状态，包含实体解析结果和追问提示。

    Returns:
        需要追问时写入 answer 的查询状态；否则原样返回。
    """
    func_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    add_running_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点开始执行，全局状态为：{sanitize_for_log(state)}")

    # 只有在 query_understanding 明确判断需要商品名时，才拦截后续检索流程。
    # 这样泛知识库问题即使没有 item_names，也会自然进入普通检索路径。
    need_item_name = bool(state.get("need_item_name", False))
    entity_resolution = state.get("entity_resolution") or {}
    confirmed_item_names = entity_resolution.get("confirmed_item_names") or []
    options_item_names = entity_resolution.get("options_item_names") or []
    clarification_hint = str(state.get("clarification_hint") or "")

    if need_item_name and not confirmed_item_names:
        clarification_text = build_clarification_question(options_item_names, clarification_hint)
        state["answer"] = clarification_text
        logger.info(f"【{func_name}】需要追问，返回澄清问题：{clarification_text}")
    else:
        logger.info(f"【{func_name}】无需追问，继续后续检索流程。")

    add_done_task(session_id, func_name, is_stream)
    return state
