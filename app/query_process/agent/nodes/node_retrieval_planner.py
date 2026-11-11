"""独立检索计划节点。"""

import sys

from app.conf.query_optimization_config import query_optimization_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.observability.events import emit_event
from app.observability.sanitizers import sanitize_for_log
from app.query_process.agent.nodes.query_understanding_utils import build_query_intelligence
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


def node_retrieval_planner(state: QueryGraphState):
    """生成查询检索计划。

    Args:
        state: 查询流程状态，包含原始问题、改写问题、实体和租户范围信息。

    Returns:
        写入 query_type、retrieval_plan、retrieval_diagnostics 和默认空召回列表后的查询状态。

    Raises:
        FileNotFoundError: 检索计划提示词文件不存在时抛出。
    """
    func_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    tenant_id = state.get("tenant_id", "default")
    kb_id = state.get("kb_id", "default")
    original_query = state.get("original_query", "")
    rewritten_query = state.get("rewritten_query") or original_query
    item_names = state.get("item_names", [])

    add_running_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点开始执行，全局状态为：{sanitize_for_log(state)}")

    # 当前版本用规则生成计划，并加载 prompt 作为后续 LLM Planner 的可回滚契约。
    planner_prompt = load_prompt(
        "retrieval_planner",
        original_query=original_query,
        rewritten_query=rewritten_query,
        query_type=state.get("query_type", "unknown"),
        intent=state.get("intent", ""),
        item_names=", ".join(item_names),
    )
    logger.debug(f"Retrieval Planner prompt loaded, chars={len(planner_prompt)}")

    query_intelligence = build_query_intelligence(
        original_query=original_query,
        rewritten_query=rewritten_query,
        explicit_query_type=state.get("query_type", ""),
        item_names=item_names,
        tenant_id=tenant_id,
        kb_id=kb_id,
        web_search_default=query_optimization_config.web_search_default,
    )
    state["query_type"] = query_intelligence["query_type"]
    state["retrieval_plan"] = query_intelligence["retrieval_plan"]

    # 条件路由只会执行被选中的检索节点；未选中的召回源在 Planner 处统一补空，避免后续汇聚 KeyError。
    state["embedding_chunks"] = []
    state["hyde_embedding_chunks"] = []
    state["web_search_docs"] = []

    retrieval_diagnostics = dict(state.get("retrieval_diagnostics") or {})
    retrieval_diagnostics.update(query_intelligence["retrieval_diagnostics"])
    retrieval_diagnostics["planner_node"] = func_name
    state["retrieval_diagnostics"] = retrieval_diagnostics

    emit_event(
        "query.retrieval_plan.generated",
        graph="query",
        node_name=func_name,
        session_id=session_id,
        tenant_id=tenant_id,
        kb_id=kb_id,
        status="ok",
        summary={
            "query_type": state["query_type"],
            "retrieval_plan": state["retrieval_plan"],
            "reason": state["retrieval_plan"].get("reason", ""),
        },
        tags=["query", func_name, "retrieval_plan"],
    )

    add_done_task(session_id, func_name, is_stream)
    logger.info(
        "检索计划生成完成："
        f"{sanitize_for_log({'query_type': state['query_type'], 'retrieval_plan': state['retrieval_plan']})}"
    )
    return state
