"""查询 LangGraph 工作流定义。"""

from langgraph.graph import StateGraph, END

from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_clarification_decision import node_clarification_decision
from app.query_process.agent.nodes.node_entity_resolution import node_entity_resolution
from app.query_process.agent.nodes.node_evidence_gate import node_evidence_gate
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_query_understanding import node_query_understanding
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rerank_gate import node_rerank_gate
from app.query_process.agent.nodes.node_retrieval_gate import node_retrieval_gate
from app.query_process.agent.nodes.node_retrieval_planner import node_retrieval_planner
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.conf.query_optimization_config import query_optimization_config
from app.core.observability import compile_graph, observe_node
from app.query_process.agent.state import QueryGraphState

builder = StateGraph(QueryGraphState)

# 节点添加完毕！！
# 在注册节点时统一包裹 observe_node，避免在每个业务节点中重复写 span/metric 逻辑。
builder.add_node("node_query_understanding", observe_node("query", "node_query_understanding", node_query_understanding))
builder.add_node("node_entity_resolution", observe_node("query", "node_entity_resolution", node_entity_resolution))
builder.add_node("node_clarification_decision", observe_node("query", "node_clarification_decision", node_clarification_decision))
builder.add_node("node_item_name_confirm", observe_node("query", "node_item_name_confirm", node_item_name_confirm))
builder.add_node("node_search_embedding", observe_node("query", "node_search_embedding", node_search_embedding))
builder.add_node("node_search_embedding_hyde", observe_node("query", "node_search_embedding_hyde", node_search_embedding_hyde))
builder.add_node("node_web_search_mcp", observe_node("query", "node_web_search_mcp", node_web_search_mcp))
builder.add_node("node_retrieval_gate", observe_node("query", "node_retrieval_gate", node_retrieval_gate))
builder.add_node("node_retrieval_planner", observe_node("query", "node_retrieval_planner", node_retrieval_planner))
builder.add_node("node_rrf", observe_node("query", "node_rrf", node_rrf))
builder.add_node("node_rerank_gate", observe_node("query", "node_rerank_gate", node_rerank_gate))
builder.add_node("node_rerank", observe_node("query", "node_rerank", node_rerank))
builder.add_node("node_evidence_gate", observe_node("query", "node_evidence_gate", node_evidence_gate))
builder.add_node("node_answer_output", observe_node("query", "node_answer_output", node_answer_output))

# 添加边
entry_node = "node_query_understanding" if query_optimization_config.use_standard_understanding else "node_item_name_confirm"
builder.set_entry_point(entry_node)


# node_item_name_confirm 可能出现，没有明确的主体 item_name 我们会提前结束返回用户提示，让他明确内容！！
# node_item_name_confirm -> (answer: str  # 最终生成的答案) -》答案生成 给前端反馈  || 多路召回
# 条件边！！！ conditional_edges

def route_after_pre_retrieval(state: QueryGraphState):
    """根据前置理解结果决定是否进入检索计划节点。

    Args:
        state: 查询 LangGraph State。

    Returns:
        下一个节点名称，或旧链路并行执行的节点名称元组。
    """
    if state['answer']:
        return "node_answer_output"
    if not query_optimization_config.use_retrieval_planner:
        # 灰度开关关闭时，保持旧的固定三路并行行为，不改变当前线上链路。
        return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"
    return "node_retrieval_planner"


def route_after_retrieval_planner(state: QueryGraphState):
    """根据检索计划选择召回节点。

    Args:
        state: 查询 LangGraph State。

    Returns:
        下一个节点名称，或并行执行的节点名称元组。
    """

    retrieval_plan = state.get("retrieval_plan") or {}
    if retrieval_plan.get("use_local_vector", True):
        return "node_search_embedding"

    next_nodes = []
    if retrieval_plan.get("use_hyde"):
        next_nodes.append("node_search_embedding_hyde")
    if retrieval_plan.get("use_web"):
        next_nodes.append("node_web_search_mcp")

    if not next_nodes:
        return "node_answer_output"
    if len(next_nodes) == 1:
        return next_nodes[0]
    return tuple(next_nodes)


def route_after_search_embedding(state: QueryGraphState):
    """根据灰度开关决定本地召回后是否进入二次门控。

    Args:
        state: 查询 LangGraph State。

    Returns:
        下一个节点名称。
    """
    if not query_optimization_config.use_retrieval_planner:
        return "node_rrf"
    return "node_retrieval_gate"


def route_after_retrieval_gate(state: QueryGraphState):
    """根据二次门控后的有效计划选择补充召回节点。

    Args:
        state: 查询 LangGraph State。

    Returns:
        下一个节点名称，或并行执行的节点名称元组。
    """
    retrieval_plan = state.get("retrieval_plan") or {}
    next_nodes = []
    if retrieval_plan.get("use_hyde"):
        next_nodes.append("node_search_embedding_hyde")
    if retrieval_plan.get("use_web"):
        next_nodes.append("node_web_search_mcp")

    if not next_nodes:
        return "node_rrf"
    if len(next_nodes) == 1:
        return next_nodes[0]
    return tuple(next_nodes)


def route_after_rerank(state: QueryGraphState):
    """根据证据门控灰度开关决定是否进入 Evidence Gate。

    Args:
        state: 查询 LangGraph State。

    Returns:
        下一个节点名称。
    """
    if query_optimization_config.enable_evidence_gate:
        return "node_evidence_gate"
    return "node_answer_output"


def route_after_rrf(state: QueryGraphState):
    """根据 Rerank Gate 灰度开关决定是否进入 Rerank 跳过门控。

    Args:
        state: 查询 LangGraph State。

    Returns:
        下一个节点名称。
    """
    if query_optimization_config.enable_rerank_gate:
        return "node_rerank_gate"
    return "node_rerank"


def route_after_rerank_gate(state: QueryGraphState):
    """根据 Rerank Gate 判断进入 Rerank 或后续生成前门控。

    Args:
        state: 查询 LangGraph State。

    Returns:
        下一个节点名称。
    """
    rerank_gate = state.get("rerank_gate") or {}
    if rerank_gate.get("should_rerank", True):
        return "node_rerank"
    return route_after_rerank(state)


builder.add_conditional_edges("node_item_name_confirm"
                              , route_after_pre_retrieval,
                              {
                                  "node_answer_output": "node_answer_output",
                                  "node_retrieval_planner": "node_retrieval_planner",
                                  "node_search_embedding": "node_search_embedding",
                                  "node_search_embedding_hyde": "node_search_embedding_hyde",
                                  "node_web_search_mcp": "node_web_search_mcp"
                              })

builder.add_edge("node_query_understanding", "node_entity_resolution")
builder.add_edge("node_entity_resolution", "node_clarification_decision")
builder.add_conditional_edges("node_clarification_decision",
                              route_after_pre_retrieval,
                              {
                                  "node_answer_output": "node_answer_output",
                                  "node_retrieval_planner": "node_retrieval_planner",
                                  "node_search_embedding": "node_search_embedding",
                                  "node_search_embedding_hyde": "node_search_embedding_hyde",
                                  "node_web_search_mcp": "node_web_search_mcp"
                              })
builder.add_conditional_edges("node_retrieval_planner",
                              route_after_retrieval_planner,
                              {
                                  "node_answer_output": "node_answer_output",
                                  "node_search_embedding": "node_search_embedding",
                                  "node_search_embedding_hyde": "node_search_embedding_hyde",
                                  "node_web_search_mcp": "node_web_search_mcp"
                              })
builder.add_conditional_edges("node_search_embedding",
                              route_after_search_embedding,
                              {
                                  "node_retrieval_gate": "node_retrieval_gate",
                                  "node_rrf": "node_rrf"
                              })
builder.add_conditional_edges("node_retrieval_gate",
                              route_after_retrieval_gate,
                              {
                                  "node_rrf": "node_rrf",
                                  "node_search_embedding_hyde": "node_search_embedding_hyde",
                                  "node_web_search_mcp": "node_web_search_mcp"
                              })

builder.add_edge("node_search_embedding_hyde", "node_rrf")
builder.add_edge("node_web_search_mcp", "node_rrf")
builder.add_conditional_edges("node_rrf",
                              route_after_rrf,
                              {
                                  "node_rerank_gate": "node_rerank_gate",
                                  "node_rerank": "node_rerank",
                              })
builder.add_conditional_edges("node_rerank_gate",
                              route_after_rerank_gate,
                              {
                                  "node_rerank": "node_rerank",
                                  "node_evidence_gate": "node_evidence_gate",
                                  "node_answer_output": "node_answer_output",
                              })
builder.add_conditional_edges("node_rerank",
                              route_after_rerank,
                              {
                                  "node_evidence_gate": "node_evidence_gate",
                                  "node_answer_output": "node_answer_output",
                              })
builder.add_edge("node_evidence_gate", "node_answer_output")
builder.add_edge("node_answer_output", END)

query_app = compile_graph("query", builder)
