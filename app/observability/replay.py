"""提供基于观测事件的节点级和流程级回放能力。"""

from __future__ import annotations

import time
import uuid
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from langgraph.graph import END, StateGraph
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.state import ImportGraphState
from app.observability.checkpoints import (
    build_langgraph_config,
    build_langgraph_thread_id,
    get_langgraph_checkpointer,
)
from app.observability.context import reset_context, set_context
from app.observability.diff import diff_snapshots
from app.observability.events import (
    emit_event,
    get_event_by_id,
    get_timeline_by_session,
    get_timeline_by_task,
    get_timeline_by_trace,
)
from app.observability.metrics import record_replay
from app.observability.replay_context import reset_replay_context, set_replay_context
from app.observability.snapshots import load_snapshot, save_snapshot
from app.conf.query_optimization_config import query_optimization_config
from app.query_process.agent.state import QueryGraphState
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

ReplayMode = Literal["timeline_only", "mock_external", "live_external", "from_node"]

_tracer = trace.get_tracer("kbase-rag.replay")
_NODE_REGISTRY: Dict[str, Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = {
    "query": {
        "node_query_understanding": node_query_understanding,
        "node_entity_resolution": node_entity_resolution,
        "node_clarification_decision": node_clarification_decision,
        "node_item_name_confirm": node_item_name_confirm,
        "node_retrieval_planner": node_retrieval_planner,
        "node_search_embedding": node_search_embedding,
        "node_search_embedding_hyde": node_search_embedding_hyde,
        "node_web_search_mcp": node_web_search_mcp,
        "node_retrieval_gate": node_retrieval_gate,
        "node_rrf": node_rrf,
        "node_rerank_gate": node_rerank_gate,
        "node_rerank": node_rerank,
        "node_evidence_gate": node_evidence_gate,
        "node_answer_output": node_answer_output,
    },
    "import": {
        "node_entry": node_entry,
        "node_pdf_to_md": node_pdf_to_md,
        "node_md_img": node_md_img,
        "node_document_split": node_document_split,
        "node_item_name_recognition": node_item_name_recognition,
        "node_bge_embedding": node_bge_embedding,
        "node_import_milvus": node_import_milvus,
    },
}
_IMPORT_NEXT_NODES = {
    "node_pdf_to_md": ["node_md_img"],
    "node_md_img": ["node_document_split"],
    "node_document_split": ["node_item_name_recognition"],
    "node_item_name_recognition": ["node_bge_embedding"],
    "node_bge_embedding": ["node_import_milvus"],
    "node_import_milvus": [],
}


def _state_schema_for_graph(graph: str):
    """获取回放子图使用的 State schema。

    Args:
        graph: 工作流名称。

    Returns:
        LangGraph State schema。

    Raises:
        ValueError: 不支持的图名称。
    """
    if graph == "query":
        return QueryGraphState
    if graph == "import":
        return ImportGraphState
    raise ValueError(f"unsupported graph for checkpoint replay: {graph}")


def _current_trace_id() -> str:
    """读取当前 OpenTelemetry trace ID。

    Returns:
        当前 trace ID；没有有效 span 时返回空字符串。
    """
    span_context = trace.get_current_span().get_span_context()
    return f"{span_context.trace_id:032x}" if span_context and span_context.is_valid else ""


def _find_completed_event(started_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """查找 `node.started` 对应的历史完成事件。

    Args:
        started_event: 原始 `node.started` 事件。

    Returns:
        匹配到的 `node.completed` 事件；不存在时返回 None。
    """
    if started_event.get("session_id"):
        timeline = get_timeline_by_session(started_event["session_id"], limit=1000)
    elif started_event.get("task_id"):
        timeline = get_timeline_by_task(started_event["task_id"], limit=1000)
    else:
        timeline = get_timeline_by_trace(started_event.get("trace_id", ""), limit=1000)
    sequence = int(started_event.get("sequence") or 0)
    for item in timeline:
        if int(item.get("sequence") or 0) <= sequence:
            continue
        if item.get("graph") != started_event.get("graph"):
            continue
        if item.get("node_name") != started_event.get("node_name"):
            continue
        if item.get("event_type") == "node.completed" and item.get("output_ref"):
            return item
    return None


def _load_timeline_for_event(event: Dict[str, Any], limit: int = 1000) -> List[Dict[str, Any]]:
    """读取原始事件所属流程的时间线。

    Args:
        event: 原始事件。
        limit: 返回事件条数上限。

    Returns:
        事件时间线列表。
    """
    if event.get("session_id"):
        return get_timeline_by_session(event["session_id"], limit=limit)
    if event.get("task_id"):
        return get_timeline_by_task(event["task_id"], limit=limit)
    return get_timeline_by_trace(event.get("trace_id", ""), limit=limit)


def _find_started_event_for_failed(
    timeline: Sequence[Dict[str, Any]],
    failed_event: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """查找失败事件对应的 `node.started` 事件。

    Args:
        timeline: 原始流程时间线。
        failed_event: `node.failed` 事件。

    Returns:
        匹配到的 `node.started` 事件；不存在时返回 None。
    """
    failed_sequence = int(failed_event.get("sequence") or 0)
    candidates = [
        item
        for item in timeline
        if int(item.get("sequence") or 0) <= failed_sequence
        and item.get("event_type") == "node.started"
        and item.get("graph") == failed_event.get("graph")
        and item.get("node_name") == failed_event.get("node_name")
        and item.get("input_ref")
    ]
    return candidates[-1] if candidates else None


def _find_latest_failed_started_event(timeline: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """查找最近失败节点对应的 started 事件。

    Args:
        timeline: 原始流程时间线。

    Returns:
        最近失败节点的 `node.started` 事件；不存在时返回 None。
    """
    failed_events = [item for item in timeline if item.get("event_type") == "node.failed"]
    for failed_event in reversed(failed_events):
        started_event = _find_started_event_for_failed(timeline, failed_event)
        if started_event:
            return started_event
    return None


def _find_historical_output_for_node(
    timeline: Sequence[Dict[str, Any]],
    original_event: Dict[str, Any],
    node_name: str,
) -> Optional[Dict[str, Any]]:
    """查找指定节点在原始时间线中的历史输出事件。

    Args:
        timeline: 原始流程时间线。
        original_event: 回放起点事件。
        node_name: 需要查找输出的节点名称。

    Returns:
        匹配到的 `node.completed` 事件；不存在时返回 None。
    """
    sequence = int(original_event.get("sequence") or 0)
    for item in timeline:
        if int(item.get("sequence") or 0) < sequence:
            continue
        if item.get("graph") != original_event.get("graph"):
            continue
        if item.get("node_name") != node_name:
            continue
        if item.get("event_type") == "node.completed" and item.get("output_ref"):
            return item
    return None


def _find_failed_event_for_started(
    timeline: Sequence[Dict[str, Any]],
    started_event: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """查找 `node.started` 对应的历史失败事件。

    Args:
        timeline: 原始流程时间线。
        started_event: 原始 `node.started` 事件。

    Returns:
        匹配到的 `node.failed` 事件；不存在时返回 None。
    """
    sequence = int(started_event.get("sequence") or 0)
    for item in timeline:
        if int(item.get("sequence") or 0) <= sequence:
            continue
        if item.get("graph") != started_event.get("graph"):
            continue
        if item.get("node_name") != started_event.get("node_name"):
            continue
        if item.get("event_type") == "node.failed":
            return item
    return None


def _safe_diff_snapshots(original_ref: str, replay_ref: str) -> Dict[str, Any]:
    """安全比较两个快照，失败时返回错误摘要。

    Args:
        original_ref: 原始快照引用。
        replay_ref: 回放快照引用。

    Returns:
        快照差异或错误摘要。
    """
    try:
        return diff_snapshots(original_ref, replay_ref)
    except Exception as exc:
        return {"error": {"type": type(exc).__name__, "message": str(exc)}}


def _validate_live_replay_allowed(mode: ReplayMode) -> None:
    """校验当前环境是否允许真实执行回放。

    Args:
        mode: 回放模式。

    Raises:
        ValueError: 环境禁止真实执行回放时抛出。
    """
    if mode not in {"from_node", "live_external"}:
        return
    allowed = os.getenv("OBSERVABILITY_REPLAY_ALLOW_LIVE", "true").strip().lower() in {"1", "true", "yes"}
    if not allowed:
        raise ValueError("live replay is disabled by OBSERVABILITY_REPLAY_ALLOW_LIVE")


def _merge_state(state: Dict[str, Any], result: Any) -> Dict[str, Any]:
    """合并节点输出到回放状态。

    Args:
        state: 当前回放状态。
        result: 节点执行结果。

    Returns:
        合并后的新状态。
    """
    merged = dict(state)
    if isinstance(result, dict):
        merged.update(result)
    return merged


def _query_next_nodes(node_name: str, state: Dict[str, Any]) -> List[str]:
    """计算查询图当前节点之后需要回放的节点链。

    路由规则与 ``app/query_process/agent/main_graph.py`` 的条件边保持一致：
    并行分支按固定顺序拍平为顺序执行，运行期产生的 ``retrieval_plan`` 与
    ``rerank_gate`` 优先于灰度开关默认值。

    Args:
        node_name: 当前节点名称。
        state: 当前回放状态。

    Returns:
        当前节点之后的完整续跑节点链；无后续节点时返回空列表。
    """

    def _after_rerank() -> List[str]:
        if query_optimization_config.enable_evidence_gate:
            return ["node_evidence_gate", "node_answer_output"]
        return ["node_answer_output"]

    def _after_rerank_gate() -> List[str]:
        rerank_gate = state.get("rerank_gate") or {}
        if rerank_gate.get("should_rerank", True):
            return ["node_rerank", *_after_rerank()]
        return _after_rerank()

    def _after_rrf() -> List[str]:
        if query_optimization_config.enable_rerank_gate:
            return ["node_rerank_gate", *_after_rerank_gate()]
        return ["node_rerank", *_after_rerank()]

    def _after_rrf_join() -> List[str]:
        return ["node_rrf", *_after_rrf()]

    def _after_retrieval_gate() -> List[str]:
        retrieval_plan = state.get("retrieval_plan") or {}
        nodes = []
        if retrieval_plan.get("use_hyde"):
            nodes.append("node_search_embedding_hyde")
        if retrieval_plan.get("use_web"):
            nodes.append("node_web_search_mcp")
        nodes.extend(_after_rrf_join())
        return nodes

    def _after_search_embedding() -> List[str]:
        if not query_optimization_config.use_retrieval_planner:
            return _after_rrf_join()
        return ["node_retrieval_gate", *_after_retrieval_gate()]

    def _after_retrieval_planner() -> List[str]:
        retrieval_plan = state.get("retrieval_plan") or {}
        if retrieval_plan.get("use_local_vector", True):
            return ["node_search_embedding", *_after_search_embedding()]
        nodes = []
        if retrieval_plan.get("use_hyde"):
            nodes.append("node_search_embedding_hyde")
        if retrieval_plan.get("use_web"):
            nodes.append("node_web_search_mcp")
        if not nodes:
            return ["node_answer_output"]
        nodes.extend(_after_rrf_join())
        return nodes

    def _after_pre_retrieval() -> List[str]:
        if state.get("answer"):
            return ["node_answer_output"]
        if not query_optimization_config.use_retrieval_planner:
            return [
                "node_search_embedding",
                "node_search_embedding_hyde",
                "node_web_search_mcp",
                *_after_rrf_join(),
            ]
        return ["node_retrieval_planner", *_after_retrieval_planner()]

    if node_name in {"node_item_name_confirm", "node_clarification_decision"}:
        return _after_pre_retrieval()
    if node_name == "node_query_understanding":
        return ["node_entity_resolution", "node_clarification_decision", *_after_pre_retrieval()]
    if node_name == "node_entity_resolution":
        return ["node_clarification_decision", *_after_pre_retrieval()]
    if node_name == "node_retrieval_planner":
        return _after_retrieval_planner()
    if node_name == "node_search_embedding":
        return _after_search_embedding()
    if node_name == "node_retrieval_gate":
        return _after_retrieval_gate()
    if node_name in {"node_search_embedding_hyde", "node_web_search_mcp"}:
        return _after_rrf_join()
    if node_name == "node_rrf":
        return _after_rrf()
    if node_name == "node_rerank_gate":
        return _after_rerank_gate()
    if node_name == "node_rerank":
        return _after_rerank()
    if node_name == "node_evidence_gate":
        return ["node_answer_output"]
    return []


def _workflow_plan(graph: str, start_node: str, state: Dict[str, Any]) -> List[str]:
    """生成从指定节点到流程结束的回放计划。

    Args:
        graph: 工作流名称。
        start_node: 起始节点。
        state: 起始状态。

    Returns:
        需要执行的节点列表。

    Raises:
        ValueError: 不支持的图或节点时抛出。
    """
    if start_node not in _NODE_REGISTRY.get(graph, {}):
        raise ValueError(f"node is not registered for workflow replay: {graph}.{start_node}")
    if graph == "import":
        if start_node == "node_entry":
            if state.get("is_md_read_enabled"):
                nodes = ["node_entry", "node_md_img"]
            elif state.get("is_pdf_read_enabled"):
                nodes = ["node_entry", "node_pdf_to_md"]
            else:
                return ["node_entry"]
            current = nodes[-1]
        else:
            nodes = [start_node]
            current = start_node
        while _IMPORT_NEXT_NODES.get(current):
            current = _IMPORT_NEXT_NODES[current][0]
            nodes.append(current)
        return nodes
    if graph == "query":
        return [start_node] + _query_next_nodes(start_node, state)
    raise ValueError(f"unsupported graph for workflow replay: {graph}")


def _invoke_node_with_langgraph_checkpoint(
    graph: str,
    node_name: str,
    node_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    state: Dict[str, Any],
    replay_id: str,
    step_index: int,
) -> Dict[str, Any]:
    """通过临时 LangGraph 子图执行单个节点并写入 checkpoint。

    Args:
        graph: 工作流名称。
        node_name: 当前回放节点名称。
        node_fn: 节点函数。
        state: 当前回放状态。
        replay_id: 本次全流程回放 ID。
        step_index: 节点在本次回放中的序号。

    Returns:
        节点执行后的状态。
    """
    replay_graph = StateGraph(_state_schema_for_graph(graph))
    replay_graph.add_node(node_name, node_fn)
    replay_graph.set_entry_point(node_name)
    replay_graph.add_edge(node_name, END)
    app = replay_graph.compile(checkpointer=get_langgraph_checkpointer())
    checkpoint_workflow_id = f"{replay_id}:{step_index}:{node_name}"
    config = build_langgraph_config(f"replay.{graph}", checkpoint_workflow_id)
    # 用临时单节点图包住真实节点执行，让 LangGraph 自身生成可查询的 checkpoint。
    return app.invoke(state, config=config)


def _replay_from_historical_output(
    event: Dict[str, Any],
    input_ref: str,
    replay_event_id: str,
) -> Dict[str, Any]:
    """使用历史输出快照模拟节点回放。

    Args:
        event: 原始 `node.started` 事件。
        input_ref: 原始输入快照引用。
        replay_event_id: 本次回放开始事件 ID。

    Returns:
        mock 回放响应。

    Raises:
        ValueError: 找不到历史输出快照时抛出。
    """
    completed_event = _find_completed_event(event)
    if completed_event is None:
        raise ValueError("mock_external requires a historical node.completed output snapshot")
    output_snapshot = load_snapshot(completed_event["output_ref"])
    output_ref = save_snapshot(
        event.get("graph", ""),
        event.get("node_name", ""),
        "replay_output",
        output_snapshot.get("payload") or {},
        source_event_id=replay_event_id,
    )
    replay_completed = emit_event(
        "replay.completed",
        graph=event.get("graph", ""),
        node_name=event.get("node_name", ""),
        session_id=event.get("session_id", ""),
        task_id=event.get("task_id", ""),
        status="ok",
        summary={
            "mode": "mock_external",
            "original_event_id": event.get("_id", ""),
            "original_trace_id": event.get("trace_id", ""),
            "mock_source_event_id": completed_event.get("_id", ""),
            "replay_trace_id": _current_trace_id(),
        },
        input_ref=input_ref,
        output_ref=output_ref,
        tags=[event.get("graph", ""), event.get("node_name", ""), "replay.completed"],
    )
    return {
        "mode": "mock_external",
        "executed": False,
        "original_event": event,
        "mock_source_event": completed_event,
        "replay_event": replay_completed,
        "input_ref": input_ref,
        "output_ref": output_ref,
        "result": output_snapshot.get("payload") or {},
    }


def replay_node(
    event_id: str,
    mode: ReplayMode = "from_node",
    override_state: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """从节点输入快照回放单个节点。

    Args:
        event_id: `node.started` 事件 ID。
        mode: 回放模式；`timeline_only` 只返回快照，不执行节点。
        override_state: 覆盖到快照输入上的状态字段。
        dry_run: 是否启用副作用隔离。

    Returns:
        回放结果、原始事件和新快照引用。

    Raises:
        ValueError: 事件、快照或节点不可回放时抛出。
        Exception: 节点回放过程中原始业务异常会继续向上抛出。
    """
    event = get_event_by_id(event_id)
    if event is None:
        raise ValueError("event not found")
    graph = event.get("graph") or ""
    node_name = event.get("node_name") or ""
    input_ref = event.get("input_ref")
    if event.get("event_type") != "node.started":
        raise ValueError("only node.started events can be replayed")
    if not input_ref:
        raise ValueError("event has no input snapshot")

    snapshot = load_snapshot(input_ref)
    state = dict(snapshot.get("payload") or {})
    if override_state:
        # 回放时只在快照状态上做显式覆盖，避免修改原始快照事实。
        state.update(override_state)

    if mode == "timeline_only":
        record_replay(graph, mode, "ok")
        return {
            "mode": mode,
            "event": event,
            "input_snapshot": snapshot,
            "executed": False,
        }

    if mode not in {"mock_external", "live_external", "from_node"}:
        raise ValueError(f"unsupported replay mode: {mode}")
    _validate_live_replay_allowed(mode)
    node_fn = _NODE_REGISTRY.get(graph, {}).get(node_name)
    if node_fn is None:
        raise ValueError(f"node is not registered for replay: {graph}.{node_name}")

    tokens = set_context(
        graph=graph,
        node_name=node_name,
        session_id=state.get("session_id") or event.get("session_id") or "",
        task_id=state.get("task_id") or event.get("task_id") or "",
    )
    replay_id = str(uuid.uuid4())
    replay_tokens = set_replay_context(replay_id=replay_id, dry_run=dry_run or mode == "mock_external")
    start = time.perf_counter()
    replay_event = emit_event(
        "replay.started",
        graph=graph,
        node_name=node_name,
        session_id=state.get("session_id") or event.get("session_id") or "",
        task_id=state.get("task_id") or event.get("task_id") or "",
        status="ok",
        summary={
            "mode": mode,
            "replay_id": replay_id,
            "dry_run": dry_run or mode == "mock_external",
            "original_event_id": event_id,
            "original_trace_id": event.get("trace_id", ""),
            "input_ref": input_ref,
        },
        input_ref=input_ref,
        tags=[graph, node_name, "replay.started"],
    )
    try:
        if mode == "mock_external":
            # mock 模式复用历史输出快照，不重新触发 LLM/Web/Milvus 等外部依赖。
            with _tracer.start_as_current_span(f"replay.{graph}.{node_name}") as span:
                span.set_attribute("rag.replay.mode", mode)
                span.set_attribute("rag.replay.original_event_id", event_id)
                span.set_attribute("rag.replay.original_trace_id", event.get("trace_id", ""))
                span.set_attribute("rag.replay.mock_external", True)
                result = _replay_from_historical_output(event, input_ref, replay_event.get("_id", ""))
                record_replay(graph, mode, "ok")
                return result
        with _tracer.start_as_current_span(f"replay.{graph}.{node_name}") as span:
            span.set_attribute("rag.replay.mode", mode)
            span.set_attribute("rag.replay.original_event_id", event_id)
            span.set_attribute("rag.replay.original_trace_id", event.get("trace_id", ""))
            result = node_fn(state)
            output_ref = save_snapshot(
                graph,
                node_name,
                "replay_output",
                result if isinstance(result, dict) else {"result": result},
                source_event_id=replay_event.get("_id"),
            )
            completed_event = _find_completed_event(event)
            diff_result = (
                _safe_diff_snapshots(completed_event["output_ref"], output_ref)
                if completed_event and completed_event.get("output_ref")
                else {}
            )
            completed = emit_event(
                "replay.completed",
                graph=graph,
                node_name=node_name,
                session_id=state.get("session_id") or event.get("session_id") or "",
                task_id=state.get("task_id") or event.get("task_id") or "",
                status="ok",
                summary={
                    "mode": mode,
                    "replay_id": replay_id,
                    "dry_run": dry_run,
                    "original_event_id": event_id,
                    "original_trace_id": event.get("trace_id", ""),
                    "replay_trace_id": _current_trace_id(),
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    "diff_count": diff_result.get("diff_count"),
                    "result_keys": list(result.keys())[:50] if isinstance(result, dict) else [],
                },
                input_ref=input_ref,
                output_ref=output_ref,
                tags=[graph, node_name, "replay.completed"],
            )
            record_replay(graph, mode, "ok")
            return {
                "mode": mode,
                "executed": True,
                "original_event": event,
                "replay_event": completed,
                "input_ref": input_ref,
                "output_ref": output_ref,
                "diff": diff_result,
                "result": result,
            }
    except Exception as exc:
        current_span = trace.get_current_span()
        current_span.record_exception(exc)
        current_span.set_status(Status(StatusCode.ERROR, str(exc)))
        emit_event(
            "replay.failed",
            graph=graph,
            node_name=node_name,
            session_id=state.get("session_id") or event.get("session_id") or "",
            task_id=state.get("task_id") or event.get("task_id") or "",
            status="error",
            summary={
                "mode": mode,
                "replay_id": replay_id,
                "dry_run": dry_run,
                "original_event_id": event_id,
                "original_trace_id": event.get("trace_id", ""),
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            },
            input_ref=input_ref,
            error={"type": type(exc).__name__, "message": str(exc)},
            tags=[graph, node_name, "replay.failed"],
        )
        record_replay(graph, mode, "error")
        raise
    finally:
        reset_replay_context(replay_tokens)
        reset_context(tokens)


def replay_workflow(
    event_id: str,
    mode: ReplayMode = "mock_external",
    override_state: Optional[Dict[str, Any]] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """从指定节点输入快照开始回放到当前工作流结束。

    Args:
        event_id: `node.started` 事件 ID。
        mode: 回放模式；`mock_external` 复用历史节点输出，`from_node`/`live_external` 真实执行节点。
        override_state: 覆盖到起点快照输入上的状态字段。
        dry_run: 是否启用副作用隔离。

    Returns:
        全流程回放结果，包含每个节点的 checkpoint。mock 模式遇到起始失败节点没有历史输出时，
        返回跳过原因并停止后续回放。

    Raises:
        ValueError: 事件、快照或节点不可回放时抛出。
        Exception: 真实执行节点时原始业务异常会继续向上抛出。
    """
    event = get_event_by_id(event_id)
    if event is None:
        raise ValueError("event not found")
    if event.get("event_type") != "node.started":
        raise ValueError("only node.started events can start workflow replay")
    input_ref = event.get("input_ref")
    if not input_ref:
        raise ValueError("event has no input snapshot")

    graph = event.get("graph") or ""
    start_node = event.get("node_name") or ""
    snapshot = load_snapshot(input_ref)
    state = dict(snapshot.get("payload") or {})
    if override_state:
        # 回放入口允许临时覆盖 state，用于验证修复或构造边界场景。
        state.update(override_state)
    if mode == "timeline_only":
        record_replay(graph, mode, "ok")
        return {
            "mode": mode,
            "event": event,
            "input_snapshot": snapshot,
            "executed": False,
            "scope": "workflow",
        }
    if mode not in {"mock_external", "live_external", "from_node"}:
        raise ValueError(f"unsupported replay mode: {mode}")
    _validate_live_replay_allowed(mode)

    timeline = _load_timeline_for_event(event)
    planned_nodes = _workflow_plan(graph, start_node, state)
    replay_id = str(uuid.uuid4())
    tokens = set_context(
        graph=graph,
        node_name=start_node,
        session_id=state.get("session_id") or event.get("session_id") or "",
        task_id=state.get("task_id") or event.get("task_id") or "",
    )
    replay_tokens = set_replay_context(replay_id=replay_id, dry_run=dry_run or mode == "mock_external")
    start = time.perf_counter()
    workflow_event = emit_event(
        "replay.workflow.started",
        graph=graph,
        node_name=start_node,
        session_id=state.get("session_id") or event.get("session_id") or "",
        task_id=state.get("task_id") or event.get("task_id") or "",
        status="ok",
        summary={
            "mode": mode,
            "original_event_id": event_id,
            "original_trace_id": event.get("trace_id", ""),
            "replay_id": replay_id,
            "dry_run": dry_run or mode == "mock_external",
            "planned_nodes": planned_nodes,
            "side_effects_possible": mode in {"from_node", "live_external"},
        },
        input_ref=input_ref,
        tags=[graph, start_node, "replay.workflow.started"],
    )
    checkpoints: List[Dict[str, Any]] = []
    try:
        with _tracer.start_as_current_span(f"replay.workflow.{graph}.{start_node}") as span:
            span.set_attribute("rag.replay.scope", "workflow")
            span.set_attribute("rag.replay.mode", mode)
            span.set_attribute("rag.replay.original_event_id", event_id)
            span.set_attribute("rag.replay.original_trace_id", event.get("trace_id", ""))
            index = 0
            while index < len(planned_nodes):
                node_name = planned_nodes[index]
                node_fn = _NODE_REGISTRY.get(graph, {}).get(node_name)
                if node_fn is None:
                    raise ValueError(f"node is not registered for workflow replay: {graph}.{node_name}")
                node_tokens = set_context(node_name=node_name)
                try:
                    node_input_ref = save_snapshot(
                        graph,
                        node_name,
                        "replay_checkpoint_input",
                        state,
                        source_event_id=workflow_event.get("_id"),
                    )
                    node_start = time.perf_counter()
                    skipped = False
                    skip_reason = ""
                    if mode == "mock_external":
                        completed_event = _find_historical_output_for_node(timeline, event, node_name)
                        if completed_event is None:
                            if node_name != start_node:
                                raise ValueError(f"mock_external requires historical output for node: {node_name}")
                            # 失败节点不会产生 completed/output_ref；mock_external 不能真实调用外部依赖，
                            # 因此保留输入状态并记录可诊断 checkpoint，避免接口直接 400。
                            failed_event = _find_failed_event_for_started(timeline, event)
                            result = {}
                            executed = False
                            skipped = True
                            mock_source_event_id = ""
                            skip_reason = f"mock_external has no historical output for failed start node: {node_name}"
                            failed_error = (failed_event.get("error") or {}) if failed_event else {}
                            if failed_error.get("message"):
                                skip_reason = f"{skip_reason}; original_error={failed_error['message']}"
                            planned_nodes[index + 1:] = []
                        else:
                            output_snapshot = load_snapshot(completed_event["output_ref"])
                            result = output_snapshot.get("payload") or {}
                            executed = False
                            mock_source_event_id = completed_event.get("_id", "")
                    else:
                        completed_event = _find_historical_output_for_node(timeline, event, node_name)
                        result = _invoke_node_with_langgraph_checkpoint(
                            graph,
                            node_name,
                            node_fn,
                            state,
                            replay_id,
                            index,
                        )
                        executed = True
                        mock_source_event_id = ""
                    state = _merge_state(state, result)
                    node_output_ref = save_snapshot(
                        graph,
                        node_name,
                        "replay_checkpoint_output",
                        state,
                        source_event_id=workflow_event.get("_id"),
                    )
                    diff_result = (
                        _safe_diff_snapshots(completed_event["output_ref"], node_output_ref)
                        if completed_event and completed_event.get("output_ref")
                        else {}
                    )
                    checkpoint_status = "skipped" if skipped else "ok"
                    checkpoint = emit_event(
                        "replay.checkpoint",
                        graph=graph,
                        node_name=node_name,
                        session_id=state.get("session_id") or event.get("session_id") or "",
                        task_id=state.get("task_id") or event.get("task_id") or "",
                        status=checkpoint_status,
                        summary={
                            "mode": mode,
                            "original_event_id": event_id,
                            "original_trace_id": event.get("trace_id", ""),
                            "node_name": node_name,
                            "replay_id": replay_id,
                            "dry_run": dry_run or mode == "mock_external",
                            "langgraph_checkpoint_thread": build_langgraph_thread_id(
                                f"replay.{graph}",
                                f"{replay_id}:{index}:{node_name}",
                            ),
                            "executed": executed,
                            "skipped": skipped,
                            "skip_reason": skip_reason,
                            "mock_source_event_id": mock_source_event_id,
                            "duration_ms": round((time.perf_counter() - node_start) * 1000, 2),
                            "diff_count": diff_result.get("diff_count"),
                            "state_keys": list(state.keys())[:50],
                        },
                        input_ref=node_input_ref,
                        output_ref=node_output_ref,
                        tags=[graph, node_name, "replay.checkpoint"],
                    )
                    checkpoint["diff"] = diff_result
                    checkpoints.append(checkpoint)
                finally:
                    reset_context(node_tokens)

                if graph == "query" and not skipped:
                    planned_nodes[index + 1:] = _query_next_nodes(node_name, state)
                index += 1
            final_ref = save_snapshot(
                graph,
                start_node,
                "replay_workflow_final",
                state,
                source_event_id=workflow_event.get("_id"),
            )
            completed = emit_event(
                "replay.workflow.completed",
                graph=graph,
                node_name=start_node,
                session_id=state.get("session_id") or event.get("session_id") or "",
                task_id=state.get("task_id") or event.get("task_id") or "",
                status="ok",
                summary={
                    "mode": mode,
                    "original_event_id": event_id,
                    "original_trace_id": event.get("trace_id", ""),
                    "replay_id": replay_id,
                    "dry_run": dry_run or mode == "mock_external",
                    "replay_trace_id": _current_trace_id(),
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    "executed_nodes": planned_nodes,
                    "checkpoint_count": len(checkpoints),
                    "side_effects_possible": mode in {"from_node", "live_external"},
                },
                input_ref=input_ref,
                output_ref=final_ref,
                tags=[graph, start_node, "replay.workflow.completed"],
            )
            record_replay(graph, mode, "ok")
            return {
                "mode": mode,
                "scope": "workflow",
                "executed": mode in {"from_node", "live_external"},
                "original_event": event,
                "replay_event": completed,
                "input_ref": input_ref,
                "output_ref": final_ref,
                "planned_nodes": planned_nodes,
                "replay_id": replay_id,
                "checkpoints": checkpoints,
                "side_effects_possible": mode in {"from_node", "live_external"},
                "dry_run": dry_run or mode == "mock_external",
            }
    except Exception as exc:
        current_span = trace.get_current_span()
        current_span.record_exception(exc)
        current_span.set_status(Status(StatusCode.ERROR, str(exc)))
        emit_event(
            "replay.workflow.failed",
            graph=graph,
            node_name=start_node,
            session_id=state.get("session_id") or event.get("session_id") or "",
            task_id=state.get("task_id") or event.get("task_id") or "",
            status="error",
            summary={
                "mode": mode,
                "original_event_id": event_id,
                "original_trace_id": event.get("trace_id", ""),
                "replay_id": replay_id,
                "dry_run": dry_run or mode == "mock_external",
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                "checkpoint_count": len(checkpoints),
            },
            input_ref=input_ref,
            error={"type": type(exc).__name__, "message": str(exc)},
            tags=[graph, start_node, "replay.workflow.failed"],
        )
        record_replay(graph, mode, "error")
        raise
    finally:
        reset_replay_context(replay_tokens)
        reset_context(tokens)


def replay_latest_failed_node(
    scope: Literal["session", "task", "trace"],
    scope_id: str,
    mode: ReplayMode = "mock_external",
    workflow: bool = True,
    override_state: Optional[Dict[str, Any]] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """一键回放指定范围内最近失败的节点。

    Args:
        scope: 查询范围，支持 session、task 或 trace。
        scope_id: 范围 ID。
        mode: 回放模式。
        workflow: 是否从失败节点继续回放到流程结束。
        override_state: 覆盖到起点快照输入上的状态字段。
        dry_run: 是否启用副作用隔离。

    Returns:
        回放结果。

    Raises:
        ValueError: 未找到失败节点或范围不支持时抛出。
    """
    if scope == "session":
        timeline = get_timeline_by_session(scope_id, limit=1000)
    elif scope == "task":
        timeline = get_timeline_by_task(scope_id, limit=1000)
    elif scope == "trace":
        timeline = get_timeline_by_trace(scope_id, limit=1000)
    else:
        raise ValueError(f"unsupported failed replay scope: {scope}")

    started_event = _find_latest_failed_started_event(timeline)
    if not started_event:
        raise ValueError("no failed node with input snapshot found")
    if workflow:
        return replay_workflow(
            event_id=started_event["_id"],
            mode=mode,
            override_state=override_state,
            dry_run=dry_run,
        )
    return replay_node(
        event_id=started_event["_id"],
        mode=mode,
        override_state=override_state,
        dry_run=dry_run,
    )
