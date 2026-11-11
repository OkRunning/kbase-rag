"""提供低侵入的可观测包装器。

该模块用于在不修改 LangGraph 节点业务逻辑的前提下，为节点统一补充
trace span、上下文注入和耗时指标。
"""

import time
from functools import wraps
from typing import Any, Callable, Dict

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.clients.langfuse import langfuse
from app.core.logger import logger
from app.observability.context import extract_workflow_ids, get_context, reset_context, set_context
from app.observability.events import emit_event
from app.observability.metrics import record_node_duration
from app.observability.snapshots import save_snapshot

_tracer = trace.get_tracer("kbase-rag.langgraph")


def _state_size(value: Any) -> int:
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    if isinstance(value, str):
        return len(value)
    return 0


def _summarize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key in (
        "embedding_chunks",
        "hyde_embedding_chunks",
        "web_search_docs",
        "rrf_chunks",
        "reranked_docs",
        "item_names",
        "chunks",
        "embeddings_content",
    ):
        if key in state:
            summary[f"{key}_size"] = _state_size(state.get(key))
    for key in ("answer", "prompt", "original_query", "rewritten_query", "md_content"):
        if key in state and state.get(key):
            summary[f"{key}_chars"] = len(str(state.get(key)))
    if state.get("query_type"):
        summary["query_type"] = str(state.get("query_type"))
    if state.get("intent"):
        summary["intent"] = str(state.get("intent"))
    if "need_item_name" in state:
        summary["need_item_name"] = bool(state.get("need_item_name"))
    if state.get("clarification_hint"):
        summary["clarification_hint"] = str(state.get("clarification_hint"))
    if state.get("evidence_status"):
        summary["evidence_status"] = str(state.get("evidence_status"))
    evidence_gate = state.get("evidence_gate")
    if isinstance(evidence_gate, dict) and evidence_gate:
        # Evidence Gate 只写状态、置信度和原因，避免把完整证据文本写入 trace。
        try:
            evidence_confidence = float(evidence_gate.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            evidence_confidence = 0.0
        summary["evidence_gate_status"] = str(evidence_gate.get("evidence_status", ""))
        summary["evidence_gate_confidence"] = evidence_confidence
        summary["evidence_gate_reason"] = str(evidence_gate.get("reason", ""))
    retrieval_plan = state.get("retrieval_plan")
    if isinstance(retrieval_plan, dict) and retrieval_plan:
        # 检索计划只记录开关和原因，避免把 filters 等完整业务字段写入 trace。
        summary["retrieval_plan_use_local_vector"] = bool(retrieval_plan.get("use_local_vector"))
        summary["retrieval_plan_use_hyde"] = bool(retrieval_plan.get("use_hyde"))
        summary["retrieval_plan_use_web"] = bool(retrieval_plan.get("use_web"))
        summary["retrieval_plan_reason"] = str(retrieval_plan.get("reason", ""))
    return summary


def _count_state_items(state: Dict[str, Any]) -> int:
    count = 0
    for key in (
        "embedding_chunks",
        "hyde_embedding_chunks",
        "web_search_docs",
        "rrf_chunks",
        "reranked_docs",
        "item_names",
        "chunks",
        "embeddings_content",
    ):
        if key in state:
            count += _state_size(state.get(key))
    return count


def observe_node(graph: str, node_name: str, fn: Callable) -> Callable:
    """包装 LangGraph 节点以记录链路和指标。

    Args:
        graph: 工作流名称，例如 query 或 import。
        node_name: LangGraph 节点名称。
        fn: 原始节点函数。

    Returns:
        带可观测能力的节点函数。

    Raises:
        Exception: 原始节点函数抛出的异常会继续向上抛出。
    """

    @wraps(fn)
    def wrapper(state: Dict[str, Any]):
        """执行带可观测能力的 LangGraph 节点。

        Args:
            state: LangGraph 节点状态。

        Returns:
            原始节点函数返回的状态更新。

        Raises:
            Exception: 原始节点函数抛出的异常会继续向上抛出。
        """
        ids = extract_workflow_ids(state or {})
        # 节点执行期间写入 graph/node/session/task 上下文，供日志和指标关联。
        tokens = set_context(graph=graph, node_name=node_name, **ids)
        start = time.perf_counter()
        status = "ok"
        try:
            with _tracer.start_as_current_span(f"node.{node_name}") as span:
                span.set_attribute("rag.graph", graph)
                span.set_attribute("rag.node_name", node_name)
                if ids.get("session_id"):
                    span.set_attribute("rag.session_id", ids["session_id"])
                if ids.get("task_id"):
                    span.set_attribute("rag.task_id", ids["task_id"])
                # 只记录状态摘要和数量，避免把 prompt、answer、chunk 全量写入 trace。
                input_summary = _summarize_state(state or {})
                input_count = _count_state_items(state or {})
                input_ref = save_snapshot(graph, node_name, "input", state or {})
                for key, value in input_summary.items():
                    span.set_attribute(f"rag.state.{key}", value)
                span.set_attribute("rag.node.input_count", input_count)
                started_event = emit_event(
                    "node.started",
                    graph=graph,
                    node_name=node_name,
                    session_id=ids.get("session_id"),
                    task_id=ids.get("task_id"),
                    status="ok",
                    summary={
                        "input_keys": list((state or {}).keys())[:50],
                        "node_name": node_name,
                        "input_count": input_count,
                        "input_summary": input_summary,
                    },
                    input_ref=input_ref,
                    tags=[graph, node_name, "node.started"],
                )

                # Langfuse 的 @observe 会读取当前 observation，因此节点 observation
                # 必须包住业务函数执行，才能形成 workflow -> node -> step 的层级。
                context = get_context()
                trace_context = None
                if context.get("langfuse_trace_id") and context.get("langfuse_parent_span_id"):
                    trace_context = {
                        "trace_id": context["langfuse_trace_id"],
                        "parent_span_id": context["langfuse_parent_span_id"],
                    }
                with langfuse.start_as_current_observation(
                    trace_context=trace_context,
                    name=node_name,
                    as_type="span",
                    metadata={
                        "graph": graph,
                        "session_id": ids.get("session_id"),
                        "task_id": ids.get("task_id"),
                        "input_summary": input_summary,
                    },
                ) as lf_node_span:
                    try:
                        result = fn(state)
                    except Exception as exc:
                        lf_node_span.update(level="ERROR", status_message=str(exc))
                        emit_event(
                            "node.failed",
                            graph=graph,
                            node_name=node_name,
                            session_id=ids.get("session_id"),
                            task_id=ids.get("task_id"),
                            status="error",
                            summary={
                                "input_summary": input_summary,
                            },
                            input_ref=input_ref,
                            error={
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                            tags=[graph, node_name, "node.failed"],
                        )
                        raise
                    if isinstance(result, dict):
                        output_summary = _summarize_state(result)
                        output_count = _count_state_items(result)
                        for key, value in output_summary.items():
                            span.set_attribute(f"rag.result.{key}", value)
                        span.set_attribute("rag.node.output_count", output_count)
                        lf_node_span.update(metadata={
                            "output_summary": output_summary,
                            "output_count": output_count,
                        })
                    else:
                        output_summary = {}
                        output_count = 0
                    output_ref = (
                        save_snapshot(
                            graph,
                            node_name,
                            "output",
                            result,
                            source_event_id=started_event.get("_id"),
                        )
                        if isinstance(result, dict)
                        else None
                    )
                    duration_ms = round((time.perf_counter() - start) * 1000, 2)
                    diagnostics = {
                        "node_name": node_name,
                        "duration_ms": duration_ms,
                        "input_count": input_count,
                        "output_count": output_count,
                        "skip_reason": "",
                        "error_type": "",
                    }
                    # 节点级诊断只写摘要字段，便于后续灰度期间对比各召回源耗时和候选数量。
                    logger.info(f"节点观测诊断：{diagnostics}")
                    emit_event(
                        "node.completed",
                        graph=graph,
                        node_name=node_name,
                        session_id=ids.get("session_id"),
                        task_id=ids.get("task_id"),
                        status="ok",
                        summary={
                            **diagnostics,
                            "input_summary": input_summary,
                            "output_summary": output_summary,
                            "result_keys": list(result.keys())[:50] if isinstance(result, dict) else [],
                        },
                        input_ref=input_ref,
                        output_ref=output_ref,
                        tags=[graph, node_name, "node.completed"],
                    )
                    return result
        except Exception as exc:
            status = "error"
            current_span = trace.get_current_span()
            current_span.record_exception(exc)
            current_span.set_status(Status(StatusCode.ERROR, str(exc)))
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "节点观测诊断："
                f"{{'node_name': '{node_name}', 'duration_ms': {duration_ms}, "
                f"'input_count': {_count_state_items(state or {})}, "
                "'output_count': 0, 'skip_reason': '', "
                f"'error_type': '{type(exc).__name__}'}}"
            )
            raise
        finally:
            duration = time.perf_counter() - start
            record_node_duration(graph, node_name, status, duration)
            # 无论节点成功、失败还是被上层捕获，都必须恢复外层上下文。
            reset_context(tokens)

    return wrapper
