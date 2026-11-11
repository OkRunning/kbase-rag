"""LangGraph 官方推荐方式的流式查询执行器（单端点 SSE）。

在请求内直接消费 ``query_app.astream(..., stream_mode=["messages", "updates"])``
的事件流并翻译为 SSE 文本，不再依赖"后台任务 + 内存队列"的推送通道。

关键设计：
- state 中 ``is_stream`` 固定为 False，生成节点"一次返回"，
  逐 token 增量由 LangGraph 的 messages 流模式通过 LLM callback 捕获，
  与老两段式接口的节点副作用推流互不干扰。
- 客户端断连时 ``CancelledError`` 直接传播进 astream，真正取消 LLM 生成。
- checkpoint 配置复用 build_langgraph_config，天然支持断点续跑与多实例。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict

from app.clients.langfuse import langfuse, propagate_attributes
from app.core.logger import logger
from app.core.observability import (
    Status,
    StatusCode,
    build_langgraph_config,
    emit_event,
    get_current_otel_ids,
    record_query,
    record_task_inflight,
    reset_context,
    sanitize_text,
    set_context,
    tracer,
)
from app.core.tenant_scope import normalize_kb_id, normalize_tenant_id
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.utils.alert_utils import CircuitOpenError, failure_monitor
from app.utils.sse_utils import SSEEvent, _sse_pack
from app.utils.task_utils import (
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    reset_task_progress,
    update_task_status,
)

_ANSWER_NODE = "node_answer_output"
_CANNED_DEGRADED_ANSWER = "系统当前访问繁忙，请稍后重试。"


def _state_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """生成最终 state 的摘要，用于观测输出，避免把大对象写进 trace。

    Args:
        state: 查询流程最终状态。

    Returns:
        精简后的状态摘要。
    """
    summary: Dict[str, Any] = {}
    if not isinstance(state, dict):
        return summary
    for key in ("reranked_docs", "sources"):
        value = state.get(key)
        if isinstance(value, (list, tuple, set, dict)):
            summary[f"{key}_size"] = len(value)
    for key in ("original_query", "answer"):
        value = state.get(key)
        if value:
            summary[f"{key}_chars"] = len(str(value))
    return summary


def _parse_sources(raw: Any) -> list:
    """解析任务结果中的结构化来源列表。

    Args:
        raw: 任务结果原始值。

    Returns:
        sources 列表；不存在或解析失败时返回空列表。
    """
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(str(raw or "[]"))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _collect_state_update(payload: Any) -> Dict[str, Any]:
    """从 updates 模式 payload 中提取节点返回的 state 更新。

    node_answer_output 返回完整 QueryGraphState，因此优先取携带
    ``answer`` 键的节点更新作为最终状态快照；并行节点超步取最后一个。

    Args:
        payload: stream_mode="updates" 单步产出。

    Returns:
        提取到的 state 字典；无法提取时返回空字典。
    """
    if not isinstance(payload, dict):
        return {}
    for update in payload.values():
        if isinstance(update, dict) and update.get("answer") is not None:
            return update
    values = [value for value in payload.values() if isinstance(value, dict)]
    return values[-1] if values else {}


def _progress_payload(session_id: str) -> Dict[str, Any]:
    """构造与老协议兼容的 progress 事件 payload。

    Args:
        session_id: 查询会话 ID。

    Returns:
        含 done_list、running_list、status 的进度字典。
    """
    return {
        "done_list": get_done_task_list(session_id),
        "running_list": get_running_task_list(session_id),
        "status": get_task_status(session_id),
    }


async def sse_query_stream(
    query: str,
    session_id: str,
    tenant_id: str = "default",
    kb_id: str = "default",
    user_id: str = "anonymous",
) -> AsyncGenerator[str, None]:
    """消费查询 astream 事件流并翻译为 SSE 文本。

    Args:
        query: 用户原始查询。
        session_id: 查询会话 ID。
        tenant_id: 租户 ID。
        kb_id: 知识库 ID。
        user_id: 用户 ID。

    Yields:
        已打包的 SSE 文本消息（ready/progress/delta/final/error）。
    """
    tenant_id = normalize_tenant_id(tenant_id)
    kb_id = normalize_kb_id(kb_id)
    tokens = set_context(graph="query", session_id=session_id, tenant_id=tenant_id, kb_id=kb_id)
    start = time.perf_counter()
    status = "ok"
    evidence_status = ""
    lf_span = None
    langfuse_tokens: Dict[str, Any] = {}
    record_task_inflight("query", 1)
    try:
        # 连接建立信号，与老协议保持一致。
        yield _sse_pack(SSEEvent.READY, {})
        reset_task_progress(session_id)
        update_task_status(session_id, "processing", False)

        state = create_query_default_state(
            session_id=session_id,
            original_query=query,
            is_stream=False,
            tenant_id=tenant_id,
            kb_id=kb_id,
            user_id=user_id,
        )
        trace_input = {
            "query": sanitize_text(query),
            "session_id": session_id,
            "is_stream": True,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "user_id": user_id,
        }
        with tracer.start_as_current_span("workflow.query.stream") as workflow_span:
            workflow_span.set_attribute("rag.graph", "query")
            workflow_span.set_attribute("rag.session_id", session_id)
            workflow_span.set_attribute("rag.tenant_id", tenant_id)
            workflow_span.set_attribute("rag.kb_id", kb_id)
            workflow_span.set_attribute("rag.is_stream", True)
            workflow_span.set_attribute("rag.endpoint", "query_stream")
            workflow_span.set_attribute("rag.query_chars", len(query or ""))
            otel_ids = get_current_otel_ids()
            emit_event(
                "workflow.started",
                graph="query",
                session_id=session_id,
                status="ok",
                summary={"query": sanitize_text(query), "is_stream": True, "tenant_id": tenant_id, "kb_id": kb_id},
                tags=["query", "workflow.started", "stream"],
            )
            final_state: Dict[str, Any] = {}
            try:
                with propagate_attributes(
                    session_id=session_id,
                    user_id=user_id,
                    tags=["query_tag", "stream"],
                    trace_name="query_rag_stream",
                    metadata={
                        "query": sanitize_text(query),
                        "tenant_id": tenant_id,
                        "kb_id": kb_id,
                        **otel_ids,
                    },
                ):
                    with langfuse.start_as_current_observation(
                        name="workflow.query.stream",
                        as_type="span",
                        input=trace_input,
                        metadata={
                            "session_id": session_id,
                            "tenant_id": tenant_id,
                            "kb_id": kb_id,
                            **otel_ids,
                        },
                    ) as lf_span:
                        langfuse.set_current_trace_io(input=trace_input)
                        langfuse_trace_id = langfuse.get_current_trace_id() or ""
                        langfuse_trace_url = langfuse.get_trace_url() or ""
                        if langfuse_trace_id:
                            workflow_span.set_attribute("langfuse.trace_id", langfuse_trace_id)
                        if langfuse_trace_url:
                            workflow_span.set_attribute("langfuse.trace_url", langfuse_trace_url)
                        langfuse_tokens = set_context(
                            langfuse_trace_id=langfuse_trace_id,
                            langfuse_parent_span_id=langfuse.get_current_observation_id(),
                            langfuse_trace_url=langfuse_trace_url,
                        )
                        config = build_langgraph_config("query", session_id, tenant_id=tenant_id, kb_id=kb_id)
                        # 客户端断连时 CancelledError 在此传播进 astream，取消底层 LLM 生成。
                        async for mode, payload in query_app.astream(
                            state, config=config, stream_mode=["messages", "updates"]
                        ):
                            if mode == "messages":
                                chunk, meta = payload
                                # 只透传答案节点的 token，过滤理解/检索等节点内部 LLM 调用。
                                if (meta or {}).get("langgraph_node") == _ANSWER_NODE:
                                    delta = getattr(chunk, "content", "")
                                    if isinstance(delta, str) and delta:
                                        yield _sse_pack(SSEEvent.DELTA, {"delta": delta})
                            elif mode == "updates":
                                final_state = _collect_state_update(payload) or final_state
                                yield _sse_pack(SSEEvent.PROGRESS, _progress_payload(session_id))

                # 图正常跑完：汇总最终结果并发送 FINAL。
                update_task_status(session_id, "completed", False)
                answer = final_state.get("answer") or get_task_result(session_id, "answer") or ""
                sources = final_state.get("sources") or _parse_sources(get_task_result(session_id, "sources", "[]"))
                image_urls = final_state.get("image_urls") or []
                evidence_status = str(final_state.get("evidence_status") or "")
                workflow_span.set_attribute("rag.answer_chars", len(answer))
                workflow_span.set_attribute("rag.sources_count", len(sources))
                trace_output = {
                    "answer": sanitize_text(answer),
                    "sources_count": len(sources),
                    "session_id": session_id,
                    "tenant_id": tenant_id,
                    "kb_id": kb_id,
                    "state_summary": _state_summary(final_state),
                }
                lf_span.update(output=trace_output)
                langfuse.set_current_trace_io(input=trace_input, output=trace_output)
                if answer:
                    langfuse.score_current_trace(name="has_answer", value=1)
                failure_monitor.record_success()
                yield _sse_pack(SSEEvent.PROGRESS, _progress_payload(session_id))
                yield _sse_pack(SSEEvent.FINAL, {
                    "answer": answer,
                    "status": "completed",
                    "image_urls": image_urls,
                    "sources": sources,
                })
                emit_event(
                    "workflow.completed",
                    graph="query",
                    session_id=session_id,
                    status="ok",
                    summary={
                        "is_stream": True,
                        "tenant_id": tenant_id,
                        "kb_id": kb_id,
                        "answer_chars": len(answer),
                        "sources_count": len(sources),
                        "evidence_status": evidence_status,
                    },
                    tags=["query", "workflow.completed", "stream"],
                )
            except CircuitOpenError as ce:
                status = "degraded"
                workflow_span.record_exception(ce)
                # 熔断降级：电路开路且无备用模型，返回兜底话术（不计为失败，不累加计数）。
                update_task_status(session_id, "completed", False)
                yield _sse_pack(SSEEvent.FINAL, {
                    "answer": _CANNED_DEGRADED_ANSWER,
                    "status": "completed",
                    "image_urls": [],
                    "sources": [],
                })
                logger.warning(f"[熔断降级] session={session_id}: {ce}")
                emit_event(
                    "workflow.completed",
                    graph="query",
                    session_id=session_id,
                    status="degraded",
                    summary={
                        "is_stream": True,
                        "tenant_id": tenant_id,
                        "kb_id": kb_id,
                        "answer_chars": len(_CANNED_DEGRADED_ANSWER),
                        "sources_count": 0,
                    },
                    error={"type": type(ce).__name__, "message": str(ce)},
                    tags=["query", "workflow.completed", "degraded"],
                )
            except Exception as e:
                status = "error"
                workflow_span.record_exception(e)
                workflow_span.set_status(Status(StatusCode.ERROR, str(e)))
                if lf_span:
                    try:
                        lf_span.update(
                            level="ERROR",
                            status_message=str(e),
                            output={"error": str(e), "error_type": type(e).__name__},
                            metadata={"traceback": __import__("traceback").format_exc()},
                        )
                    except Exception:
                        pass
                try:
                    langfuse.score_current_trace(name="error", value=1, comment=str(e))
                except Exception:
                    pass
                try:
                    trace_url = langfuse.get_trace_url()
                except Exception:
                    trace_url = None
                failure_monitor.record_failure(
                    title=f"查询失败 session={session_id}",
                    content=f"{type(e).__name__}: {e}",
                    trace_url=trace_url,
                )
                emit_event(
                    "workflow.failed",
                    graph="query",
                    session_id=session_id,
                    status="error",
                    summary={"is_stream": True, "tenant_id": tenant_id, "kb_id": kb_id},
                    error={"type": type(e).__name__, "message": str(e)},
                    tags=["query", "workflow.failed", "stream"],
                )
                logger.exception(f"---session_id = {session_id},查询流程出现异常！！{str(e)}")
                update_task_status(session_id, "failed", False)
                yield _sse_pack(SSEEvent.ERROR, {"error": str(e)})
    finally:
        record_query(status, True, time.perf_counter() - start, evidence_status)
        record_task_inflight("query", -1)
        # 先恢复内层 Langfuse 上下文，再恢复 workflow 上下文，避免嵌套状态串线。
        if langfuse_tokens:
            reset_context(langfuse_tokens)
        reset_context(tokens)
