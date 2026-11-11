"""查询服务 API。

该模块提供查询页面、问答接口、SSE 流式响应和历史记录接口，并在入口层
初始化 OpenTelemetry，记录 HTTP、workflow 和 LangGraph 节点的观测数据。
"""

import uuid
import time
import re
import json
from contextlib import asynccontextmanager

from starlette.concurrency import run_in_threadpool

from app.observability.settings import is_observability_enabled

if is_observability_enabled():
    # 仅启用可观测能力时初始化 OTel，关闭观测时业务启动不依赖 trace exporter。
    from app.core.tracing import setup_tracing

    setup_tracing()

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, StreamingResponse

from app.clients.mongo_history_utils import get_recent_messages, clear_history
from app.core.logger import logger
from app.core.observability import (
    OBSERVABILITY_ENABLED,
    Status,
    StatusCode,
    build_langgraph_config,
    build_liveness,
    build_observability_health,
    build_readiness,
    emit_event,
    get_current_otel_ids,
    instrument_fastapi,
    observability_debug_router,
    record_http_request,
    record_query,
    record_task_inflight,
    reset_context,
    sanitize_text,
    set_context,
    start_observability_automation,
    tracer,
)
from app.core.tenant_scope import normalize_kb_id, normalize_tenant_id
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.query_process.api.query_stream import sse_query_stream
from app.utils.path_util import PROJECT_ROOT
from app.utils.sse_utils import create_sse_queue, push_to_session, SSEEvent, sse_generator
from app.utils.task_utils import (
    update_task_status,
    get_task_result,
    set_task_result,
    get_task_status,
    get_done_task_list,
    get_running_task_list,
    reset_task_progress,
)

# Langfuse v4：可选观测把整条 LangGraph 执行包成一条 trace，
# 内部各节点 + LangChain LLM callback 的 generation 会挂在其下
# propagate_attributes 用于设置 trace 级属性（如 session_id）
from app.clients.langfuse import langfuse, propagate_attributes
from app.utils.alert_utils import failure_monitor, CircuitOpenError


@asynccontextmanager
async def _lifespan(app_: FastAPI):
    """管理查询服务生命周期事件。

    Args:
        app_: FastAPI 应用实例。

    Yields:
        None.
    """
    # 生命周期启动阶段拉起观测自动化线程，替代已废弃的 on_event startup。
    start_observability_automation("query")
    yield


app = FastAPI(title="query service", description="RAG查询服务！", lifespan=_lifespan)
if OBSERVABILITY_ENABLED and observability_debug_router is not None:
    app.include_router(observability_debug_router)

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# provider已初始化，获取真实tracer
instrument_fastapi(app)


def _query_state_summary(state):
    summary = {}
    if not isinstance(state, dict):
        return summary
    for key in (
            "embedding_chunks",
            "hyde_embedding_chunks",
            "web_search_docs",
            "rrf_chunks",
            "reranked_docs",
            "item_names",
            "history",
            "sources",
    ):
        value = state.get(key)
        if isinstance(value, (list, tuple, set, dict)):
            summary[f"{key}_size"] = len(value)
    for key in ("original_query", "rewritten_query", "answer", "prompt"):
        value = state.get(key)
        if value:
            summary[f"{key}_chars"] = len(str(value))
    return summary


def _get_task_sources(session_id: str) -> list:
    """读取任务结果中的结构化来源列表。

    Args:
        session_id: 会话 ID。

    Returns:
        sources 列表；不存在或解析失败时返回空列表。
    """
    raw_sources = get_task_result(session_id, "sources", "[]")
    if isinstance(raw_sources, list):
        return raw_sources
    try:
        parsed = json.loads(raw_sources or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """记录 HTTP 请求指标并注入 request_id 上下文。

    Args:
        request: FastAPI 请求对象。
        call_next: 后续中间件或路由处理函数。

    Returns:
        FastAPI 响应对象。
    """
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    tokens = set_context(request_id=request_id)
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        record_http_request(route_path, request.method, status_code, time.perf_counter() - start)
        # 请求结束后恢复上下文，避免连接复用或后台任务污染下一次请求日志。
        reset_context(tokens)


# 健康状态
@app.get("/health")
async def health():
    """返回查询服务兼容健康状态。

    Returns:
        服务健康状态。
    """
    logger.info(f"触发后台检测检查接口，数据一切正常！！")
    return build_liveness("query")


@app.get("/health/live")
async def health_live():
    """返回查询服务 liveness 状态。

    Returns:
        liveness 健康状态。
    """
    return build_liveness("query")


@app.get("/health/ready")
async def health_ready():
    """返回查询服务 readiness 状态。

    Returns:
        readiness 健康状态。
    """
    return build_readiness("query")


@app.get("/health/observability")
async def health_observability():
    """返回查询服务观测能力状态。

    Returns:
        观测能力健康状态。
    """
    return build_observability_health("query")


# 返回chat.html
@app.get("/chat")
async def chat_html():
    """返回查询聊天页面。

    Returns:
        chat.html 文件响应。

    Raises:
        HTTPException: chat.html 文件不存在。
    """
    # 查找chat.html页面的地址
    chat_html_path = PROJECT_ROOT / 'app' / 'query_process' / 'page' / 'chat.html'
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail="chat.html文件不存在")
    return FileResponse(chat_html_path)


# 发起提问接口
# 接收参数的类型
class QueryRequest(BaseModel):
    """查询请求参数。

    Attributes:
        query: 用户查询内容。
        session_id: 会话 ID，可为空。
        tenant_id: 租户 ID，老请求为空时进入默认租户。
        kb_id: 知识库 ID，老请求为空时进入默认知识库。
        user_id: 用户 ID，用于审计和历史记录。
        is_stream: 是否流式返回结果。
    """

    query: str = Field(..., min_length=1, max_length=2000, title="查询内容,必须传递")
    session_id: str | None = Field(None, max_length=128, title="会话id，可以不传递，后台uuid生成第一个！")
    tenant_id: str = Field("default", max_length=128, title="租户id，默认default")
    kb_id: str = Field("default", max_length=128, title="知识库id，默认default")
    user_id: str = Field("anonymous", max_length=128, title="用户id，默认anonymous")
    is_stream: bool = Field(False, title="是否流式返回结果")

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """校验并规范化查询内容。

        Args:
            value: 原始查询内容。

        Returns:
            去除首尾空白后的查询内容。

        Raises:
            ValueError: 查询内容为空。
        """
        value = value.strip()
        if not value:
            raise ValueError("查询内容不能为空")
        return value

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        """校验会话 ID 格式。

        Args:
            value: 会话 ID。

        Returns:
            合法会话 ID，或 None。

        Raises:
            ValueError: 会话 ID 包含非法字符。
        """
        if value is None:
            return value
        value = value.strip()
        if not value:
            return None
        if not re.fullmatch(r"[A-Za-z0-9_:\-]+", value):
            raise ValueError("session_id只能包含字母、数字、下划线、中划线和冒号")
        return value

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, value: str | None) -> str:
        """校验租户 ID。

        Args:
            value: 租户 ID。

        Returns:
            规范化后的租户 ID。

        Raises:
            ValueError: 租户 ID 格式不合法。
        """
        return normalize_tenant_id(value)

    @field_validator("kb_id")
    @classmethod
    def validate_kb_id(cls, value: str | None) -> str:
        """校验知识库 ID。

        Args:
            value: 知识库 ID。

        Returns:
            规范化后的知识库 ID。

        Raises:
            ValueError: 知识库 ID 格式不合法。
        """
        return normalize_kb_id(value)


# @observe(name="query_rag")
def run_query_graph(
    query: str,
    session_id: str,
    is_stream: bool,
    raise_on_error: bool = False,
    tenant_id: str = "default",
    kb_id: str = "default",
    user_id: str = "anonymous",
):
    """执行查询 LangGraph 工作流。

    Args:
        query: 用户原始查询。
        session_id: 查询会话 ID。
        is_stream: 是否使用 SSE 流式返回。
        raise_on_error: 是否在记录错误后继续抛出异常。
        tenant_id: 租户 ID。
        kb_id: 知识库 ID。
        user_id: 用户 ID。

    Returns:
        None.

    Raises:
        Exception: raise_on_error 为 True 且工作流执行失败。
    """
    tenant_id = normalize_tenant_id(tenant_id)
    kb_id = normalize_kb_id(kb_id)
    tokens = set_context(graph="query", session_id=session_id, tenant_id=tenant_id, kb_id=kb_id)
    start = time.perf_counter()
    status = "ok"
    evidence_status = ""
    lf_span = None
    langfuse_tokens = {}
    record_task_inflight("query", 1)
    # 一会回调用 main_graph执行
    # 本次任务开启了！ is_stream = True 把结果加入到队列，sse可以取到
    with tracer.start_as_current_span("workflow.query") as workflow_span:
        workflow_span.set_attribute("rag.graph", "query")
        workflow_span.set_attribute("rag.session_id", session_id)
        workflow_span.set_attribute("rag.tenant_id", tenant_id)
        workflow_span.set_attribute("rag.kb_id", kb_id)
        workflow_span.set_attribute("rag.is_stream", is_stream)
        workflow_span.set_attribute("rag.query_chars", len(query or ""))
        otel_ids = get_current_otel_ids()
        emit_event(
            "workflow.started",
            graph="query",
            session_id=session_id,
            status="ok",
            summary={"query": sanitize_text(query), "is_stream": is_stream, "tenant_id": tenant_id, "kb_id": kb_id},
            tags=["query", "workflow.started"],
        )
        # 同一 session_id 会跨多轮对话复用，新一轮查询开始前重置上轮的
        # running/done 列表与 result，避免 done_list 跨轮累加展示旧节点。
        reset_task_progress(session_id)
        update_task_status(session_id, "processing", is_stream)

        # 把 session_id 关联到 Langfuse trace，便于按会话检索
        # v4 用 propagate_attributes 设置 trace 级属性（替代 v3 的 update_current_trace）
        # try:
        #     propagate_attributes(session_id=session_id)
        # except Exception as e:
        #     logger.debug(f"[Langfuse] 设置 trace session_id 失败（不影响主流程）：{e}")

        state = create_query_default_state(
            session_id=session_id,
            original_query=query,
            is_stream=is_stream,
            tenant_id=tenant_id,
            kb_id=kb_id,
            user_id=user_id,
        )
        trace_input = {
            "query": sanitize_text(query),
            "session_id": session_id,
            "is_stream": is_stream,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "user_id": user_id,
        }
        try:

            with propagate_attributes(session_id=session_id,
                                      user_id=user_id,
                                      tags=["query_tag"],
                                      trace_name="query_rag",
                                      metadata={
                                          "query": sanitize_text(query),
                                          "tenant_id": tenant_id,
                                          "kb_id": kb_id,
                                          **otel_ids,
                                      }):
                with langfuse.start_as_current_observation(
                        name="workflow.query",
                        as_type="span",
                        input=trace_input,
                        metadata={
                            "session_id": session_id,
                            "tenant_id": tenant_id,
                            "kb_id": kb_id,
                            **otel_ids,
                        },
                ) as lf_span:
                    # Langfuse trace 级 input/output 需要显式设置，否则只更新 observation，
                    # trace 列表中可能仍显示为空。
                    langfuse.set_current_trace_io(input=trace_input)
                    langfuse_trace_id = langfuse.get_current_trace_id() or ""
                    try:
                        langfuse_trace_url = langfuse.get_trace_url() or ""
                    except Exception:
                        langfuse_trace_url = ""
                    if langfuse_trace_id:
                        workflow_span.set_attribute("langfuse.trace_id", langfuse_trace_id)
                    if langfuse_trace_url:
                        workflow_span.set_attribute("langfuse.trace_url", langfuse_trace_url)
                    langfuse_tokens = set_context(
                        langfuse_trace_id=langfuse_trace_id,
                        langfuse_parent_span_id=langfuse.get_current_observation_id(),
                        langfuse_trace_url=langfuse_trace_url,
                    )
                    # raise ValueError("连续异常测试")
                    # LangGraph State 在各节点间流转，节点级 span 由 observe_node 包装器记录。
                    graph_config = build_langgraph_config("query", session_id, tenant_id=tenant_id, kb_id=kb_id)
                    final_state = query_app.invoke(state, config=graph_config)
                    answer = (final_state or {}).get("answer")
                    sources = (final_state or {}).get("sources") or []
                    evidence_status = str((final_state or {}).get("evidence_status") or "")
                    workflow_span.set_attribute("rag.answer_chars", len(answer or ""))
                    workflow_span.set_attribute("rag.sources_count", len(sources))
                    trace_output = {
                        "answer": sanitize_text(answer),
                        "sources_count": len(sources),
                        "session_id": session_id,
                        "tenant_id": tenant_id,
                        "kb_id": kb_id,
                        "state_summary": _query_state_summary(final_state or {}),
                    }
                    lf_span.update(output=trace_output)
                    langfuse.set_current_trace_io(input=trace_input, output=trace_output)
                    # 可选：给这条 trace 打分（如答案非空=1）
                    if answer:
                        langfuse.score_current_trace(name="has_answer", value=1)

            # 本次任务开启了！ is_stream = True 把结果加入到队列，sse可以取到
            update_task_status(session_id, "completed", is_stream)
            # 查询成功：连续失败计数清零，关闭熔断
            failure_monitor.record_success()
            emit_event(
                "workflow.completed",
                graph="query",
                session_id=session_id,
                status="ok",
                summary={
                    "is_stream": is_stream,
                    "tenant_id": tenant_id,
                    "kb_id": kb_id,
                    "answer_chars": len(get_task_result(session_id, "answer") or ""),
                    "sources_count": len(_get_task_sources(session_id)),
                    "evidence_status": evidence_status,
                },
                tags=["query", "workflow.completed"],
            )
        except CircuitOpenError as ce:
            status = "degraded"
            workflow_span.record_exception(ce)
            # 熔断降级：电路开路且无备用模型，返回兜底话术（不计为失败，不累加计数）
            canned = "系统当前访问繁忙，请稍后重试。"
            set_task_result(session_id, "answer", canned)
            set_task_result(session_id, "sources", "[]")
            update_task_status(session_id, "completed", is_stream)
            if is_stream:
                push_to_session(session_id, SSEEvent.FINAL, {
                    "answer": canned,
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
                    "is_stream": is_stream,
                    "tenant_id": tenant_id,
                    "kb_id": kb_id,
                    "answer_chars": len(canned),
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
                    lf_span.update(level="ERROR",
                                   status_message=str(e),
                                   output={"error": str(e), "error_type": type(e).__name__},
                                   metadata={"traceback": __import__("traceback").format_exc()},
                                   )
                except Exception:
                    pass
            # 2. 给整条 trace 打个错误分，便于面板筛选/统计
            try:
                langfuse.score_current_trace(name="error", value=1, comment=str(e))
            except Exception:
                pass
            # 3. 连续异常检测：达到阈值且不在冷却期才告警（避免单次抖动刷屏）
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
                summary={"is_stream": is_stream, "tenant_id": tenant_id, "kb_id": kb_id},
                error={"type": type(e).__name__, "message": str(e)},
                tags=["query", "workflow.failed"],
            )
            logger.exception(f"---session_id = {session_id},查询流程出现异常！！{str(e)}")
            # 修改 event = process
            update_task_status(session_id, "failed", is_stream)
            # 推送指定类型的事件
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
            if raise_on_error:
                raise
        finally:
            record_query(status, is_stream, time.perf_counter() - start, evidence_status)
            record_task_inflight("query", -1)
            # 先恢复内层 Langfuse 上下文，再恢复 workflow 上下文，避免嵌套状态串线。
            if langfuse_tokens:
                reset_context(langfuse_tokens)
            # workflow 结束后恢复上下文，避免后台任务复用线程时串 session。
            reset_context(tokens)


@app.post("/query")  # 客户端 -》 问题 -》 graph开启了 -》 查到rag的结果 -》 返回即可！！
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """提交查询请求并触发查询工作流。

    Args:
        request: 查询请求参数。
        background_tasks: FastAPI 后台任务管理器。

    Returns:
        查询结果或流式任务受理信息。
    """
    query = request.query
    session_id = request.session_id or str(uuid.uuid4())
    tenant_id = request.tenant_id
    kb_id = request.kb_id
    user_id = request.user_id
    is_stream = request.is_stream
    emit_event(
        "query.received",
        graph="query",
        session_id=session_id,
        status="ok",
        summary={"query": sanitize_text(query), "is_stream": is_stream, "tenant_id": tenant_id, "kb_id": kb_id},
        tags=["query", "query.received"],
    )
    # 判断是不是流式处理 （异步 -》 先返回一个结果 开始处理 | 后台运行图，结果向前端推送）
    if is_stream:
        # 只要开启流式处理，我们业务中就是将数据，插入到队列中！ {session_id , queue [update_task_state , add_running_task,add_done_list]}
        # 创建当前session_id对应的队列 =》 _session_stream
        create_sse_queue(session_id)
        emit_event(
            "query.accepted",
            graph="query",
            session_id=session_id,
            status="ok",
            summary={"is_stream": is_stream, "mode": "async", "tenant_id": tenant_id, "kb_id": kb_id},
            tags=["query", "query.accepted"],
        )
        # 异步执行  立即返回结果前端 || 中间的过程 sse 一点一点推送给前端
        background_tasks.add_task(run_query_graph, query, session_id, is_stream, False, tenant_id, kb_id, user_id)
        logger.info(f"query:{sanitize_text(query)}已经开启了异步和流式处理！！")
        return {
            "session_id": session_id,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "message": "本次查询处理中...."
        }
    else:
        # 同步执行
        emit_event(
            "query.accepted",
            graph="query",
            session_id=session_id,
            status="ok",
            summary={"is_stream": is_stream, "mode": "sync", "tenant_id": tenant_id, "kb_id": kb_id},
            tags=["query", "query.accepted"],
        )
        try:
            await run_in_threadpool(run_query_graph, query, session_id, is_stream, True, tenant_id, kb_id, user_id)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "QUERY_WORKFLOW_FAILED",
                    "message": "查询处理失败，请稍后重试",
                    "session_id": session_id,
                    "error_type": type(e).__name__,
                },
            ) from e
        # run_query_graph(query, session_id, is_stream)
        # 获取最后一个节点插入的结果！ node_answer_output (answer)
        answer = get_task_result(session_id, "answer")  # task_utils 封装的一个存储会话结果函数
        sources = _get_task_sources(session_id)
        # 返回对应的json数据即可
        logger.info(f"query:{sanitize_text(query)}开启同步处理！处理结果为：{sanitize_text(answer)}!")
        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "message": "本次查询处理完毕！",
            "done_list": []
        }


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """LangGraph astream 单端点流式查询（SSE）。

    与两段式接口（/query + /stream/{session_id}）不同，本接口在单个
    HTTP 请求内直接返回 StreamingResponse，token 由 LangGraph
    stream_mode="messages" 捕获，不依赖内存 SSE 队列。

    Args:
        request: 查询请求参数。

    Returns:
        SSE 流式响应（ready/progress/delta/final/error）。
    """
    query = request.query
    session_id = request.session_id or str(uuid.uuid4())
    # tenant_id = "kejibu"
    tenant_id = request.tenant_id
    # kb_id = "kejibu_rag"
    kb_id = request.kb_id
    user_id = request.user_id
    emit_event(
        "query.received",
        graph="query",
        session_id=session_id,
        status="ok",
        summary={
            "query": sanitize_text(query),
            "is_stream": True,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "endpoint": "query_stream",
        },
        tags=["query", "query.received", "stream"],
    )
    emit_event(
        "query.accepted",
        graph="query",
        session_id=session_id,
        status="ok",
        summary={
            "is_stream": True,
            "mode": "stream",
            "tenant_id": tenant_id,
            "kb_id": kb_id,
        },
        tags=["query", "query.accepted", "stream"],
    )
    return StreamingResponse(
        sse_query_stream(query, session_id, tenant_id, kb_id, user_id),
        media_type="text/event-stream",
    )


@app.get("/query/status/{session_id}")
async def query_status(session_id: str):
    """查询指定 session 的查询任务状态。

    Args:
        session_id: 会话 ID。

    Returns:
        查询任务状态、节点进度和已生成答案。
    """
    return {
        "code": 200,
        "session_id": session_id,
        "status": get_task_status(session_id),
        "done_list": get_done_task_list(session_id),
        "running_list": get_running_task_list(session_id),
        "answer": get_task_result(session_id, "answer"),
        "sources": _get_task_sources(session_id),
    }


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """建立指定 session 的 SSE 流式响应。

    Args:
        session_id: 会话 ID。
        request: FastAPI 请求对象，用于检测客户端断开。

    Returns:
        StreamingResponse 响应。
    """
    logger.info(f"session_id = {session_id}客户端，已经和后台建立了长连接！")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream"
    )


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 10, tenant_id: str = "default", kb_id: str = "default"):
    """查询指定 session 的历史对话。

    Args:
        session_id: 会话 ID。
        limit: 返回的最大历史条数。
        tenant_id: 租户 ID。
        kb_id: 知识库 ID。

    Returns:
        历史对话列表。
    """
    # 获取历史对话
    tenant_id = normalize_tenant_id(tenant_id)
    kb_id = normalize_kb_id(kb_id)
    chats = get_recent_messages(session_id, limit, tenant_id=tenant_id, kb_id=kb_id)
    # chat mongodb _id -> ObjectId -> 不能直接序列化 json
    # items = []
    # for chat in chats:
    #     items.append(chat)
    logger.info(f"查询历史对话，session_id = {session_id}成功！查询数据为：{chats}")
    return {
        "session_id": session_id,
        "tenant_id": tenant_id,
        "kb_id": kb_id,
        "items": chats
    }


@app.delete("/history/{session_id}")
async def delete_history(session_id: str, tenant_id: str = "default", kb_id: str = "default"):
    """删除指定 session 的历史对话。

    Args:
        session_id: 会话 ID。
        tenant_id: 租户 ID。
        kb_id: 知识库 ID。

    Returns:
        删除数量和提示信息。
    """
    # 删除历史对话
    tenant_id = normalize_tenant_id(tenant_id)
    kb_id = normalize_kb_id(kb_id)
    delete_count = clear_history(session_id, tenant_id=tenant_id, kb_id=kb_id)
    logger.info(f"删除历史对话，session_id = {session_id}成功,删除数量：{delete_count}！")
    return {
        "deleted_count": delete_count,
        "message": f"{session_id}聊天记录删除成功！"
    }


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8001)
