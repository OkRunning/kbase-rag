"""业务层可观测能力门面。

业务模块只依赖本门面。启用观测时转发到完整实现，关闭观测时使用轻量
空实现，从而避免 API 启动依赖 debug、回放、事件存储和 checkpoint 模块。
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional

from app.observability.context import (
    get_current_otel_ids,
    reset_context,
    set_context,
)
from app.observability.sanitizers import sanitize_for_log, sanitize_text
from app.observability.settings import is_observability_enabled

OBSERVABILITY_ENABLED = is_observability_enabled()


class _NoopSpan:
    """关闭观测时使用的空 span。"""

    def __enter__(self) -> "_NoopSpan":
        """进入空 span 上下文。

        Returns:
            当前空 span。
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        """退出空 span 上下文。

        Args:
            exc_type: 异常类型。
            exc: 异常对象。
            traceback: 异常 traceback。

        Returns:
            False，保持原异常传播语义。
        """
        return False

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        """忽略 span 属性。

        Args:
            *args: span 参数。
            **kwargs: span 属性。

        Returns:
            None.
        """
        return None

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        """忽略 span 异常记录。

        Args:
            *args: 异常参数。
            **kwargs: 异常属性。

        Returns:
            None.
        """
        return None

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        """忽略 span 状态更新。

        Args:
            *args: 状态参数。
            **kwargs: 状态属性。

        Returns:
            None.
        """
        return None


class _NoopTracer:
    """关闭观测时使用的空 tracer。"""

    def start_as_current_span(self, *args: Any, **kwargs: Any) -> _NoopSpan:
        """创建空 span。

        Args:
            *args: span 参数。
            **kwargs: span 属性。

        Returns:
            空 span 上下文。
        """
        return _NoopSpan()


class _NoopStatusCode:
    """关闭观测时使用的空 span 状态码。"""

    ERROR = "ERROR"


class _NoopStatus:
    """关闭观测时使用的空 span 状态。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """初始化空状态。

        Args:
            *args: 状态参数。
            **kwargs: 状态属性。
        """


tracer: Any = _NoopTracer()
Status: Any = _NoopStatus
StatusCode: Any = _NoopStatusCode
observability_debug_router: Any = None


def emit_event(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """记录业务事件，关闭观测时返回未持久化的事件摘要。

    Args:
        *args: 透传给事件记录器的位置参数。
        **kwargs: 透传给事件记录器的关键字参数。

    Returns:
        事件摘要。
    """
    if OBSERVABILITY_ENABLED:
        from app.observability.events import emit_event as _emit_event

        return _emit_event(*args, **kwargs)
    return dict(kwargs)


def build_liveness(service: str) -> Dict[str, Any]:
    """返回 liveness 健康状态。

    Args:
        service: 服务名称。

    Returns:
        liveness 响应。
    """
    from app.core.health import build_liveness as _build_liveness

    return _build_liveness(service)


def build_readiness(service: str) -> Dict[str, Any]:
    """返回 readiness 健康状态。

    Args:
        service: 服务名称。

    Returns:
        readiness 响应。
    """
    from app.core.health import build_readiness as _build_readiness

    return _build_readiness(service)


def _record_metric(name: str, *args: Any, **kwargs: Any) -> None:
    """按开关记录指标。

    Args:
        name: 指标记录函数名称。
        *args: 指标位置参数。
        **kwargs: 指标关键字参数。

    Returns:
        None.
    """
    if not OBSERVABILITY_ENABLED:
        return None
    from app.observability import metrics

    getattr(metrics, name)(*args, **kwargs)
    return None


def record_http_request(route: str, method: str, status_code: int, duration_seconds: float) -> None:
    """记录 HTTP 请求指标。

    Args:
        route: 路由模板。
        method: HTTP 方法。
        status_code: HTTP 状态码。
        duration_seconds: 请求耗时。

    Returns:
        None.
    """
    _record_metric("record_http_request", route, method, status_code, duration_seconds)


def record_query(status: str, is_stream: bool, duration_seconds: float, evidence_status: str = "") -> None:
    """记录查询 workflow 指标。

    Args:
        status: 查询状态。
        is_stream: 是否流式。
        duration_seconds: 查询耗时。
        evidence_status: Evidence Gate 证据状态。

    Returns:
        None.
    """
    _record_metric("record_query", status, is_stream, duration_seconds, evidence_status)


def record_import(status: str, file_type: str, duration_seconds: float) -> None:
    """记录导入 workflow 指标。

    Args:
        status: 导入状态。
        file_type: 文件类型。
        duration_seconds: 导入耗时。

    Returns:
        None.
    """
    _record_metric("record_import", status, file_type, duration_seconds)


def record_task_inflight(graph: str, delta: int) -> None:
    """记录 workflow 在途任务数。

    Args:
        graph: 工作流名称。
        delta: 任务数变化。

    Returns:
        None.
    """
    _record_metric("record_task_inflight", graph, delta)


def build_langgraph_config(
    graph: str,
    workflow_id: str,
    replay_id: Optional[str] = None,
    tenant_id: str = "default",
    kb_id: str = "default",
) -> Dict[str, Any]:
    """构造 LangGraph 运行配置。

    Args:
        graph: 工作流名称。
        workflow_id: 工作流 ID。
        replay_id: 可选回放 ID。
        tenant_id: 租户 ID。
        kb_id: 知识库 ID。

    Returns:
        LangGraph config。
    """
    if OBSERVABILITY_ENABLED:
        from app.observability.checkpoints import build_langgraph_config as _build_config

        return _build_config(graph, workflow_id, replay_id, tenant_id, kb_id)
    thread_id = ":".join(
        [graph, tenant_id or "default", kb_id or "default", workflow_id or "unknown"]
        + ([replay_id] if replay_id else [])
    )
    return {"configurable": {"thread_id": thread_id}}


def observe_node(graph: str, node_name: str, fn: Callable) -> Callable:
    """为 LangGraph 节点增加可选观测包装。

    Args:
        graph: 工作流名称。
        node_name: 节点名称。
        fn: 原始节点函数。

    Returns:
        观测包装后的节点函数，或关闭观测时的原函数包装。
    """
    if OBSERVABILITY_ENABLED:
        from app.observability.decorators import observe_node as _observe_node

        return _observe_node(graph, node_name, fn)

    @wraps(fn)
    def wrapper(state: Dict[str, Any]) -> Any:
        """执行原始节点函数。

        Args:
            state: LangGraph State。

        Returns:
            原始节点返回值。
        """
        return fn(state)

    return wrapper


def compile_graph(graph_name: str, builder: Any) -> Any:
    """编译 LangGraph，并按开关决定是否启用 checkpoint。

    Args:
        graph_name: 工作流名称。
        builder: LangGraph StateGraph 构建器。

    Returns:
        可执行的 LangGraph 应用。
    """
    if not OBSERVABILITY_ENABLED:
        return builder.compile()
    from app.observability.checkpoints import get_langgraph_checkpointer
    from app.observability.graph import CheckpointedGraph

    return CheckpointedGraph(
        graph_name,
        builder.compile(checkpointer=get_langgraph_checkpointer()),
    )


def start_observability_automation(service: str) -> None:
    """按开关启动观测自动化。

    Args:
        service: 服务名称。

    Returns:
        None.
    """
    if OBSERVABILITY_ENABLED:
        from app.observability.automation import start_observability_automation as _start

        _start(service)


def build_observability_health(service: str) -> Dict[str, Any]:
    """返回观测能力健康状态。

    Args:
        service: 服务名称。

    Returns:
        观测健康状态。
    """
    if not OBSERVABILITY_ENABLED:
        return {"status": "disabled", "service": service}
    from app.observability.health import build_observability_health as _build_health

    return _build_health(service)


def instrument_fastapi(app: Any) -> None:
    """按开关执行 FastAPI instrumentation。

    Args:
        app: FastAPI 应用实例。

    Returns:
        None.
    """
    if not OBSERVABILITY_ENABLED:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)


if OBSERVABILITY_ENABLED:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    tracer = trace.get_tracer("dataset-rag.api")
    from app.observability.api.debug_server import router as observability_debug_router
