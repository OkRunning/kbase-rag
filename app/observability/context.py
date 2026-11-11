"""管理请求和工作流上下文。

该模块提供跨函数调用传递 request、session、task、graph、node 信息的
上下文变量，供日志、链路追踪和指标记录统一读取。
"""

from contextvars import ContextVar
from typing import Any, Dict

from opentelemetry import trace

_request_id: ContextVar[str] = ContextVar("request_id", default="")
_session_id: ContextVar[str] = ContextVar("session_id", default="")
_task_id: ContextVar[str] = ContextVar("task_id", default="")
_tenant_id: ContextVar[str] = ContextVar("tenant_id", default="default")
_kb_id: ContextVar[str] = ContextVar("kb_id", default="default")
_graph: ContextVar[str] = ContextVar("graph", default="")
_node_name: ContextVar[str] = ContextVar("node_name", default="")
_langfuse_trace_id: ContextVar[str] = ContextVar("langfuse_trace_id", default="")
_langfuse_parent_span_id: ContextVar[str] = ContextVar("langfuse_parent_span_id", default="")
_langfuse_trace_url: ContextVar[str] = ContextVar("langfuse_trace_url", default="")

_VARS = {
    "request_id": _request_id,
    "session_id": _session_id,
    "task_id": _task_id,
    "tenant_id": _tenant_id,
    "kb_id": _kb_id,
    "graph": _graph,
    "node_name": _node_name,
    "langfuse_trace_id": _langfuse_trace_id,
    "langfuse_parent_span_id": _langfuse_parent_span_id,
    "langfuse_trace_url": _langfuse_trace_url,
}


def get_context() -> Dict[str, str]:
    """获取当前可观测上下文。

    Returns:
        当前请求和工作流的上下文字段。
    """
    return {name: var.get() for name, var in _VARS.items()}


def get_current_otel_ids() -> Dict[str, str]:
    """获取当前 OpenTelemetry span 的关联 ID。

    Returns:
        包含 `otel_trace_id` 和 `otel_span_id` 的字典；当前没有有效 span 时返回空字符串。
    """
    span_context = trace.get_current_span().get_span_context()
    if span_context and span_context.is_valid:
        return {
            "otel_trace_id": f"{span_context.trace_id:032x}",
            "otel_span_id": f"{span_context.span_id:016x}",
        }
    return {
        "otel_trace_id": "",
        "otel_span_id": "",
    }


def set_context(**kwargs: Any) -> Dict[str, Any]:
    """设置当前调用链的可观测上下文。

    Args:
        **kwargs: 需要写入的上下文字段，例如 request_id、session_id、
            task_id、graph 或 node_name。

    Returns:
        ContextVar token 字典，调用方应在流程结束后传给 reset_context。
    """
    tokens: Dict[str, Any] = {}
    for name, value in kwargs.items():
        if name not in _VARS or value is None:
            continue
        tokens[name] = _VARS[name].set(str(value))
    return tokens


def reset_context(tokens: Dict[str, Any]) -> None:
    """恢复调用前的上下文。

    Args:
        tokens: set_context 返回的 ContextVar token 字典。

    Returns:
        None.
    """
    # 状态恢复必须反向执行，避免嵌套 workflow/node 覆盖外层上下文。
    for name, token in reversed(list(tokens.items())):
        _VARS[name].reset(token)


def extract_workflow_ids(state: Dict[str, Any]) -> Dict[str, str]:
    """从 LangGraph State 中提取工作流关联 ID。

    Args:
        state: LangGraph 节点状态。

    Returns:
        包含 session_id、task_id、tenant_id 和 kb_id 的字典。
    """
    return {
        "session_id": state.get("session_id", ""),
        "task_id": state.get("task_id", ""),
        "tenant_id": state.get("tenant_id", "default"),
        "kb_id": state.get("kb_id", "default"),
    }
