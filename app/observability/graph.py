"""提供带默认 checkpoint 配置的 LangGraph 包装器。"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from app.observability.checkpoints import build_langgraph_config


def _workflow_id_from_input(value: Any) -> str:
    """从 LangGraph 输入中提取 workflow ID。

    Args:
        value: LangGraph invoke/stream 输入。

    Returns:
        session_id、task_id 或临时 ID。
    """
    if isinstance(value, dict):
        return value.get("session_id") or value.get("task_id") or str(uuid.uuid4())
    return str(uuid.uuid4())


def _scope_from_input(value: Any) -> Dict[str, str]:
    """从 LangGraph 输入中提取租户和知识库范围。

    Args:
        value: LangGraph invoke/stream 输入。

    Returns:
        包含 tenant_id 和 kb_id 的字典。
    """
    if isinstance(value, dict):
        return {
            "tenant_id": str(value.get("tenant_id") or "default"),
            "kb_id": str(value.get("kb_id") or "default"),
        }
    return {"tenant_id": "default", "kb_id": "default"}


class CheckpointedGraph:
    """为 CompiledStateGraph 自动补齐 checkpoint config。

    Args:
        graph_name: 工作流名称。
        app: LangGraph 编译后的应用。
    """

    def __init__(self, graph_name: str, app: Any) -> None:
        self.graph_name = graph_name
        self.app = app

    def _ensure_config(self, value: Any, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """确保 LangGraph 调用携带 checkpoint config。

        Args:
            value: LangGraph 输入。
            config: 调用方传入的 config。

        Returns:
            可用于 LangGraph 的 config。
        """
        if config is not None:
            return config
        # 兼容旧脚本和测试：没有显式 config 时仍然启用 checkpoint。
        scope = _scope_from_input(value)
        return build_langgraph_config(
            self.graph_name,
            _workflow_id_from_input(value),
            tenant_id=scope["tenant_id"],
            kb_id=scope["kb_id"],
        )

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """调用 LangGraph invoke 并自动补齐 checkpoint config。

        Args:
            input: LangGraph 输入。
            config: LangGraph 运行配置。
            **kwargs: 透传给原始 invoke 的参数。

        Returns:
            LangGraph 执行结果。
        """
        return self.app.invoke(input, config=self._ensure_config(input, config), **kwargs)

    def stream(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """调用 LangGraph stream 并自动补齐 checkpoint config。

        Args:
            input: LangGraph 输入。
            config: LangGraph 运行配置。
            **kwargs: 透传给原始 stream 的参数。

        Returns:
            LangGraph stream 迭代器。
        """
        return self.app.stream(input, config=self._ensure_config(input, config), **kwargs)

    def __getattr__(self, name: str) -> Any:
        """代理 CompiledStateGraph 的其他属性和方法。

        Args:
            name: 属性名称。

        Returns:
            原始 LangGraph 应用上的属性。
        """
        return getattr(self.app, name)
