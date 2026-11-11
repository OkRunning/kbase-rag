"""提供 LangGraph Checkpointer 的统一入口。"""

from __future__ import annotations

import asyncio
import atexit
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from app.core.logger import logger

load_dotenv()

_CHECKPOINTER: Optional[Any] = None
_CHECKPOINTER_RESOURCE: Optional[Any] = None


def _checkpoint_backend() -> str:
    """读取 LangGraph checkpoint 后端类型。

    Returns:
        后端类型，支持 memory 或 postgres。
    """
    return os.getenv("LANGGRAPH_CHECKPOINT_BACKEND", "memory").strip().lower()


def get_checkpoint_backend() -> str:
    """获取当前 checkpoint 后端类型。

    Returns:
        checkpoint 后端类型。
    """
    return _checkpoint_backend()


def _is_production() -> bool:
    """判断当前是否为生产环境。

    Returns:
        生产环境返回 True，否则返回 False。
    """
    environment = os.getenv("OTEL_ENVIRONMENT") or os.getenv("ENVIRONMENT") or os.getenv("APP_ENV") or "dev"
    return environment.strip().lower() in {"prod", "production"}


def _should_setup_postgres() -> bool:
    """判断是否需要自动初始化 Postgres checkpoint 表。

    Returns:
        需要执行 setup 时返回 True。
    """
    return os.getenv("LANGGRAPH_CHECKPOINT_POSTGRES_SETUP", "true").strip().lower() in {"1", "true", "yes"}


def _close_checkpointer_resource() -> None:
    """关闭 Postgres checkpointer 资源。

    Returns:
        None.
    """
    global _CHECKPOINTER_RESOURCE
    if _CHECKPOINTER_RESOURCE is None:
        return
    close_fn = getattr(_CHECKPOINTER_RESOURCE, "close", None)
    if close_fn is not None:
        close_fn()
    _CHECKPOINTER_RESOURCE = None


def _create_memory_checkpointer() -> MemorySaver:
    """创建内存 Checkpointer。

    Returns:
        MemorySaver 实例。
    """
    return MemorySaver()


def _create_postgres_checkpointer() -> Any:
    """创建 Postgres Checkpointer。

    Returns:
        PostgresSaver 实例。

    Raises:
        RuntimeError: 配置缺失、依赖缺失或初始化失败时抛出。
    """
    global _CHECKPOINTER_RESOURCE
    dsn = os.getenv("LANGGRAPH_CHECKPOINT_POSTGRES_DSN") or os.getenv("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("LANGGRAPH_CHECKPOINT_POSTGRES_DSN is required when checkpoint backend is postgres")
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        import psycopg
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
    except ImportError as exc:
        raise RuntimeError(
            "Postgres checkpoint backend requires langgraph-checkpoint-postgres and psycopg dependencies"
        ) from exc

    try:
        connect_timeout = int(os.getenv("LANGGRAPH_CHECKPOINT_POSTGRES_CONNECT_TIMEOUT", "1"))
        # 先用短超时探活，避免 ConnectionPool 在本地 Postgres 未启动时持续后台重试并拖慢启动。
        with psycopg.connect(dsn, connect_timeout=connect_timeout):
            pass
        pool = ConnectionPool(
            conninfo=dsn,
            min_size=int(os.getenv("LANGGRAPH_CHECKPOINT_POSTGRES_POOL_MIN", "1")),
            max_size=int(os.getenv("LANGGRAPH_CHECKPOINT_POSTGRES_POOL_MAX", "10")),
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
            },
        )
        class AsyncCompatPostgresSaver(PostgresSaver):
            """为同步 PostgresSaver 补齐 LangGraph 异步 checkpoint API。

            langgraph-checkpoint 4.x 移除了基类异步方法的默认实现，
            而 langgraph-checkpoint-postgres 3.1.0 未同步实现
            aget_tuple/aput/aput_writes，导致 astream/ainvoke 等异步入口
            在读取 checkpoint 时抛出 NotImplementedError。
            这里通过 asyncio.to_thread 将异步调用委托到线程池中的同步实现，
            与内存后端（MemorySaver）的异步行为保持一致，且不影响原有同步调用路径。
            """

            async def aget_tuple(self, config: Any) -> Any:
                """异步读取 checkpoint（委托线程池中的同步实现）。

                Args:
                    config: LangGraph 运行配置。

                Returns:
                    checkpoint 元组；不存在时返回 None。
                """
                return await asyncio.to_thread(self.get_tuple, config)

            async def aput(
                self,
                config: Any,
                checkpoint: Any,
                metadata: Any,
                new_versions: Any = None,
            ) -> Any:
                """异步保存 checkpoint（委托线程池中的同步实现）。

                Args:
                    config: LangGraph 运行配置。
                    checkpoint: 待保存的 checkpoint 数据。
                    metadata: checkpoint 元数据。
                    new_versions: 写入后的 channel 版本信息。

                Returns:
                    更新后的运行配置。
                """
                return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

            async def aput_writes(
                self,
                config: Any,
                writes: Any,
                task_id: str,
                task_path: str = "",
            ) -> None:
                """异步保存节点写入（委托线程池中的同步实现）。

                Args:
                    config: LangGraph 运行配置。
                    writes: 节点写入的数据序列。
                    task_id: 任务 ID。
                    task_path: 任务路径，默认空字符串。

                Returns:
                    None.
                """
                await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

        saver = AsyncCompatPostgresSaver(pool)
        _CHECKPOINTER_RESOURCE = pool
        atexit.register(_close_checkpointer_resource)
        setup_fn = getattr(saver, "setup", None)
        if setup_fn is not None and _should_setup_postgres():
            # 生产首次部署需要初始化 LangGraph 官方 checkpoint 表结构。
            setup_fn()
        return saver
    except Exception as exc:
        raise RuntimeError(f"failed to initialize postgres checkpointer: {exc}") from exc


def get_langgraph_checkpointer() -> Any:
    """获取 LangGraph Checkpointer 单例。

    Returns:
        进程内共享的 LangGraph Checkpointer 实例。

    Raises:
        RuntimeError: 生产环境 checkpoint 初始化失败时抛出。
    """
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        backend = _checkpoint_backend()
        if backend == "memory":
            _CHECKPOINTER = _create_memory_checkpointer()
        elif backend == "postgres":
            try:
                _CHECKPOINTER = _create_postgres_checkpointer()
            except Exception as exc:
                # 生产环境必须保证 checkpoint 真正持久化，不能悄悄退回内存。
                if _is_production():
                    raise
                # 本地开发和测试环境允许降级，避免观测 Postgres 未启动时阻断业务服务。
                logger.warning(f"Postgres checkpoint 初始化失败，已降级为内存 checkpoint：{exc}")
                _CHECKPOINTER = _create_memory_checkpointer()
        else:
            raise RuntimeError(f"unsupported LangGraph checkpoint backend: {backend}")
    return _CHECKPOINTER


def build_langgraph_config(
    graph: str,
    workflow_id: str,
    replay_id: Optional[str] = None,
    tenant_id: str = "default",
    kb_id: str = "default",
) -> Dict[str, Any]:
    """构造启用 LangGraph checkpoint 的运行配置。

    Args:
        graph: 工作流名称，例如 query、import 或 replay.query。
        workflow_id: 原始 session_id、task_id 或事件 ID。
        replay_id: 回放 ID，普通业务执行不需要传。
        tenant_id: 租户标识。
        kb_id: 知识库标识。

    Returns:
        可传给 LangGraph invoke/stream 的 config。
    """
    return {
        "configurable": {
            "thread_id": build_langgraph_thread_id(graph, workflow_id, replay_id, tenant_id, kb_id)
        }
    }


def build_langgraph_thread_id(
    graph: str,
    workflow_id: str,
    replay_id: Optional[str] = None,
    tenant_id: str = "default",
    kb_id: str = "default",
) -> str:
    """生成 LangGraph checkpoint thread ID。

    Args:
        graph: 工作流名称，例如 query、import 或 replay.query。
        workflow_id: 原始 session_id、task_id 或事件 ID。
        replay_id: 回放 ID，普通业务执行不需要传。
        tenant_id: 租户标识。
        kb_id: 知识库标识。

    Returns:
        LangGraph checkpoint thread ID。
    """
    namespace = os.getenv("LANGGRAPH_CHECKPOINT_NAMESPACE", "dataset-rag")
    # checkpoint 线程 ID 必须纳入 scope，避免不同租户复用同一 session_id 时读到彼此状态。
    thread_parts = [namespace, tenant_id or "default", kb_id or "default", graph, workflow_id or "unknown"]
    if replay_id:
        thread_parts.append(replay_id)
    return ":".join(thread_parts)


def list_langgraph_checkpoints(thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """查询指定 LangGraph thread 的 checkpoint 元数据。

    Args:
        thread_id: LangGraph checkpoint thread ID。
        limit: 返回条数上限。

    Returns:
        checkpoint 元数据列表。
    """
    config = {"configurable": {"thread_id": thread_id}}
    items: List[Dict[str, Any]] = []
    for checkpoint_tuple in get_langgraph_checkpointer().list(config, limit=limit):
        checkpoint = checkpoint_tuple.checkpoint or {}
        items.append(
            {
                "config": checkpoint_tuple.config,
                "parent_config": checkpoint_tuple.parent_config,
                "metadata": checkpoint_tuple.metadata,
                "checkpoint_id": checkpoint.get("id"),
                "ts": checkpoint.get("ts"),
                "channel_versions": checkpoint.get("channel_versions", {}),
            }
        )
    return items


def audit_langgraph_checkpoints(thread_id: str, limit: int = 100) -> Dict[str, Any]:
    """生成指定 checkpoint thread 的审计摘要。

    Args:
        thread_id: LangGraph checkpoint thread ID。
        limit: 审计读取条数上限。

    Returns:
        checkpoint 审计摘要。
    """
    items = list_langgraph_checkpoints(thread_id, limit=limit)
    timestamps = [item.get("ts") for item in items if item.get("ts")]
    backend = get_checkpoint_backend()
    return {
        "thread_id": thread_id,
        "backend": backend,
        "persistent": backend == "postgres",
        "checkpoint_count": len(items),
        "oldest_ts": min(timestamps) if timestamps else "",
        "newest_ts": max(timestamps) if timestamps else "",
        "retention_days": int(os.getenv("LANGGRAPH_CHECKPOINT_RETENTION_DAYS", "30")),
        "items": items,
    }
