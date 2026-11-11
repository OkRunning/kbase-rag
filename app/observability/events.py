"""提供业务事件的持久化和查询能力。"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from opentelemetry import trace

from app.core.logger import logger
from app.observability.context import get_context
from app.observability.metrics import (
    record_event_store_fallback,
    record_event_store_write,
    record_mongo_operation,
)
from app.observability.sanitizers import sanitize_summary
from app.observability.settings import is_observability_enabled
from app.utils.path_util import PROJECT_ROOT

load_dotenv()

_COLLECTION_NAME = "observability_events"
_FALLBACK_PATH = PROJECT_ROOT / "logs" / "observability_events.jsonl"
_INDEXES = [
    [("tenant_id", 1), ("kb_id", 1), ("session_id", 1), ("ts", 1)],
    [("tenant_id", 1), ("kb_id", 1), ("task_id", 1), ("ts", 1)],
    [("session_id", 1), ("ts", 1)],
    [("task_id", 1), ("ts", 1)],
    [("trace_id", 1), ("ts", 1)],
    [("langfuse_trace_id", 1), ("ts", 1)],
    [("event_type", 1), ("ts", 1)],
    [("graph", 1), ("node_name", 1), ("ts", 1)],
]


class _EventStore:
    """封装 MongoDB 和本地 JSONL 的事件存储实现。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._collection: Optional[Any] = None
        self._fallback_path = _FALLBACK_PATH
        self._sequence: Dict[str, int] = defaultdict(int)
        self._init_error: Optional[str] = None
        self._ensure_collection()

    def _scope_key(self, document: Dict[str, Any]) -> str:
        """生成事件序列的作用域键。"""
        return (
            document.get("session_id")
            or document.get("task_id")
            or document.get("trace_id")
            or document.get("request_id")
            or "global"
        )

    def _next_sequence(self, document: Dict[str, Any]) -> int:
        """为同一作用域内的事件生成单调递增序号。"""
        scope_key = self._scope_key(document)
        with self._lock:
            self._sequence[scope_key] += 1
            return self._sequence[scope_key]

    def _ensure_collection(self) -> None:
        """懒加载 Mongo 集合并创建索引。"""
        if self._collection is not None or self._init_error is not None:
            return
        mongo_url = os.getenv("MONGO_URL")
        db_name = os.getenv("MONGO_DB_NAME")
        username = os.getenv("MONGO_DB_USERNAME")
        password = os.getenv("MONGO_DB_PASSWORD")
        if not mongo_url or not db_name:
            self._init_error = "missing mongo configuration"
            return
        try:
            from pymongo import MongoClient

            client = MongoClient(
                mongo_url,
                username=username or None,
                password=password or None,
                authSource="admin",
                serverSelectionTimeoutMS=2000,
            )
            client.admin.command("ping")
            collection = client[db_name][_COLLECTION_NAME]
            for index in _INDEXES:
                collection.create_index(index)
            self._collection = collection
        except Exception as exc:  # pragma: no cover - network and env dependent
            self._init_error = str(exc)
            logger.warning(f"事件存储 Mongo 初始化失败，改用本地 JSONL：{exc}")

    def _append_fallback(self, document: Dict[str, Any]) -> None:
        """将事件追加到本地 JSONL 兜底文件。"""
        self._fallback_path.parent.mkdir(parents=True, exist_ok=True)
        with self._fallback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(document, ensure_ascii=False, default=str))
            handle.write("\n")

    def _load_fallback(self) -> List[Dict[str, Any]]:
        """读取本地 JSONL 兜底事件。"""
        if not self._fallback_path.exists():
            return []
        items: List[Dict[str, Any]] = []
        with self._fallback_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
        return items

    def write(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """写入一条事件记录。

        Args:
            document: 待写入的事件文档。

        Returns:
            已补充 `_id` 和 `sequence` 的事件文档。
        """
        self._ensure_collection()
        payload = dict(document)
        payload.setdefault("_id", str(uuid.uuid4()))
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        payload["summary"] = sanitize_summary(payload.get("summary", {}))
        payload["sequence"] = int(payload.get("sequence") or self._next_sequence(payload))
        if payload.get("tags") is not None:
            payload["tags"] = sanitize_summary(payload.get("tags"))

        if self._collection is not None:
            start = time.perf_counter()
            status = "ok"
            try:
                self._collection.insert_one(payload)
                record_event_store_write("mongo", "ok")
                return payload
            except Exception as exc:  # pragma: no cover - network and env dependent
                status = "error"
                record_event_store_write("mongo", "error")
                record_event_store_fallback("mongo_write_failed")
                logger.warning(f"写入 observability_events 失败，降级为本地 JSONL：{exc}")
            finally:
                record_mongo_operation("insert_one", _COLLECTION_NAME, status, time.perf_counter() - start)
        else:
            record_event_store_fallback("mongo_unavailable")

        try:
            self._append_fallback(payload)
            record_event_store_write("jsonl", "ok")
        except Exception as exc:  # pragma: no cover - filesystem dependent
            record_event_store_write("jsonl", "error")
            record_event_store_fallback("jsonl_write_failed")
            logger.warning(f"事件写入本地 JSONL 失败：{exc}")
        return payload

    def _match_document(self, document: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """判断文档是否命中过滤条件。"""
        for key, value in filters.items():
            if value in (None, ""):
                continue
            if document.get(key) != value:
                return False
        return True

    def _query_fallback(self, filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """从本地 JSONL 读取并筛选事件。"""
        items = [
            item
            for item in self._load_fallback()
            if self._match_document(item, filters)
        ]
        items.sort(key=lambda item: (item.get("ts", ""), item.get("sequence", 0), item.get("_id", "")))
        return items[:limit]

    def query(self, filters: Dict[str, Any], limit: int = 200) -> List[Dict[str, Any]]:
        """按过滤条件读取事件列表。"""
        self._ensure_collection()
        if self._collection is not None:
            start = time.perf_counter()
            status = "ok"
            try:
                cursor = self._collection.find(filters).sort(
                    [("ts", 1), ("sequence", 1), ("_id", 1)]
                ).limit(limit)
                return [self._normalize_document(item) for item in cursor]
            except Exception as exc:  # pragma: no cover - network and env dependent
                status = "error"
                logger.warning(f"查询 observability_events 失败，降级为本地 JSONL：{exc}")
            finally:
                record_mongo_operation("find", _COLLECTION_NAME, status, time.perf_counter() - start)
        return self._query_fallback(filters, limit)

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """根据事件 ID 查询单条事件。"""
        if not event_id:
            return None
        self._ensure_collection()
        if self._collection is not None:
            start = time.perf_counter()
            status = "ok"
            try:
                document = self._collection.find_one({"_id": event_id})
                return self._normalize_document(document) if document else None
            except Exception as exc:  # pragma: no cover - network and env dependent
                status = "error"
                logger.warning(f"查询事件详情失败，降级为本地 JSONL：{exc}")
            finally:
                record_mongo_operation("find_one", _COLLECTION_NAME, status, time.perf_counter() - start)
        for item in self._load_fallback():
            if item.get("_id") == event_id:
                return item
        return None

    def _normalize_document(self, document: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """将 Mongo 返回文档转换为可 JSON 序列化格式。"""
        if not document:
            return None
        payload = dict(document)
        payload["_id"] = str(payload.get("_id", ""))
        if payload.get("summary") is not None:
            payload["summary"] = sanitize_summary(payload["summary"])
        if payload.get("tags") is not None:
            payload["tags"] = sanitize_summary(payload["tags"])
        return payload


_STORE: Optional[_EventStore] = None


def _get_store() -> _EventStore:
    """获取事件存储单例。"""
    global _STORE
    if _STORE is None:
        _STORE = _EventStore()
    return _STORE


def emit_event(
    event_type: str,
    graph: str,
    node_name: Optional[str] = None,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    kb_id: Optional[str] = None,
    request_id: Optional[str] = None,
    status: str = "ok",
    summary: Optional[Dict[str, Any]] = None,
    input_ref: Optional[str] = None,
    output_ref: Optional[str] = None,
    error: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """写入一条业务事件。

    Args:
        event_type: 事件类型。
        graph: 业务图名称。
        node_name: LangGraph 节点名称。
        session_id: 查询会话 ID。
        task_id: 导入任务 ID。
        tenant_id: 租户标识。
        kb_id: 知识库标识。
        request_id: HTTP 请求 ID。
        status: 事件状态。
        summary: 事件摘要。
        input_ref: 输入快照引用。
        output_ref: 输出快照引用。
        error: 错误信息。
        tags: 事件标签列表。

    Returns:
        已写入或已降级落盘的事件文档。
    """
    context = get_context()
    span_context = trace.get_current_span().get_span_context()
    trace_id = f"{span_context.trace_id:032x}" if span_context and span_context.is_valid else ""
    span_id = f"{span_context.span_id:016x}" if span_context and span_context.is_valid else ""
    document = {
        "_id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "span_id": span_id,
        "langfuse_trace_id": context.get("langfuse_trace_id", ""),
        "langfuse_parent_span_id": context.get("langfuse_parent_span_id", ""),
        "langfuse_trace_url": context.get("langfuse_trace_url", ""),
        "request_id": request_id or context.get("request_id", ""),
        "session_id": session_id or context.get("session_id", ""),
        "task_id": task_id or context.get("task_id", ""),
        "tenant_id": tenant_id or context.get("tenant_id", "default"),
        "kb_id": kb_id or context.get("kb_id", "default"),
        "graph": graph or context.get("graph", ""),
        "node_name": node_name or context.get("node_name", ""),
        "event_type": event_type,
        "status": status,
        "summary": summary or {},
        "input_ref": input_ref,
        "output_ref": output_ref,
        "error": sanitize_summary(error or {}),
        "tags": tags or [],
    }
    if not is_observability_enabled():
        return document
    return _get_store().write(document)


def get_timeline_by_session(
    session_id: str,
    limit: int = 200,
    tenant_id: str = "default",
    kb_id: str = "default",
) -> List[Dict[str, Any]]:
    """按会话 ID 获取事件时间线。

    Args:
        session_id: 查询会话 ID。
        limit: 返回条数上限。
        tenant_id: 租户标识。
        kb_id: 知识库标识。

    Returns:
        事件列表。
    """
    if not is_observability_enabled():
        return []
    return _get_store().query(
        {"session_id": session_id, "tenant_id": tenant_id, "kb_id": kb_id},
        limit,
    )


def get_timeline_by_task(
    task_id: str,
    limit: int = 200,
    tenant_id: str = "default",
    kb_id: str = "default",
) -> List[Dict[str, Any]]:
    """按任务 ID 获取事件时间线。

    Args:
        task_id: 导入任务 ID。
        limit: 返回条数上限。
        tenant_id: 租户标识。
        kb_id: 知识库标识。

    Returns:
        事件列表。
    """
    if not is_observability_enabled():
        return []
    return _get_store().query(
        {"task_id": task_id, "tenant_id": tenant_id, "kb_id": kb_id},
        limit,
    )


def get_timeline_by_trace(trace_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    """按 trace ID 获取事件时间线。

    Args:
        trace_id: OpenTelemetry trace ID。
        limit: 返回条数上限。

    Returns:
        事件列表。
    """
    if not is_observability_enabled():
        return []
    return _get_store().query({"trace_id": trace_id}, limit)


def get_event_by_id(event_id: str) -> Optional[Dict[str, Any]]:
    """根据事件 ID 获取单条事件。

    Args:
        event_id: 事件 ID。

    Returns:
        事件文档，未找到时返回 None。
    """
    if not is_observability_enabled():
        return None
    return _get_store().get_event(event_id)
