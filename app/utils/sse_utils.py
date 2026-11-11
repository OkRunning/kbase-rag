"""SSE 会话队列和事件推送工具。"""

import json
import os
import queue
import asyncio
import threading
import time
from typing import Dict, Any, Optional
from fastapi import Request
from app.observability.events import emit_event
from app.observability.metrics import (
    record_sse_connection,
    record_sse_event,
    record_sse_queue_depth,
)
from app.observability.replay_context import get_replay_context, is_dry_run


class SSEEvent:
    """定义前端 SSE 事件类型。"""

    READY = "ready"         # 连接建立
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案
    ERROR = "error"         # 错误信息
    CLOSE = "__close__"     # 关闭连接信号


# 全局 SSE 会话队列存储
# Key: session_id, Value: queue.Queue
_session_stream: Dict[str, queue.Queue] = {}
_sse_delta_batches: Dict[str, Dict[str, Any]] = {}
_sse_delta_lock = threading.Lock()
_DELTA_FLUSH_EVENTS = {SSEEvent.FINAL, SSEEvent.ERROR, SSEEvent.CLOSE}


def _get_int_env(name: str, default: int) -> int:
    """读取整数环境变量。

    Args:
        name: 环境变量名称。
        default: 解析失败时使用的默认值。

    Returns:
        解析后的整数值。
    """
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float_env(name: str, default: float) -> float:
    """读取浮点数环境变量。

    Args:
        name: 环境变量名称。
        default: 解析失败时使用的默认值。

    Returns:
        解析后的浮点数值。
    """
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _delta_event_mode() -> str:
    """读取 SSE delta timeline 记录模式。

    Returns:
        delta 记录模式，支持 aggregate、all、none。
    """
    mode = os.getenv("OBSERVABILITY_SSE_DELTA_EVENT_MODE", "aggregate").strip().lower()
    return mode if mode in {"aggregate", "all", "none"} else "aggregate"


def _delta_preview_chars() -> int:
    """读取 delta 预览文本长度。

    Returns:
        预览文本最大字符数。
    """
    return max(_get_int_env("OBSERVABILITY_SSE_DELTA_PREVIEW_CHARS", 200), 20)


def _delta_batch_size() -> int:
    """读取 delta 聚合批次大小。

    Returns:
        触发批量 flush 的 delta 数量。
    """
    return max(_get_int_env("OBSERVABILITY_SSE_DELTA_BATCH_SIZE", 50), 1)


def _delta_flush_seconds() -> float:
    """读取 delta 聚合最长 flush 间隔。

    Returns:
        触发批量 flush 的秒数。
    """
    return max(_get_float_env("OBSERVABILITY_SSE_DELTA_FLUSH_SECONDS", 2.0), 0.1)


def _extract_delta_text(data: Dict[str, Any]) -> str:
    """从 SSE delta payload 中提取增量文本。

    Args:
        data: SSE payload。

    Returns:
        delta 文本；不存在时返回空字符串。
    """
    if not isinstance(data, dict):
        return str(data or "")
    value = data.get("delta", "")
    return value if isinstance(value, str) else str(value or "")


def _append_preview(current: str, delta: str, limit: int) -> str:
    """追加并截断预览文本。

    Args:
        current: 当前预览文本。
        delta: 本次追加的文本。
        limit: 最大保留字符数。

    Returns:
        截断后的预览文本。
    """
    if len(current) >= limit:
        return current
    return (current + delta)[:limit]


def _pop_delta_batch(session_id: str, reason: str) -> Optional[Dict[str, Any]]:
    """取出并构造指定 session 的 delta 聚合事件。

    Args:
        session_id: 会话 ID。
        reason: 触发 flush 的原因。

    Returns:
        待写入 timeline 的事件参数；没有待 flush 数据时返回 None。
    """
    with _sse_delta_lock:
        batch = _sse_delta_batches.pop(session_id, None)
    if not batch or batch.get("delta_count", 0) <= 0:
        return None

    event_type = "dry_run.sse.delta_batch" if batch.get("dry_run") else "sse.delta_batch"
    tags = ["sse", "delta", "batch"]
    if batch.get("dry_run"):
        tags.insert(0, "dry_run")
    summary = {
        "event": SSEEvent.DELTA,
        "delta_count": batch.get("delta_count", 0),
        "delta_chars": batch.get("delta_chars", 0),
        "preview": batch.get("preview", ""),
        "last_delta_preview": batch.get("last_delta_preview", ""),
        "flush_reason": reason,
        "duration_ms": round((batch.get("updated_at", 0.0) - batch.get("started_at", 0.0)) * 1000, 2),
    }
    if batch.get("replay_id"):
        summary["replay_id"] = batch.get("replay_id")
    return {
        "event_type": event_type,
        "summary": summary,
        "tags": tags,
    }


def _flush_delta_batch(session_id: str, reason: str) -> None:
    """将指定 session 的 delta 聚合结果写入 timeline。

    Args:
        session_id: 会话 ID。
        reason: 触发 flush 的原因。

    Returns:
        None.
    """
    event = _pop_delta_batch(session_id, reason)
    if event is None:
        return
    emit_event(
        event["event_type"],
        graph="query",
        session_id=session_id,
        status="ok",
        summary=event["summary"],
        tags=event["tags"],
    )


def _record_delta_timeline(session_id: str, data: Dict[str, Any], dry_run: bool = False) -> None:
    """按配置记录 SSE delta timeline。

    Args:
        session_id: 会话 ID。
        data: SSE delta payload。
        dry_run: 是否处于 dry-run 副作用隔离模式。

    Returns:
        None.
    """
    mode = _delta_event_mode()
    if mode == "none":
        return

    replay_id = get_replay_context().get("replay_id") if dry_run else ""
    if mode == "all":
        emit_event(
            "dry_run.sse.skipped" if dry_run else "sse.sent",
            graph="query",
            session_id=session_id,
            status="ok",
            summary={"event": SSEEvent.DELTA, "payload": data, "replay_id": replay_id},
            tags=(["dry_run", "sse", SSEEvent.DELTA] if dry_run else ["sse", SSEEvent.DELTA]),
        )
        return

    delta = _extract_delta_text(data)
    now = time.monotonic()
    preview_limit = _delta_preview_chars()
    should_flush = False
    flush_reason = "batch_size"

    # 流式输出会高频产生 token，这里只在内存中累积摘要，避免 timeline 被逐 token 冲爆。
    with _sse_delta_lock:
        batch = _sse_delta_batches.setdefault(
            session_id,
            {
                "delta_count": 0,
                "delta_chars": 0,
                "preview": "",
                "last_delta_preview": "",
                "started_at": now,
                "updated_at": now,
                "dry_run": dry_run,
                "replay_id": replay_id,
            },
        )
        batch["delta_count"] += 1
        batch["delta_chars"] += len(delta)
        batch["preview"] = _append_preview(batch.get("preview", ""), delta, preview_limit)
        batch["last_delta_preview"] = delta[-preview_limit:]
        batch["updated_at"] = now
        batch["dry_run"] = dry_run
        batch["replay_id"] = replay_id or batch.get("replay_id", "")
        if batch["delta_count"] >= _delta_batch_size():
            should_flush = True
        elif now - batch.get("started_at", now) >= _delta_flush_seconds():
            should_flush = True
            flush_reason = "interval"

    if should_flush:
        _flush_delta_batch(session_id, flush_reason)

def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """获取指定 session 的 SSE 队列。

    Args:
        session_id: 会话 ID。

    Returns:
        SSE 队列；不存在时返回 None。
    """
    return _session_stream.get(session_id)

def create_sse_queue(session_id: str) -> "queue.Queue":
    """创建并注册新的 SSE 队列。

    Args:
        session_id: 会话 ID。

    Returns:
        新创建的 SSE 队列。
    """
    print(f"[SSE] Creating queue for session: {session_id}")
    q = queue.Queue()
    _session_stream[session_id] = q
    record_sse_queue_depth(0)
    return q

def remove_sse_queue(session_id: str):
    """移除指定 session 的 SSE 队列。

    Args:
        session_id: 会话 ID。

    Returns:
        None.
    """
    print(f"[SSE] Removing queue for session: {session_id}")
    _flush_delta_batch(session_id, "queue_removed")
    _session_stream.pop(session_id, None)

def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    # print(f"[SSE] Packing event: {event}, payload: {payload[:50]}...")
    return f"event: {event}\ndata: {payload}\n\n"

def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """向指定 session 推送 SSE 事件。

    Args:
        session_id: 会话 ID。
        event: SSE 事件类型。
        data: SSE 事件数据。

    Returns:
        None.
    """
    if is_dry_run():
        if event == SSEEvent.DELTA:
            _record_delta_timeline(session_id, data, dry_run=True)
            return
        if event in _DELTA_FLUSH_EVENTS:
            _flush_delta_batch(session_id, event)
        replay_context = get_replay_context()
        emit_event(
            "dry_run.sse.skipped",
            graph="query",
            session_id=session_id,
            status="ok",
            summary={
                "event": event,
                "payload": data,
                "replay_id": replay_context.get("replay_id"),
            },
            tags=["dry_run", "sse", event],
        )
        return

    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        # print(f"[SSE] Pushing to session {session_id}: {event}")
        stream_queue.put({"event": event, "data": data})
        # 只有真正进入队列的事件才计入发送指标，避免无连接时虚增。
        record_sse_event(event)
        record_sse_queue_depth(stream_queue.qsize())
        if event == SSEEvent.DELTA:
            _record_delta_timeline(session_id, data)
            return
        if event in _DELTA_FLUSH_EVENTS:
            _flush_delta_batch(session_id, event)
        emit_event(
            "sse.sent",
            graph="query",
            session_id=session_id,
            status="ok",
            summary={"event": event, "payload": data},
            tags=["sse", event],
        )
    else:
        print(f"[SSE] Warning: No queue found for session {session_id} when pushing {event}")
        record_sse_event("queue_missing")
        emit_event(
            "sse.queue_missing",
            graph="query",
            session_id=session_id,
            status="error",
            summary={"event": event},
            tags=["sse", "queue_missing"],
        )

async def sse_generator(session_id: str, request: Request):
    """生成 FastAPI StreamingResponse 使用的 SSE 数据流。

    Args:
        session_id: 会话 ID。
        request: FastAPI 请求对象，用于检测客户端断开。

    Yields:
        已打包的 SSE 文本消息。
    """
    print(f"[SSE] Generator started for session: {session_id}")
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        # 如果没有对应的队列，直接结束
        print(f"[SSE] Error: Queue not found for session {session_id}. Available sessions: {list(_session_stream.keys())}")
        record_sse_event("queue_missing")
        emit_event(
            "sse.queue_missing",
            graph="query",
            session_id=session_id,
            status="error",
            summary={"phase": "generator_start"},
            tags=["sse", "queue_missing"],
        )
        return

    loop = asyncio.get_running_loop()
    end_status = "completed"
    record_sse_connection("active", 1)
    try:
        # 发送连接建立信号
        print(f"[SSE] Sending ready signal for {session_id}")
        record_sse_event("connected")
        emit_event(
            "sse.connected",
            graph="query",
            session_id=session_id,
            status="ok",
            summary={},
            tags=["sse", "connected"],
        )
        yield _sse_pack("ready", {})

        while True:
            # 若客户端断开，尽快退出
            if await request.is_disconnected():
                print(f"[SSE] Client disconnected: {session_id}")
                print("-----------------------断开连接--------------------")
                end_status = "client_disconnected"
                record_sse_event("disconnected")
                emit_event(
                    "sse.disconnected",
                    graph="query",
                    session_id=session_id,
                    status="cancelled",
                    summary={"reason": "request_disconnected"},
                    tags=["sse", "disconnected"],
                )
                break

            try:
                # 使用 run_in_executor 避免阻塞 async 事件循环
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                # print(f"[SSE] Queue empty for {session_id}, waiting...")
                continue

            event = msg.get("event")
            data = msg.get("data")
            
            # print(f"[SSE] Yielding event {event} for {session_id}")

            # 特殊关闭事件
            if event == "__close__":
                print(f"[SSE] Closing signal received for {session_id}")
                end_status = "closed"
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        print(f"[SSE] Client disconnected (Cancelled/Reset/Pipe): {session_id}")
        end_status = "client_disconnected"
        record_sse_event("disconnected")
        emit_event(
            "sse.disconnected",
            graph="query",
            session_id=session_id,
            status="cancelled",
            summary={"reason": "cancelled_or_reset"},
            tags=["sse", "disconnected"],
        )
        # 生成器被取消/对端断开：静默退出
        return
    except Exception as e:
        print(f"[SSE] Exception in generator for {session_id}: {e}")
        end_status = "error"
        record_sse_event("generator_failed")
        emit_event(
            "sse.generator.failed",
            graph="query",
            session_id=session_id,
            status="error",
            summary={},
            error={"type": type(e).__name__, "message": str(e)},
            tags=["sse", "generator_failed"],
        )
    finally:
        print(f"[SSE] Generator finished for {session_id}")
        record_sse_connection("active", -1)
        if end_status in {"closed", "completed"}:
            record_sse_event("closed")
            emit_event(
                "sse.finalized",
                graph="query",
                session_id=session_id,
                status="ok",
                summary={"status": end_status},
                tags=["sse", "finalized"],
            )
        # 清理资源
        remove_sse_queue(session_id)
