"""维护查询和导入任务的运行状态。

该模块封装任务状态、运行中节点、已完成节点和任务结果的内存/Redis 双写
逻辑，并提供卡住任务扫描所需的任务快照。
"""

import threading
import time
from typing import Any, Dict, List, Optional

from app.clients.redis_utils import get_redis_client
from app.conf.redis_config import redis_config
from .sse_utils import push_to_session
from app.observability.context import get_context
from app.observability.events import emit_event
from app.observability.replay_context import get_replay_context, is_dry_run

# ---------------------------
# 内存态任务追踪（单进程）
# ---------------------------
# key: task_id
# value: 节点名列表（原始英文/节点ID）
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}

# key: task_id
# value: status 字符串（如 pending/processing/completed/failed）
_tasks_status: Dict[str, str] = {}

# key: task_id
# value: 任务结果（例如 query 的 answer）
_tasks_result: Dict[str, Dict[str, str]] = {}
_tasks_meta: Dict[str, Dict[str, str]] = {}
_tasks_lock = threading.RLock()

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 节点名 -> 中文名映射（用于前端展示）
# 说明：这里的 key 应与 LangGraph 的 add_node("xxx", ...) 中的节点名一致。
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_item_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_bge_embedding": "向量生成",
    "node_import_kg": "导入知识图谱",
    "node_import_milvus": "导入向量库",
    "__end__": "处理完成",
    "END": "处理完成",
    # --- Query 流程节点（kb/query_process/main_graph.py）---
    "node_item_name_confirm": "确认问题产品",
    "node_retrieval_planner": "生成查询检索计划",
    "node_query_understanding": "查询理解",
    "node_entity_resolution": "实体解析",
    "node_clarification_decision": "追问决策",
    "node_answer_output": "生成答案",
    "node_rerank": "重排序",
    "node_rrf": "倒排融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "切片搜索",
    "node_search_embedding_hyde": "切片搜索(假设性文档)",
    "node_multi_search": "多路搜索",
    "node_join": "多路搜索合并",
    "node_retrieval_gate": "本地召回质量门控",
    "node_evidence_gate": "证据门控",
    "node_rerank_gate": "ReRank跳过门控"
}


def _ensure_task(task_id: str) -> None:
    """确保 task_id 对应的数据结构已初始化。

    Args:
        task_id: 任务 ID。

    Returns:
        None.
    """
    with _tasks_lock:
        if task_id not in _tasks_running_list:
            _tasks_running_list[task_id] = []
        if task_id not in _tasks_done_list:
            _tasks_done_list[task_id] = []
        if task_id not in _tasks_result:
            _tasks_result[task_id] = {}
        if task_id not in _tasks_meta:
            now = str(time.time())
            _tasks_meta[task_id] = {"created_at": now, "updated_at": now}


def _to_cn(node_name: str) -> str:
    """将节点名转换为中文展示名；若无映射则返回原名。

    Args:
        node_name: 节点名称。

    Returns:
        中文展示名或原始节点名。
    """
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def _redis_keys(task_id: str) -> Dict[str, str]:
    """生成任务状态相关的 Redis key。

    Args:
        task_id: 任务 ID。

    Returns:
        Redis key 字典。
    """
    base = f"task:{task_id}"
    return {
        "meta": f"{base}:meta",
        "running": f"{base}:running",
        "done": f"{base}:done",
        "result": f"{base}:result",
    }


def _refresh_task_ttl(task_id: str) -> None:
    """刷新任务状态相关 Redis key 的过期时间。

    Args:
        task_id: 任务 ID。

    Returns:
        None.
    """
    client = get_redis_client()
    if client is None:
        return
    keys = _redis_keys(task_id)
    ttl = redis_config.task_ttl_seconds
    for key in keys.values():
        client.expire(key, ttl)


def _safe_redis_list(task_id: str, key_name: str) -> Optional[List[str]]:
    """安全读取 Redis 列表。

    Args:
        task_id: 任务 ID。
        key_name: Redis key 类型。

    Returns:
        节点名称列表；Redis 不可用时返回 None。
    """
    client = get_redis_client()
    if client is None:
        return None
    try:
        return client.lrange(_redis_keys(task_id)[key_name], 0, -1)
    except Exception:
        return None


def _safe_redis_get(task_id: str, key: str) -> Optional[str]:
    """安全读取 Redis hash 字段。

    Args:
        task_id: 任务 ID。
        key: hash 字段名称。

    Returns:
        字段值；Redis 不可用时返回 None。
    """
    client = get_redis_client()
    if client is None:
        return None
    try:
        return client.hget(_redis_keys(task_id)["meta"], key)
    except Exception:
        return None


def _safe_redis_hgetall(task_id: str) -> Optional[Dict[str, str]]:
    """安全读取 Redis 任务元信息。

    Args:
        task_id: 任务 ID。

    Returns:
        任务元信息；Redis 不可用时返回 None。
    """
    client = get_redis_client()
    if client is None:
        return None
    try:
        return client.hgetall(_redis_keys(task_id)["meta"])
    except Exception:
        return None


def _update_memory_meta(task_id: str, **fields: str) -> None:
    """更新内存任务元信息。

    Args:
        task_id: 任务 ID。
        **fields: 待更新的元信息字段。

    Returns:
        None.
    """
    _ensure_task(task_id)
    with _tasks_lock:
        meta = _tasks_meta.setdefault(task_id, {})
        now = str(time.time())
        meta.setdefault("created_at", now)
        meta["updated_at"] = now
        for key, value in fields.items():
            if value is not None:
                meta[key] = str(value)


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """添加正在运行的节点任务。

    Args:
        task_id: 任务 ID。
        node_name: 节点名称。
        is_stream: 是否需要推送 SSE 进度。

    Returns:
        None.
    """
    if is_dry_run():
        replay_context = get_replay_context()
        emit_event(
            "dry_run.task.running_skipped",
            graph=get_context().get("graph", "") or "unknown",
            node_name=node_name,
            task_id=task_id,
            status="ok",
            summary={"node_name": node_name, "is_stream": is_stream, "replay_id": replay_context.get("replay_id")},
            tags=["dry_run", "task", "node.running"],
        )
        return

    client = get_redis_client()
    if client is not None:
        try:
            keys = _redis_keys(task_id)
            # Redis 列表保留节点进入顺序，同时写入前去重，避免前端重复展示。
            if node_name not in client.lrange(keys["running"], 0, -1):
                client.rpush(keys["running"], node_name)
            now = str(time.time())
            client.hset(
                keys["meta"],
                mapping={
                    "updated_at": now,
                    "created_at": client.hget(keys["meta"], "created_at") or now,
                    "graph": get_context().get("graph", "") or "unknown",
                },
            )
            _refresh_task_ttl(task_id)
        except Exception:
            client = None

    if client is None:
        _ensure_task(task_id)
        with _tasks_lock:
            running = _tasks_running_list[task_id]
            if node_name not in running:
                running.append(node_name)
        _update_memory_meta(task_id, graph=get_context().get("graph", "") or "unknown")

    # 运行态节点变化要沉淀为事件，便于后续按 task/session 回放整个进度。
    emit_event(
        "node.running",
        graph=get_context().get("graph", "") or "unknown",
        node_name=node_name,
        task_id=task_id,
        status="ok",
        summary={"node_name": node_name, "is_stream": is_stream},
        tags=["task", "node.running"],
    )
    if is_stream:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """添加已完成的节点任务。

    Args:
        task_id: 任务 ID。
        node_name: 节点名称。
        is_stream: 是否需要推送 SSE 进度。

    Returns:
        None.
    """
    if is_dry_run():
        replay_context = get_replay_context()
        emit_event(
            "dry_run.task.done_skipped",
            graph=get_context().get("graph", "") or "unknown",
            node_name=node_name,
            task_id=task_id,
            status="ok",
            summary={"node_name": node_name, "is_stream": is_stream, "replay_id": replay_context.get("replay_id")},
            tags=["dry_run", "task", "node.done"],
        )
        return

    client = get_redis_client()
    if client is not None:
        try:
            keys = _redis_keys(task_id)
            # 状态转换：节点完成时必须从 running 移除，再按完成顺序写入 done。
            client.lrem(keys["running"], 0, node_name)
            if node_name not in client.lrange(keys["done"], 0, -1):
                client.rpush(keys["done"], node_name)
            now = str(time.time())
            client.hset(
                keys["meta"],
                mapping={
                    "updated_at": now,
                    "created_at": client.hget(keys["meta"], "created_at") or now,
                    "graph": get_context().get("graph", "") or "unknown",
                },
            )
            _refresh_task_ttl(task_id)
        except Exception:
            client = None

    if client is None:
        _ensure_task(task_id)
        with _tasks_lock:
            running = _tasks_running_list[task_id]
            _tasks_running_list[task_id] = [n for n in running if n != node_name]
            done = _tasks_done_list[task_id]
            if node_name not in done:
                done.append(node_name)
        _update_memory_meta(task_id, graph=get_context().get("graph", "") or "unknown")

    # 任务完成事件和前端进度分离存储，避免后续只剩内存态可见。
    emit_event(
        "node.done",
        graph=get_context().get("graph", "") or "unknown",
        node_name=node_name,
        task_id=task_id,
        status="ok",
        summary={"node_name": node_name, "is_stream": is_stream},
        tags=["task", "node.done"],
    )
    if is_stream:
        task_push_queue(task_id)


def set_task_result(task_id: str, key: str, value: str) -> None:
    """存储任务结果字段。

    Args:
        task_id: 任务 ID。
        key: 结果字段名称。
        value: 结果字段值。

    Returns:
        None.
    """
    if is_dry_run():
        replay_context = get_replay_context()
        emit_event(
            "dry_run.task.result_skipped",
            graph=get_context().get("graph", "") or "unknown",
            task_id=task_id,
            status="ok",
            summary={"key": key, "value_chars": len(value or ""), "replay_id": replay_context.get("replay_id")},
            tags=["dry_run", "task", "result"],
        )
        return

    client = get_redis_client()
    if client is not None:
        try:
            keys = _redis_keys(task_id)
            client.hset(keys["result"], key, value or "")
            now = str(time.time())
            client.hset(
                keys["meta"],
                mapping={
                    "updated_at": now,
                    "created_at": client.hget(keys["meta"], "created_at") or now,
                    "graph": get_context().get("graph", "") or "unknown",
                },
            )
            _refresh_task_ttl(task_id)
            return
        except Exception:
            pass

    _ensure_task(task_id)
    with _tasks_lock:
        _tasks_result[task_id][key] = value
    _update_memory_meta(task_id, graph=get_context().get("graph", "") or "unknown")


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """获取任务结果字段。

    Args:
        task_id: 任务 ID。
        key: 结果字段名称。
        default: 结果不存在时返回的默认值。

    Returns:
        任务结果字段值。
    """
    client = get_redis_client()
    if client is not None:
        try:
            value = client.hget(_redis_keys(task_id)["result"], key)
            if value is not None:
                return value
        except Exception:
            pass

    _ensure_task(task_id)
    with _tasks_lock:
        return _tasks_result.get(task_id, {}).get(key, default)


def get_task_status(task_id: str) -> str:
    """获取当前任务状态。

    Args:
        task_id: 任务 ID。

    Returns:
        状态名称；如果未设置过则返回空字符串。
    """
    value = _safe_redis_get(task_id, "status")
    if value is not None:
        return value
    with _tasks_lock:
        return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    """获取已完成节点列表。

    Args:
        task_id: 任务 ID。

    Returns:
        已完成节点中文展示名列表。
    """
    redis_done = _safe_redis_list(task_id, "done")
    if redis_done is not None:
        return [_to_cn(n) for n in redis_done]
    _ensure_task(task_id)
    with _tasks_lock:
        done = list(_tasks_done_list.get(task_id, []))
    return [_to_cn(n) for n in done]


def get_running_task_list(task_id: str) -> List[str]:
    """获取正在运行节点列表。

    Args:
        task_id: 任务 ID。

    Returns:
        正在运行节点中文展示名列表。
    """
    redis_running = _safe_redis_list(task_id, "running")
    if redis_running is not None:
        return [_to_cn(n) for n in redis_running]
    _ensure_task(task_id)
    with _tasks_lock:
        running = list(_tasks_running_list.get(task_id, []))
    return [_to_cn(n) for n in running]


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    """更新任务状态。

    Args:
        task_id: 任务 ID。
        status_name: 状态名称。
        push_queue: 是否需要推送 SSE 进度。

    Returns:
        None.
    """
    if is_dry_run():
        replay_context = get_replay_context()
        emit_event(
            "dry_run.task.status_skipped",
            graph=get_context().get("graph", "") or "unknown",
            task_id=task_id,
            status=status_name or "unknown",
            summary={"current_status": status_name, "replay_id": replay_context.get("replay_id")},
            tags=["dry_run", "task", "task.status_changed"],
        )
        return

    previous = get_task_status(task_id)
    client = get_redis_client()
    if client is not None:
        try:
            keys = _redis_keys(task_id)
            now = str(time.time())
            # 任务元信息使用 hash 存储，便于后续按字段扩展而不改接口。
            client.hset(
                keys["meta"],
                mapping={
                    "status": status_name,
                    "updated_at": now,
                    "created_at": client.hget(keys["meta"], "created_at") or now,
                    "graph": get_context().get("graph", "") or "unknown",
                },
            )
            _refresh_task_ttl(task_id)
        except Exception:
            client = None

    if client is None:
        with _tasks_lock:
            _tasks_status[task_id] = status_name
        _update_memory_meta(task_id, status=status_name, graph=get_context().get("graph", "") or "unknown")
    # 状态切换是时间线里的关键节点，必须写成事件方便排障。
    emit_event(
        "task.status_changed",
        graph=get_context().get("graph", "") or "unknown",
        task_id=task_id,
        status=status_name or "unknown",
        summary={"previous_status": previous, "current_status": status_name},
        tags=["task", "task.status_changed"],
    )
    if push_queue:
        task_push_queue(task_id)


def task_push_queue(task_id: str):
    """推送任务进度到 SSE 队列。

    Args:
        task_id: 任务 ID。

    Returns:
        None.
    """
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })




def reset_task_progress(task_id: str) -> None:
    """重置指定任务的进度态，供同一 session 发起新一轮查询时调用。

    仅清理运行中节点、已完成节点列表和上一轮结果，保留任务元信息
    （如 created_at），避免多轮对话复用 session_id 时 done_list 跨轮累加。

    Args:
        task_id: 任务 ID，查询场景即 session_id。

    Returns:
        None.
    """
    client = get_redis_client()
    if client is not None:
        try:
            keys = _redis_keys(task_id)
            # 状态转换：新一轮查询开始前，必须清空上轮的 running/done 进度
            # 与上一轮 result，避免前端展示上一轮残留节点或旧答案。
            client.delete(keys["running"], keys["done"], keys["result"])
            now = str(time.time())
            client.hset(
                keys["meta"],
                mapping={
                    "status": TASK_STATUS_PENDING,
                    "updated_at": now,
                    "created_at": client.hget(keys["meta"], "created_at") or now,
                },
            )
            _refresh_task_ttl(task_id)
        except Exception:
            client = None

    if client is None:
        _ensure_task(task_id)
        with _tasks_lock:
            # 进度列表清空，准备承载本轮节点的进入/完成顺序。
            _tasks_running_list[task_id] = []
            _tasks_done_list[task_id] = []
            _tasks_result[task_id] = {}
            _tasks_status[task_id] = TASK_STATUS_PENDING
        _update_memory_meta(task_id, status=TASK_STATUS_PENDING)

#
def clear_task(task_id: str):
    """清理任务状态。

    Args:
        task_id: 任务 ID。

    Returns:
        None.
    """
    client = get_redis_client()
    if client is not None:
        try:
            keys = _redis_keys(task_id)
            client.delete(*keys.values())
        except Exception:
            pass
    with _tasks_lock:
        _tasks_running_list.pop(task_id, None)
        _tasks_done_list.pop(task_id, None)
        _tasks_status.pop(task_id, None)
        _tasks_result.pop(task_id, None)
        _tasks_meta.pop(task_id, None)


def _to_float(value: Any) -> float:
    """将任务元信息中的时间字段转换为浮点数。

    Args:
        value: 待转换的值。

    Returns:
        转换后的时间戳；无法转换时返回 0。
    """
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _build_task_snapshot(task_id: str, meta: Dict[str, str]) -> Dict[str, Any]:
    """构造任务扫描快照。

    Args:
        task_id: 任务 ID。
        meta: 任务元信息。

    Returns:
        包含状态、节点列表和年龄的任务快照。
    """
    now = time.time()
    created_at = _to_float(meta.get("created_at"))
    updated_at = _to_float(meta.get("updated_at"))
    status = meta.get("status") or get_task_status(task_id)
    graph = meta.get("graph") or "unknown"
    return {
        "task_id": task_id,
        "graph": graph,
        "status": status or "unknown",
        "created_at": created_at,
        "updated_at": updated_at,
        "age_seconds": max(now - created_at, 0.0) if created_at else 0.0,
        "idle_seconds": max(now - updated_at, 0.0) if updated_at else 0.0,
        "running_list": get_running_task_list(task_id),
        "done_list": get_done_task_list(task_id),
    }


def list_task_snapshots() -> List[Dict[str, Any]]:
    """列出当前进程或 Redis 中可见的任务快照。

    Returns:
        任务快照列表。
    """
    client = get_redis_client()
    snapshots: List[Dict[str, Any]] = []
    if client is not None:
        try:
            # Redis scan 避免一次性拉取大量 key，适合后台卡住任务扫描。
            for key in client.scan_iter("task:*:meta", count=100):
                task_id = key[len("task:"):-len(":meta")]
                meta = client.hgetall(key)
                snapshots.append(_build_task_snapshot(task_id, meta))
            return snapshots
        except Exception:
            pass

    with _tasks_lock:
        task_ids = set(_tasks_status) | set(_tasks_running_list) | set(_tasks_done_list) | set(_tasks_result)
        meta_by_task = {task_id: dict(_tasks_meta.get(task_id, {})) for task_id in task_ids}
    return [_build_task_snapshot(task_id, meta) for task_id, meta in meta_by_task.items()]


def find_stuck_tasks(query_timeout_seconds: int = 600, import_timeout_seconds: int = 1800) -> List[Dict[str, Any]]:
    """查找可能卡住的查询或导入任务。

    Args:
        query_timeout_seconds: 查询 processing 最大空闲秒数。
        import_timeout_seconds: 导入 processing 最大空闲秒数。

    Returns:
        疑似卡住任务列表。
    """
    stuck: List[Dict[str, Any]] = []
    for snapshot in list_task_snapshots():
        status = snapshot.get("status")
        graph = snapshot.get("graph")
        timeout = import_timeout_seconds if graph == "import" else query_timeout_seconds
        if status == TASK_STATUS_PROCESSING and snapshot.get("idle_seconds", 0.0) >= timeout:
            stuck.append(snapshot)
    return stuck
