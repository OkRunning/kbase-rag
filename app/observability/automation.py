"""提供稳定性自动化后台任务。

该模块负责合成探测、卡住任务扫描和本地容量探测，避免这些稳定性检查散落
在查询或导入 API 入口中。
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.core.logger import logger
from app.observability.events import emit_event
from app.observability.health import build_observability_health, build_readiness
from app.observability.metrics import record_capacity, record_synthetic_probe, record_task_age
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import find_stuck_tasks, list_task_snapshots

_AUTOMATION_THREADS: Dict[str, threading.Thread] = {}
_AUTOMATION_LOCK = threading.Lock()
_LAST_STUCK_EMIT_AT: Dict[str, float] = {}


def _get_bool_env(name: str, default: bool) -> bool:
    """读取布尔环境变量。

    Args:
        name: 环境变量名称。
        default: 解析失败或未配置时使用的默认值。

    Returns:
        布尔配置值。
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_env(name: str, default: int) -> int:
    """读取整数环境变量。

    Args:
        name: 环境变量名称。
        default: 解析失败或未配置时使用的默认值。

    Returns:
        整数配置值。
    """
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _probe_status(payload: Dict[str, Any]) -> str:
    """从健康检查响应中提取探测状态。

    Args:
        payload: 健康检查响应。

    Returns:
        探测状态。
    """
    return str(payload.get("status") or "unknown")


def _record_probe(service: str, probe: str, func) -> Dict[str, Any]:
    """执行并记录单个合成探测。

    Args:
        service: 服务名称。
        probe: 探测名称。
        func: 探测函数。

    Returns:
        探测摘要。
    """
    start = time.perf_counter()
    status = "ok"
    summary: Dict[str, Any] = {"probe": probe}
    try:
        payload = func()
        status = _probe_status(payload)
        summary.update({"status": status, "checks": list((payload.get("checks") or {}).keys())})
    except Exception as exc:  # pragma: no cover - depends on local services
        status = "error"
        summary.update({"status": status, "error_type": type(exc).__name__, "message": str(exc)})
    duration = time.perf_counter() - start
    record_synthetic_probe(service, probe, status, duration)
    return {**summary, "duration_ms": round(duration * 1000, 2)}


def run_synthetic_probes(service: str) -> List[Dict[str, Any]]:
    """执行轻量合成探测。

    Args:
        service: 服务名称。

    Returns:
        探测摘要列表。
    """
    results = [
        _record_probe(service, "readiness", lambda: build_readiness(service)),
        _record_probe(service, "observability", lambda: build_observability_health(service)),
    ]
    worst = "ok"
    if any(item.get("status") == "error" for item in results):
        worst = "error"
    elif any(item.get("status") == "degraded" for item in results):
        worst = "degraded"
    emit_event(
        "synthetic.probe.completed",
        graph=service,
        status=worst,
        summary={"service": service, "results": results},
        tags=["synthetic", "probe", service],
    )
    return results


def _capacity_paths() -> Iterable[tuple[str, Path]]:
    """生成需要探测的本地容量路径。

    Returns:
        路径别名和路径对象迭代器。
    """
    raw_paths = os.getenv("OBSERVABILITY_CAPACITY_PATHS")
    if raw_paths:
        for item in raw_paths.split(","):
            item = item.strip()
            if item:
                yield Path(item).name or item, Path(item)
        return
    yield "project", PROJECT_ROOT
    yield "logs", PROJECT_ROOT / "logs"
    yield "output", PROJECT_ROOT / "output"


def _path_used_ratio(path: Path) -> float:
    """计算路径所在文件系统的已用比例。

    Args:
        path: 待检查路径。

    Returns:
        已用比例，范围 0 到 1。

    Raises:
        OSError: 路径或文件系统状态读取失败时抛出。
    """
    probe_path = path
    # 容量探测只读取文件系统状态，不主动创建配置目录，避免误造 Prometheus/Loki 数据目录。
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    stat = os.statvfs(probe_path)
    total = stat.f_blocks * stat.f_frsize
    available = stat.f_bavail * stat.f_frsize
    if total <= 0:
        return 0.0
    return 1.0 - (available / total)


def record_capacity_probes() -> List[Dict[str, Any]]:
    """记录本地容量探测结果。

    Returns:
        容量探测摘要列表。
    """
    warning_ratio = _get_int_env("OBSERVABILITY_CAPACITY_WARNING_PERCENT", 85) / 100
    critical_ratio = _get_int_env("OBSERVABILITY_CAPACITY_CRITICAL_PERCENT", 95) / 100
    results: List[Dict[str, Any]] = []
    for scope, path in _capacity_paths():
        status = "ok"
        try:
            used_ratio = _path_used_ratio(path)
            if used_ratio >= critical_ratio:
                status = "critical"
            elif used_ratio >= warning_ratio:
                status = "warning"
            record_capacity("filesystem", scope, status, used_ratio)
            results.append({"scope": scope, "path": str(path), "status": status, "used_ratio": round(used_ratio, 4)})
        except Exception as exc:  # pragma: no cover - filesystem dependent
            record_capacity("filesystem", scope, "critical", 1.0)
            results.append(
                {
                    "scope": scope,
                    "path": str(path),
                    "status": "critical",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return results


def scan_stuck_tasks(service: str) -> List[Dict[str, Any]]:
    """扫描并记录疑似卡住任务。

    Args:
        service: 服务名称。

    Returns:
        疑似卡住任务列表。
    """
    query_timeout = _get_int_env("OBSERVABILITY_QUERY_STUCK_SECONDS", 600)
    import_timeout = _get_int_env("OBSERVABILITY_IMPORT_STUCK_SECONDS", 1800)
    now = time.monotonic()
    emit_cooldown = _get_int_env("OBSERVABILITY_STUCK_EVENT_COOLDOWN_SECONDS", 300)
    for snapshot in list_task_snapshots():
        record_task_age(snapshot.get("graph", "unknown"), snapshot.get("status", "unknown"), snapshot.get("age_seconds", 0))
    stuck = find_stuck_tasks(query_timeout_seconds=query_timeout, import_timeout_seconds=import_timeout)
    for snapshot in stuck:
        task_id = str(snapshot.get("task_id") or "")
        previous = _LAST_STUCK_EMIT_AT.get(task_id, 0.0)
        if now - previous < emit_cooldown:
            continue
        _LAST_STUCK_EMIT_AT[task_id] = now
        # 卡住任务事件保留 task_id/session_id，指标只保留低基数 graph/status。
        emit_event(
            "task.stuck.detected",
            graph=str(snapshot.get("graph") or service),
            task_id=task_id if snapshot.get("graph") == "import" else None,
            session_id=task_id if snapshot.get("graph") == "query" else None,
            status="warning",
            summary=snapshot,
            tags=["task", "stuck", service],
        )
    return stuck


def run_automation_once(service: str) -> Dict[str, Any]:
    """执行一轮稳定性自动化。

    Args:
        service: 服务名称。

    Returns:
        本轮执行摘要。
    """
    return {
        "synthetic": run_synthetic_probes(service),
        "stuck_tasks": scan_stuck_tasks(service),
        "capacity": record_capacity_probes(),
    }


def _automation_loop(service: str) -> None:
    """后台执行稳定性自动化循环。

    Args:
        service: 服务名称。

    Returns:
        None.
    """
    interval = max(_get_int_env("OBSERVABILITY_AUTOMATION_INTERVAL_SECONDS", 60), 10)
    while True:
        try:
            run_automation_once(service)
        except Exception as exc:  # pragma: no cover - defensive background loop
            logger.warning(f"观测自动化执行失败（不影响主流程）：{type(exc).__name__}: {exc}")
        time.sleep(interval)


def start_observability_automation(service: str) -> None:
    """启动稳定性自动化后台线程。

    Args:
        service: 服务名称。

    Returns:
        None.
    """
    if not _get_bool_env("OBSERVABILITY_AUTOMATION_ENABLED", True):
        logger.info(f"观测自动化未启用，service={service}")
        return
    key = service or "unknown"
    with _AUTOMATION_LOCK:
        thread = _AUTOMATION_THREADS.get(key)
        if thread and thread.is_alive():
            return
        thread = threading.Thread(target=_automation_loop, args=(key,), name=f"observability-automation-{key}", daemon=True)
        _AUTOMATION_THREADS[key] = thread
        thread.start()
    logger.info(f"观测自动化已启动，service={key}")
