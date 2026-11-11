"""提供不依赖观测后端的业务健康检查。"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict

from pymongo import MongoClient

from app.clients.milvus_utils import get_milvus_client
from app.clients.redis_utils import get_redis_client
from app.conf.lm_config import lm_config
from app.observability.settings import is_observability_enabled


def _record_dependency_health(dependency: str, healthy: bool) -> None:
    """记录依赖健康指标。

    Args:
        dependency: 依赖名称。
        healthy: 是否健康。

    Returns:
        None.
    """
    if not is_observability_enabled():
        return
    from app.observability.metrics import record_dependency_health

    record_dependency_health(dependency, healthy)


def _duration_ms(start: float) -> int:
    """计算毫秒级耗时。

    Args:
        start: 操作开始时的 perf_counter。

    Returns:
        操作耗时，单位毫秒。
    """
    return int((time.perf_counter() - start) * 1000)


def _check_dependency(name: str, probe: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """执行单个业务依赖探测。

    Args:
        name: 依赖名称。
        probe: 依赖探测函数。

    Returns:
        依赖检查结果。
    """
    start = time.perf_counter()
    try:
        result = probe()
        status = result.get("status", "ok")
        result["duration_ms"] = _duration_ms(start)
        _record_dependency_health(name, status == "ok")
        return result
    except Exception as exc:
        _record_dependency_health(name, False)
        return {
            "status": "error",
            "duration_ms": _duration_ms(start),
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _mongo_probe() -> Dict[str, Any]:
    """探测 MongoDB 可用性。

    Returns:
        MongoDB 检查结果。
    """
    mongo_url = os.getenv("MONGO_URL")
    db_name = os.getenv("MONGO_DB_NAME")
    if not mongo_url or not db_name:
        return {"status": "degraded", "reason": "missing_configuration"}
    client = MongoClient(
        mongo_url,
        username=os.getenv("MONGO_DB_USERNAME") or None,
        password=os.getenv("MONGO_DB_PASSWORD") or None,
        authSource="admin",
        serverSelectionTimeoutMS=500,
    )
    client.admin.command("ping")
    return {"status": "ok"}


def _redis_probe() -> Dict[str, Any]:
    """探测 Redis 可用性。

    Returns:
        Redis 检查结果。
    """
    client = get_redis_client()
    if client is None:
        return {"status": "degraded", "backend": "memory"}
    client.ping()
    return {"status": "ok"}


def _milvus_probe() -> Dict[str, Any]:
    """探测 Milvus 可用性。

    Returns:
        Milvus 检查结果。
    """
    client = get_milvus_client()
    if client is None:
        return {"status": "error", "reason": "client_unavailable"}
    list_collections = getattr(client, "list_collections", None)
    if list_collections is not None:
        list_collections()
    return {"status": "ok"}


def _llm_probe() -> Dict[str, Any]:
    """检查 LLM 基础配置。

    Returns:
        LLM 检查结果。
    """
    if not lm_config.base_url or not lm_config.api_key:
        return {"status": "error", "reason": "missing_configuration"}
    return {"status": "ok", "base_url_configured": True}


def build_liveness(service: str) -> Dict[str, Any]:
    """构造 liveness 健康检查响应。

    Args:
        service: 服务名称。

    Returns:
        liveness 响应。
    """
    return {"status": "ok", "service": service}


def build_readiness(service: str) -> Dict[str, Any]:
    """构造只检查业务依赖的 readiness 响应。

    Args:
        service: 服务名称。

    Returns:
        readiness 响应。
    """
    checks = {
        "mongo": _check_dependency("mongo", _mongo_probe),
        "redis": _check_dependency("redis", _redis_probe),
        "milvus": _check_dependency("milvus", _milvus_probe),
        "llm": _check_dependency("llm", _llm_probe),
    }
    statuses = {item.get("status") for item in checks.values()}
    status = "error" if "error" in statuses else "degraded" if "degraded" in statuses else "ok"
    return {"status": status, "service": service, "checks": checks}
