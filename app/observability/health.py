"""提供服务健康检查响应构造能力。"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict

from pymongo import MongoClient

from app.clients.milvus_utils import get_milvus_client
from app.clients.redis_utils import get_redis_client
from app.clients.langfuse import is_langfuse_configured
from app.conf.lm_config import lm_config
from app.observability.checkpoints import get_checkpoint_backend
from app.observability.metrics import record_dependency_health


def _duration_ms(start: float) -> int:
    """计算毫秒级耗时。

    Args:
        start: 操作开始时的 perf_counter。

    Returns:
        操作耗时，单位毫秒。
    """
    return int((time.perf_counter() - start) * 1000)


def _check_dependency(name: str, probe: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """执行单个依赖探测并统一返回格式。

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
        record_dependency_health(name, status == "ok")
        return result
    except Exception as exc:
        record_dependency_health(name, False)
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

    Raises:
        Exception: MongoDB ping 失败时抛出原始异常。
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


def _otel_probe() -> Dict[str, Any]:
    """检查 OTLP exporter 配置。

    Returns:
        OTLP 配置检查结果。
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return {"status": "degraded", "reason": "missing_otlp_endpoint"}
    return {"status": "ok", "endpoint_configured": True}


def build_liveness(service: str) -> Dict[str, Any]:
    """构造 liveness 健康检查响应。

    Args:
        service: 服务名称。

    Returns:
        liveness 响应。
    """
    return {"status": "ok", "service": service}


def build_readiness(service: str) -> Dict[str, Any]:
    """构造 readiness 健康检查响应。

    Args:
        service: 服务名称。

    Returns:
        readiness 响应。
    """
    # readiness 只检查业务请求必须依赖的组件；观测链路单独放到 /health/observability。
    checks = {
        "mongo": _check_dependency("mongo", _mongo_probe),
        "redis": _check_dependency("redis", _redis_probe),
        "milvus": _check_dependency("milvus", _milvus_probe),
        "llm": _check_dependency("llm", _llm_probe),
    }
    statuses = {item.get("status") for item in checks.values()}
    if "error" in statuses:
        status = "error"
    elif "degraded" in statuses:
        status = "degraded"
    else:
        status = "ok"
    return {"status": status, "service": service, "checks": checks}


def build_observability_health(service: str) -> Dict[str, Any]:
    """构造观测能力健康检查响应。

    Args:
        service: 服务名称。

    Returns:
        观测能力健康响应。
    """
    checkpoint_backend = get_checkpoint_backend()
    event_backend = "mongo" if os.getenv("MONGO_URL") and os.getenv("MONGO_DB_NAME") else "jsonl"
    snapshot_backend = "minio" if (
        os.getenv("MINIO_ENDPOINT")
        and os.getenv("MINIO_ACCESS_KEY")
        and os.getenv("MINIO_SECRET_KEY")
    ) else "local"
    checks = {
        "event_store": {
            "status": "ok" if event_backend == "mongo" else "degraded",
            "backend": event_backend,
        },
        "snapshot_store": {
            "status": "ok" if snapshot_backend == "minio" else "degraded",
            "backend": snapshot_backend,
        },
        "checkpoint": {
            "status": "ok" if checkpoint_backend == "postgres" else "degraded",
            "backend": checkpoint_backend,
            "persistent": checkpoint_backend == "postgres",
        },
        "otlp_exporter": _otel_probe(),
        "langfuse": {
            "status": "ok" if is_langfuse_configured() else "degraded",
            "configured": is_langfuse_configured(),
        },
    }
    statuses = {item.get("status") for item in checks.values()}
    status = "degraded" if "degraded" in statuses else "ok"
    if "error" in statuses:
        status = "error"
    return {"status": status, "service": service, "checks": checks}
