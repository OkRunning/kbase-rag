"""Redis 客户端工具。

该模块提供懒加载 Redis 客户端，供任务状态、进度等运行时数据复用。
"""

from __future__ import annotations

import os
import time
from typing import Any

from app.conf.redis_config import redis_config
from app.core.logger import logger

_redis_client: Any = None
_redis_unavailable = False
_redis_last_failure_at = 0.0
_REDIS_RETRY_SECONDS = float(os.getenv("REDIS_RETRY_SECONDS", "30"))


def get_redis_client() -> Any:
    """获取 Redis 客户端单例。

    Returns:
        Redis 客户端；依赖不可用或连接失败时返回 None。
    """
    global _redis_client, _redis_unavailable, _redis_last_failure_at
    if _redis_unavailable and time.monotonic() - _redis_last_failure_at < _REDIS_RETRY_SECONDS:
        return None
    if _redis_client is not None:
        return _redis_client

    try:
        import redis

        client = redis.Redis.from_url(
            redis_config.url,
            decode_responses=True,
            socket_connect_timeout=0.2,
            socket_timeout=0.2,
        )
        client.ping()
        _redis_client = client
        _redis_unavailable = False
        return _redis_client
    except Exception as exc:
        _redis_unavailable = True
        _redis_last_failure_at = time.monotonic()
        logger.warning(f"Redis 不可用，任务状态将降级为进程内存存储: {exc}")
        return None


def reset_redis_client() -> None:
    """重置 Redis 客户端缓存。

    Returns:
        None.
    """
    global _redis_client, _redis_unavailable, _redis_last_failure_at
    _redis_client = None
    _redis_unavailable = False
    _redis_last_failure_at = 0.0
