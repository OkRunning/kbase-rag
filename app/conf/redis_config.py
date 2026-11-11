"""Redis 配置。

该模块集中管理任务状态等运行时数据依赖的 Redis 连接参数。
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RedisConfig:
    """Redis 连接配置。

    Args:
        url: Redis 连接 URL。
        task_ttl_seconds: 任务状态保留时间，单位秒。

    Returns:
        Redis 配置对象。
    """

    url: str = os.getenv("REDIS_URL", "redis://localhost:16379/0")
    task_ttl_seconds: int = int(os.getenv("TASK_STATE_TTL_SECONDS", "86400"))


redis_config = RedisConfig()
