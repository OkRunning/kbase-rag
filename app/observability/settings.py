"""读取可观测能力总开关。"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    """读取布尔环境变量。

    Args:
        name: 环境变量名称。
        default: 未设置时的默认值。

    Returns:
        解析后的布尔值。
    """
    raw: Optional[str] = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def is_observability_enabled() -> bool:
    """判断是否启用完整可观测能力。

    Returns:
        启用可观测能力时返回 True；关闭时事件、指标、trace、debug 和自动化降级为空实现。
    """
    return _bool_env("OBSERVABILITY_ENABLED", True)
