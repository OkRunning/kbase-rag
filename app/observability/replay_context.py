"""管理回放执行期间的上下文开关。"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict

_replay_id: ContextVar[str] = ContextVar("replay_id", default="")
_dry_run: ContextVar[bool] = ContextVar("dry_run", default=False)


def set_replay_context(replay_id: str = "", dry_run: bool = False) -> Dict[str, Any]:
    """设置当前调用链的回放上下文。

    Args:
        replay_id: 本次回放 ID。
        dry_run: 是否启用副作用隔离。

    Returns:
        ContextVar token 字典。
    """
    return {
        "replay_id": _replay_id.set(replay_id or ""),
        "dry_run": _dry_run.set(bool(dry_run)),
    }


def reset_replay_context(tokens: Dict[str, Any]) -> None:
    """恢复回放上下文。

    Args:
        tokens: `set_replay_context` 返回的 token 字典。

    Returns:
        None.
    """
    for name, token in reversed(list(tokens.items())):
        if name == "replay_id":
            _replay_id.reset(token)
        elif name == "dry_run":
            _dry_run.reset(token)


def get_replay_context() -> Dict[str, Any]:
    """获取当前回放上下文。

    Returns:
        包含 replay_id 和 dry_run 的上下文字段。
    """
    return {
        "replay_id": _replay_id.get(),
        "dry_run": _dry_run.get(),
    }


def is_dry_run() -> bool:
    """判断当前调用链是否处于 dry-run 回放。

    Returns:
        dry-run 回放返回 True，否则返回 False。
    """
    return bool(_dry_run.get())
