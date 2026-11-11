"""提供原始执行与回放执行的快照差异比较。"""

from __future__ import annotations

from typing import Any, Dict, List

from app.observability.snapshots import load_snapshot

_MAX_DIFFS = 200


def _value_type(value: Any) -> str:
    """返回值类型名称。

    Args:
        value: 任意值。

    Returns:
        类型名称。
    """
    return type(value).__name__


def _append_diff(
    diffs: List[Dict[str, Any]],
    path: str,
    change_type: str,
    original: Any = None,
    replay: Any = None,
) -> None:
    """追加一条差异记录。

    Args:
        diffs: 差异列表。
        path: 差异路径。
        change_type: 差异类型。
        original: 原始值。
        replay: 回放值。
    """
    if len(diffs) >= _MAX_DIFFS:
        return
    diffs.append(
        {
            "path": path,
            "type": change_type,
            "original_type": _value_type(original),
            "replay_type": _value_type(replay),
            "original": original,
            "replay": replay,
        }
    )


def _diff_values(original: Any, replay: Any, path: str, diffs: List[Dict[str, Any]]) -> None:
    """递归比较两个值。

    Args:
        original: 原始值。
        replay: 回放值。
        path: 当前路径。
        diffs: 差异列表。
    """
    if len(diffs) >= _MAX_DIFFS:
        return
    if type(original) is not type(replay):
        _append_diff(diffs, path, "type_changed", original, replay)
        return
    if isinstance(original, dict):
        keys = set(original.keys()) | set(replay.keys())
        for key in sorted(keys):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in original:
                _append_diff(diffs, child_path, "added", None, replay.get(key))
            elif key not in replay:
                _append_diff(diffs, child_path, "removed", original.get(key), None)
            else:
                _diff_values(original[key], replay[key], child_path, diffs)
        return
    if isinstance(original, list):
        if len(original) != len(replay):
            _append_diff(diffs, f"{path}.length", "length_changed", len(original), len(replay))
        for index, (left, right) in enumerate(zip(original, replay)):
            _diff_values(left, right, f"{path}[{index}]", diffs)
        return
    if original != replay:
        _append_diff(diffs, path, "value_changed", original, replay)


def diff_snapshots(original_ref: str, replay_ref: str) -> Dict[str, Any]:
    """比较两个快照 payload 的差异。

    Args:
        original_ref: 原始快照引用。
        replay_ref: 回放快照引用。

    Returns:
        差异比较结果。
    """
    original_snapshot = load_snapshot(original_ref)
    replay_snapshot = load_snapshot(replay_ref)
    diffs: List[Dict[str, Any]] = []
    _diff_values(
        original_snapshot.get("payload") or {},
        replay_snapshot.get("payload") or {},
        "",
        diffs,
    )
    return {
        "original_ref": original_ref,
        "replay_ref": replay_ref,
        "diff_count": len(diffs),
        "truncated": len(diffs) >= _MAX_DIFFS,
        "diffs": diffs,
    }
