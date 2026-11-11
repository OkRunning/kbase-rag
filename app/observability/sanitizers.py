"""提供可观测事件和回放数据的脱敏工具。"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

_MAX_TEXT_PREVIEW = 200
_MAX_DEPTH = 4
_MAX_LIST_ITEMS = 20
_MAX_DICT_ITEMS = 50
_MAX_LOG_LIST_ITEMS = 5
_VECTOR_KEYS = {
    "dense",
    "sparse",
    "dense_vector",
    "sparse_vector",
    "vector",
    "vectors",
    "embedding",
    "embeddings",
}


def sha256_text(text: str) -> str:
    """计算文本的 SHA256 摘要。

    Args:
        text: 需要摘要的文本。

    Returns:
        十六进制 SHA256 字符串。
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sanitize_text(text: str) -> Dict[str, Any]:
    """将长文本压缩为预览、长度和摘要。

    Args:
        text: 待脱敏文本。

    Returns:
        包含预览、长度和 hash 的字典。
    """
    text = text or ""
    return {
        "preview": text[:_MAX_TEXT_PREVIEW],
        "chars": len(text),
        "sha256": sha256_text(text),
    }


def sanitize_value(value: Any, depth: int = 0) -> Any:
    """递归脱敏任意 JSON 兼容值。

    Args:
        value: 待脱敏值。
        depth: 当前递归深度，避免过深嵌套。

    Returns:
        脱敏后的值。
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= _MAX_TEXT_PREVIEW:
            return value
        return sanitize_text(value)
    if depth >= _MAX_DEPTH:
        return str(value)
    if isinstance(value, dict):
        items = list(value.items())[:_MAX_DICT_ITEMS]
        return {str(key): sanitize_value(item_value, depth + 1) for key, item_value in items}
    if isinstance(value, (list, tuple, set)):
        items: List[Any] = list(value)[:_MAX_LIST_ITEMS]
        return [sanitize_value(item, depth + 1) for item in items]
    return str(value)


def sanitize_summary(summary: Any) -> Any:
    """对事件 summary 做统一脱敏。

    Args:
        summary: 待脱敏的 summary。

    Returns:
        脱敏后的 summary。
    """
    return sanitize_value(summary)


def _is_numeric_sequence(value: Any) -> bool:
    """判断值是否像稠密向量。

    Args:
        value: 待判断的值。

    Returns:
        值为纯数字列表或元组时返回 True。
    """
    if not isinstance(value, (list, tuple)) or not value:
        return False
    return all(isinstance(item, (int, float)) for item in value)


def _is_sparse_vector_dict(value: Any) -> bool:
    """判断值是否像稀疏向量字典。

    Args:
        value: 待判断的值。

    Returns:
        字典 key/value 形态符合稀疏向量时返回 True。
    """
    if not isinstance(value, dict) or not value:
        return False
    sample_items = list(value.items())[:20]
    return all(
        isinstance(item_value, (int, float))
        and str(item_key).replace(".", "", 1).isdigit()
        for item_key, item_value in sample_items
    )


def _summarize_vector(value: Any) -> Dict[str, Any]:
    """生成向量字段的日志摘要。

    Args:
        value: 稠密向量或稀疏向量。

    Returns:
        不包含具体向量值的摘要。
    """
    if isinstance(value, dict):
        return {"omitted": True, "reason": "vector", "type": "sparse", "size": len(value)}
    if isinstance(value, (list, tuple)):
        return {"omitted": True, "reason": "vector", "type": "dense", "size": len(value)}
    return {"omitted": True, "reason": "vector", "type": type(value).__name__}


def sanitize_for_log(value: Any, depth: int = 0) -> Any:
    """递归清洗普通日志输出，屏蔽向量和超长内容。

    Args:
        value: 待写入日志的值。
        depth: 当前递归深度，避免过深嵌套。

    Returns:
        适合 logger 打印的脱敏值。
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= _MAX_TEXT_PREVIEW:
            return value
        return sanitize_text(value)
    if _is_numeric_sequence(value) or _is_sparse_vector_dict(value):
        return _summarize_vector(value)
    if depth >= _MAX_DEPTH:
        return {"omitted": True, "reason": "max_depth", "type": type(value).__name__}
    if isinstance(value, dict):
        items = list(value.items())[:_MAX_DICT_ITEMS]
        sanitized: Dict[str, Any] = {}
        for key, item_value in items:
            key_str = str(key)
            if key_str in _VECTOR_KEYS:
                # 向量字段只保留类型和长度，避免导入 state 在节点日志里刷出完整 embedding。
                sanitized[key_str] = _summarize_vector(item_value)
            else:
                sanitized[key_str] = sanitize_for_log(item_value, depth + 1)
        if len(value) > _MAX_DICT_ITEMS:
            sanitized["_omitted_fields"] = len(value) - _MAX_DICT_ITEMS
        return sanitized
    if isinstance(value, (list, tuple, set)):
        items: List[Any] = list(value)
        sanitized_items = [sanitize_for_log(item, depth + 1) for item in items[:_MAX_LOG_LIST_ITEMS]]
        if len(items) > _MAX_LOG_LIST_ITEMS:
            sanitized_items.append({"omitted_items": len(items) - _MAX_LOG_LIST_ITEMS})
        return sanitized_items
    return str(value)
