"""提供租户与知识库隔离的通用能力。"""

from __future__ import annotations

import re

from app.utils.escape_milvus_string_utils import escape_milvus_string

DEFAULT_TENANT_ID = "default"
DEFAULT_KB_ID = "default"
_SCOPE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def normalize_scope_id(value: str | None, field_name: str, default: str) -> str:
    """规范化并校验租户或知识库标识。

    Args:
        value: 请求传入的原始标识。
        field_name: 用于生成校验错误的字段名称。
        default: 空值时使用的兼容默认值。

    Returns:
        合法的租户或知识库标识。

    Raises:
        ValueError: 标识包含不允许的字符或长度不合法。
    """
    normalized = str(value or "").strip() or default
    if not _SCOPE_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{field_name}只能包含字母、数字、点、下划线和中划线，且长度不能超过128"
        )
    return normalized


def normalize_tenant_id(value: str | None) -> str:
    """规范化租户标识。

    Args:
        value: 请求传入的租户标识。

    Returns:
        合法租户标识；空值兼容为 ``default``。

    Raises:
        ValueError: 标识格式不合法。
    """
    return normalize_scope_id(value, "tenant_id", DEFAULT_TENANT_ID)


def normalize_kb_id(value: str | None) -> str:
    """规范化知识库标识。

    Args:
        value: 请求传入的知识库标识。

    Returns:
        合法知识库标识；空值兼容为 ``default``。

    Raises:
        ValueError: 标识格式不合法。
    """
    return normalize_scope_id(value, "kb_id", DEFAULT_KB_ID)


def build_tenant_kb_filter(
    tenant_id: str,
    kb_id: str,
    extra_filter: str | None = None,
) -> str:
    """构建强制包含租户和知识库条件的 Milvus filter。

    Args:
        tenant_id: 已校验的租户标识。
        kb_id: 已校验的知识库标识。
        extra_filter: 可选的业务过滤条件，例如商品名称条件。

    Returns:
        可直接传给 Milvus 的 filter 表达式。
    """
    scope_filter = (
        f'tenant_id == "{escape_milvus_string(normalize_tenant_id(tenant_id))}"'
        f' and kb_id == "{escape_milvus_string(normalize_kb_id(kb_id))}"'
    )
    if not extra_filter or not extra_filter.strip():
        return scope_filter
    return f"({scope_filter}) and ({extra_filter.strip()})"
