"""验证租户和知识库隔离基础能力。"""

from app.core.tenant_scope import (
    build_tenant_kb_filter,
    normalize_kb_id,
    normalize_tenant_id,
)
from app.observability.checkpoints import build_langgraph_thread_id


def test_tenant_kb_filter_prevents_cross_tenant_retrieval():
    """验证相同问题在不同租户下会生成不同的强制检索范围。"""
    tenant_a_filter = build_tenant_kb_filter("tenant_a", "manuals", 'item_name in ["HAK 180"]')
    tenant_b_filter = build_tenant_kb_filter("tenant_b", "manuals", 'item_name in ["HAK 180"]')

    assert 'tenant_id == "tenant_a"' in tenant_a_filter
    assert 'tenant_id == "tenant_b"' in tenant_b_filter
    assert 'kb_id == "manuals"' in tenant_a_filter
    assert tenant_a_filter != tenant_b_filter
    assert 'item_name in ["HAK 180"]' in tenant_a_filter

    same_question_chunks = [
        {"chunk_id": "tenant-a-chunk", "tenant_id": "tenant_a", "kb_id": "manuals"},
        {"chunk_id": "tenant-b-chunk", "tenant_id": "tenant_b", "kb_id": "manuals"},
    ]
    tenant_a_results = [
        chunk
        for chunk in same_question_chunks
        if chunk["tenant_id"] == "tenant_a" and chunk["kb_id"] == "manuals"
    ]

    assert [chunk["chunk_id"] for chunk in tenant_a_results] == ["tenant-a-chunk"]


def test_tenant_kb_filter_prevents_cross_knowledge_base_retrieval():
    """验证同一租户的不同知识库使用不同检索范围。"""
    product_filter = build_tenant_kb_filter("tenant_a", "product_manual")
    policy_filter = build_tenant_kb_filter("tenant_a", "policy_manual")

    assert 'kb_id == "product_manual"' in product_filter
    assert 'kb_id == "policy_manual"' in policy_filter
    assert product_filter != policy_filter


def test_scope_ids_keep_default_compatibility_and_reject_invalid_values():
    """验证旧请求默认进入 default scope，非法标识会被拒绝。"""
    assert normalize_tenant_id(None) == "default"
    assert normalize_kb_id("") == "default"

    try:
        normalize_tenant_id("tenant a")
    except ValueError:
        pass
    else:
        raise AssertionError("tenant_id 包含空格时应被拒绝")


def test_checkpoint_thread_id_includes_tenant_and_knowledge_base():
    """验证相同 session_id 在不同 scope 下不会复用 checkpoint。"""
    tenant_a_thread = build_langgraph_thread_id(
        "query",
        "same-session",
        tenant_id="tenant_a",
        kb_id="manuals",
    )
    tenant_b_thread = build_langgraph_thread_id(
        "query",
        "same-session",
        tenant_id="tenant_b",
        kb_id="manuals",
    )

    assert tenant_a_thread != tenant_b_thread
    assert ":tenant_a:manuals:query:same-session" in tenant_a_thread
