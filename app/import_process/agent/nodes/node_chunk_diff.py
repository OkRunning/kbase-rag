"""Chunk 级增量对比节点。"""

from __future__ import annotations

import sys
from typing import Any

from app.clients.document_versioning_utils import (
    DEFAULT_CHUNK_STRATEGY,
    DEFAULT_SCHEMA_VERSION,
    compute_chunk_hash,
    get_document_version_store,
    is_document_versioning_enabled,
)
from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.core.tenant_scope import build_tenant_kb_filter, normalize_kb_id, normalize_tenant_id
from app.import_process.agent.state import ImportGraphState
from app.observability.replay_context import is_dry_run
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_done_task, add_running_task


def _mark_all_changed(state: ImportGraphState, chunks: list[dict[str, Any]]) -> ImportGraphState:
    """将当前版本所有 chunks 标记为需要重新向量化。

    Args:
        state: 导入 LangGraph State。
        chunks: 当前版本切片列表。

    Returns:
        更新增量字段后的导入 State。
    """
    doc_id = str(state.get("doc_id") or state["task_id"])
    doc_version = int(state.get("doc_version") or 0)
    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
        chunk["chunk_uid"] = chunk.get("chunk_uid") or f"{doc_id}_v{doc_version}_chunk_{index:05d}"
        chunk["diff_action"] = "changed"
    state["chunks"] = chunks
    state["changed_chunks"] = chunks
    state["reused_chunks"] = []
    state["removed_chunks"] = []
    return state


def _fetch_reused_vectors_from_milvus(
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    previous_doc_version: int,
    chunk_hashes: list[str],
) -> dict[str, dict[str, Any]]:
    """从 Milvus 查询可复用 chunk 的旧向量。

    Args:
        tenant_id: 租户标识。
        kb_id: 知识库标识。
        doc_id: 逻辑文档标识。
        previous_doc_version: 上一 active 文档版本。
        chunk_hashes: 需要查询的 chunk 指纹列表。

    Returns:
        以 chunk_hash 为键的 Milvus 旧 chunk 数据。
    """
    if not chunk_hashes:
        return {}
    client = get_milvus_client()
    if client is None:
        return {}
    escaped_hashes = ", ".join(f'"{escape_milvus_string(chunk_hash)}"' for chunk_hash in chunk_hashes)
    expr = build_tenant_kb_filter(
        tenant_id,
        kb_id,
        (
            f'doc_id == "{escape_milvus_string(doc_id)}" '
            f"and doc_version == {int(previous_doc_version)} "
            f"and chunk_hash in [{escaped_hashes}]"
        ),
    )
    try:
        result = client.query(
            collection_name=milvus_config.chunks_collection,
            filter=expr,
            output_fields=[
                "chunk_hash",
                "chunk_uid",
                "chunk_id",
                "dense_vector",
                "sparse_vector",
                "vector_version",
                "embedding_model",
                "embedding_version",
            ],
        )
    except Exception as exc:
        logger.warning(f"查询可复用chunk向量失败，将退回重新向量化：{exc}")
        return {}
    return {str(item.get("chunk_hash")): item for item in (result or []) if item.get("chunk_hash")}


def node_chunk_diff(state: ImportGraphState) -> ImportGraphState:
    """对比当前 chunks 与上一 active 版本并标记增量向量化动作。

    Args:
        state: 已完成文档切分和商品名称识别的导入 State。

    Returns:
        写入 chunk_hash、chunk_uid、changed_chunks、reused_chunks 和 removed_chunks 的 State。

    Raises:
        ValueError: State 中没有可对比的 chunks 时抛出。
    """
    func_name = sys._getframe().f_code.co_name
    logger.info(f"【{func_name}】节点开始执行")
    add_running_task(state["task_id"], func_name)
    try:
        chunks = state.get("chunks") or []
        if not chunks:
            raise ValueError("没有chunks数据，无法执行chunk级增量对比")

        tenant_id = normalize_tenant_id(state.get("tenant_id"))
        kb_id = normalize_kb_id(state.get("kb_id"))
        doc_id = str(state.get("doc_id") or state["task_id"])
        doc_version = int(state.get("doc_version") or 0)
        previous_doc_version = state.get("previous_doc_version")
        chunk_strategy = state.get("chunk_strategy") or DEFAULT_CHUNK_STRATEGY
        item_name = state.get("item_name", "")

        for index, chunk in enumerate(chunks):
            chunk["chunk_index"] = index
            chunk["chunk_uid"] = f"{doc_id}_v{doc_version}_chunk_{index:05d}"
            chunk["chunk_hash"] = compute_chunk_hash(
                chunk,
                item_name=item_name,
                chunk_strategy=chunk_strategy,
                schema_version=DEFAULT_SCHEMA_VERSION,
            )

        if (
            not is_document_versioning_enabled()
            or is_dry_run()
            or not previous_doc_version
            or int(previous_doc_version) <= 0
        ):
            logger.info("没有可用上一版本，当前 chunks 全量向量化")
            return _mark_all_changed(state, chunks)

        try:
            store = get_document_version_store()
            previous_chunks = store.list_version_chunks(tenant_id, kb_id, doc_id, int(previous_doc_version))
        except Exception as exc:
            logger.warning(f"查询上一版本chunk快照失败，当前 chunks 全量向量化：{exc}")
            return _mark_all_changed(state, chunks)
        if not previous_chunks:
            logger.info("上一版本没有chunk快照，当前 chunks 全量向量化")
            return _mark_all_changed(state, chunks)

        previous_by_hash = {item.get("chunk_hash"): item for item in previous_chunks if item.get("chunk_hash")}
        current_hashes = {chunk["chunk_hash"] for chunk in chunks}
        reused_candidates = [chunk for chunk in chunks if chunk["chunk_hash"] in previous_by_hash]
        changed_chunks = [chunk for chunk in chunks if chunk["chunk_hash"] not in previous_by_hash]
        removed_chunks = [chunk for chunk in previous_chunks if chunk.get("chunk_hash") not in current_hashes]
        for chunk in changed_chunks:
            chunk["diff_action"] = "changed"

        reused_vectors = _fetch_reused_vectors_from_milvus(
            tenant_id,
            kb_id,
            doc_id,
            int(previous_doc_version),
            [chunk["chunk_hash"] for chunk in reused_candidates],
        )

        reused_chunks = []
        for chunk in reused_candidates:
            reused_vector = reused_vectors.get(chunk["chunk_hash"])
            if not reused_vector or "dense_vector" not in reused_vector or "sparse_vector" not in reused_vector:
                chunk["diff_action"] = "changed"
                changed_chunks.append(chunk)
                continue

            # 不能只按 chunk_index 复用：章节插入会导致后续索引整体偏移，hash 才能稳定识别未变化内容。
            previous_chunk = previous_by_hash[chunk["chunk_hash"]]
            chunk["diff_action"] = "reused"
            chunk["source_chunk_uid"] = previous_chunk.get("chunk_uid") or reused_vector.get("chunk_uid")
            chunk["dense_vector"] = reused_vector["dense_vector"]
            chunk["sparse_vector"] = reused_vector["sparse_vector"]
            chunk["source_chunk_id"] = reused_vector.get("chunk_id")
            reused_chunks.append(chunk)

        state["chunks"] = chunks
        state["changed_chunks"] = changed_chunks
        state["reused_chunks"] = reused_chunks
        state["removed_chunks"] = removed_chunks
        logger.info(
            f"chunk级增量对比完成 changed={len(changed_chunks)}, "
            f"reused={len(reused_chunks)}, removed={len(removed_chunks)}"
        )
    except Exception as exc:
        logger.error(f"【{func_name}】节点执行异常，异常信息为:{exc}")
        raise
    finally:
        logger.info(f"【{func_name}】节点执行完毕")
        add_done_task(state["task_id"], func_name)
    return state
