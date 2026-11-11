"""文档版本激活节点。"""

import sys

from app.clients.document_versioning_utils import (
    compute_content_hash,
    get_document_version_store,
    is_document_versioning_enabled,
)
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.observability.events import emit_event
from app.observability.replay_context import is_dry_run
from app.utils.task_utils import add_done_task, add_running_task


def node_document_version_activate(state: ImportGraphState) -> ImportGraphState:
    """激活已完成向量写入的文档版本。

    Args:
        state: 已完成 Milvus 写入的导入 State。

    Returns:
        更新 content_hash 和版本激活结果后的导入 State。

    Raises:
        ValueError: Milvus 写入数量与 chunk 数量不一致时抛出。
        Exception: MongoDB 版本切换失败时抛出。
    """
    func_name = sys._getframe().f_code.co_name
    logger.info(f"【{func_name}】节点开始执行")
    add_running_task(state["task_id"], func_name)
    try:
        if state.get("import_action") == "noop":
            logger.info(f"文档内容未变化，跳过版本激活：{state.get('skip_reason')}")
            return state
        if is_dry_run():
            logger.info("当前为 dry-run，跳过文档版本激活")
            return state
        if not is_document_versioning_enabled():
            logger.info("文档版本治理未启用，跳过版本激活")
            return state

        doc_version = int(state.get("doc_version") or 0)
        if doc_version <= 0:
            logger.info("State中没有有效doc_version，跳过版本激活")
            return state

        chunks = state.get("chunks") or []
        content_hash = state.get("content_hash") or compute_content_hash(state.get("md_content", ""))
        state["content_hash"] = content_hash
        store = get_document_version_store()
        document = store.activate_version(
            tenant_id=state["tenant_id"],
            kb_id=state["kb_id"],
            doc_id=state["doc_id"],
            doc_version=doc_version,
            previous_doc_version=state.get("previous_doc_version"),
            content_hash=content_hash,
            item_name=state.get("item_name") or (chunks[0].get("item_name", "") if chunks else ""),
            chunk_count=len(chunks),
            milvus_insert_count=int(state.get("milvus_insert_count") or 0),
            chunks=chunks,
            changed_chunk_count=len(state.get("changed_chunks") or []),
            reused_chunk_count=len(state.get("reused_chunks") or []),
            removed_chunk_count=len(state.get("removed_chunks") or []),
        )
        emit_event(
            "document.version_activated",
            graph="import",
            task_id=state.get("task_id", ""),
            status="ok",
            summary={
                "tenant_id": state.get("tenant_id"),
                "kb_id": state.get("kb_id"),
                "doc_id": state.get("doc_id"),
                "doc_version": doc_version,
                "active_version": document.get("active_version"),
            },
            tags=["import", "document.version_activated"],
        )
        logger.info(f"文档版本激活完成：{document}")
    except Exception as exc:
        try:
            get_document_version_store().mark_version_failed(
                state.get("tenant_id", "default"),
                state.get("kb_id", "default"),
                state.get("doc_id", state.get("task_id", "")),
                state.get("doc_version"),
                str(exc),
            )
        except Exception as mark_exc:
            logger.warning(f"标记文档版本失败状态时出错：{mark_exc}")
        logger.error(f"【{func_name}】节点执行异常，异常信息为:{exc}")
        raise
    finally:
        logger.info(f"【{func_name}】节点执行完毕")
        add_done_task(state["task_id"], func_name)
    return state
