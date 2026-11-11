"""文档版本准备节点。"""

import sys
from pathlib import Path

from app.clients.document_versioning_utils import (
    compute_file_hash,
    get_configured_vector_version,
    get_document_version_store,
    is_document_versioning_enabled,
)
from app.core.logger import logger
from app.core.tenant_scope import normalize_kb_id, normalize_tenant_id
from app.import_process.agent.state import ImportGraphState
from app.observability.replay_context import is_dry_run
from app.utils.task_utils import add_done_task, add_running_task


def node_document_version_prepare(state: ImportGraphState) -> ImportGraphState:
    """创建 pending 文档版本并判断是否可以跳过导入。

    Args:
        state: 已完成入口校验并带有本地文件路径的导入 State。

    Returns:
        写入版本治理字段后的导入 State。

    Raises:
        FileNotFoundError: 本地上传文件不存在时抛出。
    """
    func_name = sys._getframe().f_code.co_name
    logger.info(f"【{func_name}】节点开始执行")
    add_running_task(state["task_id"], func_name)
    try:
        local_file_path = state.get("local_file_path", "")
        if not local_file_path or not Path(local_file_path).exists():
            raise FileNotFoundError("本地上传文件不存在，无法准备文档版本")

        tenant_id = normalize_tenant_id(state.get("tenant_id"))
        kb_id = normalize_kb_id(state.get("kb_id"))
        doc_id = str(state.get("doc_id") or state["task_id"])
        file_hash = compute_file_hash(local_file_path)
        state["tenant_id"] = tenant_id
        state["kb_id"] = kb_id
        state["doc_id"] = doc_id
        state["file_hash"] = file_hash
        state["vector_version"] = state.get("vector_version") or get_configured_vector_version()

        if not is_document_versioning_enabled():
            logger.info("文档版本治理未启用，沿用兼容导入流程")
            return state
        if is_dry_run():
            logger.info("当前为 dry-run，跳过文档版本Mongo写入")
            return state

        try:
            store = get_document_version_store()
            result = store.prepare_version(
                tenant_id=tenant_id,
                kb_id=kb_id,
                doc_id=doc_id,
                file_hash=file_hash,
                file_name=Path(local_file_path).name,
                title=state.get("file_title") or Path(local_file_path).stem,
                created_by=state.get("user_id") or "anonymous",
                vector_version=state["vector_version"],
            )
            # LangGraph State 更新：prepare 是版本治理入口，后续路由、Milvus metadata 和激活节点都依赖这些字段。
            state.update(result)
            logger.info(
                f"文档版本准备完成 doc_id={doc_id}, action={state.get('import_action')}, "
                f"doc_version={state.get('doc_version')}, previous={state.get('previous_doc_version')}"
            )
        except Exception as exc:
            logger.warning(f"文档版本治理初始化失败，沿用兼容导入流程：{exc}")
    finally:
        logger.info(f"【{func_name}】节点执行完毕")
        add_done_task(state["task_id"], func_name)
    return state
