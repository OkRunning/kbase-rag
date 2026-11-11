"""BGE-M3 文档切片向量化节点。"""

import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_done_task, add_running_task


def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """为新增或变更 chunks 生成 BGE-M3 向量。

    Args:
        state: 已完成 chunk diff 的导入 State。

    Returns:
        为缺少向量的 chunks 补充 dense_vector 和 sparse_vector 后的 State。
    """
    # 1.节点入口处打印日志以及前端埋点
    func_name = sys._getframe().f_code.co_name
    logger.info(f"【{func_name}】节点开始执行")
    add_running_task(state["task_id"], func_name)
    try:
        chunks = state['chunks']
        batch_size = 5
        embedding_targets = [
            chunk
            for chunk in chunks
            if chunk.get("diff_action") != "reused"
            or "dense_vector" not in chunk
            or "sparse_vector" not in chunk
        ]
        for i in range(0, len(embedding_targets), batch_size):
            batch_chunks = embedding_targets[i:i + batch_size]
            current_texts = []
            for chunk in batch_chunks:
                # 核心词前置，保证商品名在向量输入前段出现，降低跨文档主体误召回概率。
                context_tmp = f"商品名：{chunk['item_name']},介绍：{chunk['content']}"
                current_texts.append(context_tmp)
            result = generate_embeddings(current_texts)
            for i, chunk in enumerate(batch_chunks):
                chunk["dense_vector"] = result['dense'][i]
                chunk["sparse_vector"] = result['sparse'][i]

        state['chunks'] = chunks
        logger.info(
            f"--- BGE-M3 向量化处理完成，新生成 {len(embedding_targets)} 条，"
            f"复用 {len(chunks) - len(embedding_targets)} 条 ---"
        )
    except Exception as e:
        # 捕获节点所有异常，记录错误堆栈，不中断整体流程
        logger.error(f"BGE-M3向量化节点执行失败：{str(e)}", exc_info=True)
    finally:
        logger.info(f"【{func_name}】节点执行完毕")
        add_done_task(state["task_id"], func_name)

    return state
