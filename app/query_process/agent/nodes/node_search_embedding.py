"""普通向量检索节点。"""

import sys

from app.clients.document_versioning_utils import build_active_document_version_filter, build_active_vector_version_filter
from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
    normalize_milvus_hits,
)
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.core.tenant_scope import build_tenant_kb_filter
from app.lm.embedding_utils import generate_embeddings
from app.observability.metrics import record_retrieval_docs
from app.observability.sanitizers import sanitize_for_log
from app.query_process.agent.state import QueryGraphState
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task, add_done_task


def node_search_embedding(state: QueryGraphState):
    """执行原问题向量检索节点。

    Args:
        state: 查询流程状态，包含重写问题、商品名和会话信息。

    Returns:
        包含 `embedding_chunks` 的状态更新。
    """
    fun_name = sys._getframe().f_code.co_name
    rewritten_query = state["rewritten_query"]
    item_names = state["item_names"]
    session_id = state["session_id"]
    tenant_id = state["tenant_id"]
    kb_id = state["kb_id"]
    is_stream = state["is_stream"]

    add_running_task(session_id, fun_name, is_stream)
    logger.info(f"{fun_name}节点开始执行，全局状态为：{sanitize_for_log(state)}")

    # 1. 问题向量化，获取稀疏向量和稠密向量
    embeddings = generate_embeddings([rewritten_query])
    dense_vector = embeddings['dense'][0]
    sparse_vector = embeddings['sparse'][0]

    # 2. 获取milvus客户端
    milvus_client = get_milvus_client()
    # 3. 创建多个 AnnSearchRequest 实例
    item_names_expr = ",".join(f'"{escape_milvus_string(name)}"' for name in item_names)
    extra_filters = [f"item_name in [{item_names_expr}]"]
    active_version_filter = build_active_document_version_filter(tenant_id, kb_id, item_names)
    active_vector_filter = build_active_vector_version_filter(tenant_id, kb_id)
    if active_version_filter:
        extra_filters.extend([
            'doc_status == "active"',
            'chunk_status == "active"',
            'vector_status == "active"',
            active_version_filter,
        ])
        if active_vector_filter:
            extra_filters.append(active_vector_filter)

    search_expr = build_tenant_kb_filter(
        tenant_id,
        kb_id,
        " and ".join(extra_filters),
    )
    logger.info(f"{fun_name} Milvus search expr: {search_expr}")
    # 查询侧状态过滤必须和 Mongo active_version 指针配合，避免新版本未激活或旧版本残留造成召回污染。
    reqs = create_hybrid_search_requests(dense_vector,
                                         sparse_vector,
                                         expr=search_expr)

    # item_name_str = ', '.join(f'"{item}"' for item in item_names)
    # hybrid_search_requests = create_hybrid_search_requests(
    #     # embeddings['dense'][0],
    #     # embeddings['sparse'][0],
    #     dense_vector,
    #     sparse_vector,
    #     expr=f"item_name in [{item_name_str}]"
    # )

    # 4. 执行混合搜索
    response = hybrid_search(client=milvus_client,
                             collection_name=milvus_config.chunks_collection,
                             reqs=reqs,
                             ranker_weights=(0.9, 0.1),
                             norm_score=True,
                             output_fields=[
                                 "chunk_id",
                                 "content",
                                 "file_title",
                                 "title",
                                 "parent_title",
                                 "item_name",
                                 "tenant_id",
                                 "kb_id",
                                 "doc_id",
                                 "doc_version",
                                 "doc_status",
                                 "chunk_status",
                                 "vector_status",
                                 "vector_version",
                                 "embedding_model",
                                 "embedding_version",
                             ])

    """
       [
       
         [
            {id ,
            distance，
            entity:
               {
                  "chunk_id", "content","file_title", "title", "parent_title", "item_name"
               }
         ]
       ]
    """
    # Milvus 原始 hit 可能同时存在顶层 id 和 entity.chunk_id，先统一结构再进入快照/RRF。
    embedding_chunks = normalize_milvus_hits(response[0]) if response else []
    record_retrieval_docs("embedding", len(embedding_chunks))
    add_done_task(session_id, fun_name, is_stream)
    logger.info(f"{fun_name}节点执行完成，向量库匹配结果为：{sanitize_for_log(embedding_chunks)}")
    return {"embedding_chunks": embedding_chunks}


if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "华为显示器 B3-211H 关于儿童健康应该注意什么?",  # 模拟改写后的查询
        "item_names": ["华为显示器 B3-211H"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果,结果为：{chunks}")

    except Exception as e:
        logger.exception(f"测试运行失败: {e}", exc_info=True)
