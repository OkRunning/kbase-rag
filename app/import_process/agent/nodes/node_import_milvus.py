"""Milvus 文档切片导入节点。"""

import sys

from pymilvus import DataType

from app.clients.document_versioning_utils import (
    DEFAULT_CHUNK_STRATEGY,
    DEFAULT_CHUNK_STRATEGY_VERSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_VERSION,
    get_configured_vector_version,
)
from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.tenant_scope import build_tenant_kb_filter, normalize_kb_id, normalize_tenant_id
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.observability.events import emit_event
from app.observability.replay_context import get_replay_context, is_dry_run
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task, add_done_task


def step_2_prepare_collections():
    """初始化包含隔离字段的文档切片集合。

    Returns:
        可用于读写文档切片的 Milvus 客户端。
    """
    milvus_client = get_milvus_client()
    # 3. 判断集合是否存在，如果不存在，先创建集合
    if not milvus_client.has_collection(milvus_config.chunks_collection):
        # 2.1. Create schema
        schema = milvus_client.create_schema(
            auto_id=True,  # 主键自增长
            enable_dynamic_field=True,  # 动态字段
        )

        # 2.2. Add fields to schema
        # chunk_id、file_title、item_name、content、title、parent_title、part、dense_vector、sparse_vector
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="kb_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2.3. Prepare index parameters
        index_params = milvus_client.prepare_index_params()

        # 2.4. Add indexes
        index_params.add_index(
            field_name="dense_vector",  # 给哪个列创建索引 稠密
            index_name="dense_vector_index",  # 索引的名字
            index_type="HNSW",  # 配置查找所用的算法
            metric_type="COSINE",  # 配置向量匹配和对比的 IP COSINE
            params={"M": 32,  # Maximum number of neighbors each node can connect to in the graph
                    "efConstruction": 300},  # or "DAAT_WAND" or "TAAT_NAIVE"
        )
        """
           10000  M = 16  efConstruction = 200
           50000  M = 32  efConstruction = 300
           100000  M = 64  efConstruction = 400
           M:图中每个节点在层次结构的每个层级所能拥有的最大边数或连接数。M 越高，图的密度就越大，搜索结果的召回率和准确率也就越高，因为有更多的路径可以探索，但同时也会消耗更多内存，并由于连接数的增加而减慢插入时间。如上图所示，M = 5表示 HNSW 图中的每个节点最多与 5 个其他节点直接相连。这就形成了一个中等密度的图结构，节点有多条路径到达其他节点。
           efConstruction:索引构建过程中考虑的候选节点数量。efConstruction 越高，图的质量越好，但需要更多时间来构建。
        """

        index_params.add_index(
            field_name="sparse_vector",  # Name of the vector field to be indexed
            index_type="SPARSE_INVERTED_INDEX",  # Type of the index to create
            index_name="sparse_vector_index",  # Name of the index to create
            metric_type="IP",  # Metric type used to measure similarity
            # 只计算可能得高分的向量，跳过大量的 0
            params={"inverted_index_algo": "DAAT_MAXSCORE"},  # Algorithm used for building and querying the index
        )

        # 2.5. Create a collection with the index loaded simultaneously
        milvus_client.create_collection(
            collection_name=milvus_config.chunks_collection,
            schema=schema,  # 字段
            index_params=index_params  # 索引
        )
    return milvus_client


def step_3_delete_old_data(milvus_client, item_name, tenant_id: str, kb_id: str):
    """删除同一租户知识库内待覆盖的商品向量。

    Args:
        milvus_client: Milvus 客户端。
        item_name: 商品名称。
        tenant_id: 租户标识。
        kb_id: 知识库标识。

    Returns:
        None.
    """
    milvus_client.delete(collection_name=milvus_config.chunks_collection,
                         filter=build_tenant_kb_filter(
                             tenant_id,
                             kb_id,
                             f'item_name == "{escape_milvus_string(item_name)}"',
                         ))
    milvus_client.load_collection(collection_name=milvus_config.chunks_collection)


def step_4_insert_date(milvus_client, chunks):
    """向 Milvus 写入已完成向量化的切片。

    Args:
        milvus_client: Milvus 客户端。
        chunks: 待写入的切片列表。

    Returns:
        二元组，包含回填 Milvus 主键后的切片列表和实际插入数量。
    """
    # insert_result = {'insert_count': 2, 'ids': [467029506940733289, 467029506940733290]}
    insert_result = milvus_client.insert(collection_name=milvus_config.chunks_collection, data=chunks)
    logger.info(f"chunks插入结果 insert_result:{insert_result}")
    insert_count = insert_result.get("insert_count", 0)
    logger.info(f"完成了数据插入，成功插入了 {insert_count} 条数据")
    ids = insert_result.get("ids", [])
    if ids and len(ids) == len(chunks):
        for index, chunk in enumerate(chunks):
            chunk['chunk_id'] = ids[index]

    return chunks, insert_count


def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """将带版本治理 metadata 的文档切片写入 Milvus。

    Args:
        state: 已完成切分、商品名称识别和向量化的导入 State。

    Returns:
        回填 chunk_id、doc_version 和写入计数后的导入 State。

    Raises:
        ValueError: State 中没有可写入的切片时抛出。
        Exception: Milvus 集合初始化或写入失败时抛出。
    """
    # 1.节点入口处打印日志以及前端埋点
    func_name = sys._getframe().f_code.co_name
    logger.info(f"【{func_name}】节点开始执行")
    add_running_task(state["task_id"], func_name)
    try:
        chunks = state['chunks']
        if not chunks:
            logger.error(f">>> [{func_name}]没有chunks数据，请检查！")
            raise ValueError("没有chunks数据")

        if is_dry_run():
            replay_context = get_replay_context()
            emit_event(
                "dry_run.milvus.import_skipped",
                graph="import",
                node_name=func_name,
                task_id=state.get("task_id", ""),
                status="ok",
                summary={
                    "chunks_count": len(chunks),
                    "item_name": chunks[0].get("item_name", "") if chunks else "",
                    "replay_id": replay_context.get("replay_id"),
                },
                tags=["dry_run", "milvus", "insert"],
            )
            state["milvus_insert_count"] = len(chunks)
            return state

        tenant_id = normalize_tenant_id(state.get("tenant_id"))
        kb_id = normalize_kb_id(state.get("kb_id"))
        doc_id = str(state.get("doc_id") or state["task_id"])
        doc_version = int(state.get("doc_version") or 0)
        vector_version = state.get("vector_version") or get_configured_vector_version()
        chunk_strategy = state.get("chunk_strategy") or DEFAULT_CHUNK_STRATEGY
        chunk_strategy_version = state.get("chunk_strategy_version") or DEFAULT_CHUNK_STRATEGY_VERSION
        # 导入状态是 metadata 的唯一来源，写入前统一覆盖，禁止切片携带其他 scope。
        # P1-A 采用“插入即 active + 查询按 Mongo active_version 过滤”：新版本在 Mongo 指针切换前不可见，
        # 旧版本即使 metadata 仍是 active，也会因为 doc_version 不匹配被查询侧排除。
        for chunk in chunks:
            chunk["tenant_id"] = tenant_id
            chunk["kb_id"] = kb_id
            chunk["doc_id"] = doc_id
            chunk["doc_version"] = doc_version
            chunk["doc_status"] = "active"
            chunk["chunk_status"] = "active"
            chunk["vector_status"] = "active"
            chunk["vector_version"] = vector_version
            chunk["embedding_model"] = DEFAULT_EMBEDDING_MODEL
            chunk["embedding_version"] = DEFAULT_EMBEDDING_VERSION
            chunk["chunk_strategy"] = chunk_strategy
            chunk["chunk_strategy_version"] = chunk_strategy_version
            chunk["schema_version"] = "milvus_chunks_v2"

        # 2.没有集合，要创建集合 collection (filed,index,collection)
        milvus_client = step_2_prepare_collections()

        # 3. P1-A 不再删除旧数据，历史版本保留给回滚；查询侧通过 Mongo active_version 避免旧向量污染。
        chunks, insert_count = step_4_insert_date(milvus_client, chunks)
        state["chunks"] = chunks
        state["milvus_insert_count"] = insert_count

    except Exception as e:
        logger.error(f">>> [{func_name}]插入chunks向量节点执行出错:{e}")
        raise e
    finally:
        logger.info(f"【{func_name}】节点执行完毕")
        add_done_task(state["task_id"], func_name)
    return state


if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title": "test.pdf",
                "part": 1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
            ,
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus2",  # 必须有 item_name，用于幂等清理
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf2",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
