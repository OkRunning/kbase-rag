import os
import sys

from langchain_classic.chains.question_answering.map_reduce_prompt import messages
from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.core.tenant_scope import build_tenant_kb_filter, normalize_kb_id, normalize_tenant_id
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task, add_done_task

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500


def step_1_get_content_and_valid(state: ImportGraphState):
    file_title, chunks = state["file_title"], state["chunks"]
    if not chunks or not isinstance(chunks, list):
        raise ValueError("chunks must be a list")

    if not file_title:
        file_title = os.path.basename(state.get("md_path"))
        logger.info(f"file_title is empty, use file_title:{file_title}")
        state['file_title'] = file_title
    return file_title, chunks


def step_2_get_llm_context(chunks, file_title):
    """
    根据chunks切面的content内容进行分拼接！ （2000）
    截取内容限制： 1. 最多截取前top个 （5） 2. 最多字符不能超过 CONTEXT_TOTAL_MAX_CHARS
    截取内容处理：
          切片：{1}，标题:{title},内容：{content} \n\n
          切片：{2}，标题:{title},内容：{content} \n\n
          切片：{3}，标题:{title},内容：{content} \n\n
          切片：{4}，标题:{title},内容：{content} \n\n
          切片：{5}，标题:{title},内容：{content} \n\n
    :param chunks:
    :return:
    """
    result = []
    current_context_length = 0
    for i, chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K], start=1):
        content = chunk["content"]
        title = chunk.get("title", "")
        tmp_context = f"切片：{i}，标题:{title},内容：{content}"
        result.append(tmp_context)
        current_context_length += len(tmp_context)
        if current_context_length > CONTEXT_TOTAL_MAX_CHARS:
            logger.info(f"已经达到最大字符数:{current_context_length}，停止拼接！")
            break
    result_str = "\n\n".join(result)
    final_result = result_str[:CONTEXT_TOTAL_MAX_CHARS]
    return final_result


def step_3_get_item_name(llm_context, file_title):
    """
       1. 构建提示词
       2. 调用模型
       3. 兜底使用file_title
       4. 返回item_name
     """
    human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=llm_context)
    system_prompt = load_prompt("product_recognition_system")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]

    llm = get_llm_client()
    llm_result = llm.invoke(messages)
    logger.info(f"[step_3_get_item_name]调用模型获取结果{llm_result}")
    item_name = llm_result.content
    if not item_name:
        logger.warning(f"[step_3_get_item_name]:llm_result is empty, use file_title:{file_title}")
        item_name = file_title
    return item_name


def step_5_generate_vector(item_name):
    """
        根据item_name生成向量 -》 稠密 + 稀疏
        :param item_name:
        :return: dense_vector [稠密] ,  sparse_vector [稀疏]
        """
    """
    generate_embeddings 自己封装的嵌入式模式生成向量的函数！！ 
          embeddings list对应的向量 = model.encode_documents(texts) 传入的字符串list 
          参数：生成向量的字符串 ["1","2","3"] 
          返回结果： 
             result = {
                        "dense": [1的稠密,2的稠密,3的稠密],  #稠密向量
                        "sparse": [1的稀疏,2的稀疏,3的稀疏], #稀疏向量
                      }
    """

    result = generate_embeddings([item_name])
    return result["dense"][0], result["sparse"][0]


def step_6_save_vector(item_name, file_title, dense_vector, sparse_vector, tenant_id, kb_id, doc_id):
    """保存带租户和知识库范围的商品名称向量。

    Args:
        item_name: 商品名称。
        file_title: 文档标题。
        dense_vector: 稠密向量。
        sparse_vector: 稀疏向量。
        tenant_id: 租户标识。
        kb_id: 知识库标识。
        doc_id: 文档标识。

    Returns:
        None.
    """
    # 1. 获取milvus客户端
    milvus_client = get_milvus_client()

    # 2. 判断collection是否存在，如果不存在则创建
    if not milvus_client.has_collection(milvus_config.item_name_collection):
        # 2.1. Create schema
        schema = milvus_client.create_schema(
            auto_id=True,  # 主键自增长
            enable_dynamic_field=True,  # 动态字段
        )

        # 2.2. Add fields to schema
        #  pk file_title item_name dense_vector sparse_vector
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="kb_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
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
            params={"M": 16,  # Maximum number of neighbors each node can connect to in the graph
                    "efConstruction": 200},  # or "DAAT_WAND" or "TAAT_NAIVE"
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
            collection_name=milvus_config.item_name_collection,
            schema=schema,  # 字段
            index_params=index_params  # 索引
        )

    load_res1 = milvus_client.get_load_state(
        collection_name=milvus_config.item_name_collection,
    )
    logger.debug(f"load_milvus1结果：{load_res1}")

    # 3. 删除当前租户知识库内的同名商品向量，不能影响其他租户或知识库。
    milvus_client.delete(
        collection_name=milvus_config.item_name_collection,
        filter=build_tenant_kb_filter(
            tenant_id,
            kb_id,
            f'item_name == "{escape_milvus_string(item_name)}"',
        ),
    )
    # 4.插入数据
    # file_title item_name dense_vector sparse_vector
    milvus_client.insert(collection_name=milvus_config.item_name_collection,
                         data=[{
                             "file_title": file_title,
                             "item_name": item_name,
                             "tenant_id": tenant_id,
                             "kb_id": kb_id,
                             "doc_id": doc_id,
                             "dense_vector": dense_vector,
                             "sparse_vector": sparse_vector
                         }])
    logger.info(f"保存了item_name:{item_name}的数据到向量数据库中！！")
    load_res2 = milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    logger.debug(f"load_milvus2结果：{load_res2}")


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """识别商品名称并写入带隔离 metadata 的向量数据。

    Args:
        state: 导入 LangGraph State。

    Returns:
        补充商品名称、租户、知识库和文档标识后的 State。

    Raises:
        Exception: 商品名称识别或 Milvus 写入失败时抛出。
    """
    # 1.节点入口处打印日志以及前端埋点
    func_name = sys._getframe().f_code.co_name
    logger.info(f"【{func_name}】节点开始执行")
    add_running_task(state["task_id"], func_name)
    try:
        # 1. 校验和取值 （file_title,chunks）
        file_title, chunks = step_1_get_content_and_valid(state)

        # 2. 构建上下文环境  chunks -> top 5 -> 拼接成context文本
        llm_context = step_2_get_llm_context(chunks, file_title)

        # 3. 调用模型，拼接提示词，识别chunks对应item_name，兜底使用file_title
        item_name = step_3_get_item_name(llm_context, file_title)

        tenant_id = normalize_tenant_id(state.get("tenant_id"))
        kb_id = normalize_kb_id(state.get("kb_id"))
        doc_id = str(state.get("doc_id") or state["task_id"])
        # 将隔离范围写入 LangGraph State，后续节点只消费 State 中的统一 scope。
        for chunk in chunks:
            chunk["item_name"] = item_name
            chunk["tenant_id"] = tenant_id
            chunk["kb_id"] = kb_id
            chunk["doc_id"] = doc_id

        state["chunks"] = chunks
        state["item_name"] = item_name
        state["tenant_id"] = tenant_id
        state["kb_id"] = kb_id
        state["doc_id"] = doc_id

        # 5. item_name生成向量（稠密/稀疏）
        dense_vector, sparse_vector = step_5_generate_vector(item_name)

        # 6. 存储向量到向量数据库 kb_item_name (id / file_title / item_name / 稠密 和 稀疏)
        step_6_save_vector(item_name, file_title, dense_vector, sparse_vector, tenant_id, kb_id, doc_id)

    except Exception as e:
        logger.error(f"【{func_name}】节点执行异常，异常信息为:{e}")
        raise e
    finally:
        # 6.节点处理完成后，打印日志以及前端埋点
        logger.info(f"【{func_name}】节点节点执行结束")
        add_done_task(state["task_id"], func_name)
    return state



# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()
