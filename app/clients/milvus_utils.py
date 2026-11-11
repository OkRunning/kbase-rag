import time
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.observability.metrics import record_milvus_operation

# 全局Milvus客户端实例，实现单例复用
_milvus_client = None


def normalize_milvus_hit(hit):
    """将 Milvus search hit 统一为稳定的字典结构。

    Args:
        hit: Milvus search 返回的一条命中结果。
        eg:
        {   id ,
            distance，
            entity:
               {
                  "chunk_id", "content","file_title", "title", "parent_title", "item_name"
               }
        }

    Returns:
        包含 `id`、`milvus_id`、`chunk_id`、`distance`、`entity` 的标准命中结果。
    """
    if not isinstance(hit, dict):
        return hit

    raw_entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else {}
    milvus_id = hit.get("id")
    # Milvus 顶层 id 和 entity.chunk_id 在不同 schema / SDK 表现下可能不同。
    # 查询链路和评测统一优先使用业务 chunk_id，避免多个内容在快照里显示成同一个主键。
    chunk_id = raw_entity.get("chunk_id") or hit.get("chunk_id") or milvus_id
    entity = dict(raw_entity)
    if chunk_id is not None:
        entity["chunk_id"] = chunk_id
    for key in (
        "content",
        "file_title",
        "title",
        "parent_title",
        "item_name",
        "part",
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
    ):
        if key in hit and key not in entity:
            entity[key] = hit.get(key)

    normalized = dict(hit)
    normalized["id"] = chunk_id
    normalized["milvus_id"] = milvus_id
    normalized["chunk_id"] = chunk_id
    normalized["entity"] = entity
    return normalized


def normalize_milvus_hits(hits):
    """批量标准化 Milvus search 命中结果。

    Args:
        hits: Milvus search 返回的命中结果列表。
        eg:
        [
            {id ,
            distance，
            entity:
               {
                  "chunk_id", "content","file_title", "title", "parent_title", "item_name"
               }
            },
            {id ,
            distance，
            entity:
               {
                  "chunk_id", "content","file_title", "title", "parent_title", "item_name"
               }
            },
        ]

    Returns:
        标准化后的命中结果列表。
    """
    return [normalize_milvus_hit(hit) for hit in (hits or [])]


def get_milvus_client():
    """获取 Milvus 客户端单例。

    Returns:
        MilvusClient 实例；连接失败时返回 None。
    """
    try:
        global _milvus_client
        # 单例判断：未初始化则创建新连接
        if _milvus_client is None:
            milvus_uri = milvus_config.milvus_url
            # 校验Milvus连接地址配置
            if not milvus_uri:
                logger.error("Milvus客户端连接失败：缺少MILVUS_URL环境变量配置")
                return None
            # 初始化Milvus客户端
            _milvus_client = MilvusClient(uri=milvus_uri)
            logger.info("Milvus客户端连接成功")
        return _milvus_client
    except Exception as e:
        logger.error(f"Milvus客户端连接异常：{str(e)}", exc_info=True)
        return None


def _coerce_int64_ids(ids):
    """
    转换chunk_id为Milvus要求的INT64类型（主键字段schema为INT64）
    过滤无效ID，分离可转换/不可转换的ID
    :param ids: 待转换的chunk_id列表
    :return: 元组(ok_ids, bad_ids)，ok_ids为可转换的int64类型ID列表，bad_ids为无效ID列表
    """
    ok, bad = [], []
    for x in (ids or []):
        if x is None:
            continue
        try:
            ok.append(int(x))
        except Exception:
            bad.append(x)
    return ok, bad


def fetch_chunks_by_chunk_ids(
        client,
        collection_name: str,
        chunk_ids,
        *,
        output_fields=None,
        batch_size: int = 100,
):
    """通过 chunk_id 批量查询 Milvus 切片数据。

    Args:
        client: MilvusClient 实例。
        collection_name: 集合名称。
        chunk_ids: 待查询的 chunk_id 列表。
        output_fields: 需要返回的字段列表。
        batch_size: 分批查询大小。

    Returns:
        Milvus 实体字典列表，查询失败返回空列表。
    """
    # 前置校验：客户端/集合名无效直接返回空
    if client is None:
        return []
    if not collection_name:
        return []
    # 默认返回字段：核心切片标识与内容字段
    if output_fields is None:
        output_fields = ["chunk_id", "content", "title", "parent_title", "item_name"]

    # 转换ID为INT64类型，分离有效/无效ID
    ok_ids, bad_ids = _coerce_int64_ids(chunk_ids)
    if bad_ids:
        # 记录无效ID，跳过查询
        logger.warning(f"存在无法转换为INT64的chunk_id，将跳过查询：{bad_ids}")

    # 无有效ID直接返回空
    if not ok_ids:
        return []

    start = time.perf_counter()
    status = "ok"
    results = []
    # 分批查询降低单次请求压力，同时保留整体 Milvus 操作耗时指标。
    for i in range(0, len(ok_ids), batch_size):
        batch = ok_ids[i: i + batch_size]

        # 方式1：优先使用主键get方法查询（性能最优）
        if hasattr(client, "get"):
            try:
                got = client.get(collection_name=collection_name, ids=batch, output_fields=output_fields)
                if got:
                    results.extend(got)
                continue
            except Exception as e:
                logger.warning(f"Milvus get方法查询失败，将回退至query方法：{str(e)}")

        # 方式2：get方法失败，回退使用filter过滤查询
        try:
            expr = f"chunk_id in [{', '.join(str(x) for x in batch)}]"
            q = client.query(collection_name=collection_name, filter=expr, output_fields=output_fields)
            if q:
                results.extend(q)
        except Exception as e:
            status = "error"
            logger.error(f"Milvus query方法批量查询chunk_id失败：{str(e)}", exc_info=True)

    record_milvus_operation("fetch_chunks_by_chunk_ids", collection_name, status, time.perf_counter() - start)
    return results


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """构建 Milvus 混合搜索请求对象。

    Args:
        dense_vector: 文本生成的稠密向量。
        sparse_vector: 文本生成的稀疏向量。
        dense_params: 稠密向量搜索参数。
        sparse_params: 稀疏向量搜索参数。
        expr: 搜索过滤表达式。
        limit: 单向量搜索返回结果数量。

    Returns:
        搜索请求列表，包含 dense 和 sparse 两类请求。
    """
    # 稠密向量默认搜索参数：余弦相似度（COSINE），适配BGE-M3稠密向量并与建库参数保持一致
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # 稀疏向量默认搜索参数：内积（IP），适配BGE-M3稀疏向量
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # 构建稠密向量搜索请求，关联Milvus的dense_vector字段 近似最近邻（ANN）检索请求的核心类
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    # 构建稀疏向量搜索请求，关联Milvus的sparse_vector字段
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """执行 Milvus 稠密和稀疏向量混合搜索。

    Args:
        client: MilvusClient 实例。
        collection_name: 集合名称。
        reqs: 搜索请求列表。
        ranker_weights: 稠密和稀疏检索的加权融合权重。
        norm_score: 是否在融合前归一化评分。
        limit: 混合搜索最终返回结果数量。
        output_fields: 需要返回的字段列表。
        search_params: Milvus 搜索参数。

    Returns:
        混合搜索结果列表，搜索失败返回 None。
    """
    start = time.perf_counter()
    status = "ok"
    try:
        # 初始化加权排名器：按权重融合稠密/稀疏向量的搜索结果
        # norm_score=True：先将两个向量评分归一化到0~1区间，再加权计算
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        # 默认返回字段：文档标识字段
        if output_fields is None:
            output_fields = ["item_name"]

        # 执行混合搜索：融合稠密+稀疏向量结果，按权重重新排序
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        status = "error"
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None
    finally:
        # 失败时也记录耗时和状态，便于排查向量库抖动。
        record_milvus_operation("hybrid_search", collection_name, status, time.perf_counter() - start)
