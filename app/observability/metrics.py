"""记录业务可观测指标。

该模块封装 OpenTelemetry Metrics API，避免业务代码直接操作指标对象。
指标标签保持低基数，不把 session_id、task_id、trace_id、prompt、
chunk 文本等高基数或敏感内容写入标签。
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

from opentelemetry import metrics

from app.observability.settings import is_observability_enabled


class _NoopInstrument:
    """关闭可观测能力时使用的空指标对象。"""

    def add(self, *args, **kwargs) -> None:
        """忽略 Counter/UpDownCounter 写入。

        Args:
            *args: 指标参数。
            **kwargs: 指标属性。

        Returns:
            None.
        """
        return None

    def record(self, *args, **kwargs) -> None:
        """忽略 Histogram 写入。

        Args:
            *args: 指标参数。
            **kwargs: 指标属性。

        Returns:
            None.
        """
        return None


class _NoopMeter:
    """关闭可观测能力时使用的空 Meter。"""

    def create_counter(self, *args, **kwargs) -> _NoopInstrument:
        """创建空 Counter。

        Args:
            *args: 指标定义参数。
            **kwargs: 指标定义属性。

        Returns:
            空指标对象。
        """
        return _NoopInstrument()

    def create_histogram(self, *args, **kwargs) -> _NoopInstrument:
        """创建空 Histogram。

        Args:
            *args: 指标定义参数。
            **kwargs: 指标定义属性。

        Returns:
            空指标对象。
        """
        return _NoopInstrument()

    def create_up_down_counter(self, *args, **kwargs) -> _NoopInstrument:
        """创建空 UpDownCounter。

        Args:
            *args: 指标定义参数。
            **kwargs: 指标定义属性。

        Returns:
            空指标对象。
        """
        return _NoopInstrument()


_meter = metrics.get_meter("dataset-rag.observability") if is_observability_enabled() else _NoopMeter()
_state_lock = threading.Lock()
_circuit_state_values: Dict[Tuple[str, str], int] = {}
_consecutive_failure_values: Dict[str, int] = {}

# ---- Histogram bucket 边界 ----
# 显式声明 bucket 边界，覆盖 OpenTelemetry SDK 默认边界（按毫秒设计，
# 0/5/10/25...），避免秒级耗时几乎全部塌缩到 (0, 5] 单个桶，导致
# histogram_quantile 算出的各节点 P95 完全一致。
# 秒级通用耗时桶，覆盖毫秒到两分钟，适配节点/检索/模型/中间件等大多数操作延迟。
_DURATION_BOUNDARIES: Tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
)
# 查询工作流端到端耗时桶，覆盖亚秒到两分钟，对齐 15 秒查询延迟 SLO。
_QUERY_DURATION_BOUNDARIES: Tuple[float, ...] = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0,
)
# 导入工作流端到端耗时桶，覆盖秒到三十分钟，对齐 PDF 5 分钟、MD 2 分钟导入耗时 SLO。
_IMPORT_DURATION_BOUNDARIES: Tuple[float, ...] = (
    0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 180.0, 300.0, 600.0, 1200.0, 1800.0,
)
# 任务年龄桶，覆盖秒到一天，用于发现滞留在 processing 的查询或导入任务。
_TASK_AGE_BOUNDARIES: Tuple[float, ...] = (
    1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0, 18000.0, 86400.0,
)
# 批量大小桶，覆盖单条到两千条，用于 embedding 批量大小分布。
_BATCH_SIZE_BOUNDARIES: Tuple[float, ...] = (
    1.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,
)
# 文档计数桶，覆盖个位到千级，用于召回和 rerank 文档数量分布。
_DOC_COUNT_BOUNDARIES: Tuple[float, ...] = (
    1.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0, 200.0, 500.0, 1000.0,
)
# 队列深度桶，覆盖个位到千级，用于 SSE 队列积压观测。
_QUEUE_DEPTH_BOUNDARIES: Tuple[float, ...] = (
    1.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0,
)
# 容量已用比例桶，0 到 1 区间按 10% 步进并补充 95%/99%，用于容量告警趋势判断。
_RATIO_BOUNDARIES: Tuple[float, ...] = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
)


# 查询工作流总次数，用于计算查询吞吐、错误率和降级率。
_query_total = _meter.create_counter(
    "rag_query_total",
    description="Total query workflow executions.",
)
# 查询工作流端到端耗时，用于计算 P95/P99 延迟 SLO。
_query_duration = _meter.create_histogram(
    "rag_query_duration_seconds",
    explicit_bucket_boundaries_advisory=_QUERY_DURATION_BOUNDARIES,
    unit="s",
    description="Query workflow duration.",
)
# 导入工作流总次数，用于计算导入吞吐和失败率。
_import_total = _meter.create_counter(
    "rag_import_total",
    description="Total import workflow executions.",
)
# 导入工作流端到端耗时，用于识别慢导入和文件类型差异。
_import_duration = _meter.create_histogram(
    "rag_import_duration_seconds",
    explicit_bucket_boundaries_advisory=_IMPORT_DURATION_BOUNDARIES,
    unit="s",
    description="Import workflow duration.",
)
# LangGraph 单节点耗时，用于定位查询或导入流程中的慢节点。
_node_duration = _meter.create_histogram(
    "rag_node_duration_seconds",
    explicit_bucket_boundaries_advisory=_DURATION_BOUNDARIES,
    unit="s",
    description="LangGraph node execution duration.",
)
# SSE 事件发送次数，用于观察流式响应事件量、断连和队列缺失。
_sse_events = _meter.create_counter(
    "rag_sse_events_total",
    description="Total SSE events sent.",
)
# HTTP 请求总次数，用于 API 层吞吐和状态码分布观测。
_http_requests = _meter.create_counter(
    "rag_http_requests_total",
    description="Total HTTP requests observed by application middleware.",
)
# HTTP 请求耗时，用于识别 API 层慢请求。
_http_duration = _meter.create_histogram(
    "rag_http_request_duration_seconds",
    explicit_bucket_boundaries_advisory=_DURATION_BOUNDARIES,
    unit="s",
    description="HTTP request duration observed by application middleware.",
)
# Embedding 生成耗时，用于发现向量化模型慢调用。
_embedding_duration = _meter.create_histogram(
    "rag_embedding_duration_seconds",
    explicit_bucket_boundaries_advisory=_DURATION_BOUNDARIES,
    unit="s",
    description="Embedding generation duration.",
)
# Embedding 批量大小分布，用于关联批量大小和耗时抖动。
_embedding_batch_size = _meter.create_histogram(
    "rag_embedding_batch_size",
    explicit_bucket_boundaries_advisory=_BATCH_SIZE_BOUNDARIES,
    description="Embedding generation batch size.",
)
# Milvus 操作总次数，用于计算向量库错误率。
_milvus_operations = _meter.create_counter(
    "rag_milvus_operations_total",
    description="Total Milvus operations.",
)
# Milvus 操作耗时，用于发现向量检索或批量查询慢调用。
_milvus_operation_duration = _meter.create_histogram(
    "rag_milvus_operation_duration_seconds",
    explicit_bucket_boundaries_advisory=_DURATION_BOUNDARIES,
    unit="s",
    description="Milvus operation duration.",
)
# LLM 调用总次数，用于计算模型错误率、fallback 率和 JSON 模式差异。
_llm_requests = _meter.create_counter(
    "rag_llm_requests_total",
    description="Total LLM calls.",
)
# LLM 调用耗时，用于计算模型 P95/P99 延迟。
_llm_duration = _meter.create_histogram(
    "rag_llm_duration_seconds",
    explicit_bucket_boundaries_advisory=_DURATION_BOUNDARIES,
    unit="s",
    description="LLM call duration.",
)
# LLM token 消耗总量，用于观察成本趋势和异常 token 暴涨。
_llm_tokens = _meter.create_counter(
    "rag_llm_tokens_total",
    description="Total LLM tokens.",
)
# 流式 LLM 首 token 延迟，用于衡量用户等待首包的体验。
_llm_stream_first_token = _meter.create_histogram(
    "rag_llm_stream_first_token_seconds",
    explicit_bucket_boundaries_advisory=_DURATION_BOUNDARIES,
    unit="s",
    description="LLM stream time to first token.",
)
# Reranker 执行耗时，用于发现精排模型慢调用。
_rerank_duration = _meter.create_histogram(
    "rag_rerank_duration_seconds",
    explicit_bucket_boundaries_advisory=_DURATION_BOUNDARIES,
    unit="s",
    description="Reranker execution duration.",
)
# Rerank 输入、打分后和输出文档数量，用于观察精排截断效果。
_rerank_docs = _meter.create_histogram(
    "rag_rerank_docs_total",
    explicit_bucket_boundaries_advisory=_DOC_COUNT_BOUNDARIES,
    description="Reranker input or output document count.",
)
# 各召回来源文档数量，用于发现召回不足或某一路检索退化。
_retrieval_docs = _meter.create_histogram(
    "rag_retrieval_docs_total",
    explicit_bucket_boundaries_advisory=_DOC_COUNT_BOUNDARIES,
    description="Retrieved document count by source.",
)
# MongoDB 操作总次数，用于计算历史、事件等 Mongo 操作错误率。
_mongo_operations = _meter.create_counter(
    "rag_mongo_operations_total",
    description="Total MongoDB operations.",
)
# MongoDB 操作耗时，用于发现慢查询或写入抖动。
_mongo_operation_duration = _meter.create_histogram(
    "rag_mongo_operation_duration_seconds",
    explicit_bucket_boundaries_advisory=_DURATION_BOUNDARIES,
    unit="s",
    description="MongoDB operation duration.",
)
# MinIO 操作总次数，用于计算快照对象存储错误率。
_minio_operations = _meter.create_counter(
    "rag_minio_operations_total",
    description="Total MinIO operations.",
)
# MinIO 操作耗时，用于发现快照保存或读取变慢。
_minio_operation_duration = _meter.create_histogram(
    "rag_minio_operation_duration_seconds",
    explicit_bucket_boundaries_advisory=_DURATION_BOUNDARIES,
    unit="s",
    description="MinIO operation duration.",
)
# 事件存储写入次数，用于区分 Mongo 写入和 JSONL 兜底写入状态。
_event_store_writes = _meter.create_counter(
    "rag_event_store_writes_total",
    description="Total observability event store write attempts.",
)
# 事件存储降级次数，用于发现中心化 timeline 静默不可用。
_event_store_fallback = _meter.create_counter(
    "rag_event_store_fallback_total",
    description="Total event store fallback occurrences.",
)
# 快照保存或读取次数，用于观察节点回放所依赖的快照能力是否可用。
_snapshot_total = _meter.create_counter(
    "rag_snapshot_total",
    description="Total observability snapshot operations.",
)
# 回放执行次数，用于计算 replay API 成功率和失败率。
_replay_total = _meter.create_counter(
    "rag_replay_total",
    description="Total replay executions.",
)
# 当前 SSE 连接数，active 状态加一减一，观察流式连接并发。
_sse_connections = _meter.create_up_down_counter(
    "rag_sse_connections",
    description="Current SSE connections by state.",
)
# SSE 队列深度，用于发现客户端消费慢或后台推送积压。
_sse_queue_depth = _meter.create_histogram(
    "rag_sse_queue_depth",
    explicit_bucket_boundaries_advisory=_QUEUE_DEPTH_BOUNDARIES,
    description="SSE queue depth.",
)
# 当前 workflow 在途任务数，用于观察查询和导入后台任务并发。
_task_inflight = _meter.create_up_down_counter(
    "rag_task_inflight",
    description="Current workflow tasks in flight.",
)
# 任务年龄分布，用于发现 processing 状态长时间没有进展的查询或导入任务。
_task_age = _meter.create_histogram(
    "rag_task_age_seconds",
    explicit_bucket_boundaries_advisory=_TASK_AGE_BOUNDARIES,
    unit="s",
    description="Workflow task age observed by status scanners.",
)
# 合成探测次数，用于在用户请求前发现服务或观测能力异常。
_synthetic_probe_total = _meter.create_counter(
    "rag_synthetic_probe_total",
    description="Total synthetic probe executions.",
)
# 合成探测耗时，用于观察健康探测自身是否变慢。
_synthetic_probe_duration = _meter.create_histogram(
    "rag_synthetic_probe_duration_seconds",
    explicit_bucket_boundaries_advisory=_DURATION_BOUNDARIES,
    unit="s",
    description="Synthetic probe duration.",
)
# 容量余量状态，低于阈值时置 0，正常置 1。
_capacity_headroom = _meter.create_up_down_counter(
    "rag_capacity_headroom",
    description="Capacity headroom marker. Healthy resource has value 1.",
)
# 容量已用比例，用于本地日志、输出目录和观测栈数据目录的趋势判断。
_capacity_used_ratio = _meter.create_histogram(
    "rag_capacity_used_ratio",
    explicit_bucket_boundaries_advisory=_RATIO_BOUNDARIES,
    description="Capacity used ratio observed by automation probes.",
)
# 熔断状态标记，当前状态值为 1，其余状态值为 0。
_circuit_state = _meter.create_up_down_counter(
    "rag_circuit_state",
    description="Circuit state marker. Active state has value 1.",
)
# 连续失败计数，用于观察熔断打开前的失败趋势。
_consecutive_failures = _meter.create_up_down_counter(
    "rag_consecutive_failures",
    description="Current consecutive failure count.",
)
# 依赖健康状态，健康为 1，不健康或降级为 0。
_dependency_health = _meter.create_up_down_counter(
    "rag_dependency_health",
    description="Dependency health marker. Healthy dependency has value 1.",
)
# 文档版本物理清理次数，用于观察 archived/deleted/failed 过期数据治理效果。
_document_cleanup_total = _meter.create_counter(
    "rag_document_cleanup_total",
    description="Total document version cleanup executions.",
)
_dependency_health_values: Dict[str, int] = {}
_capacity_headroom_values: Dict[Tuple[str, str], int] = {}


def record_query(status: str, is_stream: bool, duration_seconds: float, evidence_status: str = "") -> None:
    """记录查询工作流指标。

    Args:
        status: 查询结果状态。
        is_stream: 是否为流式查询。
        duration_seconds: 查询耗时，单位秒。
        evidence_status: Evidence Gate 证据状态，用于计算据答率。

    Returns:
        None.
    """
    attrs = {
        "status": status,
        "is_stream": str(bool(is_stream)).lower(),
        "evidence_status": evidence_status or "unknown",
    }
    _query_total.add(1, attrs)
    _query_duration.record(duration_seconds, attrs)


def record_import(status: str, file_type: str, duration_seconds: float) -> None:
    """记录文档导入工作流指标。

    Args:
        status: 导入结果状态。
        file_type: 导入文件类型。
        duration_seconds: 导入耗时，单位秒。

    Returns:
        None.
    """
    attrs = {"status": status, "file_type": file_type or "unknown"}
    _import_total.add(1, attrs)
    _import_duration.record(duration_seconds, attrs)


def record_node_duration(
    graph: str,
    node_name: str,
    status: str,
    duration_seconds: float,
) -> None:
    """记录 LangGraph 节点耗时。

    Args:
        graph: 工作流名称。
        node_name: LangGraph 节点名称。
        status: 节点执行状态。
        duration_seconds: 节点耗时，单位秒。

    Returns:
        None.
    """
    _node_duration.record(
        duration_seconds,
        {
            "graph": graph or "unknown",
            "node_name": node_name or "unknown",
            "status": status,
        },
    )


def record_sse_event(event: str) -> None:
    """记录 SSE 事件发送次数。

    Args:
        event: SSE 事件类型。

    Returns:
        None.
    """
    _sse_events.add(1, {"event": event or "unknown"})


def record_http_request(route: str, method: str, status_code: int, duration_seconds: float) -> None:
    """记录 HTTP 请求指标。

    Args:
        route: FastAPI 路由模板。
        method: HTTP 方法。
        status_code: HTTP 状态码。
        duration_seconds: 请求耗时，单位秒。

    Returns:
        None.
    """
    attrs = {
        "route": route or "unknown",
        "method": method or "unknown",
        "status_code": str(status_code),
    }
    _http_requests.add(1, attrs)
    _http_duration.record(duration_seconds, attrs)


def record_embedding(model: str, device: str, status: str, batch_size: int, duration_seconds: float) -> None:
    """记录 Embedding 生成指标。

    Args:
        model: Embedding 模型名称。
        device: 模型运行设备。
        status: 生成结果状态。
        batch_size: 本次生成文本数量。
        duration_seconds: 生成耗时，单位秒。

    Returns:
        None.
    """
    attrs = {
        "model": model or "unknown",
        "device": device or "unknown",
        "status": status,
    }
    _embedding_duration.record(duration_seconds, attrs)
    _embedding_batch_size.record(batch_size, {"model": model or "unknown"})


def record_milvus_operation(
    operation: str,
    collection: str,
    status: str,
    duration_seconds: float,
) -> None:
    """记录 Milvus 操作指标。

    Args:
        operation: Milvus 操作名称。
        collection: Milvus collection 名称。
        status: 操作结果状态。
        duration_seconds: 操作耗时，单位秒。

    Returns:
        None.
    """
    attrs = {
        "operation": operation or "unknown",
        "collection": collection or "unknown",
        "status": status,
    }
    _milvus_operations.add(1, attrs)
    _milvus_operation_duration.record(duration_seconds, attrs)


def record_llm_call(
    model: str,
    status: str,
    json_mode: bool,
    fallback: bool,
    duration_seconds: float,
) -> None:
    """记录 LLM 调用结果和耗时。

    Args:
        model: 模型名称。
        status: 调用状态。
        json_mode: 是否启用 JSON 模式。
        fallback: 是否使用备用模型。
        duration_seconds: 调用耗时，单位秒。

    Returns:
        None.
    """
    attrs = {
        "model": model or "unknown",
        "status": status,
        "json_mode": str(bool(json_mode)).lower(),
        "fallback": str(bool(fallback)).lower(),
    }
    _llm_requests.add(1, attrs)
    _llm_duration.record(duration_seconds, attrs)


def record_llm_tokens(model: str, token_type: str, count: int) -> None:
    """记录 LLM token 消耗。

    Args:
        model: 模型名称。
        token_type: token 类型，例如 input、output、total 或 unknown。
        count: token 数量。

    Returns:
        None.
    """
    if count <= 0:
        return
    _llm_tokens.add(count, {"model": model or "unknown", "type": token_type or "unknown"})


def record_llm_stream_first_token(model: str, status: str, duration_seconds: float) -> None:
    """记录流式 LLM 首 token 延迟。

    Args:
        model: 模型名称。
        status: 流式调用状态。
        duration_seconds: 首 token 延迟，单位秒。

    Returns:
        None.
    """
    _llm_stream_first_token.record(
        duration_seconds,
        {"model": model or "unknown", "status": status},
    )


def record_rerank(model: str, status: str, duration_seconds: float) -> None:
    """记录 reranker 耗时。

    Args:
        model: reranker 模型名称。
        status: 执行状态。
        duration_seconds: 执行耗时，单位秒。

    Returns:
        None.
    """
    _rerank_duration.record(duration_seconds, {"model": model or "unknown", "status": status})


def record_rerank_docs(stage: str, count: int) -> None:
    """记录 rerank 输入或输出文档数。

    Args:
        stage: 阶段名称，例如 input、scored 或 output。
        count: 文档数量。

    Returns:
        None.
    """
    _rerank_docs.record(max(count, 0), {"stage": stage or "unknown"})


def record_retrieval_docs(source: str, count: int) -> None:
    """记录召回文档数量。

    Args:
        source: 召回来源，例如 embedding、hyde、web 或 rrf。
        count: 文档数量。

    Returns:
        None.
    """
    _retrieval_docs.record(max(count, 0), {"source": source or "unknown"})


def record_mongo_operation(
    operation: str,
    collection: str,
    status: str,
    duration_seconds: float,
) -> None:
    """记录 MongoDB 操作指标。

    Args:
        operation: 操作名称。
        collection: 集合名称。
        status: 操作状态。
        duration_seconds: 操作耗时，单位秒。

    Returns:
        None.
    """
    attrs = {
        "operation": operation or "unknown",
        "collection": collection or "unknown",
        "status": status,
    }
    _mongo_operations.add(1, attrs)
    _mongo_operation_duration.record(duration_seconds, attrs)


def record_document_cleanup(status: str, target: str) -> None:
    """记录文档版本物理清理结果。

    Args:
        status: 清理结果状态，例如 ok 或 error。
        target: 清理目标类型，例如 archived、deleted 或 failed。

    Returns:
        None.
    """
    _document_cleanup_total.add(1, {"status": status or "unknown", "target": target or "unknown"})


def record_minio_operation(
    operation: str,
    bucket: str,
    status: str,
    duration_seconds: float,
) -> None:
    """记录 MinIO 操作指标。

    Args:
        operation: 操作名称。
        bucket: bucket 名称。
        status: 操作状态。
        duration_seconds: 操作耗时，单位秒。

    Returns:
        None.
    """
    attrs = {
        "operation": operation or "unknown",
        "bucket": bucket or "unknown",
        "status": status,
    }
    _minio_operations.add(1, attrs)
    _minio_operation_duration.record(duration_seconds, attrs)


def record_event_store_write(backend: str, status: str) -> None:
    """记录事件存储写入结果。

    Args:
        backend: 写入后端，例如 mongo 或 jsonl。
        status: 写入状态。

    Returns:
        None.
    """
    _event_store_writes.add(1, {"backend": backend or "unknown", "status": status})


def record_event_store_fallback(reason: str) -> None:
    """记录事件存储降级次数。

    Args:
        reason: 降级原因。

    Returns:
        None.
    """
    _event_store_fallback.add(1, {"reason": reason or "unknown"})


def record_snapshot(backend: str, graph: str, kind: str, status: str) -> None:
    """记录快照保存或读取结果。

    Args:
        backend: 快照后端，例如 minio 或 local。
        graph: 工作流名称。
        kind: 快照类型。
        status: 操作状态。

    Returns:
        None.
    """
    _snapshot_total.add(
        1,
        {
            "backend": backend or "unknown",
            "graph": graph or "unknown",
            "kind": kind or "unknown",
            "status": status,
        },
    )


def record_replay(graph: str, mode: str, status: str) -> None:
    """记录回放执行结果。

    Args:
        graph: 工作流名称。
        mode: 回放模式。
        status: 回放状态。

    Returns:
        None.
    """
    _replay_total.add(1, {"graph": graph or "unknown", "mode": mode or "unknown", "status": status})


def record_sse_connection(state: str, delta: int) -> None:
    """记录 SSE 连接状态变化。

    Args:
        state: 连接状态，例如 active。
        delta: 连接数变化量。

    Returns:
        None.
    """
    _sse_connections.add(delta, {"state": state or "unknown"})


def record_sse_queue_depth(depth: int) -> None:
    """记录 SSE 队列深度。

    Args:
        depth: 当前队列深度。

    Returns:
        None.
    """
    _sse_queue_depth.record(max(depth, 0))


def record_task_inflight(graph: str, delta: int) -> None:
    """记录 workflow 并发任务数变化。

    Args:
        graph: 工作流名称。
        delta: 并发任务数变化量。

    Returns:
        None.
    """
    _task_inflight.add(delta, {"graph": graph or "unknown"})


def record_task_age(graph: str, status: str, age_seconds: float) -> None:
    """记录任务年龄。

    Args:
        graph: 工作流名称。
        status: 任务状态。
        age_seconds: 任务从创建到当前的年龄，单位秒。

    Returns:
        None.
    """
    _task_age.record(max(float(age_seconds or 0.0), 0.0), {"graph": graph or "unknown", "status": status or "unknown"})


def record_synthetic_probe(service: str, probe: str, status: str, duration_seconds: float) -> None:
    """记录合成探测结果。

    Args:
        service: 服务名称。
        probe: 探测类型。
        status: 探测状态。
        duration_seconds: 探测耗时，单位秒。

    Returns:
        None.
    """
    attrs = {"service": service or "unknown", "probe": probe or "unknown", "status": status or "unknown"}
    _synthetic_probe_total.add(1, attrs)
    _synthetic_probe_duration.record(max(float(duration_seconds or 0.0), 0.0), attrs)


def record_capacity(resource: str, scope: str, status: str, used_ratio: float) -> None:
    """记录容量状态和已用比例。

    Args:
        resource: 资源类型，例如 filesystem、prometheus、loki。
        scope: 资源范围或路径别名。
        status: 容量状态，支持 ok、warning、critical。
        used_ratio: 已用比例，范围 0 到 1。

    Returns:
        None.
    """
    normalized_resource = resource or "unknown"
    normalized_scope = scope or "unknown"
    normalized_status = status or "unknown"
    ratio = min(max(float(used_ratio or 0.0), 0.0), 1.0)
    _capacity_used_ratio.record(
        ratio,
        {"resource": normalized_resource, "scope": normalized_scope, "status": normalized_status},
    )
    target = 1 if normalized_status == "ok" else 0
    key = (normalized_resource, normalized_scope)
    with _state_lock:
        previous = _capacity_headroom_values.get(key, 0)
        delta = target - previous
        if delta:
            _capacity_headroom.add(delta, {"resource": normalized_resource, "scope": normalized_scope})
            _capacity_headroom_values[key] = target


def record_circuit_state(name: str, state: str, consecutive_failures: int) -> None:
    """记录熔断器状态和连续失败数。

    Args:
        name: 熔断器名称。
        state: 当前状态，支持 closed、open、half_open。
        consecutive_failures: 连续失败次数。

    Returns:
        None.
    """
    normalized_name = name or "unknown"
    normalized_state = state or "unknown"
    with _state_lock:
        for candidate in ("closed", "open", "half_open"):
            key = (normalized_name, candidate)
            previous = _circuit_state_values.get(key, 0)
            target = 1 if candidate == normalized_state else 0
            delta = target - previous
            if delta:
                _circuit_state.add(delta, {"name": normalized_name, "state": candidate})
                _circuit_state_values[key] = target

        previous_failures = _consecutive_failure_values.get(normalized_name, 0)
        target_failures = max(int(consecutive_failures or 0), 0)
        delta_failures = target_failures - previous_failures
        if delta_failures:
            _consecutive_failures.add(delta_failures, {"name": normalized_name})
            _consecutive_failure_values[normalized_name] = target_failures


def record_dependency_health(dependency: str, healthy: bool) -> None:
    """记录依赖健康状态。

    Args:
        dependency: 依赖名称。
        healthy: 是否健康。

    Returns:
        None.
    """
    normalized = dependency or "unknown"
    target = 1 if healthy else 0
    with _state_lock:
        previous = _dependency_health_values.get(normalized, 0)
        delta = target - previous
        if delta:
            _dependency_health.add(delta, {"dependency": normalized})
            _dependency_health_values[normalized] = target
