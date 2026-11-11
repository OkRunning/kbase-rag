"""初始化 dataset-rag 的 OpenTelemetry 能力。

该模块负责配置 trace、metric、上下文传播和基础自动 instrumentation。
入口服务必须在导入业务模块前调用 setup_tracing，确保节点包装器和客户端
instrumentation 使用真实 provider。
"""

import os
from typing import Dict

from dotenv import load_dotenv
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3Format
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_ON,
    ParentBased,
    TraceIdRatioBased,
)

_initialized = False


def _parse_otlp_headers(raw: str) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            continue
        headers[key.strip()] = value.strip()
    return headers


def _build_resource() -> Resource:
    return Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", "dataset-rag"),
            "service.version": os.getenv("OTEL_SERVICE_VERSION", "0.1.0"),
            "deployment.environment": os.getenv(
                "OTEL_ENVIRONMENT",
                os.getenv("ENVIRONMENT", "development"),
            ),
        }
    )


def _build_sampler():
    sampler = os.getenv("OTEL_TRACES_SAMPLER", "always_on").lower()
    if sampler in {"parentbased_traceidratio", "traceidratio"}:
        ratio = float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0"))
        ratio_sampler = TraceIdRatioBased(ratio)
        if sampler.startswith("parentbased"):
            return ParentBased(ratio_sampler)
        return ratio_sampler
    return ALWAYS_ON


def setup_tracing():
    """初始化 OpenTelemetry tracing、metrics 和自动 instrumentation。

    Returns:
        当前模块的 OpenTelemetry tracer。
    """
    global _initialized
    # 多个 API 服务或 reload 进程可能重复调用，初始化必须保持幂等。
    if _initialized:
        return trace.get_tracer(__name__)

    load_dotenv()
    resource = _build_resource()
    headers = _parse_otlp_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", ""))
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    # 采样策略集中在 provider 层配置，业务代码只负责创建 span。
    trace_provider = TracerProvider(resource=resource, sampler=_build_sampler())
    trace.set_tracer_provider(trace_provider)
    trace_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{endpoint.rstrip('/')}/v1/traces",
                headers=headers,
            )
        )
    )

    if os.getenv("OTEL_CONSOLE_EXPORTER", "false").lower() == "true":
        trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(
            endpoint=f"{endpoint.rstrip('/')}/v1/metrics",
            headers=headers,
        )
    )
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))

    set_global_textmap(B3Format())

    # 自动 instrumentation 是辅助能力，失败不能影响服务启动。
    for instrumentor in (RequestsInstrumentor(), PymongoInstrumentor()):
        try:
            instrumentor.instrument()
        except Exception:
            pass

    _initialized = True
    return trace.get_tracer(__name__)
