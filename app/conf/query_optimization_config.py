"""查询链路优化开关配置。"""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔型环境变量。

    Args:
        name: 环境变量名称。
        default: 环境变量不存在时使用的默认值。

    Returns:
        解析后的布尔值。
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    """读取浮点型环境变量。

    Args:
        name: 环境变量名称。
        default: 环境变量不存在或解析失败时使用的默认值。

    Returns:
        解析后的浮点数。
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """读取整型环境变量。

    Args:
        name: 环境变量名称。
        default: 环境变量不存在或解析失败时使用的默认值。

    Returns:
        解析后的整数。
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class QueryOptimizationConfig:
    """查询链路优化灰度开关。

    Attributes:
        use_standard_understanding: 是否启用标准 Query Understanding 流程。
        use_retrieval_planner: 是否启用检索计划器。
        web_search_default: 是否默认启用 Web Search。
        web_search_on_low_confidence: 是否在本地低置信度时允许 Web 补充。
        enable_evidence_gate: 是否启用证据门控。
        enable_citations: 是否启用结构化引用。
        enable_rerank_gate: 是否启用 Rerank 前置跳过门控。
        rerank_fallback_enabled: 是否启用 Rerank 降级回退。
        low_confidence_refusal_enabled: 是否启用低置信度拒答。
        evidence_min_rerank_score: Evidence Gate 判定可回答的最低 top rerank 分数。
        evidence_min_docs: Evidence Gate 判定可回答的最低证据数量。
        allow_external_only: 是否允许仅依赖外部 Web 证据生成答案。
    """

    use_standard_understanding: bool
    use_retrieval_planner: bool
    web_search_default: bool
    web_search_on_low_confidence: bool
    enable_evidence_gate: bool
    enable_citations: bool
    enable_rerank_gate: bool
    rerank_fallback_enabled: bool
    low_confidence_refusal_enabled: bool
    evidence_min_rerank_score: float
    evidence_min_docs: int
    allow_external_only: bool


# 默认关闭新流程能力，只保留不改变旧链路行为的保护型开关。
query_optimization_config = QueryOptimizationConfig(
    use_standard_understanding=_env_bool("QUERY_USE_STANDARD_UNDERSTANDING", False),
    use_retrieval_planner=_env_bool("QUERY_USE_RETRIEVAL_PLANNER", False),
    web_search_default=_env_bool("QUERY_WEB_SEARCH_DEFAULT", False),
    web_search_on_low_confidence=_env_bool("QUERY_WEB_SEARCH_ON_LOW_CONFIDENCE", True),
    enable_evidence_gate=_env_bool("QUERY_ENABLE_EVIDENCE_GATE", False),
    enable_citations=_env_bool("QUERY_ENABLE_CITATIONS", False),
    enable_rerank_gate=_env_bool("QUERY_ENABLE_RERANK_GATE", False),
    rerank_fallback_enabled=_env_bool("QUERY_RERANK_FALLBACK_ENABLED", True),
    low_confidence_refusal_enabled=_env_bool("QUERY_LOW_CONFIDENCE_REFUSAL_ENABLED", False),
    evidence_min_rerank_score=_env_float("QUERY_EVIDENCE_MIN_RERANK_SCORE", 0.45),
    evidence_min_docs=_env_int("QUERY_EVIDENCE_MIN_DOCS", 1),
    allow_external_only=_env_bool("QUERY_ALLOW_EXTERNAL_ONLY", False),
)
