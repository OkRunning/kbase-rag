"""Langfuse 评测上报适配层。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator

from app.core.logger import logger
from app.evaluation.schemas import EvalCase, EvalResult


def _get_langfuse_client() -> Any:
    """获取 Langfuse 客户端。

    Returns:
        Langfuse 客户端对象。

    Raises:
        RuntimeError: Langfuse 客户端不可用时抛出。
    """
    try:
        from app.clients.langfuse import langfuse
    except Exception as exc:
        raise RuntimeError(f"Langfuse client is unavailable: {exc}") from exc
    return langfuse


@contextmanager
def eval_trace(case: EvalCase, profile: str, enabled: bool) -> Iterator[Any]:
    """创建可选的 Langfuse 评测 trace。

    Args:
        case: 当前评测样本。
        profile: 评测配置名称。
        enabled: 是否启用 Langfuse 上报。

    Yields:
        Langfuse observation；未启用或创建失败时为 None。
    """
    if not enabled:
        yield None
        return

    try:
        from app.clients.langfuse import propagate_attributes

        langfuse = _get_langfuse_client()
        metadata = {
            "case_id": case.case_id,
            "profile": profile,
            "tenant_id": case.tenant_id,
            "kb_id": case.kb_id,
            "query_type": case.query_type,
            "difficulty": case.difficulty,
            "tags": case.tags,
        }
        with propagate_attributes(
            session_id=f"eval:{profile}:{case.case_id}",
            user_id="rag_eval",
            tags=["rag_eval", profile, case.query_type, case.difficulty],
            trace_name="rag_eval",
            metadata=metadata,
        ):
            with langfuse.start_as_current_observation(
                name="rag_eval.case",
                as_type="span",
                input=case.to_langfuse_input(),
                metadata=metadata,
            ) as observation:
                langfuse.set_current_trace_io(input=case.to_langfuse_input())
                yield observation
    except Exception as exc:
        # Langfuse 只是评测展示层，失败不应阻断本地质量基线生成。
        logger.warning(f"[Langfuse Eval] 创建评测 trace 失败，已降级为本地评测：{exc}")
        yield None


def score_result(result: EvalResult, enabled: bool) -> None:
    """将评测指标写入当前 Langfuse trace。

    Args:
        result: 单条样本评测结果。
        enabled: 是否启用 Langfuse 上报。

    Returns:
        None.
    """
    if not enabled:
        return

    try:
        langfuse = _get_langfuse_client()
        output: Dict[str, Any] = {
            "answer": result.answer,
            "status": result.status,
            "retrieved_chunk_ids": result.retrieved_chunk_ids,
            "reranked_chunk_ids": result.reranked_chunk_ids,
            "metrics": result.metrics,
            "error": result.error,
        }
        langfuse.set_current_trace_io(input=result.case.to_langfuse_input(), output=output)
        for name, value in result.metrics.items():
            langfuse.score_current_trace(name=name, value=float(value))
        langfuse.score_current_trace(name="eval_passed", value=1.0 if result.passed() else 0.0)
    except Exception as exc:
        logger.warning(f"[Langfuse Eval] 写入评测 score 失败，已忽略：{exc}")


def flush(enabled: bool) -> None:
    """刷新 Langfuse 客户端缓冲区。

    Args:
        enabled: 是否启用 Langfuse 上报。

    Returns:
        None.
    """
    if not enabled:
        return
    try:
        langfuse = _get_langfuse_client()
        flush_fn = getattr(langfuse, "flush", None)
        if flush_fn:
            flush_fn()
    except Exception as exc:
        logger.warning(f"[Langfuse Eval] flush 失败，已忽略：{exc}")
