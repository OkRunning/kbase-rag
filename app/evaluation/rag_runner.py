"""RAG 离线评测运行器。"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List

from app.core.tracing import setup_tracing

setup_tracing()

from app.core.logger import logger
from app.evaluation.langfuse_eval import eval_trace, score_result
from app.evaluation.metrics import evaluate_case_result, extract_chunk_ids
from app.evaluation.schemas import EvalCase, EvalResult
from app.observability.checkpoints import build_langgraph_config
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state


def _merge_retrieval_ids(final_state: Dict[str, Any]) -> List[str]:
    """合并查询链路中各检索阶段的 chunk ID。

    Args:
        final_state: LangGraph 最终状态。

    Returns:
        按阶段顺序去重后的 chunk ID 列表。
    """
    merged: List[str] = []
    seen = set()
    # 评测关注召回链路的覆盖能力：先保留 RRF 排名，再补充两路原始召回。
    for key in ("rrf_chunks", "embedding_chunks", "hyde_embedding_chunks"):
        for chunk_id in extract_chunk_ids(final_state.get(key) or []):
            if chunk_id not in seen:
                merged.append(chunk_id)
                seen.add(chunk_id)
    return merged


def run_eval_case(
    case: EvalCase,
    profile: str,
    k: int = 5,
    enable_langfuse: bool = False,
) -> EvalResult:
    """运行单条 RAG 评测样本。

    Args:
        case: 评测样本。
        profile: 评测配置名称。
        k: TopK 指标截断位置。
        enable_langfuse: 是否把 trace 和 score 上报到 Langfuse。

    Returns:
        单条样本评测结果。
    """
    session_id = f"eval_{profile}_{case.case_id}"
    state = create_query_default_state(
        session_id=session_id,
        original_query=case.question,
        is_stream=False,
        eval_mode=True,
        tenant_id=case.tenant_id,
        kb_id=case.kb_id,
    )
    result = EvalResult(case=case, status="ok", session_id=session_id)

    with eval_trace(case, profile, enable_langfuse):
        try:
            # 离线评测直接调用 LangGraph，才能拿到检索、RRF、Rerank 等中间状态。
            final_state = query_app.invoke(
                state,
                config=build_langgraph_config(
                    "query.eval",
                    session_id,
                    tenant_id=case.tenant_id,
                    kb_id=case.kb_id,
                ),
            )
            final_state = final_state or {}
            answer = str(final_state.get("answer") or "")
            retrieved_ids = _merge_retrieval_ids(final_state)
            reranked_ids = extract_chunk_ids(final_state.get("reranked_docs") or [])
            result.answer = answer
            result.retrieved_chunk_ids = retrieved_ids
            result.reranked_chunk_ids = reranked_ids
            result.metrics = evaluate_case_result(
                case=case,
                answer=answer,
                retrieved_chunk_ids=retrieved_ids,
                reranked_chunk_ids=reranked_ids,
                k=k,
            )
        except Exception as exc:
            result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}"
            logger.error(f"评测样本 {case.case_id} 运行失败：{result.error}\n{traceback.format_exc()}")
        finally:
            score_result(result, enable_langfuse)
    return result


def run_eval_cases(
    cases: List[EvalCase],
    profile: str,
    k: int = 5,
    enable_langfuse: bool = False,
) -> List[EvalResult]:
    """批量运行 RAG 评测样本。

    Args:
        cases: 评测样本列表。
        profile: 评测配置名称。
        k: TopK 指标截断位置。
        enable_langfuse: 是否把 trace 和 score 上报到 Langfuse。

    Returns:
        单条样本评测结果列表。
    """
    results: List[EvalResult] = []
    for index, case in enumerate(cases, start=1):
        logger.info(f"开始运行评测样本 {index}/{len(cases)}：{case.case_id}")
        results.append(run_eval_case(case, profile=profile, k=k, enable_langfuse=enable_langfuse))
    return results
