"""RAG 评测报告生成。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.evaluation.schemas import EvalResult


def _average(values: Iterable[float]) -> float:
    """计算平均值。

    Args:
        values: 数值序列。

    Returns:
        平均值；空序列返回 0.0。
    """
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def summarize_results(results: List[EvalResult]) -> Dict[str, Any]:
    """汇总评测结果。

    Args:
        results: 单条样本评测结果列表。

    Returns:
        报告摘要字典。
    """
    metric_names = sorted({name for result in results for name in result.metrics})
    summary: Dict[str, Any] = {
        "total_cases": len(results),
        "ok_cases": sum(1 for result in results if result.status == "ok"),
        "failed_cases": sum(1 for result in results if result.status != "ok"),
        "pass_rate": _average(1.0 if result.passed() else 0.0 for result in results),
        "metrics": {},
    }
    for name in metric_names:
        summary["metrics"][name] = _average(result.metrics.get(name, 0.0) for result in results)
    return summary


def _first_metric(result: EvalResult, prefix: str) -> float:
    """读取指定前缀的首个指标。

    Args:
        result: 单条样本评测结果。
        prefix: 指标名前缀。

    Returns:
        指标值；不存在时返回 0.0。
    """
    for name, value in result.metrics.items():
        if name.startswith(prefix):
            return value
    return 0.0


def results_to_json(
    results: List[EvalResult],
    profile: str,
    baseline_comparison: Dict[str, Any] | None = None,
    threshold_violations: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """转换为 JSON 报告结构。

    Args:
        results: 单条样本评测结果列表。
        profile: 评测配置名称。
        baseline_comparison: 和 baseline 报告的对比结果。
        threshold_violations: 未满足阈值的指标列表。

    Returns:
        可序列化的 JSON 报告。
    """
    return {
        "profile": profile,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summarize_results(results),
        "baseline_comparison": baseline_comparison or {},
        "threshold_violations": threshold_violations or [],
        "cases": [asdict(result) for result in results],
    }


def render_markdown_report(
    results: List[EvalResult],
    profile: str,
    baseline_comparison: Dict[str, Any] | None = None,
    threshold_violations: List[Dict[str, Any]] | None = None,
) -> str:
    """渲染 Markdown 评测报告。

    Args:
        results: 单条样本评测结果列表。
        profile: 评测配置名称。
        baseline_comparison: 和 baseline 报告的对比结果。
        threshold_violations: 未满足阈值的指标列表。

    Returns:
        Markdown 报告文本。
    """
    summary = summarize_results(results)
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- profile: `{profile}`",
        f"- generated_at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- total_cases: `{summary['total_cases']}`",
        f"- ok_cases: `{summary['ok_cases']}`",
        f"- failed_cases: `{summary['failed_cases']}`",
        f"- pass_rate: `{summary['pass_rate']:.4f}`",
        "",
        "## Metrics",
        "",
        "| metric | average |",
        "| --- | ---: |",
    ]
    for name, value in summary["metrics"].items():
        lines.append(f"| `{name}` | {value:.4f} |")

    if baseline_comparison:
        lines.extend(
            [
                "",
                "## Baseline Comparison",
                "",
                "| metric | baseline | current | delta |",
                "| --- | ---: | ---: | ---: |",
                "| `pass_rate` | {baseline:.4f} | {current:.4f} | {delta:.4f} |".format(
                    baseline=baseline_comparison.get("pass_rate", {}).get("baseline", 0.0),
                    current=baseline_comparison.get("pass_rate", {}).get("current", 0.0),
                    delta=baseline_comparison.get("pass_rate", {}).get("delta", 0.0),
                ),
            ]
        )
        for name, values in (baseline_comparison.get("metrics") or {}).items():
            lines.append(
                "| `{name}` | {baseline:.4f} | {current:.4f} | {delta:.4f} |".format(
                    name=name,
                    baseline=values.get("baseline", 0.0),
                    current=values.get("current", 0.0),
                    delta=values.get("delta", 0.0),
                )
            )

    if threshold_violations:
        lines.extend(
            [
                "",
                "## Threshold Violations",
                "",
                "| metric | threshold | current | delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for item in threshold_violations:
            lines.append(
                "| `{metric}` | {threshold:.4f} | {current:.4f} | {delta:.4f} |".format(
                    metric=item.get("metric", ""),
                    threshold=item.get("threshold", 0.0),
                    current=item.get("current", 0.0),
                    delta=item.get("delta", 0.0),
                )
            )

    lines.extend(
        [
            "",
            "## Failed Cases",
            "",
            "| case_id | status | hit@5 | mrr@5 | reason |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for result in results:
        if result.passed():
            continue
        reason = result.error or "metric threshold not met"
        lines.append(
            "| `{case}` | {status} | {hit:.4f} | {mrr:.4f} | {reason} |".format(
                case=result.case.case_id,
                status=result.status,
                hit=result.metrics.get("hit_at_5", _first_metric(result, "hit_at_")),
                mrr=result.metrics.get("mrr_at_5", _first_metric(result, "mrr_at_")),
                reason=reason.replace("|", "\\|"),
            )
        )

    lines.extend(
        [
            "",
            "## Case Details",
            "",
        ]
    )
    for result in results:
        lines.extend(
            [
                f"### {result.case.case_id}",
                "",
                f"- status: `{result.status}`",
                f"- passed: `{result.passed()}`",
                f"- question: {result.case.question}",
                f"- expected_chunk_ids: `{result.case.expected_chunk_ids}`",
                f"- reranked_chunk_ids: `{result.reranked_chunk_ids}`",
                f"- retrieved_chunk_ids: `{result.retrieved_chunk_ids}`",
                f"- answer_chars: `{len(result.answer or '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_reports(
    results: List[EvalResult],
    profile: str,
    output_dir: str | Path,
    baseline_comparison: Dict[str, Any] | None = None,
    threshold_violations: List[Dict[str, Any]] | None = None,
) -> Dict[str, Path]:
    """写入 Markdown 和 JSON 评测报告。

    Args:
        results: 单条样本评测结果列表。
        profile: 评测配置名称。
        output_dir: 报告输出目录。
        baseline_comparison: 和 baseline 报告的对比结果。
        threshold_violations: 未满足阈值的指标列表。

    Returns:
        报告类型到文件路径的映射。
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"rag_eval_{profile}_{timestamp}"
    markdown_path = target_dir / f"{base_name}.md"
    json_path = target_dir / f"{base_name}.json"
    markdown_path.write_text(
        render_markdown_report(
            results,
            profile,
            baseline_comparison=baseline_comparison,
            threshold_violations=threshold_violations,
        ),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            results_to_json(
                results,
                profile,
                baseline_comparison=baseline_comparison,
                threshold_violations=threshold_violations,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"markdown": markdown_path, "json": json_path}
