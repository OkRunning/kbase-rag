"""RAG 评测 baseline 对比工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_report_summary(report_path: str | Path) -> Dict[str, Any]:
    """加载历史评测报告摘要。

    Args:
        report_path: JSON 评测报告路径。

    Returns:
        报告中的 summary 字典。

    Raises:
        FileNotFoundError: 报告文件不存在。
        ValueError: 报告缺少 summary 时抛出。
    """
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"baseline report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"baseline report missing summary: {path}")
    return summary


def compare_summary(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """对比当前评测摘要和 baseline 摘要。

    Args:
        current: 当前报告 summary。
        baseline: 历史 baseline summary。

    Returns:
        包含 pass_rate 和指标 delta 的对比结果。
    """
    current_metrics = current.get("metrics") or {}
    baseline_metrics = baseline.get("metrics") or {}
    metric_names = sorted(set(current_metrics) | set(baseline_metrics))
    metric_deltas = {}
    for name in metric_names:
        current_value = float(current_metrics.get(name, 0.0))
        baseline_value = float(baseline_metrics.get(name, 0.0))
        metric_deltas[name] = {
            "current": current_value,
            "baseline": baseline_value,
            "delta": current_value - baseline_value,
        }
    current_pass_rate = float(current.get("pass_rate", 0.0))
    baseline_pass_rate = float(baseline.get("pass_rate", 0.0))
    return {
        "pass_rate": {
            "current": current_pass_rate,
            "baseline": baseline_pass_rate,
            "delta": current_pass_rate - baseline_pass_rate,
        },
        "metrics": metric_deltas,
    }

