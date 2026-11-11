"""RAG 评测阈值门禁。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "pass_rate": 0.8,
    "hit_at_5": 0.8,
    "recall_at_5": 0.8,
    "mrr_at_5": 0.5,
    "required_keywords_pass": 0.9,
    "forbidden_keywords_pass": 1.0,
}


def load_thresholds(thresholds_path: str | Path | None) -> Dict[str, float]:
    """加载评测阈值配置。

    Args:
        thresholds_path: 阈值 JSON 文件路径；为空时返回默认阈值。

    Returns:
        指标名称到最低阈值的映射。

    Raises:
        FileNotFoundError: 阈值文件不存在。
        ValueError: 阈值文件格式不合法。
    """
    if not thresholds_path:
        return dict(DEFAULT_THRESHOLDS)
    path = Path(thresholds_path)
    if not path.exists():
        raise FileNotFoundError(f"thresholds file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("thresholds file must be a JSON object")
    thresholds: Dict[str, float] = {}
    for name, value in payload.items():
        thresholds[str(name)] = float(value)
    return thresholds


def evaluate_thresholds(summary: Dict[str, Any], thresholds: Dict[str, float]) -> List[Dict[str, float | str]]:
    """检查评测摘要是否满足阈值。

    Args:
        summary: 当前评测报告 summary。
        thresholds: 指标名称到最低阈值的映射。

    Returns:
        未满足阈值的指标列表。
    """
    metrics = summary.get("metrics") or {}
    violations: List[Dict[str, float | str]] = []
    for name, threshold in thresholds.items():
        if name == "pass_rate":
            current_value = float(summary.get("pass_rate", 0.0))
        else:
            current_value = float(metrics.get(name, 0.0))
        if current_value < threshold:
            violations.append(
                {
                    "metric": name,
                    "current": current_value,
                    "threshold": float(threshold),
                    "delta": current_value - float(threshold),
                }
            )
    return violations

