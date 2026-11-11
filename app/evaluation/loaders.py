"""评测样本加载工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from app.evaluation.schemas import EvalCase


def load_jsonl_cases(dataset_path: str | Path) -> List[EvalCase]:
    """从 JSONL 文件加载评测样本。

    Args:
        dataset_path: 评测集文件路径。

    Returns:
        评测样本列表。

    Raises:
        FileNotFoundError: 评测集文件不存在。
        ValueError: JSON 格式错误或样本字段缺失。
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"eval dataset not found: {path}")

    cases: List[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        try:
            cases.append(EvalCase.from_dict(raw))
        except ValueError as exc:
            raise ValueError(f"invalid eval case at {path}:{line_number}: {exc}") from exc
    return cases

