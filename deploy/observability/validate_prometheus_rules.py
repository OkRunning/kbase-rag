"""校验 Prometheus 配置和告警规则。

优先调用 `promtool` 做 PromQL 语法校验；本地没有 promtool 时退化为结构校验，
保证 CI 可以通过安装 promtool 获得完整规则检查。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
PROMETHEUS_CONFIG = ROOT / "deploy" / "observability" / "prometheus.yml"
RULE_FILE = ROOT / "deploy" / "observability" / "rules" / "kbase-rag-alerts.yml"


def _run_promtool(promtool: str) -> int:
    """使用 promtool 校验配置和规则。

    Args:
        promtool: promtool 可执行文件路径。

    Returns:
        promtool 退出码。
    """
    commands = [
        [promtool, "check", "config", str(PROMETHEUS_CONFIG)],
        [promtool, "check", "rules", str(RULE_FILE)],
    ]
    for command in commands:
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


def _fallback_validate() -> List[str]:
    """执行无依赖结构校验。

    Returns:
        错误信息列表。
    """
    text = RULE_FILE.read_text(encoding="utf-8")
    errors: List[str] = []
    alert_names = re.findall(r"^\s*- alert:\s*(\S+)", text, flags=re.MULTILINE)
    expr_count = len(re.findall(r"^\s*expr:\s*(?:\||\S)", text, flags=re.MULTILINE))
    runbook_count = len(re.findall(r"^\s*runbook_url:\s*\S+", text, flags=re.MULTILINE))
    if not alert_names:
        errors.append("no alert rules found")
    if len(alert_names) != expr_count:
        errors.append(f"alert/expr count mismatch: alerts={len(alert_names)} expr={expr_count}")
    if len(alert_names) != runbook_count:
        errors.append(f"alert/runbook count mismatch: alerts={len(alert_names)} runbook={runbook_count}")
    duplicated = sorted({name for name in alert_names if alert_names.count(name) > 1})
    if duplicated:
        errors.append(f"duplicated alert names: {', '.join(duplicated)}")
    if "rule_files:" not in PROMETHEUS_CONFIG.read_text(encoding="utf-8"):
        errors.append("prometheus.yml missing rule_files")
    return errors


def main() -> int:
    """执行 Prometheus 规则校验。

    Returns:
        进程退出码。
    """
    promtool = shutil.which("promtool")
    if promtool:
        return _run_promtool(promtool)
    errors = _fallback_validate()
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("promtool not found; fallback rule structure validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
