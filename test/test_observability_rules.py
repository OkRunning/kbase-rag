import subprocess
import sys


def test_prometheus_rule_validation_script_passes():
    """验证 Prometheus 告警规则至少通过本地结构校验。"""
    result = subprocess.run(
        [sys.executable, "deploy/observability/validate_prometheus_rules.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
