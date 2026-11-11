import json
from pathlib import Path

from app.observability import automation
from app.observability.jsonl_backfill import backfill_events_jsonl_to_mongo


def test_scan_stuck_tasks_emits_low_cardinality_event(monkeypatch):
    """验证卡住任务扫描写入事件但不把 task_id 放入指标标签。"""
    emitted = []
    ages = []
    snapshot = {
        "task_id": "session-1",
        "graph": "query",
        "status": "processing",
        "age_seconds": 700,
        "idle_seconds": 700,
        "running_list": ["生成答案"],
        "done_list": [],
    }

    monkeypatch.setattr(automation, "list_task_snapshots", lambda: [snapshot])
    monkeypatch.setattr(automation, "find_stuck_tasks", lambda **kwargs: [snapshot])
    monkeypatch.setattr(automation, "record_task_age", lambda graph, status, age: ages.append((graph, status, age)))
    monkeypatch.setattr(automation, "emit_event", lambda event_type, **kwargs: emitted.append((event_type, kwargs)))
    automation._LAST_STUCK_EMIT_AT.clear()

    stuck = automation.scan_stuck_tasks("query")

    assert stuck == [snapshot]
    assert ages == [("query", "processing", 700)]
    assert emitted[0][0] == "task.stuck.detected"
    assert emitted[0][1]["session_id"] == "session-1"
    assert emitted[0][1]["task_id"] is None


def test_backfill_events_jsonl_dry_run_counts_valid_events(tmp_path: Path):
    """验证 JSONL 回灌 dry-run 只统计有效事件，不连接 Mongo。"""
    path = tmp_path / "observability_events.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"_id": "event-1", "summary": {"query": "hello"}, "tags": ["query"]}),
                "{bad-json",
                json.dumps({"summary": {"missing": "id"}}),
            ]
        ),
        encoding="utf-8",
    )

    result = backfill_events_jsonl_to_mongo(path=path, dry_run=True)

    assert result["processed"] == 2
    assert result["written"] == 0
    assert result["skipped"] == 1
    assert result["dry_run"] == 1
