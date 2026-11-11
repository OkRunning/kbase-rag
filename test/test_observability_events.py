from app.observability import events
from app.observability import replay
from app.observability.api import debug_server
from app.observability.context import reset_context, set_context
from app.observability.sanitizers import sanitize_summary


class _FakeStore:
    def __init__(self):
        self.written = None

    def write(self, document):
        document = dict(document)
        document["summary"] = sanitize_summary(document.get("summary", {}))
        document["tags"] = sanitize_summary(document.get("tags", []))
        self.written = document
        return document

    def query(self, filters, limit=200):
        return []

    def get_event(self, event_id):
        return None


def test_sanitize_summary_truncates_long_strings():
    payload = sanitize_summary({"query": "a" * 250})

    assert payload["query"]["chars"] == 250
    assert payload["query"]["preview"] == "a" * 200
    assert len(payload["query"]["sha256"]) == 64


def test_emit_event_writes_event_document(monkeypatch):
    fake_store = _FakeStore()
    monkeypatch.setattr(events, "_STORE", fake_store)

    tokens = set_context(
        langfuse_trace_id="lf-trace-1",
        langfuse_parent_span_id="lf-parent-1",
        langfuse_trace_url="https://langfuse.example/trace/lf-trace-1",
    )
    try:
        document = events.emit_event(
            "query.received",
            graph="query",
            session_id="session-1",
            summary={"query": "b" * 250},
            tags=["query"],
        )
    finally:
        reset_context(tokens)

    assert document["event_type"] == "query.received"
    assert document["session_id"] == "session-1"
    assert document["langfuse_trace_id"] == "lf-trace-1"
    assert document["langfuse_parent_span_id"] == "lf-parent-1"
    assert document["langfuse_trace_url"] == "https://langfuse.example/trace/lf-trace-1"
    assert document["summary"]["query"]["chars"] == 250
    assert fake_store.written is not None


def test_replay_failed_mock_external_skips_failed_start_without_output(monkeypatch):
    """验证失败起点缺少历史输出时 mock workflow 返回诊断 checkpoint。"""
    started_event = {
        "_id": "event-started",
        "event_type": "node.started",
        "graph": "import",
        "node_name": "node_pdf_to_md",
        "task_id": "task-1",
        "sequence": 1,
        "input_ref": "snapshot-input",
        "trace_id": "trace-1",
    }
    failed_event = {
        "_id": "event-failed",
        "event_type": "node.failed",
        "graph": "import",
        "node_name": "node_pdf_to_md",
        "task_id": "task-1",
        "sequence": 2,
        "input_ref": "snapshot-input",
        "error": {"message": "mineru failed"},
    }
    emitted_events = []

    def fake_emit_event(event_type, **kwargs):
        document = {"_id": f"emitted-{len(emitted_events)}", "event_type": event_type, **kwargs}
        emitted_events.append(document)
        return document

    monkeypatch.setattr(replay, "get_timeline_by_task", lambda task_id, limit=1000: [started_event, failed_event])
    monkeypatch.setattr(replay, "get_event_by_id", lambda event_id: started_event)
    monkeypatch.setattr(replay, "load_snapshot", lambda ref: {"payload": {"task_id": "task-1", "is_pdf_read_enabled": True}})
    monkeypatch.setattr(replay, "save_snapshot", lambda *args, **kwargs: f"snapshot-{len(emitted_events)}")
    monkeypatch.setattr(replay, "emit_event", fake_emit_event)

    result = replay.replay_latest_failed_node(
        scope="task",
        scope_id="task-1",
        mode="mock_external",
        workflow=True,
    )

    assert result["scope"] == "workflow"
    assert result["checkpoints"][0]["status"] == "skipped"
    assert result["checkpoints"][0]["summary"]["skipped"] is True
    assert "node_pdf_to_md" in result["checkpoints"][0]["summary"]["skip_reason"]


def test_replay_failed_endpoint_uses_threadpool():
    """验证异步回放接口不会在 FastAPI 事件循环里同步执行节点。"""
    names = debug_server.replay_failed_endpoint.__code__.co_names

    assert "run_in_threadpool" in names
    assert "replay_latest_failed_node" in names
