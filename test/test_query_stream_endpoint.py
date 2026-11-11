"""测试 LangGraph astream 单端点流式执行器（query_stream）。"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import Mock, patch

from app.query_process.api import query_stream as module


@contextmanager
def _nullcontext(*args, **kwargs):
    yield None


class _FakeLangfuse:
    """空 langfuse 适配，避免真实导出。"""

    def start_as_current_observation(self, *args, **kwargs):
        @contextmanager
        def _cm():
            yield _FakeObservation()

        return _cm()

    def set_current_trace_io(self, *args, **kwargs):
        return None

    def get_current_trace_id(self):
        return ""

    def get_current_observation_id(self):
        return ""

    def get_trace_url(self):
        return ""

    def score_current_trace(self, *args, **kwargs):
        return None


class _FakeObservation:
    def update(self, *args, **kwargs):
        return None


class _FakeTracer:
    """空 tracer，避免真实 OTel 导出。"""

    def start_as_current_span(self, *args, **kwargs):
        @contextmanager
        def _cm():
            yield _FakeSpan()

        return _cm()


class _FakeSpan:
    def set_attribute(self, *args, **kwargs):
        return None

    def record_exception(self, *args, **kwargs):
        return None

    def set_status(self, *args, **kwargs):
        return None


def _collect(agen):
    """同步收集 async generator 的全部输出。"""

    return asyncio.run(_collect_async(agen))


async def _collect_async(agen):
    out = []
    async for item in agen:
        out.append(item)
    return out


def _message_chunk(content, node="node_answer_output"):
    class _Chunk:
        def __init__(self, value):
            self.content = value

    return _Chunk(content), {"langgraph_node": node}


def _updates(payload):
    return "updates", payload


class _FakeApp:
    """模拟 CompiledStateGraph，astream 是 async generator，产出预设事件。"""

    def __init__(self, events):
        self._events = events

    async def astream(self, *args, **kwargs):
        for event in self._events:
            yield event


def _patch_deps(app):
    """构造公共 mock 上下文，替换外部依赖。app 为 query_app 的替身（事件列表或异常 app）。"""
    return [
        patch.object(module, "query_app", app),
        patch.object(module, "record_query", return_value=None),
        patch.object(module, "record_task_inflight", return_value=None),
        patch.object(module, "reset_task_progress", return_value=None),
        patch.object(module, "update_task_status", return_value=None),
        patch.object(module, "get_task_status", return_value="processing"),
        patch.object(module, "get_done_task_list", return_value=[]),
        patch.object(module, "get_running_task_list", return_value=["node_answer_output"]),
        patch.object(module, "get_task_result", side_effect=lambda _sid, key, default="": {"answer": "答案内容", "sources": "[]"}.get(key, default)),
        patch.object(module, "set_context", return_value={}),
        patch.object(module, "reset_context", return_value=None),
        patch.object(module, "emit_event", return_value={}),
        patch.object(module, "failure_monitor", Mock()),
        patch.object(module, "propagate_attributes", _nullcontext),
        patch.object(module, "langfuse", _FakeLangfuse()),
        patch.object(module, "tracer", _FakeTracer()),
    ]


def _error_app(exc):
    """构造首轮即抛异常的 astream app。"""
    class _Boom:
        async def astream(self, *args, **kwargs):
            raise exc
            yield  # pragma: no cover

    return _Boom()


def test_delta_only_from_answer_node():
    """messages 模式只透传答案节点 token，忽略其他节点 LLM 输出。"""
    events = [
        ("messages", _message_chunk("检索节点内部token", node="node_query_understanding")),
        ("messages", _message_chunk("答案")),
        ("messages", _message_chunk("内容")),
        _updates({"node_answer_output": {"answer": "答案内容", "sources": [], "image_urls": ["img1"]}}),
    ]

    with _context(*_patch_deps(_FakeApp(events))):
        out = _collect(module.sse_query_stream("q", "s1"))

    deltas = [line for line in out if "event: delta" in line]
    progress = [line for line in out if "event: progress" in line]
    final = [line for line in out if "event: final" in line]

    assert len(deltas) == 2, deltas
    assert "答案" in deltas[0] and "内容" in deltas[1]
    # 非答案节点 token 不推送
    assert not any("检索节点内部token" in d for d in deltas)
    assert len(progress) >= 1
    assert len(final) == 1
    assert '"answer": "答案内容"' in final[0]
    assert '"image_urls": ["img1"]' in final[0]


def test_insufficient_no_llm_answer_in_final():
    """拒答/无 LLM token 场景下，FINAL 带出完整答案。"""
    events = [_updates({"node_answer_output": {"answer": "资料不足...", "sources": [], "image_urls": []}})]

    with _context(*_patch_deps(_FakeApp(events))):
        out = _collect(module.sse_query_stream("q", "s2"))

    final = [line for line in out if "event: final" in line]
    assert len(final) == 1
    assert "资料不足..." in final[0]


def test_error_event_on_exception():
    """图执行抛出异常时发送 error 事件。"""
    with _context(*_patch_deps(_error_app(RuntimeError("llm exploded")))):
        out = _collect(module.sse_query_stream("q", "s3"))

    err = [line for line in out if "event: error" in line]
    assert len(err) == 1
    assert "llm exploded" in err[0]


class _context:
    """多 patch 的 context manager 便捷封装。"""

    def __init__(self, *patches):
        self._patches = patches

    def __enter__(self):
        for p in self._patches:
            p.__enter__()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.__exit__(*exc)
        return False

