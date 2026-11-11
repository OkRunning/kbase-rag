"""提供可观测事件的调试和时间线接口。"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.observability.checkpoints import audit_langgraph_checkpoints, list_langgraph_checkpoints
from app.observability.diff import diff_snapshots
from app.observability.events import (
    get_event_by_id,
    get_timeline_by_session,
    get_timeline_by_task,
    get_timeline_by_trace,
)
from app.observability.replay import replay_latest_failed_node, replay_node, replay_workflow
from app.observability.snapshots import load_snapshot

router = APIRouter(prefix="/debug", tags=["debug"])


class ReplayNodeRequest(BaseModel):
    """节点回放请求体。

    Args:
        event_id: `node.started` 事件 ID。
        mode: 回放模式。
        override_state: 覆盖快照输入状态的字段。
        dry_run: 是否启用副作用隔离。
    """

    event_id: str
    mode: Literal["timeline_only", "mock_external", "live_external", "from_node"] = "from_node"
    override_state: Dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False


class ReplayWorkflowRequest(BaseModel):
    """流程回放请求体。

    Args:
        event_id: 作为流程回放起点的 `node.started` 事件 ID。
        mode: 回放模式，默认复用历史输出快照。
        override_state: 覆盖起点快照输入状态的字段。
        dry_run: 是否启用副作用隔离。
    """

    event_id: str
    mode: Literal["timeline_only", "mock_external", "live_external", "from_node"] = "mock_external"
    override_state: Dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = True


class ReplayFailedRequest(BaseModel):
    """失败节点一键回放请求体。

    Args:
        scope: 查询范围，支持 session、task 或 trace。
        scope_id: 范围 ID。
        mode: 回放模式。
        workflow: 是否继续回放到流程结束。
        override_state: 覆盖起点快照输入状态的字段。
        dry_run: 是否启用副作用隔离。
    """

    scope: Literal["session", "task", "trace"]
    scope_id: str
    mode: Literal["timeline_only", "mock_external", "live_external", "from_node"] = "mock_external"
    workflow: bool = True
    override_state: Dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = True


class SnapshotDiffRequest(BaseModel):
    """快照差异比较请求体。

    Args:
        original_ref: 原始执行快照引用。
        replay_ref: 回放执行快照引用。
    """

    original_ref: str
    replay_ref: str


def _timeline_payload(scope: str, scope_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """构造 timeline 响应体。

    Args:
        scope: 查询维度名称。
        scope_id: 维度取值。
        items: 事件列表。

    Returns:
        可直接返回给 API 的响应体。
    """
    return {
        "scope": scope,
        "scope_id": scope_id,
        "count": len(items),
        "items": items,
    }


@router.get("/", response_class=HTMLResponse)
async def debug_home() -> str:
    """返回一个最小化的 timeline 调试页面。

    Returns:
        HTML 页面内容。
    """
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>kbase-rag debug timeline</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
    .path { background: #f3f4f6; border-left: 4px solid #2563eb; padding: 12px 16px; margin: 16px 0; border-radius: 6px; }
    .path ol { margin: 8px 0 0 20px; }
    input, select, button { padding: 8px 10px; font-size: 14px; }
    .links { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0 12px; }
    .links a { color: #2563eb; font-size: 14px; }
    pre { background: #111827; color: #e5e7eb; padding: 16px; border-radius: 6px; overflow: auto; }
  </style>
</head>
<body>
  <h1>kbase-rag debug timeline</h1>
  <div class="path">
    <strong>告警排障路径</strong>
    <ol>
      <li>从 Alertmanager 或 Grafana 打开告警，先看 dashboard_url。</li>
      <li>用 trace_id、session_id 或 task_id 打开对应 timeline。</li>
      <li>定位到异常节点后，用 replay 接口复现并对比快照。</li>
    </ol>
  </div>
  <div class="row">
    <select id="scope">
      <option value="session">session</option>
      <option value="task">task</option>
      <option value="trace">trace</option>
    </select>
    <input id="scopeId" placeholder="输入 session_id / task_id / trace_id" size="48" />
    <button onclick="loadTimeline()">加载 timeline</button>
  </div>
  <div class="row">
    <input id="eventId" placeholder="输入 event_id" size="48" />
    <button onclick="loadEvent()">加载事件详情</button>
  </div>
  <div id="links" class="links"></div>
  <pre id="output">等待查询...</pre>
  <script>
    function renderTraceLinks(payload) {
      const links = document.getElementById('links');
      const urls = new Set();
      const collect = (value) => {
        if (!value) return;
        if (Array.isArray(value)) {
          value.forEach(collect);
          return;
        }
        if (typeof value === 'object') {
          if (value.langfuse_trace_url) urls.add(value.langfuse_trace_url);
          Object.values(value).forEach(collect);
        }
      };
      collect(payload);
      links.innerHTML = '';
      for (const url of urls) {
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.target = '_blank';
        anchor.rel = 'noreferrer';
        anchor.textContent = '打开 Langfuse trace';
        links.appendChild(anchor);
      }
    }
    async function loadTimeline() {
      const scope = document.getElementById('scope').value;
      const scopeId = document.getElementById('scopeId').value.trim();
      const output = document.getElementById('output');
      if (!scopeId) {
        output.textContent = '请先输入 ID';
        return;
      }
      const path = scope === 'session'
        ? `/debug/sessions/${scopeId}/timeline`
        : scope === 'task'
          ? `/debug/tasks/${scopeId}/timeline`
          : `/debug/traces/${scopeId}/timeline`;
      const resp = await fetch(path);
      const payload = await resp.json();
      renderTraceLinks(payload);
      output.textContent = JSON.stringify(payload, null, 2);
    }
    async function loadEvent() {
      const eventId = document.getElementById('eventId').value.trim();
      const output = document.getElementById('output');
      if (!eventId) {
        output.textContent = '请先输入 event_id';
        return;
      }
      const resp = await fetch(`/debug/events/${eventId}`);
      const payload = await resp.json();
      renderTraceLinks(payload);
      output.textContent = JSON.stringify(payload, null, 2);
    }
  </script>
</body>
</html>
"""


@router.get("/sessions/{session_id}/timeline")
async def session_timeline(
    session_id: str,
    limit: int = 200,
    tenant_id: str = "default",
    kb_id: str = "default",
) -> Dict[str, Any]:
    """查询指定会话的事件时间线。

    Args:
        session_id: 查询会话 ID。
        limit: 返回事件条数上限。
        tenant_id: 租户标识。
        kb_id: 知识库标识。

    Returns:
        会话时间线响应。
    """
    return _timeline_payload(
        "session",
        session_id,
        get_timeline_by_session(session_id, limit, tenant_id=tenant_id, kb_id=kb_id),
    )


@router.get("/tasks/{task_id}/timeline")
async def task_timeline(
    task_id: str,
    limit: int = 200,
    tenant_id: str = "default",
    kb_id: str = "default",
) -> Dict[str, Any]:
    """查询指定导入任务的事件时间线。

    Args:
        task_id: 导入任务 ID。
        limit: 返回事件条数上限。
        tenant_id: 租户标识。
        kb_id: 知识库标识。

    Returns:
        任务时间线响应。
    """
    return _timeline_payload(
        "task",
        task_id,
        get_timeline_by_task(task_id, limit, tenant_id=tenant_id, kb_id=kb_id),
    )


@router.get("/traces/{trace_id}/timeline")
async def trace_timeline(trace_id: str, limit: int = 200) -> Dict[str, Any]:
    """查询指定 trace 的事件时间线。

    Args:
        trace_id: OpenTelemetry trace ID。
        limit: 返回事件条数上限。

    Returns:
        trace 时间线响应。
    """
    return _timeline_payload("trace", trace_id, get_timeline_by_trace(trace_id, limit))


@router.get("/events/{event_id}")
async def event_detail(event_id: str) -> Dict[str, Any]:
    """查询单条事件详情。

    Args:
        event_id: 事件 ID。

    Returns:
        事件详情。

    Raises:
        HTTPException: 事件不存在时抛出 404。
    """
    event = get_event_by_id(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    return event


@router.get("/snapshots/{snapshot_id}")
async def snapshot_detail(snapshot_id: str) -> Dict[str, Any]:
    """查询节点输入或输出快照。

    Args:
        snapshot_id: 快照 ID 或快照引用。

    Returns:
        快照详情。

    Raises:
        HTTPException: 快照不存在时抛出 404。
    """
    try:
        return load_snapshot(snapshot_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/checkpoints/{thread_id}")
async def checkpoint_timeline(thread_id: str, limit: int = 20) -> Dict[str, Any]:
    """查询 LangGraph checkpoint 列表。

    Args:
        thread_id: LangGraph checkpoint thread ID。
        limit: 返回条数上限。

    Returns:
        checkpoint 列表响应。
    """
    items = list_langgraph_checkpoints(thread_id, limit=limit)
    return {
        "thread_id": thread_id,
        "count": len(items),
        "items": items,
    }


@router.get("/checkpoints/{thread_id}/audit")
async def checkpoint_audit(thread_id: str, limit: int = 100) -> Dict[str, Any]:
    """查询 LangGraph checkpoint 长期审计摘要。

    Args:
        thread_id: LangGraph checkpoint thread ID。
        limit: 审计读取条数上限。

    Returns:
        checkpoint 审计摘要。
    """
    return audit_langgraph_checkpoints(thread_id, limit=limit)


@router.post("/snapshots/diff")
async def snapshot_diff(request: SnapshotDiffRequest) -> Dict[str, Any]:
    """比较原始执行与回放执行的快照差异。

    Args:
        request: 快照 diff 请求体。

    Returns:
        快照差异。
    """
    try:
        return diff_snapshots(request.original_ref, request.replay_ref)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/replay/node")
async def replay_node_endpoint(request: ReplayNodeRequest) -> Dict[str, Any]:
    """从 `node.started` 事件的输入快照回放单个节点。

    Args:
        request: 节点回放请求体。

    Returns:
        回放结果。

    Raises:
        HTTPException: 请求不可回放或回放失败时抛出。
    """
    try:
        return await run_in_threadpool(
            replay_node,
            event_id=request.event_id,
            mode=request.mode,
            override_state=request.override_state,
            dry_run=request.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"replay failed: {exc}") from exc


@router.post("/replay/workflow")
async def replay_workflow_endpoint(request: ReplayWorkflowRequest) -> Dict[str, Any]:
    """从指定节点输入快照开始回放后续工作流。

    Args:
        request: 流程回放请求体。

    Returns:
        流程回放结果。

    Raises:
        HTTPException: 请求不可回放或回放失败时抛出。
    """
    try:
        return await run_in_threadpool(
            replay_workflow,
            event_id=request.event_id,
            mode=request.mode,
            override_state=request.override_state,
            dry_run=request.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"workflow replay failed: {exc}") from exc


@router.post("/replay/failed")
async def replay_failed_endpoint(request: ReplayFailedRequest) -> Dict[str, Any]:
    """一键回放指定范围内最近失败的节点。

    Args:
        request: 失败节点回放请求体。

    Returns:
        回放结果。

    Raises:
        HTTPException: 没有可回放失败节点或回放失败时抛出。
    """
    try:
        return await run_in_threadpool(
            replay_latest_failed_node,
            scope=request.scope,
            scope_id=request.scope_id,
            mode=request.mode,
            workflow=request.workflow,
            override_state=request.override_state,
            dry_run=request.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed-node replay failed: {exc}") from exc
