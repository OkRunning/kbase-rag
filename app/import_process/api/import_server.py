"""文档导入服务 API。

该模块提供导入页面、文件上传和任务状态查询接口，并在入口层初始化
OpenTelemetry，记录 HTTP、导入 workflow 和 LangGraph 节点的观测数据。
"""

import shutil
import uuid
import time
import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Dict, Any

from app.observability.settings import is_observability_enabled

if is_observability_enabled():
    # 仅启用可观测能力时初始化 OTel，关闭观测时业务启动不依赖 trace exporter。
    from app.core.tracing import setup_tracing

    setup_tracing()

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Query, Request
from fastapi.responses import FileResponse
from bson import json_util
from starlette.middleware.cors import CORSMiddleware

from app.clients.langfuse import langfuse, propagate_attributes
from app.clients.document_versioning_utils import get_document_version_store
from app.core.logger import logger  # 项目统一日志工具
from app.core.observability import (
    OBSERVABILITY_ENABLED,
    Status,
    StatusCode,
    build_langgraph_config,
    build_liveness,
    build_observability_health,
    build_readiness,
    emit_event,
    get_current_otel_ids,
    instrument_fastapi,
    observability_debug_router,
    record_http_request,
    record_import,
    record_task_inflight,
    reset_context,
    sanitize_for_log,
    set_context,
    start_observability_automation,
    tracer,
)
from app.core.tenant_scope import normalize_kb_id, normalize_tenant_id
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import get_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_running_task, add_done_task, update_task_status, get_task_status, \
    get_done_task_list, get_running_task_list

@asynccontextmanager
async def _lifespan(app_: FastAPI):
    """管理导入服务生命周期事件。

    Args:
        app_: FastAPI 应用实例。

    Yields:
        None.
    """
    # 生命周期启动阶段拉起观测自动化线程，替代已废弃的 on_event startup。
    start_observability_automation("import")
    yield


# 初始化FastAPI应用实例
# 标题和描述会在Swagger文档(http://ip:port/docs)中展示
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus入库)",
    lifespan=_lifespan,
)
if OBSERVABILITY_ENABLED and observability_debug_router is not None:
    app.include_router(observability_debug_router)

# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)

instrument_fastapi(app)


@app.get("/health")
async def health():
    """返回导入服务兼容健康状态。

    Returns:
        服务健康状态。
    """
    return build_liveness("import")


@app.get("/health/live")
async def health_live():
    """返回导入服务 liveness 状态。

    Returns:
        liveness 健康状态。
    """
    return build_liveness("import")


@app.get("/health/ready")
async def health_ready():
    """返回导入服务 readiness 状态。

    Returns:
        readiness 健康状态。
    """
    return build_readiness("import")


@app.get("/health/observability")
async def health_observability():
    """返回导入服务观测能力状态。

    Returns:
        观测能力健康状态。
    """
    return build_observability_health("import")


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """记录 HTTP 请求指标并注入 request_id 上下文。

    Args:
        request: FastAPI 请求对象。
        call_next: 后续中间件或路由处理函数。

    Returns:
        FastAPI 响应对象。
    """
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    tokens = set_context(request_id=request_id)
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        record_http_request(route_path, request.method, status_code, time.perf_counter() - start)
        # 请求级上下文只在当前 HTTP 生命周期内有效，结束后必须恢复。
        reset_context(tokens)




# # 然后应用instrumentation
# FlaskInstrumentor().instrument_app(app)  # 如果使用Flask

# 如果使用FastAPI
# FastAPIInstrumentor.instrument_app(app)


#  8080/import  -> import.html
@app.get("/import",response_class=FileResponse)
async def get_import_page():
    """返回文档导入页面。

    Returns:
        import.html 文件响应。

    Raises:
        HTTPException: 导入页面文件不存在。
    """
    import_html_path = PROJECT_ROOT / "app" / "import_process" / "page"/ "import.html"
    if not import_html_path.exists():
        raise HTTPException(status_code=404, detail="导入页面不存在！！")
    return FileResponse(path = import_html_path , media_type="text/html")


@app.get("/version-diff", response_class=FileResponse)
async def get_version_diff_page():
    """返回文档版本 Diff 页面。

    Returns:
        version_diff.html 文件响应。

    Raises:
        HTTPException: 页面文件不存在。
    """
    version_diff_html_path = PROJECT_ROOT / "app" / "import_process" / "page" / "version_diff.html"
    if not version_diff_html_path.exists():
        raise HTTPException(status_code=404, detail="文档版本Diff页面不存在")
    return FileResponse(path=version_diff_html_path, media_type="text/html")



# 定义调用import_graph的函数，像之前测试一样执行的图！！
# local_file_path (str)  task_id  local_dir(str)
def _normalize_user_id(value: str | None) -> str:
    """规范化导入请求中的用户标识。

    Args:
        value: 用户标识。

    Returns:
        合法用户标识；空值时返回 anonymous。

    Raises:
        ValueError: 用户标识包含非法字符。
    """
    user_id = str(value or "").strip() or "anonymous"
    if not re.fullmatch(r"[A-Za-z0-9_.:\-]{1,128}", user_id):
        raise ValueError("user_id只能包含字母、数字、点、下划线、中划线和冒号")
    return user_id


def _normalize_doc_id(value: str | None, default: str) -> str:
    """规范化导入请求中的逻辑文档标识。

    Args:
        value: 请求传入的文档标识。
        default: 空值时使用的默认文档标识。

    Returns:
        合法文档标识。

    Raises:
        ValueError: 文档标识包含非法字符。
    """
    doc_id = str(value or "").strip() or default
    if not re.fullmatch(r"[A-Za-z0-9_.:\-]{1,128}", doc_id):
        raise ValueError("doc_id只能包含字母、数字、点、下划线、中划线和冒号")
    return doc_id


def _mongo_document_to_json(document: Dict[str, Any]) -> Dict[str, Any]:
    """将 MongoDB 文档转换为 API 可返回的 JSON 字典。

    Args:
        document: MongoDB 原始文档。

    Returns:
        可 JSON 序列化的字典。
    """
    return json_util.loads(json_util.dumps(document or {}))


def run_import_graph(
    task_id: str,
    local_file_path: str,
    local_dir: str,
    tenant_id: str = "default",
    kb_id: str = "default",
    user_id: str = "anonymous",
    doc_id: str | None = None,
):
    """执行文档导入 LangGraph 工作流。

    Args:
        task_id: 文件导入任务 ID。
        local_file_path: 上传文件在本地的保存路径。
        local_dir: 本次导入任务的本地工作目录。
        tenant_id: 租户 ID。
        kb_id: 知识库 ID。
        user_id: 用户 ID。
        doc_id: 文档 ID，空值时使用 task_id。

    Returns:
        None.
    """
    # 本次任务对应的节点状态
    # _tasks_running_list: Dict[str, List[str]] = {}
    # _tasks_done_list: Dict[str, List[str]] = {}
    # key=task_id 本次上传文件任务的唯一标识
    # value = list [文件上传，检查文件 节点名字] （正在进行 | 已经完成）
    # add_done_task(task_id, "upload_file")
    # add_running_task(task_id, "upload_file")
    tenant_id = normalize_tenant_id(tenant_id)
    kb_id = normalize_kb_id(kb_id)
    user_id = _normalize_user_id(user_id)
    doc_id = str(doc_id or task_id)
    tokens = set_context(graph="import", task_id=task_id, tenant_id=tenant_id, kb_id=kb_id)
    start = time.perf_counter()
    status = "ok"
    langfuse_tokens = {}
    file_type = local_file_path.rsplit(".", 1)[-1].lower() if "." in local_file_path else "unknown"
    record_task_inflight("import", 1)
    with tracer.start_as_current_span("workflow.import") as workflow_span:
        workflow_span.set_attribute("rag.graph", "import")
        workflow_span.set_attribute("rag.task_id", task_id)
        workflow_span.set_attribute("rag.tenant_id", tenant_id)
        workflow_span.set_attribute("rag.kb_id", kb_id)
        workflow_span.set_attribute("rag.doc_id", doc_id)
        workflow_span.set_attribute("rag.file_type", file_type)
        workflow_span.set_attribute("rag.local_file_path", local_file_path)
        otel_ids = get_current_otel_ids()
        emit_event(
            "workflow.started",
            graph="import",
            task_id=task_id,
            status="ok",
            summary={
                "local_file_path": local_file_path,
                "local_dir": local_dir,
                "file_type": file_type,
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "doc_id": doc_id,
            },
            tags=["import", "workflow.started"],
        )
        with langfuse.start_as_current_observation(
            name="workflow.import_execution",
            metadata={
                "task_id": task_id,
                "file_type": file_type,
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "doc_id": doc_id,
                **otel_ids,
            },
        ) as span:
            try:
                latest_state: Dict[str, Any] | None = None
                langfuse_trace_id = langfuse.get_current_trace_id() or ""
                try:
                    langfuse_trace_url = langfuse.get_trace_url() or ""
                except Exception:
                    langfuse_trace_url = ""
                if langfuse_trace_id:
                    workflow_span.set_attribute("langfuse.trace_id", langfuse_trace_id)
                if langfuse_trace_url:
                    workflow_span.set_attribute("langfuse.trace_url", langfuse_trace_url)
                langfuse_tokens = set_context(
                    langfuse_trace_id=langfuse_trace_id,
                    langfuse_parent_span_id=langfuse.get_current_observation_id(),
                    langfuse_trace_url=langfuse_trace_url,
                )
                span.update(input={
                    "task_id":task_id,
                    "local_file_path":local_file_path,
                    "local_dir":local_dir
                })

                with propagate_attributes(
                        user_id=user_id,
                        session_id=task_id,
                        metadata={
                            "file_name": local_file_path.split("/")[-1],
                            "tenant_id": tenant_id,
                            "kb_id": kb_id,
                            "doc_id": doc_id,
                            **otel_ids,
                        },
                        tags=["import_process_tag"],
                    ):
                    # 本次任务的总状态
                    # _tasks_status: Dict[str, str] = {}
                    # key= task_id
                    # value = task_id任务的状态
                    update_task_status(task_id,"processing")
                    init_state = get_default_state()
                    # LangGraph State 更新是导入链路的核心：后续节点依赖这些路径字段。
                    init_state["task_id"] = task_id
                    init_state["tenant_id"] = tenant_id
                    init_state["kb_id"] = kb_id
                    init_state["user_id"] = user_id
                    init_state["doc_id"] = doc_id
                    init_state["local_file_path"] = local_file_path
                    init_state["local_dir"] = local_dir
                    latest_state = init_state
                    # 执行我们图
                    graph_config = build_langgraph_config("import", task_id, tenant_id=tenant_id, kb_id=kb_id)
                    for event in kb_import_app.stream(init_state, config=graph_config):
                    # event {节点名 : state}
                        for node_name, result in event.items():
                            if isinstance(result, dict):
                                latest_state = result
                            logger.info(f"节点：{node_name}已经完成执行，执行结果为：{sanitize_for_log(result)}")
                            # add_done_task(task_id, node_name)
                    update_task_status(task_id,"completed")
                    logger.info(f"{task_id}:图状态执行完毕！！")
                    emit_event(
                        "workflow.completed",
                        graph="import",
                        task_id=task_id,
                        status="ok",
                        summary={"file_type": file_type, "tenant_id": tenant_id, "kb_id": kb_id, "doc_id": doc_id},
                        tags=["import", "workflow.completed"],
                    )
                span.update(output={
                    "task_id": task_id,
                    "result": "completed"
                })
            except Exception as e:
                status = "error"
                workflow_span.record_exception(e)
                workflow_span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.exception("====图执行失败！发生异常====")
                try:
                    store = get_document_version_store()
                    if store is not None and doc_id and latest_state:
                        state_doc_version = latest_state.get("doc_version")
                        if state_doc_version:
                            store.mark_version_failed(tenant_id, kb_id, doc_id, int(state_doc_version), str(e))
                except Exception as version_exc:
                    logger.warning(f"回填失败版本状态时出错：{version_exc}")
                update_task_status(task_id, "failed")
                emit_event(
                    "workflow.failed",
                    graph="import",
                    task_id=task_id,
                    status="error",
                    summary={"file_type": file_type, "tenant_id": tenant_id, "kb_id": kb_id, "doc_id": doc_id},
                    error={"type": type(e).__name__, "message": str(e)},
                    tags=["import", "workflow.failed"],
                )
                span.update(output={
                    "task_id": task_id,
                    "result": "failed",
                    "error": str(e)
                })
            finally:
                record_import(status, file_type, time.perf_counter() - start)
                record_task_inflight("import", -1)
                # 先恢复内层 Langfuse 上下文，再恢复 workflow 上下文，避免异步任务串线。
                if langfuse_tokens:
                    reset_context(langfuse_tokens)
                # 后台任务执行完必须恢复上下文，避免后续任务日志串 task_id。
                reset_context(tokens)




# 8080/upload post -> 文件上传 + 开启导入流程
"""
  1. 接收文件存储到output文件夹！  /output/当天的日期/uuid(taskid)/文件名
  2. 异步开启，import_graph图的执行 【1. 整个任务的状态（开始和结束） 2. 每个节点的状态 add_running add_done】
"""
@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks,
                      files: List[UploadFile] = File(...),
                      tenant_id: str = Form("default"),
                      kb_id: str = Form("default"),
                      user_id: str = Form("anonymous"),
                      doc_id: str | None = Form(None)):
    """上传文件并触发导入工作流。

    Args:
        background_tasks: FastAPI 后台任务管理器。
        files: 上传文件列表。
        tenant_id: 租户 ID，默认进入 default 租户。
        kb_id: 知识库 ID，默认进入 default 知识库。
        user_id: 上传用户 ID。
        doc_id: 逻辑文档 ID，仅单文件上传时建议传入。

    Returns:
        上传受理结果和任务 ID 列表。

    Raises:
        HTTPException: 多文件上传同时指定 doc_id 时抛出。
    """
    # 1. 整理下输出的位置 output / 日期文件夹
    today_str = datetime.now().strftime("%Y%m%d")
    tenant_id = normalize_tenant_id(tenant_id)
    kb_id = normalize_kb_id(kb_id)
    user_id = _normalize_user_id(user_id)
    if doc_id and len(files) > 1:
        raise HTTPException(status_code=400, detail="多文件上传不能共用同一个doc_id，请拆分上传或不传doc_id")
    # 本地工作目录按租户和知识库分层，方便审计、清理和后续迁移到对象存储。
    base_out_path = PROJECT_ROOT / "output" / tenant_id / kb_id / today_str
    emit_event(
        "import.upload.received",
        graph="import",
        status="ok",
        summary={"file_count": len(files), "tenant_id": tenant_id, "kb_id": kb_id},
        tags=["import", "import.upload.received"],
    )
    # 2. 记录下每个文件上传的任务id [taskid,taskid]
    task_ids = []
    # 3. 循环处理每个上传的文件（存储到本地） + 进行异步图任务调用
    # UploadFile:
    #   .file 上传文件的输入流
    #   .filename 上传文件名
    #   .read 可以直接读取
    #   .contenttype 获取我们文件mimetype类型
    for file in files:
        task_id = str(uuid.uuid4())  # uuid不重复
        task_ids.append(task_id)
        current_doc_id = _normalize_doc_id(doc_id, task_id)
        upload_tokens = set_context(graph="import", task_id=task_id, tenant_id=tenant_id, kb_id=kb_id)
        try:
            # 记录下进行文件上传了
            add_running_task(task_id, "upload_file")
            # 文件的dir_path
            dir_path = base_out_path / task_id
            # 没有文件夹
            dir_path.mkdir(parents=True, exist_ok=True)
            # 文件的local_file_path
            local_file_path = dir_path / file.filename
            # 将上传的文件写入到 local_file_path
            with local_file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            emit_event(
                "file.saved",
                graph="import",
                task_id=task_id,
                status="ok",
                summary={
                    "file_name": file.filename,
                    "file_size": local_file_path.stat().st_size,
                    "tenant_id": tenant_id,
                    "kb_id": kb_id,
                    "doc_id": current_doc_id,
                },
                tags=["import", "file.saved"],
            )
            # 异步执行
            # 参数1： run_import_graph 执行的方法
            # 参数2： *args  参数列表 task_id, local_file_path, dir_path - 》 run_import_graph
            background_tasks.add_task(
                run_import_graph,
                task_id,
                str(local_file_path),
                str(dir_path),
                tenant_id,
                kb_id,
                user_id,
                current_doc_id,
            )
            logger.info(f"{task_id}:完成文件上传，并开启了对应的异步任务！！")
            add_done_task(task_id, "upload_file")
        finally:
            reset_context(upload_tokens)
    # 4. 最终返回结果即可
    return {
        "code": 200,
        "message": f"完成了文件上传，并开启了异步任务！文件数量为: {len(files)}",
        "tenant_id": tenant_id,
        "kb_id": kb_id,
        "task_ids": task_ids
    }


@app.get("/kb/{kb_id}/documents")
async def list_documents(kb_id: str, tenant_id: str = "default", status: str | None = None):
    """查询知识库文档列表。

    Args:
        kb_id: 知识库 ID。
        tenant_id: 租户 ID。
        status: 可选文档状态过滤。

    Returns:
        文档列表。

    Raises:
        HTTPException: MongoDB 不可用时抛出。
    """
    try:
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        documents = get_document_version_store().list_documents(tenant_id, kb_id, status=status)
        return {"code": 200, "tenant_id": tenant_id, "kb_id": kb_id, "documents": _mongo_document_to_json(documents)}
    except Exception as exc:
        logger.error(f"查询文档列表失败：{exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/kb/{kb_id}/documents/{doc_id}/versions")
async def list_document_versions(kb_id: str, doc_id: str, tenant_id: str = "default"):
    """查询逻辑文档版本列表。

    Args:
        kb_id: 知识库 ID。
        doc_id: 逻辑文档 ID。
        tenant_id: 租户 ID。

    Returns:
        文档版本列表。

    Raises:
        HTTPException: MongoDB 不可用时抛出。
    """
    try:
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        doc_id = _normalize_doc_id(doc_id, doc_id)
        versions = get_document_version_store().list_document_versions(tenant_id, kb_id, doc_id)
        return {
            "code": 200,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "doc_id": doc_id,
            "versions": _mongo_document_to_json(versions),
        }
    except Exception as exc:
        logger.error(f"查询文档版本失败：{exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/kb/{kb_id}/documents/{doc_id}/diff")
async def diff_document_versions(
    kb_id: str,
    doc_id: str,
    from_version: int = Query(...),
    to_version: int = Query(...),
    tenant_id: str = "default",
):
    """对比同一文档的两个版本。

    Args:
        kb_id: 知识库 ID。
        doc_id: 逻辑文档 ID。
        from_version: 基准版本号。
        to_version: 目标版本号。
        tenant_id: 租户 ID。

    Returns:
        版本 diff 摘要。

    Raises:
        HTTPException: 版本不存在或 MongoDB 不可用时抛出。
    """
    try:
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        doc_id = _normalize_doc_id(doc_id, doc_id)
        diff = get_document_version_store().diff_document_versions(
            tenant_id,
            kb_id,
            doc_id,
            from_version,
            to_version,
        )
        return {"code": 200, "diff": _mongo_document_to_json(diff)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"查询文档版本diff失败：{exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/kb/{kb_id}/version-audit")
async def document_version_audit(kb_id: str, tenant_id: str = "default"):
    """生成知识库版本审计报表。

    Args:
        kb_id: 知识库 ID。
        tenant_id: 租户 ID。

    Returns:
        版本审计报表。

    Raises:
        HTTPException: MongoDB 不可用时抛出。
    """
    try:
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        report = get_document_version_store().build_audit_report(tenant_id, kb_id)
        return {"code": 200, "audit": _mongo_document_to_json(report)}
    except Exception as exc:
        logger.error(f"生成版本审计报表失败：{exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/kb/{kb_id}/version-cleanup")
async def cleanup_document_versions(
    kb_id: str,
    tenant_id: str = Form("default"),
    dry_run: bool = Form(True),
    archive_retention_days: int | None = Form(None),
    deleted_retention_days: int | None = Form(None),
    failed_retention_days: int | None = Form(None),
    limit: int = Form(100),
):
    """清理过期文档版本向量。

    Args:
        kb_id: 知识库 ID。
        tenant_id: 租户 ID。
        dry_run: 是否只返回候选不删除。
        archive_retention_days: archived 版本保留天数。
        deleted_retention_days: deleted 版本保留天数。
        failed_retention_days: failed/pending 半成品保留天数。
        limit: 单次最多处理数量。

    Returns:
        清理候选或执行结果。

    Raises:
        HTTPException: 清理失败或依赖不可用时抛出。
    """
    try:
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        result = get_document_version_store().cleanup_expired_versions(
            tenant_id,
            kb_id,
            archive_retention_days=archive_retention_days,
            deleted_retention_days=deleted_retention_days,
            failed_retention_days=failed_retention_days,
            limit=limit,
            dry_run=dry_run,
        )
        emit_event(
            "document.cleanup_completed",
            graph="import",
            status="ok" if result.get("failed_count", 0) == 0 else "error",
            summary={
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "dry_run": bool(dry_run),
                "candidate_count": result.get("candidate_count"),
                "cleaned_count": result.get("cleaned_count"),
                "failed_count": result.get("failed_count"),
            },
            tags=["import", "document.cleanup_completed"],
        )
        return {"code": 200, "cleanup": _mongo_document_to_json(result)}
    except Exception as exc:
        logger.error(f"文档版本清理失败：{exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/documents/{doc_id}")
async def soft_delete_document(doc_id: str, tenant_id: str = "default", kb_id: str = "default"):
    """软删除逻辑文档。

    Args:
        doc_id: 逻辑文档 ID。
        tenant_id: 租户 ID。
        kb_id: 知识库 ID。

    Returns:
        软删除后的文档状态。

    Raises:
        HTTPException: 文档不存在或 MongoDB 不可用时抛出。
    """
    try:
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        doc_id = _normalize_doc_id(doc_id, doc_id)
        document = get_document_version_store().soft_delete_document(tenant_id, kb_id, doc_id)
        emit_event(
            "document.deleted",
            graph="import",
            status="ok",
            summary={"tenant_id": tenant_id, "kb_id": kb_id, "doc_id": doc_id},
            tags=["import", "document.deleted"],
        )
        return {"code": 200, "document": _mongo_document_to_json(document)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"软删除文档失败：{exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/documents/{doc_id}/rollback")
async def rollback_document(
    doc_id: str,
    target_version: int = Form(...),
    tenant_id: str = Form("default"),
    kb_id: str = Form("default"),
):
    """回滚逻辑文档到指定版本。

    Args:
        doc_id: 逻辑文档 ID。
        target_version: 目标版本号。
        tenant_id: 租户 ID。
        kb_id: 知识库 ID。

    Returns:
        回滚后的文档状态。

    Raises:
        HTTPException: 目标版本不存在或 MongoDB 不可用时抛出。
    """
    try:
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        doc_id = _normalize_doc_id(doc_id, doc_id)
        document = get_document_version_store().rollback_document(tenant_id, kb_id, doc_id, target_version)
        emit_event(
            "document.version_rolled_back",
            graph="import",
            status="ok",
            summary={
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "doc_id": doc_id,
                "target_version": target_version,
                "active_version": document.get("active_version"),
            },
            tags=["import", "document.version_rolled_back"],
        )
        return {"code": 200, "document": _mongo_document_to_json(document)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"回滚文档失败：{exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/kb/{kb_id}/vector-versions")
async def list_vector_versions(kb_id: str, tenant_id: str = "default"):
    """查询知识库向量版本配置。

    Args:
        kb_id: 知识库 ID。
        tenant_id: 租户 ID。

    Returns:
        向量版本配置列表。

    Raises:
        HTTPException: MongoDB 不可用时抛出。
    """
    try:
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        versions = get_document_version_store().list_vector_versions(tenant_id, kb_id)
        return {
            "code": 200,
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "vector_versions": _mongo_document_to_json(versions),
        }
    except Exception as exc:
        logger.error(f"查询向量版本失败：{exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/kb/{kb_id}/vector-versions/{version}/activate")
async def activate_vector_version(
    kb_id: str,
    version: str,
    tenant_id: str = Form("default"),
    fallback_vector_versions: str = Form(""),
    evaluation_report: str = Form(""),
    evaluation_status: str = Form(""),
):
    """激活知识库向量版本。

    Args:
        kb_id: 知识库 ID。
        version: 待激活向量版本。
        tenant_id: 租户 ID。
        fallback_vector_versions: 逗号分隔的 fallback 向量版本列表。
        evaluation_report: 发布前评测报告路径。
        evaluation_status: 发布前评测状态。

    Returns:
        激活后的向量版本配置。

    Raises:
        HTTPException: 参数非法或 MongoDB 不可用时抛出。
    """
    try:
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        vector_version = _normalize_doc_id(version, version)
        # fallback 列表用于灰度回退，保持顺序并过滤空白值，避免查询 filter 混入空版本。
        fallback_versions = [
            _normalize_doc_id(item.strip(), item.strip())
            for item in fallback_vector_versions.split(",")
            if item.strip()
        ]
        vector_config = get_document_version_store().activate_vector_version(
            tenant_id=tenant_id,
            kb_id=kb_id,
            vector_version=vector_version,
            fallback_vector_versions=fallback_versions,
            evaluation_report=evaluation_report or None,
            evaluation_status=evaluation_status or None,
        )
        return {"code": 200, "vector_version": _mongo_document_to_json(vector_config)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"激活向量版本失败：{exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """查询导入任务状态。

    Args:
        task_id: 全局唯一任务 ID。

    Returns:
        包含任务状态、已完成节点和运行中节点的响应。
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
