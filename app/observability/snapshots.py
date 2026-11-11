"""提供节点调试快照的保存和读取能力。"""

from __future__ import annotations

import io
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from minio import Minio

from app.core.logger import logger
from app.observability.metrics import record_minio_operation, record_snapshot
from app.observability.sanitizers import sanitize_summary
from app.utils.path_util import PROJECT_ROOT

_BUCKET_NAME = os.getenv("OBSERVABILITY_SNAPSHOT_BUCKET", "observability-snapshots")
_LOCAL_DIR = PROJECT_ROOT / "logs" / "observability_snapshots"
_REF_PREFIX = "observability-snapshot://"
_MAX_LIST_ITEMS = int(os.getenv("OBSERVABILITY_SNAPSHOT_MAX_LIST_ITEMS", "50"))
_SNAPSHOT_LEVELS = {"off", "safe", "debug", "full"}
_MINIO_CLIENT: Optional[Minio] = None
_MINIO_INIT_ATTEMPTED = False

_QUERY_STATE_KEYS = {
    "session_id",
    "tenant_id",
    "kb_id",
    "original_query",
    "embedding_chunks",
    "hyde_embedding_chunks",
    "web_search_docs",
    "rrf_chunks",
    "reranked_docs",
    "prompt",
    "answer",
    "item_names",
    "rewritten_query",
    "is_stream",
}
_IMPORT_STATE_KEYS = {
    "task_id",
    "tenant_id",
    "kb_id",
    "doc_id",
    "is_md_read_enabled",
    "is_pdf_read_enabled",
    "is_normal_split_enabled",
    "is_silicon_flow_api_enabled",
    "is_advanced_split_enabled",
    "is_vllm_enabled",
    "local_dir",
    "local_file_path",
    "pdf_path",
    "md_path",
    "file_title",
    "split_path",
    "embeddings_path",
    "md_content",
    "chunks",
    "item_name",
    "embeddings_content",
}
_FULL_TEXT_KEYS = {"original_query", "rewritten_query", "item_name", "file_title"}
_HASH_ONLY_KEYS = {"prompt", "answer", "md_content"}
_VECTOR_KEYS = {"dense_vector", "sparse_vector", "vector", "embeddings", "embedding"}


def _stringify_identifier(value: Any) -> str:
    """将调试快照中的标识符统一转为字符串。

    Args:
        value: 原始 ID 值。

    Returns:
        字符串形式的 ID。
    """
    return "" if value is None else str(value)


def get_snapshot_level() -> str:
    """读取当前快照保存策略。

    Returns:
        快照级别，支持 off、safe、debug、full。
    """
    level = os.getenv("OBSERVABILITY_SNAPSHOT_LEVEL", "safe").strip().lower()
    return level if level in _SNAPSHOT_LEVELS else "safe"


def snapshot_ref(snapshot_id: str) -> str:
    """生成快照引用字符串。

    Args:
        snapshot_id: 快照 ID。

    Returns:
        可写入事件的快照引用。
    """
    return f"{_REF_PREFIX}{snapshot_id}"


def snapshot_id_from_ref(ref: str) -> str:
    """从快照引用中解析快照 ID。

    Args:
        ref: 快照引用或快照 ID。

    Returns:
        快照 ID。
    """
    return ref.removeprefix(_REF_PREFIX)


def _get_minio_client() -> Optional[Minio]:
    """按需创建 MinIO 客户端。

    Returns:
        MinIO 客户端；配置缺失或连接失败时返回 None。
    """
    global _MINIO_CLIENT, _MINIO_INIT_ATTEMPTED
    if _MINIO_INIT_ATTEMPTED:
        return _MINIO_CLIENT
    _MINIO_INIT_ATTEMPTED = True
    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    secure = os.getenv("MINIO_SECURE") == "True"
    if not endpoint or not access_key or not secret_key:
        return None
    try:
        client = Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        if not client.bucket_exists(_BUCKET_NAME):
            client.make_bucket(_BUCKET_NAME)
        _MINIO_CLIENT = client
        return _MINIO_CLIENT
    except Exception as exc:  # pragma: no cover - depends on local MinIO
        logger.warning(f"初始化观测快照 MinIO 失败，改用本地文件：{exc}")
        return None


def _compact_external_document(document: Dict[str, Any], level: str) -> Dict[str, Any]:
    """压缩检索文档，保留节点回放所需的最小文本和元数据。

    Args:
        document: 检索文档或 Web 搜索结果。
        level: 快照保存级别。

    Returns:
        压缩后的文档。
    """
    compact: Dict[str, Any] = {}
    entity = document.get("entity") if isinstance(document.get("entity"), dict) else {}

    # Milvus search result 常见结构是 {id, distance, entity:{chunk_id, content...}}。
    # 快照里同时保留 milvus_id 和业务 chunk_id，避免人工标注时把两类 ID 混用。
    if document.get("id") is not None:
        compact["milvus_id"] = _stringify_identifier(document.get("id"))
    chunk_id = _stringify_identifier(entity.get("chunk_id") or document.get("chunk_id") or document.get("id"))
    if chunk_id:
        compact["chunk_id"] = chunk_id
    if document.get("distance") is not None:
        compact["distance"] = document.get("distance")

    for key in (
        "title",
        "url",
        "source",
        "score",
        "rerank_score",
        "item_name",
        "file_title",
        "parent_title",
        "tenant_id",
        "kb_id",
        "doc_id",
    ):
        if key in document:
            compact[key] = document.get(key)
        elif key in entity:
            compact[key] = entity.get(key)
    # 节点回放需要一定文本上下文，但不能把超长 chunk 原文完整写入快照。
    for key in ("text", "content", "snippet"):
        value = document.get(key) or entity.get(key)
        if value:
            text = str(value)
            if level == "full":
                compact[key] = text
            elif level == "debug":
                compact[key] = text[:8000]
            else:
                compact[key] = text[:2000]
            compact[f"{key}_chars"] = len(text)
    return compact or sanitize_summary(document)


def _sanitize_snapshot_value(key: str, value: Any, level: str) -> Any:
    """按快照策略清洗单个字段。

    Args:
        key: 字段名。
        value: 字段值。

    Returns:
        可 JSON 序列化且已脱敏的字段值。
    """
    if level == "off":
        return None
    if key in _HASH_ONLY_KEYS:
        if level in {"debug", "full"}:
            return value
        return sanitize_summary(value)
    if key in _FULL_TEXT_KEYS:
        return value
    if key in _VECTOR_KEYS:
        if level == "full":
            return value
        return {"omitted": True, "reason": "vector"}
    if isinstance(value, list):
        if level == "full":
            items = value
        elif level == "debug":
            items = value[: max(_MAX_LIST_ITEMS, 200)]
        else:
            items = value[:_MAX_LIST_ITEMS]
        if key in {"embedding_chunks", "hyde_embedding_chunks", "web_search_docs", "rrf_chunks", "reranked_docs", "chunks"}:
            return [_compact_external_document(item, level) if isinstance(item, dict) else sanitize_summary(item) for item in items]
        if key == "embeddings_content":
            return [
                {
                    item_key: _sanitize_snapshot_value(item_key, item_value, level)
                    for item_key, item_value in item.items()
                    if level == "full" or item_key not in _VECTOR_KEYS
                }
                if isinstance(item, dict)
                else sanitize_summary(item)
                for item in items
            ]
        return sanitize_summary(items)
    if isinstance(value, dict):
        return {
            item_key: _sanitize_snapshot_value(item_key, item_value, level)
            for item_key, item_value in value.items()
            if level == "full" or item_key not in _VECTOR_KEYS
        }
    return sanitize_summary(value)


def build_state_snapshot(graph: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """按图类型和白名单构建节点状态快照。

    Args:
        graph: 工作流名称。
        state: LangGraph State 或节点输出。

    Returns:
        可保存的状态快照。
    """
    level = get_snapshot_level()
    if level == "off":
        return {}
    allowed_keys = _QUERY_STATE_KEYS if graph == "query" else _IMPORT_STATE_KEYS
    # 只保存白名单字段，避免将临时客户端、完整 prompt 或向量对象写入快照。
    return {
        key: _sanitize_snapshot_value(key, value, level)
        for key, value in (state or {}).items()
        if key in allowed_keys
    }


def save_snapshot(
    graph: str,
    node_name: str,
    kind: str,
    payload: Dict[str, Any],
    source_event_id: Optional[str] = None,
) -> str:
    """保存节点输入或输出快照。

    Args:
        graph: 工作流名称。
        node_name: 节点名称。
        kind: 快照类型，例如 input 或 output。
        payload: 快照内容。
        source_event_id: 关联的原始事件 ID。

    Returns:
        快照引用。
    """
    snapshot_id = str(uuid.uuid4())
    document = {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "graph": graph,
        "node_name": node_name,
        "kind": kind,
        "snapshot_level": get_snapshot_level(),
        "source_event_id": source_event_id or "",
        "payload": build_state_snapshot(graph, payload),
    }
    body = json.dumps(document, ensure_ascii=False, default=str).encode("utf-8")
    object_name = f"{graph}/{node_name}/{snapshot_id}.json"
    client = _get_minio_client()
    if client is not None:
        start = time.perf_counter()
        status = "ok"
        try:
            client.put_object(
                _BUCKET_NAME,
                object_name,
                io.BytesIO(body),
                length=len(body),
                content_type="application/json",
            )
            document["storage"] = "minio"
            document["bucket"] = _BUCKET_NAME
            document["object_name"] = object_name
            _write_local_manifest(document)
            record_snapshot("minio", graph, kind, "ok")
            return snapshot_ref(snapshot_id)
        except Exception as exc:  # pragma: no cover - depends on local MinIO
            status = "error"
            record_snapshot("minio", graph, kind, "error")
            logger.warning(f"写入观测快照 MinIO 失败，改用本地文件：{exc}")
        finally:
            record_minio_operation("put_object", _BUCKET_NAME, status, time.perf_counter() - start)
    try:
        _write_local_snapshot(document)
        record_snapshot("local", graph, kind, "ok")
    except Exception:
        record_snapshot("local", graph, kind, "error")
        raise
    return snapshot_ref(snapshot_id)


def load_snapshot(ref: str) -> Dict[str, Any]:
    """读取快照内容。

    Args:
        ref: 快照 ID 或快照引用。

    Returns:
        快照文档。

    Raises:
        FileNotFoundError: 未找到快照。
    """
    snapshot_id = snapshot_id_from_ref(ref)
    local_path = _local_snapshot_path(snapshot_id)
    if local_path.exists():
        try:
            with local_path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
            document["_loaded_from"] = {
                "storage": "local",
                "path": str(local_path),
            }
            record_snapshot("local", document.get("graph", "unknown"), document.get("kind", "load"), "ok")
            return document
        except Exception:
            record_snapshot("local", "unknown", "load", "error")
            raise
    manifest = _load_local_manifest(snapshot_id)
    if manifest:
        client = _get_minio_client()
        if client is not None:
            start = time.perf_counter()
            status = "ok"
            response = None
            try:
                response = client.get_object(manifest["bucket"], manifest["object_name"])
                document = json.loads(response.read().decode("utf-8"))
                document["_loaded_from"] = {
                    "storage": "minio",
                    "bucket": manifest["bucket"],
                    "object_name": manifest["object_name"],
                }
                record_snapshot("minio", document.get("graph", "unknown"), document.get("kind", "load"), "ok")
                return document
            except Exception:
                status = "error"
                record_snapshot("minio", "unknown", "load", "error")
                raise
            finally:
                record_minio_operation("get_object", manifest.get("bucket", _BUCKET_NAME), status, time.perf_counter() - start)
                if response is not None:
                    response.close()
                    response.release_conn()
    record_snapshot("unknown", "unknown", "load", "error")
    raise FileNotFoundError(f"snapshot not found: {snapshot_id}")


def _local_snapshot_path(snapshot_id: str) -> Path:
    """返回本地快照路径。

    Args:
        snapshot_id: 快照 ID。

    Returns:
        本地快照文件路径。
    """
    return _LOCAL_DIR / f"{snapshot_id}.json"


def _write_local_snapshot(document: Dict[str, Any]) -> None:
    """将快照写入本地文件。

    Args:
        document: 快照文档。
    """
    _LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    document["storage"] = "local"
    with _local_snapshot_path(document["snapshot_id"]).open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, default=str)


def _manifest_path() -> Path:
    """返回本地快照索引路径。

    Returns:
        本地索引文件路径。
    """
    return _LOCAL_DIR / "manifest.jsonl"


def _write_local_manifest(document: Dict[str, Any]) -> None:
    """记录 MinIO 快照索引，便于按 ID 回读。

    Args:
        document: 快照元数据。
    """
    _LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    with _manifest_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({key: document[key] for key in ("snapshot_id", "bucket", "object_name", "storage")}))
        handle.write("\n")


def _load_local_manifest(snapshot_id: str) -> Optional[Dict[str, Any]]:
    """读取本地快照索引。

    Args:
        snapshot_id: 快照 ID。

    Returns:
        快照索引；不存在时返回 None。
    """
    path = _manifest_path()
    if not path.exists():
        return None
    matched_item: Optional[Dict[str, Any]] = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except Exception:
                continue
            if item.get("snapshot_id") == snapshot_id:
                # manifest 是追加写入，调试读取同 ID 时使用最后一条记录更符合“最新索引”的直觉。
                matched_item = item
    return matched_item
