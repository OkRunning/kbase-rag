"""提供 RAG 文档版本治理的 MongoDB 读写能力。"""

from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection

from app.core.logger import logger
from app.core.tenant_scope import normalize_kb_id, normalize_tenant_id
from app.observability.metrics import record_mongo_operation
from app.utils.escape_milvus_string_utils import escape_milvus_string

load_dotenv()

DOCUMENTS_COLLECTION = "rag_documents"
DOCUMENT_VERSIONS_COLLECTION = "rag_document_versions"
CHUNKS_COLLECTION = "rag_chunks"
VECTOR_VERSIONS_COLLECTION = "rag_vector_versions"
DEFAULT_VECTOR_VERSION = "bge_m3_v1_legacy"
DEFAULT_CHUNK_STRATEGY = "markdown_heading"
DEFAULT_CHUNK_STRATEGY_VERSION = "v1"
DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_EMBEDDING_VERSION = "v1"
DEFAULT_SCHEMA_VERSION = "milvus_chunks_v2"
DEFAULT_EMBEDDING_DIMENSION = 1024
DEFAULT_DENSE_WEIGHT = 0.9
DEFAULT_SPARSE_WEIGHT = 0.1
DEFAULT_ARCHIVE_RETENTION_DAYS = 30
DEFAULT_DELETED_RETENTION_DAYS = 30
DEFAULT_FAILED_RETENTION_DAYS = 7

_document_version_store: DocumentVersionStore | None = None
_document_version_store_error: str | None = None


def is_document_versioning_enabled() -> bool:
    """判断是否启用文档版本治理。

    Returns:
        配置启用时返回 True；显式配置为 false/0/no/off 时返回 False。
    """
    value = os.getenv("RAG_DOCUMENT_VERSIONING_ENABLED", "true")
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def utc_now_iso() -> str:
    """获取当前 UTC 时间字符串。

    Returns:
        ISO 8601 格式的 UTC 时间。
    """
    return datetime.now(timezone.utc).isoformat()


def get_configured_vector_version() -> str:
    """获取当前配置的默认向量版本。

    Returns:
        环境变量配置的向量版本；未配置时返回兼容默认版本。
    """
    return os.getenv("RAG_VECTOR_VERSION", DEFAULT_VECTOR_VERSION).strip() or DEFAULT_VECTOR_VERSION


def get_retention_days(env_name: str, default_days: int) -> int:
    """读取版本治理保留期配置。

    Args:
        env_name: 环境变量名称。
        default_days: 配置缺失或非法时使用的默认天数。

    Returns:
        非负整数保留天数。
    """
    try:
        return max(int(os.getenv(env_name, str(default_days))), 0)
    except ValueError:
        logger.warning(f"保留期配置非法，使用默认值：{env_name}={default_days}")
        return default_days


def parse_iso_datetime(value: str | None) -> datetime | None:
    """解析 ISO 8601 时间字符串。

    Args:
        value: 待解析的时间字符串。

    Returns:
        带时区的 datetime；空值或非法值返回 None。
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def build_version_milvus_filter(tenant_id: str, kb_id: str, doc_id: str, doc_version: int) -> str:
    """构建单个文档版本的 Milvus 删除过滤表达式。

    Args:
        tenant_id: 租户标识。
        kb_id: 知识库标识。
        doc_id: 逻辑文档标识。
        doc_version: 文档版本号。

    Returns:
        可用于 Milvus delete 的过滤表达式。
    """
    return (
        f'tenant_id == "{escape_milvus_string(normalize_tenant_id(tenant_id))}" '
        f'and kb_id == "{escape_milvus_string(normalize_kb_id(kb_id))}" '
        f'and doc_id == "{escape_milvus_string(str(doc_id))}" '
        f"and doc_version == {int(doc_version)}"
    )


def compute_file_hash(file_path: str) -> str:
    """计算本地文件的 SHA256 指纹。

    Args:
        file_path: 待计算指纹的文件路径。

    Returns:
        带 sha256 前缀的文件指纹。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
    """
    path = Path(file_path)
    hasher = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(block)
    return f"sha256:{hasher.hexdigest()}"


def compute_content_hash(content: str) -> str:
    """计算规范化文本内容的 SHA256 指纹。

    Args:
        content: 文档解析后的 Markdown 文本。

    Returns:
        带 sha256 前缀的内容指纹。
    """
    normalized = "\n".join(line.rstrip() for line in (content or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    return f"sha256:{hashlib.sha256(normalized.strip().encode('utf-8')).hexdigest()}"


def normalize_chunk_content(content: str) -> str:
    """规范化 chunk 正文用于稳定指纹计算。

    Args:
        content: chunk 原始正文。

    Returns:
        规范化后的 chunk 正文。
    """
    lines = (content or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def compute_chunk_hash(
    chunk: dict[str, Any],
    *,
    item_name: str = "",
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> str:
    """计算 chunk 的稳定 SHA256 指纹。

    Args:
        chunk: 待计算指纹的切片。
        item_name: 当前文档识别出的商品名称。
        chunk_strategy: 当前切分策略名称。
        schema_version: Milvus metadata schema 版本。

    Returns:
        带 sha256 前缀的 chunk 指纹。
    """
    hash_input = "\n".join(
        [
            schema_version,
            chunk_strategy,
            item_name or str(chunk.get("item_name") or ""),
            str(chunk.get("title") or ""),
            str(chunk.get("parent_title") or ""),
            normalize_chunk_content(str(chunk.get("content") or "")),
        ]
    )
    return f"sha256:{hashlib.sha256(hash_input.encode('utf-8')).hexdigest()}"


def build_active_document_version_filter(
    tenant_id: str,
    kb_id: str,
    item_names: list[str] | None = None,
) -> str | None:
    """构建当前 active 文档版本的 Milvus 过滤表达式。

    Args:
        tenant_id: 租户标识。
        kb_id: 知识库标识。
        item_names: 可选的商品名称列表，用于收窄 active 文档范围。

    Returns:
        可拼接到 Milvus filter 的表达式；版本库不可用或无记录时返回 None。
    """
    if not is_document_versioning_enabled():
        return None
    try:
        store = get_document_version_store()
        active_docs = store.list_active_documents(tenant_id, kb_id, item_names=item_names)
    except Exception as exc:
        logger.warning(f"构建 active 文档版本过滤条件失败，将使用兼容检索：{exc}")
        return None
    if not active_docs:
        return None

    clauses = []
    for document in active_docs:
        doc_id = document.get("doc_id")
        active_version = document.get("active_version")
        if doc_id and active_version is not None:
            clauses.append(
                f'(doc_id == "{escape_milvus_string(str(doc_id))}" and doc_version == {int(active_version)})'
            )
    if not clauses:
        return None
    # 查询侧必须按 Mongo active 指针过滤版本，避免 Milvus 中历史 active metadata 污染线上召回。
    return "(" + " or ".join(clauses) + ")"


def build_active_vector_version_filter(tenant_id: str, kb_id: str) -> str | None:
    """构建当前 active 与 fallback 向量版本的 Milvus 过滤表达式。

    Args:
        tenant_id: 租户标识。
        kb_id: 知识库标识。

    Returns:
        可拼接到 Milvus filter 的向量版本表达式；版本治理不可用时返回 None。
    """
    if not is_document_versioning_enabled():
        return None
    try:
        store = get_document_version_store()
        versions = store.list_queryable_vector_versions(tenant_id, kb_id)
    except Exception as exc:
        logger.warning(f"构建 active 向量版本过滤条件失败，将使用兼容检索：{exc}")
        return None
    if not versions:
        return None
    values = ",".join(f'"{escape_milvus_string(version)}"' for version in versions)
    if not values:
        return None
    # P2 只允许当前 active/fallback 向量版本进入查询，避免不同 embedding 空间的分数被混排。
    return f"vector_version in [{values}]"


class DocumentVersionStore:
    """封装 RAG 文档版本治理集合的基础读写。

    该类管理逻辑文档和文档版本两个集合，用于导入时创建 pending 版本、
    导入完成后激活版本，以及查询侧解析 active 版本过滤条件。
    """

    def __init__(self):
        """初始化 MongoDB 连接和索引。

        Raises:
            RuntimeError: MongoDB 配置缺失时抛出。
            Exception: MongoDB 连接或索引创建失败时抛出。
        """
        mongo_url = os.getenv("MONGO_URL")
        db_name = os.getenv("MONGO_DB_NAME")
        if not mongo_url or not db_name:
            raise RuntimeError("缺少 MONGO_URL 或 MONGO_DB_NAME，无法启用文档版本治理")

        username = os.getenv("MONGO_DB_USERNAME")
        password = os.getenv("MONGO_DB_PASSWORD")
        self.client = MongoClient(mongo_url, username=username, password=password, authSource="admin")
        self.client.admin.command("ping")
        self.db = self.client[db_name]
        self.documents: Collection = self.db[DOCUMENTS_COLLECTION]
        self.document_versions: Collection = self.db[DOCUMENT_VERSIONS_COLLECTION]
        self.chunks: Collection = self.db[CHUNKS_COLLECTION]
        self.vector_versions: Collection = self.db[VECTOR_VERSIONS_COLLECTION]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """创建文档版本治理所需索引。

        Returns:
            None.
        """
        self.documents.create_index(
            [("tenant_id", ASCENDING), ("kb_id", ASCENDING), ("doc_id", ASCENDING)],
            unique=True,
        )
        self.documents.create_index([("tenant_id", ASCENDING), ("kb_id", ASCENDING), ("status", ASCENDING)])
        self.documents.create_index([("tenant_id", ASCENDING), ("kb_id", ASCENDING), ("item_name", ASCENDING)])
        self.document_versions.create_index(
            [
                ("tenant_id", ASCENDING),
                ("kb_id", ASCENDING),
                ("doc_id", ASCENDING),
                ("doc_version", ASCENDING),
            ],
            unique=True,
        )
        self.document_versions.create_index([("tenant_id", ASCENDING), ("kb_id", ASCENDING), ("doc_id", ASCENDING), ("status", ASCENDING)])
        self.document_versions.create_index([("tenant_id", ASCENDING), ("kb_id", ASCENDING), ("file_hash", ASCENDING)])
        self.chunks.create_index(
            [
                ("tenant_id", ASCENDING),
                ("kb_id", ASCENDING),
                ("doc_id", ASCENDING),
                ("doc_version", ASCENDING),
                ("chunk_uid", ASCENDING),
            ],
            unique=True,
        )
        self.chunks.create_index(
            [
                ("tenant_id", ASCENDING),
                ("kb_id", ASCENDING),
                ("doc_id", ASCENDING),
                ("doc_version", ASCENDING),
                ("chunk_hash", ASCENDING),
            ]
        )
        self.vector_versions.create_index(
            [("tenant_id", ASCENDING), ("kb_id", ASCENDING), ("vector_version", ASCENDING)],
            unique=True,
        )
        self.vector_versions.create_index([("tenant_id", ASCENDING), ("kb_id", ASCENDING), ("status", ASCENDING)])

    def ensure_vector_version(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        vector_version: str,
        status: str = "active",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        embedding_version: str = DEFAULT_EMBEDDING_VERSION,
        chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
        chunk_strategy_version: str = DEFAULT_CHUNK_STRATEGY_VERSION,
        schema_version: str = DEFAULT_SCHEMA_VERSION,
        dense_weight_default: float = DEFAULT_DENSE_WEIGHT,
        sparse_weight_default: float = DEFAULT_SPARSE_WEIGHT,
        fallback_vector_versions: list[str] | None = None,
        evaluation_report: str | None = None,
        evaluation_status: str | None = None,
    ) -> dict[str, Any]:
        """创建或更新向量版本配置。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            vector_version: 向量版本名称。
            status: 版本状态，例如 active、candidate、archived。
            embedding_model: embedding 模型名称。
            embedding_dimension: embedding 向量维度。
            embedding_version: embedding 模型版本。
            chunk_strategy: chunk 切分策略名称。
            chunk_strategy_version: chunk 切分策略版本。
            schema_version: Milvus metadata schema 版本。
            dense_weight_default: 默认稠密向量权重。
            sparse_weight_default: 默认稀疏向量权重。
            fallback_vector_versions: 查询异常时允许回退的向量版本列表。
            evaluation_report: 发布前评测报告路径。
            evaluation_status: 发布前评测状态。

        Returns:
            创建或更新后的向量版本配置。
        """
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        vector_version = vector_version.strip() or get_configured_vector_version()
        now = utc_now_iso()
        key = f"{tenant_id}:{kb_id}:{vector_version}"
        doc = {
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "vector_version": vector_version,
            "status": status,
            "embedding_model": embedding_model,
            "embedding_dimension": int(embedding_dimension),
            "embedding_version": embedding_version,
            "chunk_strategy": chunk_strategy,
            "chunk_strategy_version": chunk_strategy_version,
            "schema_version": schema_version,
            "dense_weight_default": float(dense_weight_default),
            "sparse_weight_default": float(sparse_weight_default),
            "fallback_vector_versions": fallback_vector_versions or [],
            "evaluation_report": evaluation_report,
            "evaluation_status": evaluation_status,
            "updated_at": now,
        }
        if status == "active":
            doc["activated_at"] = now
        self.vector_versions.update_one(
            {"_id": key},
            {
                "$setOnInsert": {"_id": key, "created_at": now},
                "$set": doc,
            },
            upsert=True,
        )
        return self.vector_versions.find_one({"_id": key}) or {}

    def get_active_vector_version(self, tenant_id: str, kb_id: str) -> dict[str, Any]:
        """查询当前 active 向量版本配置，不存在时创建默认配置。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。

        Returns:
            当前 active 向量版本配置。
        """
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        active = self.vector_versions.find_one({"tenant_id": tenant_id, "kb_id": kb_id, "status": "active"})
        if active:
            return active
        # 首次启用 P2 时自动补齐默认 active 配置，保持导入链路无额外初始化步骤。
        return self.ensure_vector_version(
            tenant_id=tenant_id,
            kb_id=kb_id,
            vector_version=get_configured_vector_version(),
            status="active",
        )

    def list_vector_versions(self, tenant_id: str, kb_id: str) -> list[dict[str, Any]]:
        """查询知识库内全部向量版本配置。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。

        Returns:
            按更新时间倒序排列的向量版本配置列表。
        """
        return list(
            self.vector_versions.find(
                {"tenant_id": normalize_tenant_id(tenant_id), "kb_id": normalize_kb_id(kb_id)}
            ).sort("updated_at", -1)
        )

    def list_queryable_vector_versions(self, tenant_id: str, kb_id: str) -> list[str]:
        """查询线上检索允许使用的向量版本列表。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。

        Returns:
            active 向量版本及其 fallback 版本列表。
        """
        active = self.get_active_vector_version(tenant_id, kb_id)
        versions = [str(active.get("vector_version") or get_configured_vector_version())]
        for fallback in active.get("fallback_vector_versions") or []:
            if fallback and fallback not in versions:
                versions.append(str(fallback))
        return versions

    def activate_vector_version(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        vector_version: str,
        fallback_vector_versions: list[str] | None = None,
        evaluation_report: str | None = None,
        evaluation_status: str | None = None,
    ) -> dict[str, Any]:
        """激活指定向量版本并归档同知识库下旧 active 版本。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            vector_version: 待激活向量版本。
            fallback_vector_versions: 新版本异常时允许检索回退的向量版本。
            evaluation_report: 发布前评测报告路径。
            evaluation_status: 发布前评测状态。

        Returns:
            激活后的向量版本配置。
        """
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        vector_version = vector_version.strip() or get_configured_vector_version()
        now = utc_now_iso()
        # 先归档旧 active，再 upsert 目标版本为 active；单知识库只保留一个主 active 版本。
        self.vector_versions.update_many(
            {
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "vector_version": {"$ne": vector_version},
                "status": "active",
            },
            {"$set": {"status": "archived", "updated_at": now}},
        )
        return self.ensure_vector_version(
            tenant_id=tenant_id,
            kb_id=kb_id,
            vector_version=vector_version,
            status="active",
            fallback_vector_versions=fallback_vector_versions,
            evaluation_report=evaluation_report,
            evaluation_status=evaluation_status,
        )

    def prepare_version(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        doc_id: str,
        file_hash: str,
        file_name: str,
        title: str,
        created_by: str,
        vector_version: str = DEFAULT_VECTOR_VERSION,
    ) -> dict[str, Any]:
        """创建 pending 文档版本或判定本次导入无需处理。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            doc_id: 逻辑文档标识。
            file_hash: 上传文件指纹。
            file_name: 上传文件名。
            title: 文档标题。
            created_by: 上传用户标识。
            vector_version: 当前导入使用的向量版本。

        Returns:
            包含导入动作、版本号和 previous_doc_version 的状态字典。
        """
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        doc_key = f"{tenant_id}:{kb_id}:{doc_id}"
        now = utc_now_iso()
        start = time.perf_counter()
        status = "ok"
        try:
            active_vector = self.get_active_vector_version(tenant_id, kb_id)
            vector_version = str(
                active_vector.get("vector_version") or vector_version or get_configured_vector_version()
            )
            document = self.documents.find_one({"_id": doc_key})
            previous_doc_version = document.get("active_version") if document else None
            active_version = None
            if previous_doc_version is not None:
                active_version = self.document_versions.find_one(
                    {
                        "tenant_id": tenant_id,
                        "kb_id": kb_id,
                        "doc_id": doc_id,
                        "doc_version": previous_doc_version,
                    }
                )
            if active_version and active_version.get("file_hash") == file_hash:
                return {
                    "import_action": "noop",
                    "skip_reason": "file_hash_unchanged",
                    "doc_version": previous_doc_version,
                    "previous_doc_version": previous_doc_version,
                    "file_hash": file_hash,
                    "vector_version": active_version.get("vector_version", vector_version),
                }

            latest_version = int(document.get("latest_version", 0)) if document else 0
            doc_version = latest_version + 1
            version_key = f"{doc_key}:{doc_version}"

            # pending 版本先落 Mongo，后续解析/向量写入失败时不会影响当前 active 指针。
            self.documents.update_one(
                {"_id": doc_key},
                {
                    "$setOnInsert": {
                        "_id": doc_key,
                        "tenant_id": tenant_id,
                        "kb_id": kb_id,
                        "doc_id": doc_id,
                        "created_at": now,
                    },
                    "$set": {
                        "title": title,
                        "file_name": file_name,
                        "latest_version": doc_version,
                        "status": "active",
                        "deleted_at": None,
                        "updated_at": now,
                    },
                },
                upsert=True,
            )
            self.document_versions.replace_one(
                {"_id": version_key},
                {
                    "_id": version_key,
                    "tenant_id": tenant_id,
                    "kb_id": kb_id,
                    "doc_id": doc_id,
                    "doc_version": doc_version,
                    "status": "pending",
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "content_hash": None,
                    "parser": {"type": "mineru_or_markdown", "version": "default"},
                    "chunk_strategy": {
                        "name": DEFAULT_CHUNK_STRATEGY,
                        "version": DEFAULT_CHUNK_STRATEGY_VERSION,
                    },
                    "vector_version": vector_version,
                    "chunk_count": 0,
                    "milvus_insert_count": 0,
                    "created_by": created_by,
                    "created_at": now,
                    "updated_at": now,
                    "error": None,
                },
                upsert=True,
            )
            return {
                "import_action": "create" if previous_doc_version is None else "update",
                "skip_reason": "",
                "doc_version": doc_version,
                "previous_doc_version": previous_doc_version,
                "file_hash": file_hash,
                "vector_version": vector_version,
            }
        except Exception:
            status = "error"
            raise
        finally:
            record_mongo_operation("prepare_version", DOCUMENT_VERSIONS_COLLECTION, status, time.perf_counter() - start)

    def activate_version(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        doc_id: str,
        doc_version: int,
        previous_doc_version: int | None,
        content_hash: str | None,
        item_name: str,
        chunk_count: int,
        milvus_insert_count: int,
        chunks: list[dict[str, Any]] | None = None,
        changed_chunk_count: int | None = None,
        reused_chunk_count: int | None = None,
        removed_chunk_count: int | None = None,
    ) -> dict[str, Any]:
        """激活已写入 Milvus 的文档版本。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            doc_id: 逻辑文档标识。
            doc_version: 待激活的文档版本号。
            previous_doc_version: 原 active 版本号。
            content_hash: 解析后内容指纹。
            item_name: 文档识别出的商品名称。
            chunk_count: 当前版本切片数量。
            milvus_insert_count: Milvus 写入数量。
            chunks: 当前版本 chunk 元数据列表。
            changed_chunk_count: 新增或变化 chunk 数量。
            reused_chunk_count: 复用上一 active 版本向量的 chunk 数量。
            removed_chunk_count: 上一 active 版本中被移除的 chunk 数量。

        Returns:
            更新后的逻辑文档记录。

        Raises:
            ValueError: Milvus 写入数量与 chunk 数量不一致，或 active 指针已被并发导入切换时抛出。
        """
        if chunk_count != milvus_insert_count:
            raise ValueError(f"Milvus写入数量({milvus_insert_count})与chunk数量({chunk_count})不一致")

        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        doc_key = f"{tenant_id}:{kb_id}:{doc_id}"
        now = utc_now_iso()
        start = time.perf_counter()
        status = "ok"
        try:
            document_filter: dict[str, Any] = {"_id": doc_key}
            if previous_doc_version is None:
                document_filter["active_version"] = {"$exists": False}
            else:
                document_filter["active_version"] = previous_doc_version

            # 并发导入同一个 doc_id 时，只有仍持有旧 active 指针的任务可以切换。
            # 失败任务保留为 ready_conflict，方便运维用 diff/audit 判断是否手动重试或清理。
            switch_result = self.documents.update_one(
                document_filter,
                {
                    "$set": {
                        "active_version": doc_version,
                        "latest_version": doc_version,
                        "item_name": item_name,
                        "status": "active",
                        "deleted_at": None,
                        "updated_at": now,
                    }
                },
            )
            if switch_result.matched_count == 0:
                current_document = self.documents.find_one({"_id": doc_key}) or {}
                self.document_versions.update_one(
                    {"tenant_id": tenant_id, "kb_id": kb_id, "doc_id": doc_id, "doc_version": doc_version},
                    {
                        "$set": {
                            "status": "ready_conflict",
                            "content_hash": content_hash,
                            "item_name": item_name,
                            "chunk_count": chunk_count,
                            "changed_chunk_count": changed_chunk_count,
                            "reused_chunk_count": reused_chunk_count,
                            "removed_chunk_count": removed_chunk_count,
                            "milvus_insert_count": milvus_insert_count,
                            "conflict_active_version": current_document.get("active_version"),
                            "updated_at": now,
                            "error": "active_version_conflict",
                        }
                    },
                )
                raise ValueError("文档 active_version 已被其他导入任务切换，请通过版本 diff 审核后处理")

            self.document_versions.update_one(
                {"tenant_id": tenant_id, "kb_id": kb_id, "doc_id": doc_id, "doc_version": doc_version},
                {
                    "$set": {
                        "status": "active",
                        "content_hash": content_hash,
                        "item_name": item_name,
                        "chunk_count": chunk_count,
                        "changed_chunk_count": changed_chunk_count,
                        "reused_chunk_count": reused_chunk_count,
                        "removed_chunk_count": removed_chunk_count,
                        "milvus_insert_count": milvus_insert_count,
                        "activated_at": now,
                        "updated_at": now,
                        "error": None,
                    }
                },
            )
            if previous_doc_version is not None and previous_doc_version != doc_version:
                self.document_versions.update_one(
                    {"tenant_id": tenant_id, "kb_id": kb_id, "doc_id": doc_id, "doc_version": previous_doc_version},
                    {"$set": {"status": "archived", "updated_at": now}},
                )
            if chunks is not None:
                self.replace_chunks(
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    doc_id=doc_id,
                    doc_version=doc_version,
                    chunks=chunks,
                )
            return self.documents.find_one({"_id": doc_key}) or {}
        except Exception:
            status = "error"
            raise
        finally:
            record_mongo_operation("activate_version", DOCUMENT_VERSIONS_COLLECTION, status, time.perf_counter() - start)

    def mark_version_failed(
        self,
        tenant_id: str,
        kb_id: str,
        doc_id: str,
        doc_version: int | None,
        error: str,
    ) -> None:
        """将 pending 版本标记为 failed。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            doc_id: 逻辑文档标识。
            doc_version: 版本号，为空时不处理。
            error: 失败原因。

        Returns:
            None.
        """
        if doc_version is None:
            return
        now = utc_now_iso()
        self.document_versions.update_one(
            {
                "tenant_id": normalize_tenant_id(tenant_id),
                "kb_id": normalize_kb_id(kb_id),
                "doc_id": doc_id,
                "doc_version": doc_version,
                "status": "pending",
            },
            {"$set": {"status": "failed", "error": error, "updated_at": now}},
        )

    def list_active_documents(
        self,
        tenant_id: str,
        kb_id: str,
        *,
        item_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """查询当前可见的逻辑文档。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            item_names: 可选商品名列表。

        Returns:
            当前 active 文档列表。
        """
        query: dict[str, Any] = {
            "tenant_id": normalize_tenant_id(tenant_id),
            "kb_id": normalize_kb_id(kb_id),
            "status": "active",
            "active_version": {"$ne": None},
        }
        if item_names:
            query["item_name"] = {"$in": item_names}
        return list(self.documents.find(query))

    def list_documents(self, tenant_id: str, kb_id: str, status: str | None = None) -> list[dict[str, Any]]:
        """查询知识库逻辑文档列表。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            status: 可选文档状态过滤。

        Returns:
            按更新时间倒序排列的逻辑文档列表。
        """
        query: dict[str, Any] = {
            "tenant_id": normalize_tenant_id(tenant_id),
            "kb_id": normalize_kb_id(kb_id),
        }
        if status:
            query["status"] = status
        return list(self.documents.find(query).sort("updated_at", -1))

    def list_document_versions(self, tenant_id: str, kb_id: str, doc_id: str) -> list[dict[str, Any]]:
        """查询逻辑文档的版本列表。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            doc_id: 逻辑文档标识。

        Returns:
            按版本号倒序排列的文档版本列表。
        """
        return list(
            self.document_versions.find(
                {
                    "tenant_id": normalize_tenant_id(tenant_id),
                    "kb_id": normalize_kb_id(kb_id),
                    "doc_id": doc_id,
                }
            ).sort("doc_version", -1)
        )

    def list_version_chunks(self, tenant_id: str, kb_id: str, doc_id: str, doc_version: int) -> list[dict[str, Any]]:
        """查询指定文档版本的 chunk 元数据。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            doc_id: 逻辑文档标识。
            doc_version: 文档版本号。

        Returns:
            按 chunk_index 排序的 chunk 元数据列表。
        """
        query = {
            "tenant_id": normalize_tenant_id(tenant_id),
            "kb_id": normalize_kb_id(kb_id),
            "doc_id": doc_id,
            "doc_version": int(doc_version),
        }
        return list(self.chunks.find(query).sort("chunk_index", ASCENDING))

    def diff_document_versions(
        self,
        tenant_id: str,
        kb_id: str,
        doc_id: str,
        from_version: int,
        to_version: int,
    ) -> dict[str, Any]:
        """对比同一逻辑文档的两个 chunk 版本。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            doc_id: 逻辑文档标识。
            from_version: 基准文档版本号。
            to_version: 目标文档版本号。

        Returns:
            包含 added、removed、unchanged 和 index_changed 的差异摘要。

        Raises:
            ValueError: 任一目标版本不存在时抛出。
        """
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        from_doc = self.document_versions.find_one(
            {"tenant_id": tenant_id, "kb_id": kb_id, "doc_id": doc_id, "doc_version": int(from_version)}
        )
        to_doc = self.document_versions.find_one(
            {"tenant_id": tenant_id, "kb_id": kb_id, "doc_id": doc_id, "doc_version": int(to_version)}
        )
        if not from_doc or not to_doc:
            raise ValueError("待对比版本不存在")

        from_chunks = self.list_version_chunks(tenant_id, kb_id, doc_id, int(from_version))
        to_chunks = self.list_version_chunks(tenant_id, kb_id, doc_id, int(to_version))
        from_by_hash = {chunk.get("chunk_hash"): chunk for chunk in from_chunks if chunk.get("chunk_hash")}
        to_by_hash = {chunk.get("chunk_hash"): chunk for chunk in to_chunks if chunk.get("chunk_hash")}
        added_hashes = [chunk_hash for chunk_hash in to_by_hash if chunk_hash not in from_by_hash]
        removed_hashes = [chunk_hash for chunk_hash in from_by_hash if chunk_hash not in to_by_hash]
        shared_hashes = [chunk_hash for chunk_hash in to_by_hash if chunk_hash in from_by_hash]
        index_changed = [
            {
                "chunk_hash": chunk_hash,
                "from_index": from_by_hash[chunk_hash].get("chunk_index"),
                "to_index": to_by_hash[chunk_hash].get("chunk_index"),
                "title": to_by_hash[chunk_hash].get("title") or from_by_hash[chunk_hash].get("title"),
            }
            for chunk_hash in shared_hashes
            if from_by_hash[chunk_hash].get("chunk_index") != to_by_hash[chunk_hash].get("chunk_index")
        ]

        def _chunk_summary(chunk: dict[str, Any]) -> dict[str, Any]:
            """生成版本 diff 返回中的 chunk 摘要。

            Args:
                chunk: MongoDB chunk 元数据。

            Returns:
                面向 API 的低敏 chunk 摘要。
            """
            return {
                "chunk_uid": chunk.get("chunk_uid"),
                "chunk_index": chunk.get("chunk_index"),
                "chunk_hash": chunk.get("chunk_hash"),
                "title": chunk.get("title"),
                "parent_title": chunk.get("parent_title"),
                "content_preview": chunk.get("content_preview"),
            }

        # diff 以 chunk_hash 为主，避免章节插入导致 chunk_index 整体偏移时误判所有后续切片变化。
        return {
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "doc_id": doc_id,
            "from_version": int(from_version),
            "to_version": int(to_version),
            "summary": {
                "added": len(added_hashes),
                "removed": len(removed_hashes),
                "unchanged": len(shared_hashes),
                "index_changed": len(index_changed),
                "from_chunk_count": len(from_chunks),
                "to_chunk_count": len(to_chunks),
            },
            "added_chunks": [_chunk_summary(to_by_hash[chunk_hash]) for chunk_hash in added_hashes],
            "removed_chunks": [_chunk_summary(from_by_hash[chunk_hash]) for chunk_hash in removed_hashes],
            "index_changed_chunks": index_changed,
        }

    def build_audit_report(self, tenant_id: str, kb_id: str) -> dict[str, Any]:
        """生成知识库文档版本审计报表。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。

        Returns:
            包含文档、版本、chunk 和回滚线索的审计摘要。
        """
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        documents = list(self.documents.find({"tenant_id": tenant_id, "kb_id": kb_id}))
        versions = list(self.document_versions.find({"tenant_id": tenant_id, "kb_id": kb_id}))
        status_counts: dict[str, int] = {}
        vector_version_counts: dict[str, int] = {}
        total_chunks = 0
        total_changed_chunks = 0
        total_reused_chunks = 0
        rollback_records = []
        conflict_versions = []
        for version in versions:
            version_status = str(version.get("status") or "unknown")
            status_counts[version_status] = status_counts.get(version_status, 0) + 1
            vector_version = str(version.get("vector_version") or "unknown")
            vector_version_counts[vector_version] = vector_version_counts.get(vector_version, 0) + 1
            total_chunks += int(version.get("chunk_count") or 0)
            total_changed_chunks += int(version.get("changed_chunk_count") or 0)
            total_reused_chunks += int(version.get("reused_chunk_count") or 0)
            if version.get("rolled_back_at"):
                rollback_records.append(
                    {
                        "doc_id": version.get("doc_id"),
                        "target_version": version.get("doc_version"),
                        "rollback_from_version": version.get("rollback_from_version"),
                        "rolled_back_at": version.get("rolled_back_at"),
                    }
                )
            if version_status == "ready_conflict":
                conflict_versions.append(
                    {
                        "doc_id": version.get("doc_id"),
                        "doc_version": version.get("doc_version"),
                        "conflict_active_version": version.get("conflict_active_version"),
                        "updated_at": version.get("updated_at"),
                    }
                )
        return {
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "document_count": len(documents),
            "active_document_count": sum(1 for document in documents if document.get("status") == "active"),
            "deleted_document_count": sum(1 for document in documents if document.get("status") == "deleted"),
            "version_count": len(versions),
            "version_status_counts": status_counts,
            "vector_version_counts": vector_version_counts,
            "chunk_totals": {
                "chunk_count": total_chunks,
                "changed_chunk_count": total_changed_chunks,
                "reused_chunk_count": total_reused_chunks,
            },
            "rollback_count": len(rollback_records),
            "rollback_records": rollback_records,
            "conflict_count": len(conflict_versions),
            "conflict_versions": conflict_versions,
        }

    def replace_chunks(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        doc_id: str,
        doc_version: int,
        chunks: list[dict[str, Any]],
    ) -> None:
        """替换指定文档版本的 chunk 元数据快照。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            doc_id: 逻辑文档标识。
            doc_version: 文档版本号。
            chunks: 当前版本 chunk 列表。

        Returns:
            None.
        """
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        doc_version = int(doc_version)
        now = utc_now_iso()
        docs = []
        for index, chunk in enumerate(chunks):
            chunk_uid = str(chunk.get("chunk_uid") or f"{doc_id}_v{doc_version}_chunk_{index:05d}")
            docs.append(
                {
                    "_id": f"{tenant_id}:{kb_id}:{doc_id}:{doc_version}:{chunk_uid}",
                    "tenant_id": tenant_id,
                    "kb_id": kb_id,
                    "doc_id": doc_id,
                    "doc_version": doc_version,
                    "chunk_uid": chunk_uid,
                    "chunk_index": int(chunk.get("chunk_index", index)),
                    "chunk_hash": chunk.get("chunk_hash", ""),
                    "content_hash": compute_content_hash(str(chunk.get("content") or "")),
                    "title": chunk.get("title", ""),
                    "parent_title": chunk.get("parent_title", ""),
                    "item_name": chunk.get("item_name", ""),
                    "content_preview": str(chunk.get("content") or "")[:200],
                    "status": chunk.get("chunk_status", "active"),
                    "source_chunk_uid": chunk.get("source_chunk_uid"),
                    "diff_action": chunk.get("diff_action", "changed"),
                    "milvus_chunk_id": chunk.get("chunk_id"),
                    "created_at": now,
                }
            )
        self.chunks.delete_many({"tenant_id": tenant_id, "kb_id": kb_id, "doc_id": doc_id, "doc_version": doc_version})
        if docs:
            self.chunks.insert_many(docs, ordered=False)

    def soft_delete_document(self, tenant_id: str, kb_id: str, doc_id: str) -> dict[str, Any]:
        """软删除逻辑文档。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            doc_id: 逻辑文档标识。

        Returns:
            删除后的逻辑文档记录。

        Raises:
            ValueError: 文档不存在时抛出。
        """
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        doc_key = f"{tenant_id}:{kb_id}:{doc_id}"
        now = utc_now_iso()
        document = self.documents.find_one({"_id": doc_key})
        if not document:
            raise ValueError("文档不存在")
        self.documents.update_one(
            {"_id": doc_key},
            {"$set": {"status": "deleted", "deleted_at": now, "updated_at": now}},
        )
        self.document_versions.update_many(
            {"tenant_id": tenant_id, "kb_id": kb_id, "doc_id": doc_id, "status": "active"},
            {"$set": {"status": "deleted", "deleted_at": now, "updated_at": now}},
        )
        return self.documents.find_one({"_id": doc_key}) or {}

    def rollback_document(self, tenant_id: str, kb_id: str, doc_id: str, target_version: int) -> dict[str, Any]:
        """回滚逻辑文档到指定历史版本。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            doc_id: 逻辑文档标识。
            target_version: 目标文档版本号。

        Returns:
            回滚后的逻辑文档记录。

        Raises:
            ValueError: 文档或目标版本不存在时抛出。
        """
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        doc_key = f"{tenant_id}:{kb_id}:{doc_id}"
        document = self.documents.find_one({"_id": doc_key})
        if not document:
            raise ValueError("文档不存在")
        target = self.document_versions.find_one(
            {
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "doc_id": doc_id,
                "doc_version": int(target_version),
                "status": {"$in": ["active", "archived"]},
                "physical_deleted_at": {"$exists": False},
            }
        )
        if not target:
            raise ValueError("目标版本不存在或不可回滚")
        now = utc_now_iso()
        current_version = document.get("active_version")
        if current_version is not None and current_version != target_version:
            self.document_versions.update_one(
                {"tenant_id": tenant_id, "kb_id": kb_id, "doc_id": doc_id, "doc_version": current_version},
                {"$set": {"status": "archived", "updated_at": now}},
            )
        self.document_versions.update_one(
            {"tenant_id": tenant_id, "kb_id": kb_id, "doc_id": doc_id, "doc_version": int(target_version)},
            {
                "$set": {
                    "status": "active",
                    "rollback_from_version": current_version,
                    "rolled_back_at": now,
                    "updated_at": now,
                }
            },
        )
        self.documents.update_one(
            {"_id": doc_key},
            {
                "$set": {
                    "active_version": int(target_version),
                    "item_name": target.get("item_name", document.get("item_name", "")),
                    "status": "active",
                    "deleted_at": None,
                    "updated_at": now,
                }
            },
        )
        return self.documents.find_one({"_id": doc_key}) or {}

    def _version_cleanup_target(
        self,
        version: dict[str, Any],
        *,
        now: datetime,
        archive_retention_days: int,
        deleted_retention_days: int,
        failed_retention_days: int,
    ) -> str | None:
        """判断文档版本是否到达物理清理窗口。

        Args:
            version: 文档版本记录。
            now: 当前时间。
            archive_retention_days: archived 版本保留天数。
            deleted_retention_days: deleted 版本保留天数。
            failed_retention_days: failed/pending 半成品保留天数。

        Returns:
            清理目标类型；未到期时返回 None。
        """
        status = str(version.get("status") or "")
        if status == "archived":
            base_time = parse_iso_datetime(version.get("updated_at") or version.get("activated_at"))
            return "archived" if base_time and base_time <= now - timedelta(days=archive_retention_days) else None
        if status == "deleted":
            base_time = parse_iso_datetime(version.get("deleted_at") or version.get("updated_at"))
            return "deleted" if base_time and base_time <= now - timedelta(days=deleted_retention_days) else None
        if status in {"failed", "pending", "ready_conflict"}:
            base_time = parse_iso_datetime(version.get("updated_at") or version.get("created_at"))
            return "failed" if base_time and base_time <= now - timedelta(days=failed_retention_days) else None
        return None

    def list_cleanup_candidates(
        self,
        tenant_id: str,
        kb_id: str,
        *,
        archive_retention_days: int | None = None,
        deleted_retention_days: int | None = None,
        failed_retention_days: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """查询到期的版本物理清理候选。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            archive_retention_days: archived 版本保留天数。
            deleted_retention_days: deleted 版本保留天数。
            failed_retention_days: failed/pending 半成品保留天数。
            limit: 最多返回候选数量。

        Returns:
            清理候选列表，包含 Milvus 过滤表达式。
        """
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        archive_days = (
            archive_retention_days
            if archive_retention_days is not None
            else get_retention_days("RAG_ARCHIVE_RETENTION_DAYS", DEFAULT_ARCHIVE_RETENTION_DAYS)
        )
        deleted_days = (
            deleted_retention_days
            if deleted_retention_days is not None
            else get_retention_days("RAG_DELETED_RETENTION_DAYS", DEFAULT_DELETED_RETENTION_DAYS)
        )
        failed_days = (
            failed_retention_days
            if failed_retention_days is not None
            else get_retention_days("RAG_FAILED_RETENTION_DAYS", DEFAULT_FAILED_RETENTION_DAYS)
        )
        max_candidates = max(int(limit), 0)
        if max_candidates == 0:
            return []
        now = datetime.now(timezone.utc)
        versions = self.document_versions.find(
            {
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "status": {"$in": ["archived", "deleted", "failed", "pending", "ready_conflict"]},
                "physical_deleted_at": {"$exists": False},
            }
        ).sort("updated_at", ASCENDING)
        candidates = []
        for version in versions:
            target = self._version_cleanup_target(
                version,
                now=now,
                archive_retention_days=archive_days,
                deleted_retention_days=deleted_days,
                failed_retention_days=failed_days,
            )
            if not target:
                continue
            doc_id = str(version.get("doc_id"))
            doc_version = int(version.get("doc_version"))
            candidates.append(
                {
                    "target": target,
                    "tenant_id": tenant_id,
                    "kb_id": kb_id,
                    "doc_id": doc_id,
                    "doc_version": doc_version,
                    "status": version.get("status"),
                    "updated_at": version.get("updated_at"),
                    "milvus_filter": build_version_milvus_filter(tenant_id, kb_id, doc_id, doc_version),
                }
            )
            if len(candidates) >= max_candidates:
                break
        return candidates

    def cleanup_expired_versions(
        self,
        tenant_id: str,
        kb_id: str,
        *,
        archive_retention_days: int | None = None,
        deleted_retention_days: int | None = None,
        failed_retention_days: int | None = None,
        limit: int = 100,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """清理过期文档版本的向量和 chunk 元数据。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            archive_retention_days: archived 版本保留天数。
            deleted_retention_days: deleted 版本保留天数。
            failed_retention_days: failed/pending 半成品保留天数。
            limit: 单次最多处理候选数量。
            dry_run: 为 True 时只返回候选，不执行删除。

        Returns:
            清理执行结果。

        Raises:
            Exception: Milvus 删除或 MongoDB 状态更新失败时抛出。
        """
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        candidates = self.list_cleanup_candidates(
            tenant_id,
            kb_id,
            archive_retention_days=archive_retention_days,
            deleted_retention_days=deleted_retention_days,
            failed_retention_days=failed_retention_days,
            limit=limit,
        )
        result: dict[str, Any] = {
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "dry_run": bool(dry_run),
            "candidate_count": len(candidates),
            "cleaned_count": 0,
            "failed_count": 0,
            "candidates": candidates,
            "errors": [],
        }
        if dry_run or not candidates:
            return result

        from app.clients.milvus_utils import get_milvus_client
        from app.conf.milvus_config import milvus_config
        from app.observability.metrics import record_document_cleanup

        milvus_client = get_milvus_client()
        now = utc_now_iso()
        for candidate in candidates:
            try:
                # 物理清理先删 Milvus，再标记 Mongo，避免 Mongo 显示已清理但向量仍可被兼容查询召回。
                milvus_client.delete(
                    collection_name=milvus_config.chunks_collection,
                    filter=candidate["milvus_filter"],
                )
                self.chunks.delete_many(
                    {
                        "tenant_id": tenant_id,
                        "kb_id": kb_id,
                        "doc_id": candidate["doc_id"],
                        "doc_version": candidate["doc_version"],
                    }
                )
                self.document_versions.update_one(
                    {
                        "tenant_id": tenant_id,
                        "kb_id": kb_id,
                        "doc_id": candidate["doc_id"],
                        "doc_version": candidate["doc_version"],
                    },
                    {"$set": {"physical_deleted_at": now, "updated_at": now}},
                )
                result["cleaned_count"] += 1
                record_document_cleanup("ok", candidate["target"])
            except Exception as exc:
                result["failed_count"] += 1
                result["errors"].append(
                    {
                        "doc_id": candidate["doc_id"],
                        "doc_version": candidate["doc_version"],
                        "error": str(exc),
                    }
                )
                record_document_cleanup("error", candidate["target"])
        return result


def get_document_version_store() -> DocumentVersionStore:
    """获取文档版本治理 Mongo Store 单例。

    Returns:
        DocumentVersionStore 单例。

    Raises:
        Exception: MongoDB 配置或连接不可用时抛出。
    """
    global _document_version_store, _document_version_store_error
    if _document_version_store is not None:
        return _document_version_store
    if _document_version_store_error is not None:
        raise RuntimeError(_document_version_store_error)
    try:
        _document_version_store = DocumentVersionStore()
        return _document_version_store
    except Exception as exc:
        _document_version_store_error = str(exc)
        raise
