"""文档版本 P3 运维治理增强测试。"""

from __future__ import annotations

import pytest

from app.clients.document_versioning_utils import DocumentVersionStore


class _UpdateResult:
    """模拟 pymongo 更新结果。

    Args:
        matched_count: 命中的文档数量。
    """

    def __init__(self, matched_count: int):
        """初始化更新结果。

        Args:
            matched_count: 命中的文档数量。
        """
        self.matched_count = matched_count


class _QueryResult(list):
    """提供 pymongo cursor 的 sort 子集能力。"""

    def sort(self, field: str, direction: int):
        """按字段排序。

        Args:
            field: 排序字段。
            direction: 排序方向，负数表示倒序。

        Returns:
            排序后的查询结果。
        """
        reverse = direction < 0
        return _QueryResult(
            sorted(self, key=lambda item: "" if item.get(field) is None else item.get(field), reverse=reverse)
        )


class _FakeCollection:
    """提供版本治理测试所需的 Mongo collection 子集能力。

    Args:
        docs: 初始文档列表。
    """

    def __init__(self, docs: list[dict] | None = None):
        """初始化内存集合。

        Args:
            docs: 初始文档列表。
        """
        self.docs = list(docs or [])

    def find_one(self, query: dict):
        """查询单条文档。

        Args:
            query: Mongo 风格查询条件。

        Returns:
            命中的文档；没有命中时返回 None。
        """
        for doc in self.docs:
            if self._matches(doc, query):
                return doc
        return None

    def find(self, query: dict):
        """查询多条文档。

        Args:
            query: Mongo 风格查询条件。

        Returns:
            可排序的文档列表。
        """
        return _QueryResult([doc for doc in self.docs if self._matches(doc, query)])

    def update_one(self, query: dict, update: dict, upsert: bool = False):
        """更新单条文档。

        Args:
            query: Mongo 风格查询条件。
            update: Mongo 风格更新内容。
            upsert: 是否在未命中时插入。

        Returns:
            模拟 pymongo 更新结果。
        """
        doc = self.find_one(query)
        if not doc:
            return _UpdateResult(0)
        doc.update(update.get("$set", {}))
        return _UpdateResult(1)

    def update_many(self, query: dict, update: dict):
        """更新多条文档。

        Args:
            query: Mongo 风格查询条件。
            update: Mongo 风格更新内容。

        Returns:
            模拟 pymongo 更新结果。
        """
        count = 0
        for doc in self.docs:
            if self._matches(doc, query):
                doc.update(update.get("$set", {}))
                count += 1
        return _UpdateResult(count)

    def delete_many(self, query: dict):
        """删除多条文档。

        Args:
            query: Mongo 风格查询条件。

        Returns:
            模拟 pymongo 更新结果。
        """
        before = len(self.docs)
        self.docs = [doc for doc in self.docs if not self._matches(doc, query)]
        return _UpdateResult(before - len(self.docs))

    def insert_many(self, docs: list[dict], ordered: bool = False):
        """插入多条文档。

        Args:
            docs: 待插入文档列表。
            ordered: 是否顺序插入。

        Returns:
            None.
        """
        self.docs.extend(docs)

    def _matches(self, doc: dict, query: dict) -> bool:
        """判断文档是否匹配查询条件。

        Args:
            doc: 待判断文档。
            query: Mongo 风格查询条件。

        Returns:
            匹配时返回 True。
        """
        for key, expected in query.items():
            actual = doc.get(key)
            if isinstance(expected, dict):
                if "$in" in expected and actual not in expected["$in"]:
                    return False
                if "$ne" in expected and actual == expected["$ne"]:
                    return False
                if "$exists" in expected and (key in doc) != bool(expected["$exists"]):
                    return False
                continue
            if actual != expected:
                return False
        return True


def _build_store() -> DocumentVersionStore:
    """创建不连接 MongoDB 的文档版本 store。

    Returns:
        注入内存集合的 DocumentVersionStore。
    """
    store = DocumentVersionStore.__new__(DocumentVersionStore)
    store.documents = _FakeCollection()
    store.document_versions = _FakeCollection()
    store.chunks = _FakeCollection()
    store.vector_versions = _FakeCollection()
    return store


def test_activate_version_marks_ready_conflict_when_active_pointer_changed():
    """验证并发导入冲突不会覆盖已切换的 active 指针。

    Returns:
        None.
    """
    store = _build_store()
    store.documents.docs = [{"_id": "tenant-a:kb-a:doc-a", "active_version": 2, "latest_version": 3}]
    store.document_versions.docs = [
        {"tenant_id": "tenant-a", "kb_id": "kb-a", "doc_id": "doc-a", "doc_version": 3, "status": "pending"}
    ]

    with pytest.raises(ValueError, match="active_version"):
        store.activate_version(
            tenant_id="tenant-a",
            kb_id="kb-a",
            doc_id="doc-a",
            doc_version=3,
            previous_doc_version=1,
            content_hash="sha256:new",
            item_name="设备",
            chunk_count=1,
            milvus_insert_count=1,
        )

    assert store.documents.docs[0]["active_version"] == 2
    assert store.document_versions.docs[0]["status"] == "ready_conflict"
    assert store.document_versions.docs[0]["conflict_active_version"] == 2


def test_diff_document_versions_reports_added_removed_and_moved_chunks():
    """验证版本 diff 按 chunk_hash 识别新增、删除和位置变化。

    Returns:
        None.
    """
    store = _build_store()
    store.document_versions.docs = [
        {"tenant_id": "tenant-a", "kb_id": "kb-a", "doc_id": "doc-a", "doc_version": 1, "status": "archived"},
        {"tenant_id": "tenant-a", "kb_id": "kb-a", "doc_id": "doc-a", "doc_version": 2, "status": "active"},
    ]
    store.chunks.docs = [
        {"tenant_id": "tenant-a", "kb_id": "kb-a", "doc_id": "doc-a", "doc_version": 1, "chunk_index": 0, "chunk_hash": "same", "title": "A"},
        {"tenant_id": "tenant-a", "kb_id": "kb-a", "doc_id": "doc-a", "doc_version": 1, "chunk_index": 1, "chunk_hash": "old", "title": "B"},
        {"tenant_id": "tenant-a", "kb_id": "kb-a", "doc_id": "doc-a", "doc_version": 2, "chunk_index": 1, "chunk_hash": "same", "title": "A"},
        {"tenant_id": "tenant-a", "kb_id": "kb-a", "doc_id": "doc-a", "doc_version": 2, "chunk_index": 2, "chunk_hash": "new", "title": "C"},
    ]

    diff = store.diff_document_versions("tenant-a", "kb-a", "doc-a", 1, 2)

    assert diff["summary"]["added"] == 1
    assert diff["summary"]["removed"] == 1
    assert diff["summary"]["unchanged"] == 1
    assert diff["summary"]["index_changed"] == 1


def test_audit_report_counts_status_rollbacks_and_conflicts():
    """验证审计报表汇总版本增长、回滚记录和冲突版本。

    Returns:
        None.
    """
    store = _build_store()
    store.documents.docs = [
        {"tenant_id": "tenant-a", "kb_id": "kb-a", "doc_id": "doc-a", "status": "active"},
        {"tenant_id": "tenant-a", "kb_id": "kb-a", "doc_id": "doc-b", "status": "deleted"},
    ]
    store.document_versions.docs = [
        {
            "tenant_id": "tenant-a",
            "kb_id": "kb-a",
            "doc_id": "doc-a",
            "doc_version": 1,
            "status": "active",
            "vector_version": "v1",
            "chunk_count": 3,
            "changed_chunk_count": 1,
            "reused_chunk_count": 2,
            "rolled_back_at": "2026-07-01T00:00:00+00:00",
            "rollback_from_version": 2,
        },
        {
            "tenant_id": "tenant-a",
            "kb_id": "kb-a",
            "doc_id": "doc-a",
            "doc_version": 3,
            "status": "ready_conflict",
            "vector_version": "v2",
            "conflict_active_version": 2,
        },
    ]

    report = store.build_audit_report("tenant-a", "kb-a")

    assert report["document_count"] == 2
    assert report["version_status_counts"] == {"active": 1, "ready_conflict": 1}
    assert report["chunk_totals"]["reused_chunk_count"] == 2
    assert report["rollback_count"] == 1
    assert report["conflict_count"] == 1


def test_cleanup_expired_versions_dry_run_returns_candidates():
    """验证清理 dry-run 只返回到期候选和 Milvus filter。

    Returns:
        None.
    """
    store = _build_store()
    store.document_versions.docs = [
        {
            "tenant_id": "tenant-a",
            "kb_id": "kb-a",
            "doc_id": "doc-a",
            "doc_version": 1,
            "status": "archived",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "tenant_id": "tenant-a",
            "kb_id": "kb-a",
            "doc_id": "doc-a",
            "doc_version": 2,
            "status": "active",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
    ]

    result = store.cleanup_expired_versions("tenant-a", "kb-a", archive_retention_days=0, dry_run=True)

    assert result["dry_run"] is True
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["target"] == "archived"
    assert 'doc_id == "doc-a"' in result["candidates"][0]["milvus_filter"]
