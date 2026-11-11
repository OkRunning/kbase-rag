"""向量版本升级治理测试。"""

from app.clients import document_versioning_utils as versioning
from app.import_process.agent.nodes import node_document_version_prepare as prepare_node
from app.import_process.agent.state import create_default_state


class FakeVectorVersionStore:
    """提供向量版本治理测试所需的 store 行为。

    Args:
        queryable_versions: 查询侧允许使用的向量版本列表。
        active_vector_version: 导入侧当前 active 向量版本。
    """

    def __init__(self, queryable_versions=None, active_vector_version="bge_m3_v2_md_heading_v1"):
        """初始化 fake store。

        Args:
            queryable_versions: 查询侧允许使用的向量版本列表。
            active_vector_version: 导入侧当前 active 向量版本。
        """
        self.queryable_versions = queryable_versions or []
        self.active_vector_version = active_vector_version
        self.prepare_calls = []

    def list_queryable_vector_versions(self, tenant_id, kb_id):
        """返回 active 与 fallback 向量版本。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。

        Returns:
            可参与查询的向量版本列表。
        """
        return self.queryable_versions

    def prepare_version(self, **kwargs):
        """记录导入准备参数并返回 pending 版本状态。

        Args:
            **kwargs: 文档版本准备参数。

        Returns:
            导入 State 需要合并的版本治理字段。
        """
        self.prepare_calls.append(kwargs)
        return {
            "import_action": "create",
            "skip_reason": "",
            "doc_version": 1,
            "previous_doc_version": None,
            "file_hash": kwargs["file_hash"],
            "vector_version": self.active_vector_version,
        }


def test_build_active_vector_version_filter_includes_fallback_versions(monkeypatch):
    """验证查询 filter 限制在 active 与 fallback 向量版本内。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        None.
    """
    store = FakeVectorVersionStore(
        queryable_versions=["bge_m3_v2_md_heading_v1", "bge_m3_v1_md_heading_v1"]
    )
    monkeypatch.setattr(versioning, "is_document_versioning_enabled", lambda: True)
    monkeypatch.setattr(versioning, "get_document_version_store", lambda: store)

    expr = versioning.build_active_vector_version_filter("tenant-a", "kb-a")

    assert expr == 'vector_version in ["bge_m3_v2_md_heading_v1","bge_m3_v1_md_heading_v1"]'


def test_node_document_version_prepare_uses_active_vector_version(monkeypatch, tmp_path):
    """验证导入准备节点会采用当前 active 向量版本。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest 临时目录 fixture。

    Returns:
        None.
    """
    upload = tmp_path / "manual.md"
    upload.write_text("# 测试手册\n", encoding="utf-8")
    store = FakeVectorVersionStore(active_vector_version="bge_m3_v2_md_heading_v1")

    monkeypatch.setattr(prepare_node, "is_document_versioning_enabled", lambda: True)
    monkeypatch.setattr(prepare_node, "is_dry_run", lambda: False)
    monkeypatch.setattr(prepare_node, "get_document_version_store", lambda: store)

    state = create_default_state(
        task_id="task-vector",
        tenant_id="tenant-a",
        kb_id="kb-a",
        doc_id="doc-a",
        local_file_path=str(upload),
    )

    result = prepare_node.node_document_version_prepare(state)

    assert result["vector_version"] == "bge_m3_v2_md_heading_v1"
    assert store.prepare_calls[0]["vector_version"] == versioning.DEFAULT_VECTOR_VERSION
