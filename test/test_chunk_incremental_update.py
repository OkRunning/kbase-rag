"""Chunk 级增量更新测试。"""

from app.clients.document_versioning_utils import compute_chunk_hash
from app.import_process.agent.nodes import node_bge_embedding as embedding_node
from app.import_process.agent.nodes import node_chunk_diff as chunk_diff_node
from app.import_process.agent.state import create_default_state


class FakeChunkStore:
    """提供 chunk diff 测试所需的上一版本快照。

    Args:
        previous_chunks: 上一 active 版本 chunk 元数据列表。
    """

    def __init__(self, previous_chunks):
        """初始化 fake store。

        Args:
            previous_chunks: 上一 active 版本 chunk 元数据列表。
        """
        self.previous_chunks = previous_chunks

    def list_version_chunks(self, tenant_id, kb_id, doc_id, doc_version):
        """返回上一 active 版本 chunk 元数据。

        Args:
            tenant_id: 租户标识。
            kb_id: 知识库标识。
            doc_id: 逻辑文档标识。
            doc_version: 文档版本号。

        Returns:
            上一 active 版本 chunk 元数据列表。
        """
        return self.previous_chunks


def test_node_chunk_diff_reuses_hash_matched_vectors(monkeypatch):
    """验证 hash 命中的 chunk 会复用上一版本向量。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        None.
    """
    unchanged = {
        "content": "相同内容",
        "title": "## 使用说明",
        "parent_title": "手册 > 使用说明",
        "item_name": "测试设备",
    }
    removed = {
        "content": "旧版本已删除内容",
        "title": "## 删除章节",
        "parent_title": "手册 > 删除章节",
        "item_name": "测试设备",
    }
    unchanged_hash = compute_chunk_hash(unchanged, item_name="测试设备")
    removed_hash = compute_chunk_hash(removed, item_name="测试设备")
    previous_chunks = [
        {"chunk_hash": unchanged_hash, "chunk_uid": "doc_v1_chunk_00000", "chunk_index": 0},
        {"chunk_hash": removed_hash, "chunk_uid": "doc_v1_chunk_00001", "chunk_index": 1},
    ]

    monkeypatch.setattr(chunk_diff_node, "is_document_versioning_enabled", lambda: True)
    monkeypatch.setattr(chunk_diff_node, "is_dry_run", lambda: False)
    monkeypatch.setattr(chunk_diff_node, "get_document_version_store", lambda: FakeChunkStore(previous_chunks))
    monkeypatch.setattr(
        chunk_diff_node,
        "_fetch_reused_vectors_from_milvus",
        lambda *args, **kwargs: {
            unchanged_hash: {
                "chunk_hash": unchanged_hash,
                "chunk_id": 101,
                "dense_vector": [0.1, 0.2],
                "sparse_vector": {1: 0.5},
            }
        },
    )

    state = create_default_state(
        task_id="task-1",
        tenant_id="tenant-a",
        kb_id="kb-a",
        doc_id="doc-a",
        doc_version=2,
        previous_doc_version=1,
        item_name="测试设备",
        chunks=[
            dict(unchanged),
            {
                "content": "新增内容",
                "title": "## 新增章节",
                "parent_title": "手册 > 新增章节",
                "item_name": "测试设备",
            },
        ],
    )

    result = chunk_diff_node.node_chunk_diff(state)

    assert len(result["reused_chunks"]) == 1
    assert len(result["changed_chunks"]) == 1
    assert len(result["removed_chunks"]) == 1
    assert result["chunks"][0]["diff_action"] == "reused"
    assert result["chunks"][0]["dense_vector"] == [0.1, 0.2]
    assert result["chunks"][1]["diff_action"] == "changed"


def test_node_bge_embedding_only_generates_missing_vectors(monkeypatch):
    """验证 embedding 节点跳过已复用向量的 chunk。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        None.
    """
    calls = []

    def fake_generate_embeddings(texts):
        """返回测试向量并记录调用输入。

        Args:
            texts: 待向量化文本列表。

        Returns:
            模拟 BGE-M3 输出。
        """
        calls.extend(texts)
        return {"dense": [[0.3, 0.4]], "sparse": [{2: 0.6}]}

    monkeypatch.setattr(embedding_node, "generate_embeddings", fake_generate_embeddings)
    state = create_default_state(
        task_id="task-2",
        chunks=[
            {
                "content": "相同内容",
                "item_name": "测试设备",
                "diff_action": "reused",
                "dense_vector": [0.1, 0.2],
                "sparse_vector": {1: 0.5},
            },
            {"content": "新增内容", "item_name": "测试设备", "diff_action": "changed"},
        ],
    )

    result = embedding_node.node_bge_embedding(state)

    assert calls == ["商品名：测试设备,介绍：新增内容"]
    assert result["chunks"][0]["dense_vector"] == [0.1, 0.2]
    assert result["chunks"][1]["dense_vector"] == [0.3, 0.4]
