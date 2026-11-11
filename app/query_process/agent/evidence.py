"""查询证据文档标准结构与转换工具。"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, TypedDict


class SourceHit(TypedDict, total=False):
    """记录单个召回源对同一证据的命中信息。

    Attributes:
        source: 召回源名称。
        rank: 当前召回源内的排名。
        score: 当前召回源返回的原始分数。
    """

    source: str
    rank: int
    score: float


class EvidenceDocument(TypedDict, total=False):
    """标准化证据文档。

    Attributes:
        evidence_id: 证据唯一标识。
        source_type: 来源类型。
        source: 召回源名称。
        text: 可用于 Rerank 和 Prompt 的证据正文。
        chunk_id: 本地知识库 chunk ID。
        title: 标题。
        file_title: 文件标题。
        section: 章节标题。
        parent_title: 父级标题。
        item_name: 商品或实体名称。
        doc_id: 文档 ID。
        url: Web 来源 URL。
        domain: Web 来源域名。
        score: 原始召回分数。
        rrf_score: RRF 融合分数。
        source_hits: 多源命中详情。
        metadata: 其他透传元数据。
    """

    evidence_id: str
    source_type: str
    source: str
    text: str
    chunk_id: str
    title: str
    file_title: str
    section: str
    parent_title: str
    item_name: str
    doc_id: str
    url: str
    domain: str
    score: float
    rrf_score: float
    source_hits: List[SourceHit]
    metadata: Dict[str, Any]


def _to_float(value: Any, default: float = 0.0) -> float:
    """安全转换浮点分数。

    Args:
        value: 原始分数值。
        default: 转换失败时的默认值。

    Returns:
        浮点分数。
    """
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _hash_text(text: str) -> str:
    """生成文本哈希标识。

    Args:
        text: 原始文本。

    Returns:
        文本 SHA256 短标识。
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def build_evidence_id(doc: EvidenceDocument) -> str:
    """生成证据去重 ID。

    Args:
        doc: 标准化证据文档。

    Returns:
        优先基于 chunk_id 或 URL 的证据 ID。
    """
    source_type = doc.get("source_type") or doc.get("source") or "unknown"
    chunk_id = str(doc.get("chunk_id") or "").strip()
    if chunk_id:
        return f"chunk:{chunk_id}"
    url = str(doc.get("url") or "").strip()
    if url:
        return f"web:{url}"
    return f"{source_type}:text:{_hash_text(str(doc.get('text') or ''))}"


def local_hit_to_evidence(hit: Dict[str, Any], source: str = "local_vector", rank: int = 0) -> EvidenceDocument | None:
    """将本地向量召回结果转换为标准证据。

    Args:
        hit: Milvus 标准化或原始 hit。
            eg:
                {   id ,
                    milvus_id,
                    chunk_id,
                    distance，
                    entity:
                       {
                          "chunk_id", "content","file_title", "title", "parent_title", "item_name"
                       }
                }
        source: 召回源名称。
        rank: 召回源内排名。

    Returns:
        标准化证据；缺少正文时返回 None。
    """
    entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else {}
    text = str(hit.get("content") or entity.get("content") or "").strip()
    if not text:
        return None
    score = _to_float(hit.get("distance", hit.get("score", 0.0)))
    doc: EvidenceDocument = {
        "source_type": "local",
        "source": source,
        "text": text,
        "chunk_id": str(hit.get("chunk_id") or entity.get("chunk_id") or hit.get("id") or ""),
        "title": str(hit.get("title") or entity.get("title") or ""),
        "file_title": str(hit.get("file_title") or entity.get("file_title") or ""),
        "section": str(hit.get("section") or entity.get("section") or entity.get("title") or ""),
        "parent_title": str(hit.get("parent_title") or entity.get("parent_title") or ""),
        "item_name": str(hit.get("item_name") or entity.get("item_name") or ""),
        "doc_id": str(hit.get("doc_id") or entity.get("doc_id") or ""),
        "url": "",
        "domain": "",
        "score": score,
        "rrf_score": 0.0,
        "source_hits": [{"source": source, "rank": rank, "score": score}],
        "metadata": {"raw_id": hit.get("id")},
    }
    doc["evidence_id"] = build_evidence_id(doc)
    return doc


def hyde_hit_to_evidence(hit: Dict[str, Any], rank: int = 0) -> EvidenceDocument | None:
    """将 HyDE 召回结果转换为标准证据。

    Args:
        hit: HyDE 向量召回 hit。
        rank: HyDE 召回源内排名。

    Returns:
        标准化证据；缺少正文时返回 None。
    """
    return local_hit_to_evidence(hit, source="hyde", rank=rank)


def web_doc_to_evidence(doc: Dict[str, Any], rank: int = 0) -> EvidenceDocument | None:
    """将 Web Search 结果转换为标准证据。

    Args:
        doc: Web Search 标准化文档。
        rank: Web 召回源内排名。

    Returns:
        标准化证据；缺少正文时返回 None。
    """
    text = str(doc.get("snippet") or doc.get("text") or doc.get("content") or "").strip()
    if not text:
        return None
    score = _to_float(doc.get("score", doc.get("distance", 0.0)))
    evidence: EvidenceDocument = {
        "source_type": "web",
        "source": "web",
        "text": text,
        "chunk_id": "",
        "title": str(doc.get("title") or ""),
        "file_title": "",
        "section": "",
        "parent_title": "",
        "item_name": str(doc.get("item_name") or ""),
        "doc_id": "",
        "url": str(doc.get("url") or ""),
        "domain": str(doc.get("domain") or ""),
        "score": score,
        "rrf_score": 0.0,
        "source_hits": [{"source": "web", "rank": rank, "score": score}],
        "metadata": {"trust_level": doc.get("trust_level", "external")},
    }
    evidence["evidence_id"] = build_evidence_id(evidence)
    return evidence


def is_evidence_document(doc: Dict[str, Any]) -> bool:
    """判断字典是否为标准证据结构。

    Args:
        doc: 待判断文档。

    Returns:
        符合 EvidenceDocument 基础字段时返回 True。
    """
    return isinstance(doc, dict) and bool(doc.get("evidence_id")) and "source_hits" in doc and "text" in doc
