from typing import Annotated, Any, Dict, List, TypedDict
import copy


def merge_retrieval_diagnostics(left: Dict[str, Any] | None, right: Dict[str, Any] | None) -> Dict[str, Any]:
    """合并并行检索分支产生的诊断信息。

    Args:
        left: 当前 State 中已有的检索诊断信息。
        right: 当前分支新增的检索诊断信息。

    Returns:
        合并后的检索诊断信息。
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class QueryGraphState(TypedDict):
    """定义查询流程中流转的数据结构。

    Attributes:
        session_id: 会话唯一标识。
        tenant_id: 租户标识。
        kb_id: 知识库标识。
        user_id: 用户标识。
        original_query: 用户原始问题。
        embedding_chunks: 普通向量检索回来的切片。
        hyde_embedding_chunks: HyDE 检索回来的切片。
        web_search_docs: 网络搜索回来的文档。
        rrf_chunks: Evidence Fusion 融合排序后的标准证据。
        reranked_docs: 重排序后的最终 Top-K 文档。
        prompt: 组装好的 Prompt。
        answer: 最终生成的答案。
        item_names: 提取出的商品名称。
        rewritten_query: 改写后的问题。
        history: 历史对话记录。
        is_stream: 是否流式输出。
        eval_mode: 是否为离线评测模式。
        query_type: 检索问题类型。
        intent: 查询意图。
        need_item_name: 是否需要商品名称才能继续回答。
        clarification_hint: 追问提示。
        entities: 结构化实体抽取结果。
        entity_resolution: 商品实体消歧和确认结果。
        retrieval_plan: 检索计划。
        retrieval_diagnostics: 检索诊断信息。
        rerank_gate: Rerank 跳过门控结果。
        rerank_quality: Rerank 质量诊断。
        evidence_status: 证据状态。
        evidence_gate: 证据门控结果。
        sources: 前端可展示的结构化来源。
        image_urls: 从候选文档中提取的图片 URL 列表。
    """
    session_id: str  # 会话唯一标识
    tenant_id: str  # 租户标识
    kb_id: str  # 知识库标识
    user_id: str  # 用户标识
    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE 检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # Evidence Fusion 融合排序后的标准证据
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: List[str]  # 提取出的商品名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记
    eval_mode: bool  # 是否离线评测模式，评测时避免写入真实用户历史

    # RAG 标准化改造扩展字段，默认仅承载诊断和后续灰度数据，不改变旧链路路由。
    query_type: str  # 检索问题类型
    intent: str  # 查询意图
    need_item_name: bool  # 是否需要商品名称才能继续回答
    clarification_hint: str  # 追问提示
    entities: Dict[str, Any]  # 结构化实体抽取结果
    entity_resolution: Dict[str, Any]  # 商品实体消歧和确认结果
    retrieval_plan: Dict[str, Any]  # 检索计划
    retrieval_diagnostics: Annotated[Dict[str, Any], merge_retrieval_diagnostics]  # 检索诊断信息
    rerank_gate: Dict[str, Any]  # Rerank 跳过门控结果
    rerank_quality: Dict[str, Any]  # Rerank 质量诊断
    evidence_status: str  # 证据状态
    evidence_gate: Dict[str, Any]  # 证据门控结果
    sources: list  # 前端可展示的结构化来源
    image_urls: list  # 从候选文档中提取的图片 URL 列表


# ========================
# 默认状态（全部为空）
# ========================
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "tenant_id": "default",
    "kb_id": "default",
    "user_id": "anonymous",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
    "eval_mode": False,
    "query_type": "unknown",
    "intent": "",
    "need_item_name": False,
    "clarification_hint": "",
    "entities": {},
    "entity_resolution": {},
    "retrieval_plan": {},
    "retrieval_diagnostics": {},
    "rerank_gate": {},
    "rerank_quality": {},
    "evidence_status": "",
    "evidence_gate": {},
    "sources": [],
    "image_urls": []
}


# ========================
# 创建默认状态（可覆盖）
# ========================
def create_query_default_state(**overrides) -> QueryGraphState:
    """创建查询流程的默认状态。

    Args:
        **overrides: 需要覆盖的默认字段。

    Returns:
        查询流程默认状态副本。
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


# ========================
# 获取干净状态
# ========================
def get_query_default_state() -> QueryGraphState:
    """获取查询流程的干净默认状态。

    Returns:
        查询流程默认状态副本。
    """
    return copy.deepcopy(query_graph_default_state)


# ========================
# ✅ 状态复制函数（你要的）
# ========================
def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """复制现有查询状态并覆盖指定字段。

    Args:
        state: 原始查询流程状态。
        **overrides: 需要覆盖的新字段。

    Returns:
        深拷贝后的查询流程状态。
    """
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


if __name__ == "__main__":
    # 测试
    state = create_query_default_state(
        session_id="test_001",
        original_query="华为P60怎么样?",
        is_stream=False
    )
    print("初始化状态：", state)

    # 复制状态
    new_state = copy_query_state(
        state,
        original_query="修改后的问题"
    )
    print("复制后的状态：", new_state)
