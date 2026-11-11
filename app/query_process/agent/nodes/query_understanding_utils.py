"""查询理解阶段通用工具。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from app.clients.langfuse import langfuse
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from app.conf.milvus_config import milvus_config
from app.core.tenant_scope import build_tenant_kb_filter
from app.query_process.agent.state import QueryGraphState
from app.utils.alert_utils import CircuitOpenError, llm_failure_monitor

_QUERY_TYPE_ORDER = (
    "fact",
    "procedure",
    "troubleshooting",
    "safety",
    "comparison",
    "summary",
    "follow_up",
    "realtime",
    "chitchat",
    "unknown",
)
_QUERY_TYPE_KEYWORDS = {
    "realtime": (
        "最新",
        "现在",
        "今天",
        "实时",
        "最近",
        "当下",
        "当前",
        "新闻",
        "行情",
        "价格",
        "天气",
    ),
    "chitchat": (
        "你好",
        "您好",
        "在吗",
        "谢谢",
        "辛苦了",
        "早上好",
        "晚上好",
        "hello",
        "hi",
    ),
    "comparison": (
        "对比",
        "比较",
        "区别",
        "差异",
        "哪个好",
        "哪个更",
        "vs",
    ),
    "troubleshooting": (
        "故障",
        "报错",
        "异常",
        "无法",
        "不能",
        "打不开",
        "开不了",
        "不工作",
        "坏了",
        "怎么修",
        "维修",
    ),
    "safety": (
        "安全",
        "注意事项",
        "危险",
        "禁止",
        "禁忌",
        "防护",
        "小心",
    ),
    "procedure": (
        "怎么",
        "如何",
        "步骤",
        "操作",
        "使用",
        "设置",
        "安装",
        "流程",
        "教程",
        "指南",
    ),
    "summary": (
        "总结",
        "概括",
        "摘要",
        "梳理",
        "概述",
        "汇总",
        "讲一下",
    ),
    "fact": (
        "是什么",
        "是什么?",
        "是什么？",
        "参数",
        "规格",
        "型号",
        "定义",
        "说明",
        "什么意思",
    ),
}
_QUERY_TYPE_PRIORITY = (
    "realtime",
    "comparison",
    "troubleshooting",
    "safety",
    "procedure",
    "summary",
    "fact",
)
_CHITCHAT_TOKENS = (
    "早上好",
    "晚上好",
    "hello",
    "辛苦了",
    "你好",
    "您好",
    "在吗",
    "谢谢",
    "hi",
)
_CHITCHAT_SEPARATOR_PATTERN = re.compile(r"[\s,，。！!？?~～、；;：:]+")


def build_history_text(history: list) -> str:
    """将历史消息拼接为提示词可用文本。

    Args:
        history: 历史消息列表。

    Returns:
        拼接后的历史文本。
    """
    if not history:
        return ""
    return "\n".join([f"{m.get('role', '')}: {m.get('text', '')}" for m in history if m])


def parse_llm_json_content(content: str) -> Dict[str, Any]:
    """解析大模型返回的 JSON 内容。

    Args:
        content: 大模型返回文本。

    Returns:
        解析后的字典。

    Raises:
        ValueError: JSON 解析失败。
    """
    raw_content = (content or "").strip()
    if raw_content.startswith("```json"):
        raw_content = raw_content.replace("```json", "", 1).replace("```", "").strip()
    elif raw_content.startswith("```"):
        raw_content = raw_content.replace("```", "").strip()
    return json.loads(raw_content or "{}")


def normalize_string_list(values: Any) -> List[str]:
    """规范化字符串列表。

    Args:
        values: 原始值。

    Returns:
        去重后的字符串列表。
    """
    if not isinstance(values, list):
        return []
    seen = set()
    normalized: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def extract_entity_payload(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """提取并规范化实体结构。

    Args:
        data: 大模型返回的数据。

    Returns:
        规范化后的实体字段。
    """
    entities = data.get("entities") or {}
    return {
        "item_names": normalize_string_list(entities.get("item_names")),
        "model_numbers": normalize_string_list(entities.get("model_numbers")),
        "parts": normalize_string_list(entities.get("parts")),
        "error_codes": normalize_string_list(entities.get("error_codes")),
    }


def extract_item_names_from_state(state: QueryGraphState) -> List[str]:
    """从 State 中提取商品名。

    Args:
        state: 查询状态。

    Returns:
        商品名列表。
    """
    item_names = state.get("item_names") or []
    return normalize_string_list(item_names)


def step_1_query_understanding(history: list, original_query: str) -> Dict[str, Any]:
    """执行查询理解。

    Args:
        history: 历史消息。
        original_query: 用户原始问题。

    Returns:
        结构化理解结果。
    """
    history_text = build_history_text(history)
    prompt = load_prompt("query_understanding", history_text=history_text, query=original_query)
    messages = [HumanMessage(content=prompt)]

    try:
        llm = get_llm_client(json_mode=True)
        response = llm.invoke(messages)
    except Exception as exc:
        if not isinstance(exc, CircuitOpenError):
            try:
                trace_url = langfuse.get_trace_url() if 'langfuse' in globals() else None
            except Exception:
                trace_url = None
            llm_failure_monitor.record_failure(
                title="节点node_query_understanding-LLM调用失败",
                content=f"{type(exc).__name__}: {exc}",
                trace_url=trace_url,
            )
        raise

    if not response:
        return {}

    resp_content = response.content
    logger.info(f"【step_1_query_understanding】节点处理，获取结果：{resp_content}")
    return parse_llm_json_content(resp_content)


def step_2_embedding_search_item_names(
    model_item_names: List[str],
    tenant_id: str = "default",
    kb_id: str = "default",
) -> List[Dict[str, Any]]:
    """对商品名执行向量检索。

    Args:
        model_item_names: 模型抽取的商品名列表。
        tenant_id: 租户标识。
        kb_id: 知识库标识。

    Returns:
        每个商品名的匹配结果列表。
    """
    final_result: List[Dict[str, Any]] = []
    milvus_client = get_milvus_client()
    embeddings = generate_embeddings(model_item_names)
    for index, item_name in enumerate(model_item_names):
        dense_vector = embeddings["dense"][index]
        sparse_vector = embeddings["sparse"][index]
        reqs = create_hybrid_search_requests(
            dense_vector,
            sparse_vector,
            expr=build_tenant_kb_filter(tenant_id, kb_id),
        )
        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs=reqs,
            ranker_weights=(0.8, 0.2),
            norm_score=True,
            output_fields=["item_name", "tenant_id", "kb_id", "doc_id"],
        )
        matches: List[Dict[str, Any]] = []
        if response and len(response) > 0:
            hits = response[0]
            for hit in hits:
                entity = hit.get("entity", {})
                hit_item_name = entity.get("item_name", "")
                score = hit.get("distance", 0.0)
                if hit_item_name:
                    matches.append({"item_name": hit_item_name, "score": score})
        final_result.append({"expected": item_name, "matches": matches})

    logger.info(f"查询向量数据库结果为：{final_result}")
    return final_result


def step_3_classify_item_names(embedding_result: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """对商品名检索结果进行分类。

    Args:
        embedding_result: 商品名检索结果。

    Returns:
        包含 confirmed_item_names 和 options_item_names 的结果。
    """
    confirmed_item_names: List[str] = []
    options_item_names: List[str] = []

    for item in embedding_result:
        high_score_match = []
        middle_score_match = []
        model_item_name = item["expected"]
        matches = item.get("matches", [])
        if matches:
            matches.sort(key=lambda x: x["score"], reverse=True)
            high_score_match = [it["item_name"] for it in matches if it["score"] > 0.85]
            middle_score_match = [it["item_name"] for it in matches if it["score"] >= 0.6]

        if high_score_match:
            the_same_item_name = ""
            for high_name in high_score_match:
                if high_name == model_item_name:
                    the_same_item_name = high_name
                    break
            confirmed_item_names.append(the_same_item_name or high_score_match[0])
            continue
        if middle_score_match:
            options_item_names.extend(middle_score_match[:2])
            continue

    final_result = {
        "confirmed_item_names": list(dict.fromkeys(confirmed_item_names)),
        "options_item_names": list(dict.fromkeys(options_item_names)),
    }
    logger.info(f"【step_3_classify_item_names】节点处理，最终结果为：{final_result}")
    return final_result


def build_clarification_question(item_names: List[str], clarification_hint: str = "") -> str:
    """构建澄清问题。

    Args:
        item_names: 可选商品名列表。
        clarification_hint: 额外提示。

    Returns:
        澄清用户的提示语。
    """
    if item_names:
        option_names = "、".join(item_names)
        if clarification_hint:
            return f"请选择您想咨询的商品名称：{option_names}。{clarification_hint}"
        return f"请选择您想咨询的商品名称：{option_names}"
    if clarification_hint:
        return clarification_hint
    return "请补充更具体的商品名称或型号。"


def normalize_query_type(query_type: Any) -> str:
    """规范化查询类型。

    Args:
        query_type: 原始查询类型值。

    Returns:
        规范化后的 query_type；不在枚举内时返回 ``unknown``。
    """
    value = str(query_type or "").strip().lower()
    if value in _QUERY_TYPE_ORDER:
        return value
    return "unknown"


def is_chitchat_query(text: str) -> bool:
    """判断查询是否为纯闲聊。

    Args:
        text: 用户问题或改写后的问题文本。

    Returns:
        仅包含寒暄表达时返回 True。
    """
    normalized_text = (text or "").strip().lower()
    if not normalized_text:
        return False
    compact_text = _CHITCHAT_SEPARATOR_PATTERN.sub("", normalized_text)
    if not compact_text:
        return False

    # 纯寒暄可以由多个寒暄短语拼接而成，例如“你好，在吗”。
    while compact_text:
        matched_token = ""
        for token in _CHITCHAT_TOKENS:
            if compact_text.startswith(token):
                matched_token = token
                break
        if not matched_token:
            return False
        compact_text = compact_text[len(matched_token):]
    return True


def infer_query_type(
    original_query: str,
    rewritten_query: str = "",
    explicit_query_type: str = "",
) -> str:
    """根据显式结果和规则推断查询类型。

    Args:
        original_query: 用户原始问题。
        rewritten_query: 改写后的问题。
        explicit_query_type: LLM 显式返回的 query_type。

    Returns:
        推断出的查询类型。
    """
    normalized_explicit = normalize_query_type(explicit_query_type)
    if normalized_explicit != "unknown":
        return normalized_explicit

    text = f"{original_query or ''} {rewritten_query or ''}".strip().lower()
    if not text:
        return "unknown"

    # 只有纯寒暄才判为 chitchat，避免“你好，烫金机如何使用”这类业务问题被闲聊词截断。
    if is_chitchat_query(text):
        return "chitchat"

    for query_type in _QUERY_TYPE_PRIORITY:
        for keyword in _QUERY_TYPE_KEYWORDS.get(query_type, ()):
            if keyword and keyword.lower() in text:
                return query_type

    if any(token in text for token in ("怎么", "如何", "步骤", "使用", "设置", "操作")):
        return "procedure"
    if any(token in text for token in ("报错", "异常", "无法", "不能", "故障")):
        return "troubleshooting"
    if any(token in text for token in ("最新", "实时", "今天", "现在", "当前")):
        return "realtime"
    return "unknown"


def build_retrieval_plan(
    query_type: str,
    item_names: List[str] | None = None,
    tenant_id: str = "default",
    kb_id: str = "default",
    web_search_default: bool = False,
) -> Dict[str, Any]:
    """构建第一版检索计划。

    Args:
        query_type: 规范化后的查询类型。
        item_names: 归一化后的商品名称列表。
        tenant_id: 租户标识。
        kb_id: 知识库标识。
        web_search_default: 是否允许默认启用 Web Search。

    Returns:
        第一版检索计划字典。
    """
    normalized_query_type = normalize_query_type(query_type)
    normalized_item_names = normalize_string_list(item_names)
    use_local_vector = normalized_query_type != "chitchat"
    use_hyde = False
    use_web = False
    use_kg = False
    dense_weight = 0.85
    sparse_weight = 0.15
    local_top_k = 8
    hyde_top_k = 5
    web_top_k = 3
    rrf_top_k = 8
    rerank_top_k = 5
    reason = "默认走本地知识库召回"

    if normalized_query_type in {"fact", "safety", "procedure", "comparison", "summary"}:
        reason = f"{normalized_query_type} 问题优先本地知识库，不默认启用外部召回"
    elif normalized_query_type == "troubleshooting":
        use_hyde = True
        dense_weight = 0.8
        sparse_weight = 0.2
        reason = "故障排查问题通常表述更模糊，优先本地召回并补充 HyDE"
    elif normalized_query_type == "realtime":
        use_web = True
        use_local_vector = True
        dense_weight = 0.7
        sparse_weight = 0.3
        reason = "实时问题需要联网补充，保留本地召回作为上下文底座"
    elif normalized_query_type == "chitchat":
        use_local_vector = False
        reason = "闲聊问题不进入知识库检索"
    else:
        use_hyde = not normalized_item_names
        reason = "未知问题默认先走本地召回，缺少明确实体时补充 HyDE"

    if web_search_default and normalized_query_type != "chitchat":
        use_web = True
        reason = f"{reason}；管理端允许 Web 作为默认补充"

    return {
        "use_local_vector": use_local_vector,
        "use_hyde": use_hyde,
        "use_web": use_web,
        "use_kg": use_kg,
        "local_top_k": local_top_k,
        "hyde_top_k": hyde_top_k,
        "web_top_k": web_top_k,
        "rrf_top_k": rrf_top_k,
        "rerank_top_k": rerank_top_k,
        "dense_weight": dense_weight,
        "sparse_weight": sparse_weight,
        "filters": {
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            "item_name": normalized_item_names,
            "doc_status": "active",
            "vector_version": "active",
            "language": "",
        },
        "source_weights": {
            "local_vector": 1.0,
            "hyde": 0.7,
            "web": 0.5,
            "kg": 0.8,
        },
        "reason": reason,
    }


def build_query_intelligence(
    original_query: str,
    rewritten_query: str = "",
    explicit_query_type: str = "",
    item_names: List[str] | None = None,
    tenant_id: str = "default",
    kb_id: str = "default",
    web_search_default: bool = False,
) -> Dict[str, Any]:
    """构建查询理解和检索计划的基础结果。

    Args:
        original_query: 用户原始问题。
        rewritten_query: 改写后的问题。
        explicit_query_type: LLM 显式返回的查询类型。
        item_names: 归一化后的商品名称列表。
        tenant_id: 租户标识。
        kb_id: 知识库标识。
        web_search_default: 是否允许默认启用 Web Search。

    Returns:
        包含 query_type、retrieval_plan 和 retrieval_diagnostics 的字典。
    """
    query_type = infer_query_type(original_query, rewritten_query, explicit_query_type)
    normalized_item_names = normalize_string_list(item_names)
    retrieval_plan = build_retrieval_plan(
        query_type=query_type,
        item_names=normalized_item_names,
        tenant_id=tenant_id,
        kb_id=kb_id,
        web_search_default=web_search_default,
    )
    retrieval_diagnostics = {
        "query_type_source": "llm" if normalize_query_type(explicit_query_type) != "unknown" else "rule",
        "explicit_query_type": normalize_query_type(explicit_query_type),
        "query_type_reason": retrieval_plan["reason"],
        "retrieval_plan_reason": retrieval_plan["reason"],
    }
    return {
        "query_type": query_type,
        "retrieval_plan": retrieval_plan,
        "retrieval_diagnostics": retrieval_diagnostics,
    }
