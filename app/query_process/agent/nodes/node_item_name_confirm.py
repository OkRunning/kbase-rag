import json
import sys

from langchain_core.messages import HumanMessage

from app.clients.langfuse import langfuse, observe
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.core.tenant_scope import build_tenant_kb_filter
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.observability.sanitizers import sanitize_for_log
from app.query_process.agent.nodes.query_understanding_utils import (
    normalize_string_list,
)
from app.query_process.agent.state import QueryGraphState
from app.utils.alert_utils import CircuitOpenError, llm_failure_monitor
from app.utils.task_utils import add_running_task, add_done_task

# @observe(name="step_1_get_item_name_and_rewrite_query")
def step_1_get_item_name_and_rewrite_query(history, original_query):
    """提取商品名、改写问题并读取可选查询类型。

    Args:
        history: 历史对话记录。
        original_query: 用户原始问题。

    Returns:
        包含 item_names、rewritten_query 和 query_type 的结构化结果。

    Raises:
        Exception: LLM 调用失败或响应 JSON 解析失败时抛出。
    """
    # raise Exception("来个异常验证下")
    # 1. 加载提示词
    # 1.1 拼接history
    history_text = ""
    if history:
        history_text = "\n".join([f"{m['role']}: {m['text']}" for m in history])

    prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=original_query)

    messages = [
        HumanMessage(content=prompt)
    ]

    try:
        # 2. 获取llm_client
        llm = get_llm_client(json_mode=True)
        response = llm.invoke(messages)
    except Exception as e:
        if not isinstance(e, CircuitOpenError):  # 只记录真实失败
            try:
                trace_url = langfuse.get_trace_url() if 'langfuse' in globals() else None
            except:
                trace_url = None
            llm_failure_monitor.record_failure(
                title="节点node_item_name_confirm-LLM调用失败",
                content=f"{type(e).__name__}: {e}",
                trace_url=trace_url
            )
        raise  # 确保异常继续向上抛

    dict_content = {}
    if response:
        resp_content = response.content
        logger.info(f"【step_1_get_item_name_and_rewrite_query】节点处理，获取结果：{resp_content}")

        # json -> ```json   json  ```
        if resp_content.startswith("```json"):
            resp_content = resp_content.replace("```json", "").replace("```", "")
        dict_content = json.loads(resp_content)
        logger.info(f"========load_result_type :{type(dict_content)}")

    item_names = dict_content.get("item_names", [])
    rewritten_query = dict_content.get("rewritten_query", original_query)
    query_type = dict_content.get("query_type", "")

    return {
        "item_names": normalize_string_list(item_names),
        "rewritten_query": rewritten_query,
        "query_type": query_type,
    }


# @observe(name="step_2_embedding_search_item_names")
def step_2_embedding_search_item_names(model_item_names, tenant_id="default", kb_id="default"):
    """在指定租户和知识库中匹配商品名称。

    Args:
        model_item_names: 模型提取的商品名称。
        tenant_id: 租户标识。
        kb_id: 知识库标识。

    Returns:
        商品名称匹配结果列表。
    """
    final_result = []
    # 1. 获取milvus客户端
    milvus_client = get_milvus_client()

    # 2. 创建多个 AnnSearchRequest 实例
    # 2.1 将model_item_names先向量化
    embeddings = generate_embeddings(model_item_names)
    for index, item_name in enumerate(model_item_names):
        # 获取当前item_name的稠密和稀疏向量
        dense_vector = embeddings['dense'][index]
        sparse_vector = embeddings['sparse'][index]
        # 2.2 创建 AnnSearchRequest 实例
        reqs = create_hybrid_search_requests(
            dense_vector,
            sparse_vector,
            expr=build_tenant_kb_filter(tenant_id, kb_id),
        )

        # 3. 配置 Rerankers 策略 ranker_weights=(0.5, 0.5)
        # 4. 执行混合搜索
        response = hybrid_search(client=milvus_client,
                                 collection_name=milvus_config.item_name_collection,
                                 reqs=reqs,
                                 ranker_weights=(0.8, 0.2),
                                 norm_score=True,
                                 output_fields=["item_name", "tenant_id", "kb_id", "doc_id"])
        """
          [
            [
              {id:xx , distance: 0.x,entity:{item_name:xxx} } ,
              {id:xx , distance: 0.x,entity:{item_name:xxx} } 
             ]
          ]
        """
        # 5. 处理结果
        matches = []  # 当前item对应的匹配结果
        if response and len(response) > 0:
            hits = response[0]
            for hit in hits:
                entity = hit.get("entity", {})
                hit_item_name = entity.get("item_name", "")
                score = hit.get("distance", 0.0)
                if hit_item_name:
                    matches.append({"item_name": hit_item_name, "score": score})

        final_result.append({"expected": item_name, "matches": matches})

    logger.info(f"查询向量数据库结果为：{sanitize_for_log(final_result)}")
    return final_result


@observe(name="step_3_classify_item_names")
def step_3_classify_item_names(embedding_result):
    """
    对向量匹配结果进行分类
    根据分数进行判断：
     分数>0.85表示可以确定
     0.85>分数>=0.6 表示可选
    :param embedding_result: [
                                {"expected":"model_item_name","matches":[{"item_name":"xx","score":0.x},{"item_name":"xx","score":0.x}]},
                                {"expected":"model_item_name","matches":[{"item_name":"xx","score":0.x},{"item_name":"xx","score":0.x}]},
                            ]
    :return:{
                "confirmed_item_names":[],
                "options_item_names":[]
            }
    """
    confirmed_item_names = []
    options_item_names = []

    for item in embedding_result:
        high_score_match = []
        middle_score_match = []
        model_item_name = item['expected']
        matches = item.get("matches", [])
        if len(matches) > 0:
            matches.sort(key=lambda x: x['score'], reverse=True)
            high_score_match = [it['item_name'] for it in matches if it['score'] > 0.85]
            middle_score_match = [it['item_name'] for it in matches if it['score'] >= 0.6]

        if len(high_score_match) > 0:
            the_same_item_name = ""
            for high_name in high_score_match:
                if high_name == model_item_name:
                    the_same_item_name = high_name
                    break
            if the_same_item_name:
                confirmed_item_names.append(the_same_item_name)
            else:
                confirmed_item_names.append(high_score_match[0])
            continue
        if len(middle_score_match) > 0:
            options_item_names.extend(middle_score_match[:2])
            continue

    final_result = {
        "confirmed_item_names": list(set(confirmed_item_names)),
        "options_item_names": list(set(options_item_names)),
    }
    logger.info(f"【step_3_classify_item_names】节点处理，最终结果为：{sanitize_for_log(final_result)}")
    return final_result


@observe(name="step_4_deal_with_item_names")
def step_4_deal_with_item_names(state: QueryGraphState, final_item_names):
    """
        对final_item_names 最终结果处理
    :param state:
    :param final_item_names:  {
                "confirmed_item_names":[],
                "options_item_names":[]
            }
    :return:
    """
    confirmed_item_names = final_item_names.get("confirmed_item_names", [])
    options_item_names = final_item_names.get("options_item_names", [])
    if confirmed_item_names:
        state["item_names"] = confirmed_item_names
        logger.info(f"提取到商品名称为：{confirmed_item_names}")
    elif options_item_names:
        option_names = "、".join(options_item_names)
        state["answer"] = f"请选择您想咨询的商品名称：{option_names}"
        logger.info(f"未确定商品名称，可选的商品名称为：{option_names}")
    else:
        state["answer"] = "没有找到对应的商品名称，请重新输入"
        logger.info(f"没有找到对应的商品名称")
    return state


def node_item_name_confirm(state: QueryGraphState):
    """确认商品名称并在当前 scope 内保存用户问题。

    Args:
        state: 包含查询文本、会话及租户范围的查询 State。

    Returns:
        补充历史、商品名称和改写问题后的查询 State。

    Raises:
        Exception: 大模型、Milvus 或 MongoDB 调用失败时抛出。
    """

    func_name = sys._getframe().f_code.co_name
    logger.info(f"【{func_name}】节点开始执行，全局状态为：{sanitize_for_log(state)}")

    session_id = state["session_id"]
    is_stream = state["is_stream"]
    original_query = state["original_query"]
    tenant_id = state.get("tenant_id", "default")
    kb_id = state.get("kb_id", "default")
    user_id = state.get("user_id", "anonymous")

    # 记录任务开始
    add_running_task(session_id, func_name, is_stream)

    # 1. 获取历史条件记录（作为依据）
    history = get_recent_messages(session_id, tenant_id=tenant_id, kb_id=kb_id)
    state['history'] = history

    # 2. 调用模型提取item_name，重写问题
    model_result = step_1_get_item_name_and_rewrite_query(history, original_query)
    model_item_names = model_result.get("item_names", [])
    rewritten_query = model_result.get("rewritten_query", original_query)
    state['rewritten_query'] = rewritten_query
    final_item_names = {}
    if model_item_names:
        # 3. 进行item_name的向量数据库查询
        embedding_result = step_2_embedding_search_item_names(model_item_names, tenant_id, kb_id)
        # 4. 对item_name结果进行打分分类处理 A 【确认集合】  B【可选集合】
        final_item_names = step_3_classify_item_names(embedding_result)

    # 5. 处理确认和可选集合！ 有确认 =》 继续下个节点执行  || 有可选 or 没有item_names -> answer赋值结果
    state = step_4_deal_with_item_names(state, final_item_names)

    state["query_type"] = model_result.get("query_type", "") or state.get("query_type", "unknown")

    # 6.保存当前次的聊天记录
    save_chat_message(
        session_id=session_id,
        role="user",
        text=original_query,
        tenant_id=tenant_id,
        kb_id=kb_id,
        user_id=user_id,
    )
    add_done_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点结束执行，全局状态为：{sanitize_for_log(state)}")
    return state


if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "华为B3-211H显示器好用么？？",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False, default=str))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        logger.exception("==========")
