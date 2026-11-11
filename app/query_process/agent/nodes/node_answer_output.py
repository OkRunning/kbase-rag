import json
import re
import sys

from app.clients.mongo_history_utils import save_chat_message
from app.conf.query_optimization_config import query_optimization_config
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.observability.sanitizers import sanitize_text
from app.query_process.agent.state import QueryGraphState
from app.utils.sse_utils import push_to_session, SSEEvent
from app.utils.task_utils import add_running_task, set_task_result, add_done_task
from app.core.logger import logger

_IMAGE_BLOCK_MARKER = "【图片】"
MAX_CONTEXT_CHARS = 12000  # 限制 prompt的长度
_ANSWERABILITY_ALLOW_LIST = {"answerable", "low_confidence", "insufficient"}
_SOURCE_PREVIEW_CHARS = 800


def _get_answerability(state: QueryGraphState) -> str:
    """读取当前回答置信度等级。

    Args:
        state: 查询 LangGraph State。

    Returns:
        标准化后的回答置信度等级。
    """
    rerank_quality = state.get("rerank_quality") or {}
    answerability = str(rerank_quality.get("answerability") or "answerable")
    if answerability not in _ANSWERABILITY_ALLOW_LIST:
        return "answerable"
    return answerability


def _build_answer_policy_instruction(answerability: str) -> str:
    """生成与证据置信度匹配的回答约束说明。

    Args:
        answerability: rerank 质量给出的回答置信度等级。

    Returns:
        可直接拼入 prompt 的策略说明文本。
    """
    base_instruction = (
        "回答要求：只能依据给定资料回答。"
        "没有资料支撑时必须明确说明资料不足。"
        "不要编造参数、步骤、型号、法规、网址。"
    )
    if query_optimization_config.enable_citations:
        base_instruction += "回答中的关键结论必须标注对应资料编号，例如 [1]、[2]。"
    if answerability == "low_confidence":
        if query_optimization_config.low_confidence_refusal_enabled:
            return base_instruction + "当前证据置信度较低，请优先给出资料中能直接确认的内容，并明确提示结论可能不完整，避免把推测写成结论。"
        return base_instruction + "当前证据置信度较低，请优先给出资料中能直接确认的内容，并明确提示结论可能不完整。"
    if answerability == "insufficient":
        return base_instruction + "当前证据明显不足时，不要输出确定性结论。"
    return base_instruction


def _build_insufficient_answer(state: QueryGraphState) -> str:
    """构造资料不足时的固定回复。

    Args:
        state: 查询 LangGraph State。

    Returns:
        资料不足场景下的拒答文本。
    """
    question = state.get("rewritten_query") or state.get("original_query") or "当前问题"
    return (
        f"资料不足，当前检索到的内容无法支撑对“{question}”给出确定性回答。"
        "请补充更具体的型号、故障现象、操作步骤、截图或说明书页码，我再继续帮你判断。"
    )


def _emit_answer(session_id: str, answer: str) -> None:
    """将答案写入任务结果。

    Args:
        session_id: 会话唯一标识。
        answer: 最终答案文本。

    Returns:
        None.
    """
    set_task_result(session_id, "answer", answer)


def _emit_sources(session_id: str, sources: list) -> None:
    """将结构化来源写入任务结果。

    Args:
        session_id: 会话唯一标识。
        sources: 前端可展示的结构化来源列表。

    Returns:
        None.
    """
    set_task_result(session_id, "sources", json.dumps(sources or [], ensure_ascii=False))


def _normalize_source_text(text: object) -> str:
    """规范化来源正文预览。

    Args:
        text: 原始正文。

    Returns:
        截断后的来源正文。
    """
    normalized = str(text or "").strip()
    if len(normalized) <= _SOURCE_PREVIEW_CHARS:
        return normalized
    return normalized[:_SOURCE_PREVIEW_CHARS] + "..."


def _get_doc_metadata(doc: dict) -> dict:
    """读取证据文档元数据。

    Args:
        doc: 重排序后的证据文档。

    Returns:
        元数据字典；旧格式或脏数据返回空字典。
    """
    metadata = doc.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def build_answer_sources(state: QueryGraphState) -> list:
    """构建答案可追溯来源列表。

    Args:
        state: 查询 LangGraph State。

    Returns:
        与 prompt 资料编号对应的 sources 列表。
    """
    sources = []
    docs = state.get("reranked_docs") or []
    answerability = _get_answerability(state)
    evidence_status = state.get("evidence_status", "")
    for index, doc in enumerate(docs, start=1):
        metadata = _get_doc_metadata(doc)
        source_type = doc.get("source_type") or doc.get("source") or ""
        source = {
            "index": index,
            "source_id": f"[{index}]",
            "source_type": source_type,
            "source": doc.get("source", source_type),
            "file_title": doc.get("file_title") or doc.get("title") or "",
            "title": doc.get("title", ""),
            "section": doc.get("section", ""),
            "page": doc.get("page") or doc.get("page_number") or metadata.get("page", ""),
            "score": doc.get("score", 0.0),
            "chunk_id": doc.get("chunk_id", ""),
            "evidence_id": doc.get("evidence_id", ""),
            "url": doc.get("url", ""),
            "domain": doc.get("domain", ""),
            "item_name": doc.get("item_name", ""),
            "text": _normalize_source_text(doc.get("text", "")),
        }
        if answerability == "insufficient":
            source["usage"] = "reference"
            source["note"] = "当前证据未通过充分性门控，仅作为参考证据。"
        if evidence_status:
            source["evidence_status"] = evidence_status
        sources.append(source)
    return sources


def step_1_check_answer(state):
    """检查 State 中是否已有答案。

    Args:
        state: 查询 LangGraph State。

    Returns:
        已有答案返回 True，否则返回 False。
    """
    answer = state.get("answer")
    is_stream = state.get("is_stream", False)
    if not answer:
        return False

    session_id = state["session_id"]
    sources = build_answer_sources(state)
    state["sources"] = sources
    _emit_sources(session_id, sources)
    if is_stream:
        # 流式： 推送到sse
        # session_id: str, event: str, data: Dict[str, Any]
        push_to_session(session_id, SSEEvent.DELTA, {"delta": answer})
        set_task_result(session_id, "answer", answer)
    else:
        # 非流式： 设置任务结果
        set_task_result(session_id, "answer", answer)
    return True


def step_2_get_prompt(state: QueryGraphState):
    """生成最终回答使用的 Prompt。

    Args:
        state: 查询 LangGraph State。

    Returns:
        组装后的 Prompt 文本。
    """
    # 数据从state中获取
    rewritten_query = state.get("rewritten_query") or state.get("original_query")  # question
    reranked_docs = state.get("reranked_docs", [])
    item_names = state.get("item_names", [])
    history = state.get("history", [])
    answerability = _get_answerability(state)
    answer_policy_instruction = _build_answer_policy_instruction(answerability)

    # ============开始处理reranked_docs============
    # [
    #          {
    #                text:内容 snippet content,
    #                chunk_id: chunk_id rrf有  mcp None,
    #                title: title ,
    #                url : rrfNone mcp url ,
    #                source: web -> mcp  || local -> rrf ,
    #                score: rerank打的分
    #          }
    # ]

    context_list = []
    used_length = 0  # 记录使用的长度
    # reranked_docs => [{text,chunk_id,score,url,title,source local | web },{}]
    # Prompt 使用编号资料块，答案中的 [1]、[2] 可以与 sources 返回值一一对应。
    for index, doc in enumerate(reranked_docs, start=1):
        metadata = _get_doc_metadata(doc)
        text = doc.get("text", "")
        append_content = (
            f"[{index}]\n"
            f"source_type: {doc.get('source_type') or doc.get('source') or ''}\n"
            f"file_title: {doc.get('file_title') or doc.get('title') or ''}\n"
            f"section: {doc.get('section', '')}\n"
            f"page: {doc.get('page') or doc.get('page_number') or metadata.get('page', '')}\n"
            f"score: {doc.get('score')}\n"
            f"text: {text}"
        )

        if used_length + len(append_content) > MAX_CONTEXT_CHARS:
            logger.info(f"本次内容停止追加了！已经大于限制长度！")
            break

        context_list.append(append_content)
        used_length += len(append_content)

    final_context = "\n\n".join(context_list)

    # ========处理history============
    history_str = ""
    if history and len(history) > 0:
        for i, message in enumerate(history, start=1):
            role = message.get("role")
            text = message.get("text")
            current_history = ""
            if role == "user" and text:
                current_history = f"【用户】: {text}\n"
            elif role == "assistant" and text:
                current_history = f"【助手】: {text}\n"
            history_str += current_history
            used_length += len(current_history)  # 使用长度
            if used_length > MAX_CONTEXT_CHARS:
                logger.info(f"本次内容停止追加了！已经大于限制长度！")
                break
    else:
        history_str = "没有历史对话记录！"

    #  ================处理 item_name================
    item_names_str = ",".join(item_names)

    prompt = load_prompt("answer_out",
                         context=final_context,
                         history=history_str,
                         item_names=item_names_str,
                         question=rewritten_query,
                         answer_policy_instruction=answer_policy_instruction
                         )
    logger.info(f"已经完成了提示词生成：{sanitize_text(prompt)}")
    return prompt


def step_3_create_answer(state, prompt):
    """调用大模型生成答案。

    Args:
        state: 查询 LangGraph State。
        prompt: 回答生成 Prompt。

    Returns:
        最终答案文本。

    说明:
        当 rerank 质量判定为 insufficient 时，直接返回资料不足的固定回复，
        不再调用大模型生成确定性答案。
    """
    session_id = state["session_id"]
    is_stream = state.get("is_stream", False)
    answerability = _get_answerability(state)
    sources = state.get("sources") or build_answer_sources(state)
    state["sources"] = sources
    _emit_sources(session_id, sources)

    if answerability == "insufficient":
        # 资料不足时不再调用大模型，避免在证据空洞的情况下生成确定性答案。
        final_answer = _build_insufficient_answer(state)
        if is_stream:
            # 只在拒答分支补发一条完整结果，保证流式前端仍能收到最终内容。
            push_to_session(session_id, SSEEvent.DELTA, {"delta": final_answer})
        _emit_answer(session_id, final_answer)
        state["answer"] = final_answer
        logger.info(f"资料不足，直接返回拒答结果：{sanitize_text(final_answer)}")
        return final_answer

    # 1. 获取大模型
    llm = get_llm_client()

    # 2. 获取流式状态【sse | set_result】
    final_answer = ""
    # 调用模型进行生成 sse . stream  ||  set_result . invoke
    if is_stream:
        for chunk in llm.stream(prompt):
            delta = chunk.content
            push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
            final_answer += delta
    else:
        response = llm.invoke(prompt)
        final_answer = response.content

    _emit_answer(session_id, final_answer)
    state["answer"] = final_answer
    logger.info(f"lm模型最终返回的结果：{sanitize_text(final_answer)}")
    return final_answer


def step_4_extract_images_url(state: QueryGraphState):
    """从候选文档中提取图片 URL。

    Args:
        state: 查询 LangGraph State。

    Returns:
        去重后的图片 URL 列表。
    """
    images = []  # -> 存储图片 (On) (想要先后顺序)
    set_images = set()  # -> 图片重复判断  （重复时间复杂度 O1）

    # 1. 定义正则
    image_reg = re.compile(r"!\[.*?\]\((.*?)\)")

    docs = state.get("reranked_docs") or []
    for doc in docs:
        url = doc.get("url")
        if (url and url.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))
                and url not in set_images):
            images.append(url)
            set_images.add(url)
        text = doc.get("text")
        if text:
            # 正在匹配的所有图片
            matches = image_reg.findall(text)
            for image_url in matches:
                if image_url not in set_images:
                    images.append(image_url)
                    set_images.add(image_url)
    logger.info(f"已经完成图片提取。数量:{len(images)},提取内容：{images}")
    return images


def step_5_write_history(state):
    """将助手答案写入历史对话。

    Args:
        state: 查询 LangGraph State。

    Returns:
        None.
    """
    if state.get("eval_mode"):
        # 离线评测会批量重放样本，跳过历史写入可以避免污染真实用户会话。
        logger.info("当前为离线评测模式，跳过 Mongo 历史写入。")
        return

    session_id = state.get("session_id")
    tenant_id = state.get("tenant_id", "default")
    kb_id = state.get("kb_id", "default")
    user_id = state.get("user_id", "anonymous")
    answer = state.get("answer")
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    item_names = state.get("item_names", [])

    # if rewritten_query:
    #     # user
    #     save_chat_message(
    #         session_id = session_id,
    #         role = "user",
    #         text = rewritten_query,
    #         item_names = item_names
    #     )
    if answer:
        # assistant
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            item_names=item_names,
            rewritten_query=rewritten_query,
            tenant_id=tenant_id,
            kb_id=kb_id,
            user_id=user_id,
        )
    logger.info(f"完成了本次对话的记录存储！")


def node_answer_output(state):
    """输出查询答案并记录会话历史。

    Args:
        state: 查询 LangGraph State。

    Returns:
        更新后的查询 LangGraph State。

    说明:
        根据 rerank_quality.answerability 决定正常生成、谨慎回答或直接拒答。
    """
    func_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state["is_stream"]
    add_running_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点开始执行")

    images_url = []
    sources = build_answer_sources(state)
    state["sources"] = sources
    _emit_sources(session_id, sources)
    has_answer = step_1_check_answer(state)
    if not has_answer:
        logger.info(f"【{func_name}】没有答案，开始生成答案")
        logger.info(f"【{func_name}】当前回答置信度：{_get_answerability(state)}")
        # 2. 加载提示词
        prompt = step_2_get_prompt(state)

        # 3. 调用大模型润色答案
        answer = step_3_create_answer(state, prompt)

        # 4. 提取原来topk list中的图片地址，单独返回【see】
        images_url = step_4_extract_images_url(state)

    # 图片列表写入 state，供流式传输层从最终 state 构造 final 事件。
    state["image_urls"] = images_url

    if is_stream:
        # 流式协议必须稳定发送 FINAL，前端依赖该事件判断本轮回答已结束。
        push_to_session(session_id, SSEEvent.FINAL, {
            "answer": state.get("answer", ""),
            "status": "completed",
            "image_urls": state.get("image_urls", images_url),
            "sources": state.get("sources", []),
        })

    # 6. 添加聊天对话到mongo
    step_5_write_history(state)
    add_done_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点执行完成")
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_answer_output 本地测试")
    print("=" * 50)

    # 1. 构造模拟数据
    # 模拟重排序后的文档列表 (reranked_docs)
    # 包含：本地文档（带Markdown图片）、联网结果（带URL字段）、纯文本文档
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://www.baidu.com/img/bd_logo.png)

            如果是进行局部烫金，请调节侧面的旋钮。
            ![侧面旋钮细节](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",  # 这是一个直接指向图片的URL（虽然少见，但用于测试提取）
            "text": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "安全注意事项",
            "score": 0.82,
            "text": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    # 模拟历史记录
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"}
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,  # 测试非流式
        # "is_stream": True, # 若要测试流式，需确保 SSE 环境或 mock 相关函数
        "answer": None  # 初始无答案
    }

    try:
        # 运行节点
        result = node_answer_output(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")

        # 1. 验证 Prompt 构建
        if "prompt" in result:
            print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
            # print(f"Prompt 预览:\n{result['prompt'][:200]}...")
        else:
            print("[FAIL] Prompt 未构建")

        # 2. 验证答案生成
        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
            print(f"答案预览: {answer[:50]}...")
        else:
            print(f"[WARN] 答案生成可能异常 (Content: {answer})")

        # 3. 验证图片提取
        # 我们期望提取到 3 张图片：
        # 1. http://local-server/images/panel_view.jpg (来自 local_101)
        # 2. http://local-server/images/knob_detail.png (来自 local_101)
        # 3. http://example.com/hak180_troubleshooting.jpeg (来自 web 结果的 url 字段)

        # 注意：这里我们没办法直接从 result state 里拿到 image_urls，因为它是作为 SSE 推送出去的，或者存库了
        # 但我们可以通过日志观察 _extract_images_from_docs 的输出
        # 如果需要验证，可以临时修改 node_answer_output 返回 image_urls
        print("\n[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
