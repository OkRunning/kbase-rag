"""查询流程 Web Search MCP 召回节点。"""

import asyncio
import json
import time
import sys
from typing import Any, Dict
from urllib.parse import urlparse

from agents.mcp import MCPServerStreamableHttp

from app.conf.bailian_mcp_config import mcp_config
from app.core.logger import logger
from app.observability.metrics import record_retrieval_docs
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task

WEB_CONNECT_TIMEOUT_SECONDS = 3
WEB_TOOL_CALL_TIMEOUT_SECONDS = 8
WEB_MAX_RETRY_ATTEMPTS = 1
WEB_MAX_RESULTS = 5


def _merge_web_diagnostics(state: QueryGraphState, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """合并 Web Search 检索诊断信息。

    Args:
        state: 查询流程状态。
        diagnostics: 当前 Web Search 节点诊断信息。

    Returns:
        合并后的检索诊断信息。
    """
    retrieval_diagnostics = dict(state.get("retrieval_diagnostics") or {})
    retrieval_diagnostics["web"] = diagnostics
    return retrieval_diagnostics


def _empty_web_result(state: QueryGraphState, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """构建空 Web Search 召回结果。

    Args:
        state: 查询流程状态。
        diagnostics: 当前 Web Search 节点诊断信息。

    Returns:
        包含空 Web 结果和诊断信息的状态更新。
    """
    return {
        "web_search_docs": [],
        "retrieval_diagnostics": _merge_web_diagnostics(state, diagnostics),
    }


def _extract_domain(url: str) -> str:
    """从 URL 中提取域名。

    Args:
        url: Web 搜索结果 URL。

    Returns:
        URL 域名；无法解析时返回空字符串。
    """
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed.netloc or parsed.path.split("/")[0]


def _normalize_web_documents(raw_documents: list) -> list:
    """标准化 Web Search 返回结果。

    Args:
        raw_documents: MCP 工具返回的原始页面列表。

    Returns:
        补齐来源字段后的 Web 文档列表。
    """
    normalized_documents = []
    for doc in raw_documents or []:
        if not isinstance(doc, dict):
            continue
        url = doc.get("url", "") or ""
        normalized_doc = dict(doc)
        normalized_doc["source_type"] = "web"
        normalized_doc["source"] = "web"
        normalized_doc["url"] = url
        normalized_doc["domain"] = doc.get("domain") or _extract_domain(url)
        normalized_doc["trust_level"] = doc.get("trust_level") or "external"
        normalized_documents.append(normalized_doc)
    return normalized_documents


async def mcp_tool_call(query: str):
    """调用阿里云百炼网络搜索工具。

    Args:
        query: 联网搜索的问题。

    Returns:
        MCP 工具调用结果；失败时返回 None。
    """

    # 1. 创建MCPServerStreamableHttp对象
    search_mcp = MCPServerStreamableHttp(
        name="bailian_web_search",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
            "timeout": WEB_CONNECT_TIMEOUT_SECONDS,
        },
        cache_tools_list=True,
        max_retry_attempts=WEB_MAX_RETRY_ATTEMPTS,
    )

    # 2. 连接 - 调用 - 关闭
    try:
        await search_mcp.connect()
        tools = await search_mcp.list_tools()
        logger.info(f"mcp_tools:{tools}")
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": query,
                "max_results": WEB_MAX_RESULTS,
                "timeout": WEB_TOOL_CALL_TIMEOUT_SECONDS,
            },
        )
        return result
    except Exception as e:
        logger.exception(f"MCP_tools 连接百炼网络搜索工具失败: {e}")
    finally:
        await search_mcp.cleanup()


def node_web_search_mcp(state: QueryGraphState):
    """按检索计划执行 Web 搜索节点。

    Args:
        state: 查询流程状态，包含重写问题、会话信息和检索计划。

    Returns:
        包含 `web_search_docs` 和 Web 诊断信息的状态更新。
    """
    start = time.perf_counter()
    func_name = sys._getframe().f_code.co_name
    session_id = state["session_id"]
    is_stream = state["is_stream"]

    add_running_task(session_id, func_name, is_stream)
    logger.info(f"【{func_name}】节点开始执行")
    retrieval_plan = state.get("retrieval_plan") or {}
    use_web = retrieval_plan.get("use_web", True)
    if not use_web:
        # 旧固定三路召回仍可能触发 Web 节点；这里用 plan 做二次门控，避免无效联网调用。
        diagnostics = {
            "status": "skipped",
            "skip_reason": "disabled_by_plan",
            "error_type": "",
            "input_count": 0,
            "output_count": 0,
            "duration_ms": int((time.perf_counter() - start) * 1000),
        }
        record_retrieval_docs("web", 0)
        add_done_task(session_id, func_name, is_stream)
        logger.info(f"【{func_name}】按检索计划跳过，诊断信息：{diagnostics}")
        return _empty_web_result(state, diagnostics)

    try:
        # mcp调用网络搜索；Web 是补充召回源，超时或异常时必须降级为空结果。
        result = asyncio.run(mcp_tool_call(state["rewritten_query"]))

        # 处理返回结果
        # {
        #   "isError": false,
        #   "content": [{"text": "{\"pages\":[{\"snippet\":\"内容\",\"title\":\"标题\",\"url\":\"https://example.com\"}]}"}]
        # }
        web_documents = []
        if result and not result.isError:
            web_documents = json.loads(result.content[0].text).get("pages", [])
            web_documents = _normalize_web_documents(web_documents)
            logger.info(f"mcp联网搜索返回结果：{web_documents}")
        record_retrieval_docs("web", len(web_documents))
        diagnostics = {
            "status": "ok",
            "skip_reason": "",
            "error_type": "",
            "input_count": 1,
            "output_count": len(web_documents),
            "duration_ms": int((time.perf_counter() - start) * 1000),
            "tool_call_timeout_seconds": WEB_TOOL_CALL_TIMEOUT_SECONDS,
            "max_retry_attempts": WEB_MAX_RETRY_ATTEMPTS,
        }

        return {
            "web_search_docs": web_documents,
            "retrieval_diagnostics": _merge_web_diagnostics(state, diagnostics),
        }
    except Exception as exc:
        diagnostics = {
            "status": "error",
            "skip_reason": "web_search_failed",
            "error_type": type(exc).__name__,
            "input_count": 1,
            "output_count": 0,
            "duration_ms": int((time.perf_counter() - start) * 1000),
            "tool_call_timeout_seconds": WEB_TOOL_CALL_TIMEOUT_SECONDS,
            "max_retry_attempts": WEB_MAX_RETRY_ATTEMPTS,
        }
        record_retrieval_docs("web", 0)
        logger.exception(f"【{func_name}】执行失败，已降级为空 Web 结果：{exc}")
        return _empty_web_result(state, diagnostics)
    finally:
        add_done_task(session_id, func_name, is_stream)
        logger.info(f"【{func_name}】节点执行完成")


from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    test_state = {
        "session_id": "mcp_01",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream": True
    }

    # 调用 websearch_node 函数
    result_state = node_web_search_mcp(test_state)

    # 验证结果
    print("测试结果:")
    print(f"查询内容: {test_state.get('rewritten_query')}")

    # 输出搜索结果
    search_results = result_state.get('web_search_docs', [])
    print(f"搜索结果数量: {len(search_results)}")
    print("search_results", search_results)
