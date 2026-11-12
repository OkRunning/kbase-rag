# 环境配置与依赖导入
import os
import time
import uuid

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

# 项目内部依赖
from app.clients.langfuse import get_langfuse_callback
from app.conf.lm_config import lm_config
from app.core.logger import logger
from app.observability.metrics import (
    record_llm_call,
    record_llm_stream_first_token,
    record_llm_tokens,
)
# 熔断降级：电路开路时切备用模型或抛 CircuitOpenError 由上层兜底
from app.utils.alert_utils import llm_failure_monitor, CircuitOpenError


# 全局缓存：键为(模型名, JSON输出模式)元组，值为ChatOpenAI实例
# 作用：避免重复初始化客户端，提升性能，统一实例管理
_llm_client_cache = {}

# Langfuse 回调单例由适配层按配置创建；未启用时不传 callback，LLM 主流程保持可用。
_langfuse_callback = get_langfuse_callback()


def _record_response_tokens(model: str, response: Any) -> None:
    """从 LLM 响应元数据中提取并记录 token。

    Args:
        model: 模型名称。
        response: LangChain LLM 响应对象。

    Returns:
        None.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
    merged_usage = {**token_usage, **usage}
    input_tokens = int(merged_usage.get("input_tokens") or merged_usage.get("prompt_tokens") or 0)
    output_tokens = int(merged_usage.get("output_tokens") or merged_usage.get("completion_tokens") or 0)
    total_tokens = int(merged_usage.get("total_tokens") or (input_tokens + output_tokens) or 0)
    record_llm_tokens(model, "input", input_tokens)
    record_llm_tokens(model, "output", output_tokens)
    record_llm_tokens(model, "total", total_tokens)


class _ObservedLLMClient:
    """为 LangChain LLM 客户端增加 Prometheus 指标观测。"""

    def __init__(self, client: ChatOpenAI, model: str, json_mode: bool, fallback: bool) -> None:
        """初始化 LLM 观测代理。

        Args:
            client: 原始 ChatOpenAI 客户端。
            model: 模型名称。
            json_mode: 是否启用 JSON 模式。
            fallback: 是否为备用模型调用。

        Returns:
            None.
        """
        self._client = client
        self._model = model
        self._json_mode = json_mode
        self._fallback = fallback

    def __getattr__(self, name: str) -> Any:
        """代理原始客户端的其他属性。

        Args:
            name: 属性名。

        Returns:
            原始客户端属性。

        Raises:
            AttributeError: 原始客户端不存在该属性。
        """
        return getattr(self._client, name)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """执行同步 LLM 调用并记录指标。

        Args:
            *args: 透传给 LangChain invoke 的位置参数。
            **kwargs: 透传给 LangChain invoke 的关键字参数。

        Returns:
            LLM 响应对象。

        Raises:
            Exception: 原始 LLM 调用失败时继续抛出。
        """
        start = time.perf_counter()
        status = "ok"
        try:
            response = self._client.invoke(*args, **kwargs)
            _record_response_tokens(self._model, response)
            llm_failure_monitor.record_success()
            return response
        except Exception as exc:
            status = "error"
            llm_failure_monitor.record_failure(
                title=f"LLM 调用失败 model={self._model}",
                content=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            record_llm_call(
                self._model,
                status,
                self._json_mode,
                self._fallback,
                time.perf_counter() - start,
            )

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """执行流式 LLM 调用并记录指标。

        Args:
            *args: 透传给 LangChain stream 的位置参数。
            **kwargs: 透传给 LangChain stream 的关键字参数。

        Yields:
            LLM 流式响应块。

        Raises:
            Exception: 原始 LLM 流式调用失败时继续抛出。
        """
        start = time.perf_counter()
        status = "ok"
        first_token_recorded = False
        try:
            for chunk in self._client.stream(*args, **kwargs):
                if not first_token_recorded:
                    record_llm_stream_first_token(self._model, "ok", time.perf_counter() - start)
                    first_token_recorded = True
                _record_response_tokens(self._model, chunk)
                yield chunk
            llm_failure_monitor.record_success()
        except Exception as exc:
            status = "error"
            if not first_token_recorded:
                record_llm_stream_first_token(self._model, "error", time.perf_counter() - start)
            llm_failure_monitor.record_failure(
                title=f"LLM 流式调用失败 model={self._model}",
                content=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            record_llm_call(
                self._model,
                status,
                self._json_mode,
                self._fallback,
                time.perf_counter() - start,
            )
def _is_opencode_api_base(api_base: Optional[str]) -> bool:
    """
    判断 API 地址是否指向 OpenCode 服务。

    Args:
        api_base: OpenAI 兼容服务的基础地址。

    Returns:
        地址属于 OpenCode 域名时返回 True，否则返回 False。
    """
    if not api_base:
        return False

    hostname = urlparse(api_base).hostname or ""
    return hostname == "opencode.ai" or hostname.endswith(".opencode.ai")


def get_llm_client(model: Optional[str] = None, json_mode: bool = False, _fallback: bool = False) -> _ObservedLLMClient:
    """
    获取带全局缓存和指标观测的 LangChain ChatOpenAI 客户端实例。

    Args:
        model: 模型名称，优先级为传入参数、配置文件默认模型、内置默认模型。
        json_mode: 是否开启 JSON 输出模式。
        _fallback: 是否为熔断降级后的备用模型客户端。

    Returns:
        已初始化的 LLM 观测代理客户端。

    Raises:
        ValueError: 缺失 API 密钥或基础地址等核心配置。
        CircuitOpenError: 熔断开路且无可用备用模型。
        Exception: 模型初始化失败。
    """

    # 1. 确定目标模型（优先级递减，保证模型名非空）
    target_model = model or lm_config.llm_model or "qwen3-32b"
    fallback_model = lm_config.fallback_model or "qwen3-32b"  # 内置默认值防 None

    # 1.1 熔断降级：电路开路时，切到备用模型；未配备用模型则抛 CircuitOpenError
    # if failure_monitor.should_bypass:
    #     fallback_model = lm_config.fallback_model
    #     if fallback_model and fallback_model != target_model:
    #         logger.warning(
    #             f"[熔断降级] 电路开路，主模型 {target_model} 降级到备用模型 {fallback_model}"
    #         )
    #         target_model = fallback_model
    #     else:
    #         raise CircuitOpenError(
    #             f"电路开路且未配置可用备用模型（LLM_FALLBACK_MODEL），主模型={target_model}"
    #         )

    # 原子检查：是否必须旁路 -- 是否需要降级
    if llm_failure_monitor.should_bypass():
        # 1. 有备用模型 → 直接返回备用模型客户端
        if fallback_model and fallback_model != target_model:
            logger.warning(f"[熔断] 主模型 {target_model} → 降级到 {fallback_model}")
            return get_llm_client(fallback_model, json_mode=json_mode, _fallback=True)

        # 2. 无备用模型 → 抛出错误由上层兜底
        raise CircuitOpenError(
            f"电路开路且未配置备用模型（current={target_model}, fallback={fallback_model})"
        )
    # 缓存键：模型名+JSON模式，唯一标识不同配置的客户端
    # cache_key = (target_model, json_mode)
    # 缓存键增加是否降级标识
    cache_key = (target_model, json_mode, "fallback" if _fallback else "normal")

    # 2. 缓存命中：直接返回已初始化的实例，避免重复创建
    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM客户端] 缓存命中，直接返回实例：模型={target_model}，JSON模式={json_mode}")
        return _llm_client_cache[cache_key]

    # 3. 核心配置校验：拦截缺失的API关键配置，提前抛出明确异常
    if not lm_config.api_key:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_API_KEY（大模型API密钥）")
    if not lm_config.base_url:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_API_BASE（API接口基础地址）")
    logger.info(f"[LLM客户端] 开始初始化新实例：模型={target_model}，JSON模式={json_mode}")

    # 4. 配置参数组装：区分「国产模型私有参数」和「OpenAI通用参数」
    # extra_body：千问/即梦等国产模型专属私有参数（LangChain透传至API）
    extra_body = {"enable_thinking": False}  # 千问专属：关闭思考链输出，减少冗余内容
    # model_kwargs：OpenAI通用参数，所有兼容API均支持
    model_kwargs = {}
    if json_mode:
        # 开启JSON标准输出模式，强制模型返回可解析的json_object
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug(f"[LLM客户端] 已开启JSON输出模式，模型将返回标准JSON结构")

    # 添加模拟 LLM 失败逻辑 👇
    # if os.getenv("MOCK_LLM_FAILURE", "false").lower() == "true":
    #     raise Exception("模拟 LLM 调用失败")  # ✅ 触发 LLM 层失败
    #

    # OpenCode 依赖会话头做请求路由。default_headers 会同时传给
    # OpenAI 的同步与异步客户端，避免 ainvoke 请求丢失该请求头。
    if _is_opencode_api_base(lm_config.base_url):
        default_headers = {}
        default_headers.setdefault(
            "x-opencode-session",str(uuid.uuid4()),
        )
        default_headers.setdefault("User-Agent", "opencode-langchain")
        # llm_kwargs["default_headers"] = default_headers

    # 5. 客户端初始化：捕获LangChain封装层异常，抛出更友好的提示
    try:
        callbacks = [_langfuse_callback] if _langfuse_callback is not None else None
        llm_client = ChatOpenAI(
            model=target_model,  # 目标模型名
            temperature=lm_config.llm_temperature or 0.1,  # 低温度保证输出确定性（0~1）
            api_key=lm_config.api_key,  # API密钥
            base_url=lm_config.base_url,  # API基础地址（适配国产模型代理地址）
            streaming=True,  # 全程开启流式：invoke 也逐 chunk 触发 on_chat_model_stream，供 LangGraph stream_mode="messages" 捕获
            extra_body=extra_body,  # 国产模型私有参数透传
            model_kwargs=model_kwargs,  # OpenAI通用参数
            callbacks=callbacks,  # Langfuse 未启用时为 None，不影响 LLM 调用。
            default_headers=default_headers
        )
    except LangChainException as e:
        raise Exception(f"[LLM客户端] 模型【{target_model}】初始化失败（LangChain层）：{str(e)}") from e

    # 6. 新实例存入全局缓存，供后续调用复用
    _llm_client_cache[cache_key] = _ObservedLLMClient(llm_client, target_model, json_mode, _fallback)
    logger.info(f"[LLM客户端] 实例初始化成功并缓存：模型={target_model}，JSON模式={json_mode}")

    return _llm_client_cache[cache_key]


# 测试示例：验证客户端创建、缓存机制及日志输出
if __name__ == "__main__":
    logger.info("===== 开始执行LLM客户端工具测试 =====")
    try:
        # 测试1：默认配置（默认模型+普通模式）
        client1 = get_llm_client()
        logger.info("✅ 测试1通过：默认配置客户端创建成功")

        # 测试2：指定多模态模型（qwen-vl-plus）+ 普通模式
        client2 = get_llm_client(model="qwen-vl-plus")
        logger.info("✅ 测试2通过：指定多模态模型客户端创建成功")

        # 测试3：同一模型+模式，验证缓存命中
        client3 = get_llm_client(model="qwen-vl-plus")
        logger.info(f"✅ 测试3通过：缓存机制验证成功，client2与client3为同一实例：{client2 is client3}")

        # 测试4：开启JSON输出模式
        client4 = get_llm_client(model="qwen3-32b", json_mode=True)
        logger.info("✅ 测试4通过：JSON输出模式客户端创建成功")

    except Exception as e:
        logger.error(f"❌ LLM客户端工具测试失败：{str(e)}", exc_info=True)
    finally:
        logger.info("===== LLM客户端工具测试结束 =====")
