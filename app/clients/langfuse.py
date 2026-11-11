"""提供可选 Langfuse 客户端适配层。

该模块把 Langfuse 从业务启动链路中解耦：配置完整且依赖可导入时启用
真实客户端，否则退回空实现，保证查询和导入主流程不依赖观测服务。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, Optional

from app.conf.langfuse_config import langfuse_config
from app.core.logger import logger
from app.observability.settings import is_observability_enabled

_LANGFUSE_CLIENT: Optional[Any] = None
_LANGFUSE_CLIENT_INITIALIZED = False
_LANGFUSE_CALLBACK: Optional[Any] = None
_LANGFUSE_CALLBACK_INITIALIZED = False


def _flag_enabled(value: Optional[str], default: bool) -> bool:
    """解析布尔环境开关。

    Args:
        value: 环境变量原始值。
        default: 未设置时的默认值。

    Returns:
        解析后的布尔值。
    """
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_langfuse_configured() -> bool:
    """判断 Langfuse 是否具备启用条件。

    Returns:
        配置完整且未显式关闭时返回 True。
    """
    if not is_observability_enabled():
        return False
    explicit_enabled = os.getenv("LANGFUSE_ENABLED")
    enabled = _flag_enabled(explicit_enabled, default=True)
    if not enabled:
        return False
    return bool(
        langfuse_config.langfuse_public_key
        and langfuse_config.langfuse_secret_key
        and langfuse_config.langfuse_base_url
    )


class _NoopObservation:
    """Langfuse observation 的空实现。"""

    def __enter__(self) -> "_NoopObservation":
        """进入空 observation 上下文。

        Returns:
            当前空 observation。
        """
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        """退出空 observation 上下文。

        Args:
            exc_type: 异常类型。
            exc: 异常对象。
            traceback: 异常 traceback。

        Returns:
            False，保持原异常传播语义。
        """
        return False

    def update(self, *args: Any, **kwargs: Any) -> None:
        """忽略 observation 更新。

        Args:
            *args: 位置参数。
            **kwargs: 关键字参数。

        Returns:
            None.
        """
        return None


class OptionalLangfuseClient:
    """按需代理 Langfuse 客户端，缺配置时提供空实现。"""

    def _client(self) -> Optional[Any]:
        """获取真实 Langfuse 客户端。

        Returns:
            Langfuse 客户端；不可用时返回 None。
        """
        global _LANGFUSE_CLIENT, _LANGFUSE_CLIENT_INITIALIZED
        if _LANGFUSE_CLIENT_INITIALIZED:
            return _LANGFUSE_CLIENT
        _LANGFUSE_CLIENT_INITIALIZED = True
        if not is_langfuse_configured():
            logger.info("[Langfuse] 未启用或配置不完整，已使用空观测适配器")
            return None
        try:
            from langfuse import Langfuse

            _LANGFUSE_CLIENT = Langfuse(
                # org_id=langfuse_config.org_id,
                public_key=langfuse_config.langfuse_public_key,
                secret_key=langfuse_config.langfuse_secret_key,
                base_url=langfuse_config.langfuse_base_url,
            )
        except Exception as exc:
            logger.warning(f"[Langfuse] 初始化失败，已降级为空观测适配器：{exc}")
            _LANGFUSE_CLIENT = None
        return _LANGFUSE_CLIENT

    def start_as_current_observation(self, *args: Any, **kwargs: Any) -> Any:
        """创建当前 Langfuse observation。

        Args:
            *args: 透传给 Langfuse 的位置参数。
            **kwargs: 透传给 Langfuse 的关键字参数。

        Returns:
            Langfuse observation 上下文；不可用时返回空上下文。
        """
        client = self._client()
        if client is None:
            return _NoopObservation()
        try:
            return client.start_as_current_observation(*args, **kwargs)
        except Exception as exc:
            logger.warning(f"[Langfuse] 创建 observation 失败，已忽略：{exc}")
            return _NoopObservation()

    def get_current_trace_id(self) -> str:
        """获取当前 Langfuse trace ID。

        Returns:
            trace ID；不可用时返回空字符串。
        """
        client = self._client()
        if client is None:
            return ""
        try:
            return client.get_current_trace_id() or ""
        except Exception:
            return ""

    def get_current_observation_id(self) -> str:
        """获取当前 Langfuse observation ID。

        Returns:
            observation ID；不可用时返回空字符串。
        """
        client = self._client()
        if client is None:
            return ""
        try:
            return client.get_current_observation_id() or ""
        except Exception:
            return ""

    def get_trace_url(self) -> str:
        """获取当前 Langfuse trace URL。

        Returns:
            trace URL；不可用时返回空字符串。
        """
        client = self._client()
        if client is None:
            return ""
        try:
            return client.get_trace_url() or ""
        except Exception:
            return ""

    def set_current_trace_io(self, *args: Any, **kwargs: Any) -> None:
        """更新当前 Langfuse trace 的输入输出。

        Args:
            *args: 透传给 Langfuse 的位置参数。
            **kwargs: 透传给 Langfuse 的关键字参数。

        Returns:
            None.
        """
        client = self._client()
        if client is None:
            return None
        try:
            client.set_current_trace_io(*args, **kwargs)
        except Exception as exc:
            logger.warning(f"[Langfuse] 更新 trace IO 失败，已忽略：{exc}")
        return None

    def score_current_trace(self, *args: Any, **kwargs: Any) -> None:
        """给当前 Langfuse trace 评分。

        Args:
            *args: 透传给 Langfuse 的位置参数。
            **kwargs: 透传给 Langfuse 的关键字参数。

        Returns:
            None.
        """
        client = self._client()
        if client is None:
            return None
        try:
            client.score_current_trace(*args, **kwargs)
        except Exception as exc:
            logger.warning(f"[Langfuse] 写入 trace score 失败，已忽略：{exc}")
        return None

    def flush(self) -> None:
        """刷新 Langfuse 客户端缓冲区。

        Returns:
            None.
        """
        client = self._client()
        if client is None:
            return None
        flush_fn = getattr(client, "flush", None)
        if flush_fn:
            flush_fn()
        return None

    def __getattr__(self, name: str) -> Any:
        """代理真实 Langfuse 客户端的其他属性。

        Args:
            name: 属性名。

        Returns:
            真实客户端属性。

        Raises:
            AttributeError: Langfuse 未启用或真实客户端不存在该属性。
        """
        client = self._client()
        if client is None:
            raise AttributeError(f"Langfuse is not enabled: {name}")
        return getattr(client, name)


@contextmanager
def propagate_attributes(*args: Any, **kwargs: Any) -> Iterator[None]:
    """设置 Langfuse trace 级属性。

    Args:
        *args: 透传给 Langfuse propagate_attributes 的位置参数。
        **kwargs: 透传给 Langfuse propagate_attributes 的关键字参数。

    Yields:
        None.
    """
    if not is_langfuse_configured():
        yield None
        return
    try:
        from langfuse import propagate_attributes as _propagate_attributes

        with _propagate_attributes(*args, **kwargs):
            yield None
    except Exception as exc:
        # Langfuse 上下文传播只是观测层能力，失败时继续执行业务流程。
        logger.warning(f"[Langfuse] 设置 trace 属性失败，已忽略：{exc}")
        yield None


def observe(*args: Any, **kwargs: Any) -> Callable:
    """创建可选 Langfuse observe 装饰器。

    Args:
        *args: 透传给 Langfuse observe 的位置参数。
        **kwargs: 透传给 Langfuse observe 的关键字参数。

    Returns:
        装饰器；Langfuse 不可用时返回保持函数原样的装饰器。
    """
    if is_langfuse_configured():
        try:
            from langfuse import observe as _observe

            return _observe(*args, **kwargs)
        except Exception as exc:
            logger.warning(f"[Langfuse] 创建 observe 装饰器失败，已使用空装饰器：{exc}")

    def decorator(fn: Callable) -> Callable:
        """返回原函数的空装饰器。

        Args:
            fn: 被装饰函数。

        Returns:
            保持行为不变的包装函数。
        """

        @wraps(fn)
        def wrapper(*fn_args: Any, **fn_kwargs: Any) -> Any:
            """执行原函数。

            Args:
                *fn_args: 原函数位置参数。
                **fn_kwargs: 原函数关键字参数。

            Returns:
                原函数返回值。

            Raises:
                Exception: 原函数异常会继续抛出。
            """
            return fn(*fn_args, **fn_kwargs)

        return wrapper

    return decorator


def get_langfuse_callback() -> Optional[Any]:
    """获取 LangChain Langfuse 回调。

    Returns:
        CallbackHandler 实例；Langfuse 未启用或初始化失败时返回 None。
    """
    global _LANGFUSE_CALLBACK, _LANGFUSE_CALLBACK_INITIALIZED
    if _LANGFUSE_CALLBACK_INITIALIZED:
        return _LANGFUSE_CALLBACK
    _LANGFUSE_CALLBACK_INITIALIZED = True
    if not is_langfuse_configured():
        return None
    try:
        from langfuse.langchain import CallbackHandler

        _LANGFUSE_CALLBACK = CallbackHandler()
    except Exception as exc:
        logger.warning(f"[Langfuse] 初始化 LangChain 回调失败，已忽略：{exc}")
        _LANGFUSE_CALLBACK = None
    return _LANGFUSE_CALLBACK


langfuse = OptionalLangfuseClient()
langfuse_handler = get_langfuse_callback()
