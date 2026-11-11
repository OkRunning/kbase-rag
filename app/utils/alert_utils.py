"""告警通知工具：支持邮件 + 个人微信(Server酱)

通过环境变量配置启用哪些渠道（默认关闭）：
- 邮件: ALERT_EMAIL_ENABLED, SMTP_HOST, SMTP_PORT, SMTP_USER,
         SMTP_PASSWORD, SMTP_SECURE, ALERT_EMAIL_TO
- 微信: ALERT_WECHAT_ENABLED, SERVERCHAN_SENDKEY
         （Server酱 https://sct.ftqq.com，免费推送到个人微信）

设计原则：尽力而为、绝不影响主流程。所有发送失败都被吞掉并记日志；
发送在守护线程里异步执行，不阻塞请求。
"""
import os
import smtplib
import threading
import time
from email.mime.text import MIMEText
from email.utils import formataddr
from typing import Optional

import httpx

from app.core.logger import logger
from app.observability.metrics import record_circuit_state


def _build_text(title: str, content: str, trace_url: Optional[str]) -> str:
    text = f"【RAG告警】{title}\n{content}"
    if trace_url:
        text += f"\nLangfuse Trace: {trace_url}"
    return text


def _send_email(text: str) -> None:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "465"))
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASSWORD")
    secure = os.getenv("SMTP_SECURE", "true").lower() == "true"
    to = os.getenv("ALERT_EMAIL_TO")
    if not all([host, user, pwd, to]):
        logger.debug("[告警] 邮件渠道未完整配置，跳过")
        return

    msg = MIMEText(text, "plain", "utf-8")
    msg["Subject"] = "【RAG告警】"
    msg["From"] = formataddr(("RAG告警", user))
    msg["To"] = to

    smtp_cls = smtplib.SMTP_SSL if secure else smtplib.SMTP
    with smtp_cls(host, port, timeout=10) as s:
        s.login(user, pwd)
        s.sendmail(user, [to], msg.as_string())
    logger.info(f"[告警] 邮件已发送至 {to}")


def _send_wechat(text: str) -> None:
    """通过 Server酱 推送到个人微信"""
    key = os.getenv("SERVERCHAN_SENDKEY")
    if not key:
        logger.debug("[告警] 微信(Server酱)未配置 SERVERCHAN_SENDKEY，跳过")
        return

    # Server酱：title 限 32 字，正文放 desp
    lines = text.splitlines()
    title = lines[0][:32] if lines else "RAG告警"
    desp = "\n".join(lines[1:]) or title

    resp = httpx.post(
        f"https://sctapi.ftqq.com/{key}.send",
        data={"title": title, "desp": desp},
        timeout=10,
    )
    if resp.status_code == 200 and resp.json().get("code", -1) == 0:
        logger.info("[告警] 微信(Server酱)已发送")
    else:
        logger.warning(f"[告警] 微信发送失败: {resp.status_code} {resp.text}")


def send_alert(title: str, content: str, trace_url: Optional[str] = None) -> None:
    """发送告警。

    Args:
        title: 告警标题。
        content: 告警正文。
        trace_url: 可选的 Langfuse trace 链接。

    Returns:
        None.
    """
    text = _build_text(title, content, trace_url)

    def _worker():
        try:
            if os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true":
                _send_email(text)
            if os.getenv("ALERT_WECHAT_ENABLED", "false").lower() == "true":
                _send_wechat(text)
        except Exception as e:
            logger.warning(f"[告警] 发送失败（不影响主流程）：{e}")

    threading.Thread(target=_worker, daemon=True).start()


class CircuitState:
    """熔断器状态"""
    CLOSED = "closed"      # 正常，放行
    OPEN = "open"          # 熔断，降级/拒绝
    HALF_OPEN = "half_open"  # 半开，放行一个探测请求


class CircuitOpenError(Exception):
    """电路开路（熔断中）且无可用降级方案时抛出，供上层捕获做兜底"""
    pass


class ConsecutiveFailureMonitor:
    """连续异常检测器 + 熔断器

    检测部分：
    - 成功一次 → 连续失败计数清零
    - 失败一次 → 计数 +1
    - 连续失败达到告警阈值 且 不在告警冷却期 → 触发告警

    熔断部分：
    - 连续失败达到开路阈值(open_threshold) → 电路 OPEN，调用方应降级
    - OPEN 持续 open_duration 秒后 → 自动转 HALF_OPEN，放行一个探测请求
    - 探测成功 → CLOSED；探测失败 → 重新 OPEN

    线程安全（FastAPI 并发调用安全）。

    配置（环境变量）：
    - ALERT_FAILURE_THRESHOLD: 连续失败几次告警，默认 3
    - ALERT_FAILURE_COOLDOWN: 告警冷却秒数，默认 300
    - CIRCUIT_OPEN_THRESHOLD: 连续失败几次熔断开路，默认 5（应 > 告警阈值）
    - CIRCUIT_OPEN_DURATION: 开路持续秒数，默认 60
    """

    def __init__(
        self,
        threshold: Optional[int] = None,
        cooldown_seconds: Optional[int] = None,
        open_threshold: Optional[int] = None,
        open_duration: Optional[int] = None,
        name: str = "query",
    ):
        self.threshold = threshold or int(os.getenv("ALERT_FAILURE_THRESHOLD", "3"))
        self.cooldown = cooldown_seconds or int(os.getenv("ALERT_FAILURE_COOLDOWN", "300"))
        self.open_threshold = open_threshold or int(os.getenv("CIRCUIT_OPEN_THRESHOLD", "5"))
        self.open_duration = open_duration or int(os.getenv("CIRCUIT_OPEN_DURATION", "60"))
        self.name = name
        self._lock = threading.Lock()
        self._consecutive = 0
        self._last_alert_ts = 0.0  # 上次告警的 monotonic 时间
        self._state = CircuitState.CLOSED
        self._opened_at = 0.0  # 进入 OPEN 的 monotonic 时间
        self._sync_metrics()

    def _sync_metrics(self) -> None:
        """同步熔断状态到 Prometheus 指标。

        Returns:
            None.
        """
        record_circuit_state(self.name, self._state, self._consecutive)

    def record_success(self) -> None:
        """记录成功调用并关闭电路。

        Returns:
            None.
        """
        with self._lock:
            self._consecutive = 0
            self._state = CircuitState.CLOSED
            self._sync_metrics()

    def record_failure(
        self,
        title: str,
        content: str,
        trace_url: Optional[str] = None,
    ) -> bool:
        """记录失败调用并在必要时触发告警和熔断。

        Args:
            title: 告警标题。
            content: 告警正文。
            trace_url: 可选的 Langfuse trace 链接。

        Returns:
            本次是否真正触发了告警。
        """
        with self._lock:
            self._consecutive += 1
            count = self._consecutive
            now = time.monotonic()
            # 熔断：达到开路阈值 → OPEN
            if count >= self.open_threshold and self._state != CircuitState.OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = now
                logger.warning(
                    f"[熔断] {self.name} 连续失败 {count} 次达到开路阈值 "
                    f"{self.open_threshold}，电路 OPEN（持续 {self.open_duration}s）"
                )
            self._sync_metrics()
            # 告警：达到告警阈值且不在冷却期
            in_cooldown = (now - self._last_alert_ts) < self.cooldown
            should_alert = count >= self.threshold and not in_cooldown
            if should_alert:
                self._last_alert_ts = now

        if not should_alert:
            logger.debug(
                f"[连续异常] {self.name} 失败累计 {count} 次"
                f"（告警阈值 {self.threshold}，冷却中 {in_cooldown}），本次不告警"
            )
            return False

        logger.warning(f"[连续异常] {self.name} 连续失败 {count} 次，触发告警")
        send_alert(
            title=f"[连续异常] {title}（连续 {count} 次）",
            content=content,
            trace_url=trace_url,
        )
        return True

    @property
    def state(self) -> str:
        """读取当前电路状态。

        Returns:
            当前电路状态；会自动处理 OPEN 到 HALF_OPEN 的转换。
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._opened_at >= self.open_duration:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"[熔断] {self.name} 进入 HALF_OPEN，放行探测请求")
                    self._sync_metrics()
            return self._state

    # @property
    # def is_open(self) -> bool:
    #     """电路是否处于熔断开路状态（OPEN）。HALF_OPEN 视为放行探测，不算开路。"""
    #     return self.state == CircuitState.OPEN

    # @property
    def should_bypass(self) -> bool:
        """判断是否应直接旁路。

        Returns:
            熔断开路且无可用降级时返回 True。
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._opened_at >= self.open_duration:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"[熔断] {self.name} 进入 HALF_OPEN，放行探测请求")
                    self._sync_metrics()
                    return False
                return True
            return False

    @property
    def consecutive_failures(self) -> int:
        """读取连续失败次数。

        Returns:
            连续失败次数。
        """
        with self._lock:
            return self._consecutive


# 查询流程的全局单例（多请求共享同一计数器与熔断状态）
failure_monitor = ConsecutiveFailureMonitor(name="query")

# 在 ConsecutiveFailureMonitor 定义后新增
llm_failure_monitor = ConsecutiveFailureMonitor(
    name="llm",
    threshold=int(os.getenv("LLM_FAILURE_THRESHOLD", "3")),
    open_threshold=int(os.getenv("LLM_OPEN_THRESHOLD", "5")),
    cooldown_seconds=int(os.getenv("LLM_FAILURE_COOLDOWN", "300")),
    open_duration=int(os.getenv("LLM_OPEN_DURATION", "60"))
)

