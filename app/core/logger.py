"""
项目日志工具类
基于loguru实现，支持.env配置控制台/文件双输出，自动生成logs/app_年月日.log
特性：
1. 配置驱动：通过.env开关输出、修改日志级别
2. 自动路径：文件日志默认输出到 项目根/logs/app_YYYYMMDD.log
3. 自动清理：按配置保留日志，自动删除过期文件
4. 中文友好：utf-8编码，彻底解决中文乱码
5. 异步安全：开启异步入队，支持多线程/异步场景，避免日志错乱
6. 开箱即用：项目所有模块直接导入logger即可使用
7. 位置终极精准：穿透loguru内部+工具类自身，完美显示业务模块实际调用位置
"""
import sys
import inspect
from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger
from opentelemetry import trace

from app.observability.context import get_context


# -------------------------- 第一步：加载.env配置文件 --------------------------
load_dotenv()

# -------------------------- 第二步：读取.env配置（带默认值，防止配置缺失） --------------------------
LOG_CONSOLE_ENABLE = os.getenv("LOG_CONSOLE_ENABLE", "True").lower() == "true"
LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
LOG_FILE_ENABLE = os.getenv("LOG_FILE_ENABLE", "True").lower() == "true"
LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "INFO").upper()
LOG_FILE_RETENTION = os.getenv("LOG_FILE_RETENTION", "7 days")
LOG_JSON_ENABLE = os.getenv("LOG_JSON_ENABLE", "True").lower() == "true"
LOG_JSON_LEVEL = os.getenv("LOG_JSON_LEVEL", LOG_FILE_LEVEL).upper()

# -------------------------- 第三步：定义日志路径（自动推导项目根） --------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE_NAME = "app_{time:YYYYMMDD}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME
LOG_JSON_FILE_NAME = "app_{time:YYYYMMDD}.jsonl"
LOG_JSON_FILE_PATH = LOG_DIR / LOG_JSON_FILE_NAME

# -------------------------- 第四步：定义日志格式（彩色、结构化、易读） --------------------------
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name: <20}</cyan>:<cyan>{function: <15}</cyan>:<cyan>{line: <4}</cyan> - "
    "<level>{message}</level>"
)

# -------------------------- 第五步：初始化日志配置（核心方法） --------------------------
def init_logger():
    """初始化全局日志配置。

    Returns:
        配置完成的 loguru logger 实例。
    """
    # 1. 移除loguru默认的控制台输出
    logger.remove()

    # 2. 配置控制台输出（若.env开启）
    if LOG_CONSOLE_ENABLE:
        logger.add(
            sink=sys.stdout,
            level=LOG_CONSOLE_LEVEL,
            format=LOG_FORMAT,
            colorize=True,
            enqueue=True
        )

    # 3. 配置文件输出（若.env开启）
    if LOG_FILE_ENABLE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=LOG_FILE_PATH,
            level=LOG_FILE_LEVEL,
            format=LOG_FORMAT,
            rotation="00:00",
            retention=LOG_FILE_RETENTION,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )

    # JSON Lines 输出给 Promtail/Loki 采集，文本日志继续服务本地排查。
    if LOG_JSON_ENABLE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=LOG_JSON_FILE_PATH,
            level=LOG_JSON_LEVEL,
            serialize=True,
            rotation="00:00",
            retention=LOG_FILE_RETENTION,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )

    return logger

# -------------------------- 第六步：初始化并终极修正全局logger --------------------------
base_logger = init_logger()

def fix_log_position(record):
    """修正日志调用位置并注入可观测上下文。

    Args:
        record: loguru 当前日志记录。

    Returns:
        None.
    """
    # 将当前 span 和业务上下文写入 extra，JSON 日志可按 trace/session/task 检索。
    span_context = trace.get_current_span().get_span_context()
    if span_context and span_context.is_valid:
        record["extra"]["trace_id"] = f"{span_context.trace_id:032x}"
        record["extra"]["span_id"] = f"{span_context.span_id:016x}"
    else:
        record["extra"]["trace_id"] = ""
        record["extra"]["span_id"] = ""

    for key, value in get_context().items():
        record["extra"][key] = value

    for frame in inspect.stack():
        # 终极过滤：排除loguru内部 + 排除工具类logger.py自身，直接定位业务模块
        if ("_logger.py" in frame.filename or frame.function == "_log") or "logger.py" in frame.filename:
            continue
        # 更新日志字段为业务代码实际位置
        record.update(
            name=frame.filename.split("/")[-1].split("\\")[-1],
            function=frame.function,
            line=frame.lineno
        )
        break

# 应用终极修复，导出全局可用的logger
logger = base_logger.patch(fix_log_position)

# -------------------------- 测试代码（验证修复效果） --------------------------
if __name__ == '__main__':
    logger.info("【测试】logger.py内部调用（仅测试，业务模块调用会显示正确文件名）")
    logger.error("【测试】error。报错了2222")
    # print(f"日志文件输出路径：{LOG_FILE_PATH}")
