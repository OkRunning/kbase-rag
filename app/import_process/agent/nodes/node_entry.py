import sys
from pathlib import Path

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.observability.sanitizers import sanitize_for_log
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """根据上传文件类型初始化导入路由状态。

    Args:
        state: 导入流程状态，需包含 local_file_path 和 task_id。

    Returns:
        写入文件标题、文件类型路由标记和文件路径后的导入流程状态。
    """

    # 1.节点入口处打印日志以及前端埋点
    # 动态获取函数名避免硬编码
    func_name = sys._getframe().f_code.co_name
    logger.info(f"【{func_name}】节点开始执行，全局状态为:{sanitize_for_log(state)}")
    add_running_task(state["task_id"], func_name)

    # 2.节点处理逻辑-判断上传文件类型以及路由设置
    local_file_path = state.get("local_file_path","")
    if not local_file_path:
        logger.info(f"【{func_name}】节点处理，上传文件为空，无法解析")
        return state

    if local_file_path.endswith(".pdf"):
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path

    elif local_file_path.endswith(".md"):
        state["is_md_read_enabled"] = True
        state["md_path"] = local_file_path
    else:
        logger.info(f"【{func_name}】节点处理，不支持的文件类型:{local_file_path}")

    # 3.节点处理逻辑-设置文件标题
    # .stem 去掉后缀的文件名
    # .name 获取完整文件名
    # .suffix 获取文件后缀
    state["file_title"] = Path(local_file_path).stem

    # 4.节点处理完成后，打印日志以及前端埋点
    logger.info(f"【{func_name}】节点节点执行结束，全局状态为:{sanitize_for_log(state)}")
    add_done_task(state["task_id"], func_name)
    return state
