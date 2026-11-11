"""验证 task_utils 在同一 session_id 跨轮查询时的进度重置能力。"""

from app.utils.task_utils import (
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    reset_task_progress,
    set_task_result,
    update_task_status,
)


def test_reset_task_progress_clears_done_list_for_reused_session():
    """同一 session_id 发起第二轮查询前调用 reset，上轮 done_list 不应残留。"""
    session_id = "test-reset-session"

    # 模拟第一轮查询：写入运行中节点、已完成节点、结果与终态。
    add_running_task(session_id, "node_search")
    add_done_task(session_id, "node_search")
    add_done_task(session_id, "node_answer")
    set_task_result(session_id, "answer", "上一轮的旧答案")
    update_task_status(session_id, "completed")

    assert get_done_task_list(session_id) != []
    assert get_task_result(session_id, "answer") == "上一轮的旧答案"

    # 第二轮查询开始前重置进度态。
    reset_task_progress(session_id)

    assert get_done_task_list(session_id) == []
    assert get_running_task_list(session_id) == []
    assert get_task_result(session_id, "answer") == ""
    assert get_task_status(session_id) == "pending"
