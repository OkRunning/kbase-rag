import asyncio
import uuid
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()
# 跨域配置（保持不变）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ✅ 替换列表缓存：用异步队列存储每个会话的消息（key: session_id, value: asyncio.Queue）
task_queues = {}


class QueryRequest(BaseModel):
    query: str
    session_id: str = None


@app.post("/submit_query")
async def submit_query(req: QueryRequest, background_tasks: BackgroundTasks):
    # 生成或使用用户传入的session_id
    session_id = req.session_id or str(uuid.uuid4())
    # 将耗时任务加入后台执行
    background_tasks.add_task(long_task, session_id, req.query)
    return {"message": "任务已经启动", "session_id": session_id}


async def long_task(session_id: str, query: str):
    """模拟耗时的异步任务，分阶段往队列丢消息（替代列表累加）"""
    # 为当前会话创建专属异步队列
    queue = asyncio.Queue()
    task_queues[session_id] = queue

    # 模拟5个进度步骤，往队列丢进度消息
    for i in range(5):
        progress_msg = {
            "event": "progress",
            "data": f"【{query}】的第{i + 1}段回答:xxx{i + 1}"
        }
        await queue.put(progress_msg)  # 进度消息入队
        await asyncio.sleep(1)

    # 任务完成，往队列丢完成消息
    complete_msg = {
        "event": "complete",
        "data": f"【{session_id}】查询完成！所有结果已返回"
    }
    await queue.put(complete_msg)
    # 关键：丢结束标记，告诉SSE可以停止监听
    await queue.put(None)


@app.get("/stream/{session_id}")
async def stream(session_id: str):
    """SSE流式返回任务结果，基于队列实现（无轮询）"""

    async def event_generator():
        # 等待当前会话的队列创建（防止SSE比任务先启动）
        while session_id not in task_queues:
            await asyncio.sleep(0.1)
        queue = task_queues[session_id]

        # 循环从队列取消息，有消息就推，收到结束标记就停
        while True:
            msg = await queue.get()  # 异步阻塞等待消息（无轮询）
            if msg is None:  # 收到结束标记，退出循环
                break

            # 拼接自定义Event的SSE格式（和你原逻辑一致）
            yield f"event: {msg['event']}\n"
            yield f"data: {msg['data']}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8010)
