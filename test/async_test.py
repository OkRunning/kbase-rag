import asyncio
from starlette.concurrency import run_in_threadpool


def sync_func():
    try:
        asyncio.get_running_loop()
        print("有 running loop")
    except RuntimeError:
        print("没有 running loop")


async def endpoint_like():
    print("主 async 里：", asyncio.get_running_loop())
    await run_in_threadpool(sync_func)


asyncio.run(endpoint_like())
