# 导入系统模块：用于读取环境变量
import json
import os
# 导入日志模块：用于记录程序运行日志（成功/失败/错误信息）
import logging
import time
# 导入类型注解模块：用于函数参数/返回值的类型提示，提升代码可读性和规范性
from typing import List, Dict, Any, Optional
# 导入时间模块：用于生成时间戳，记录对话的创建时间
from datetime import datetime
# 导入pymongo核心模块：MongoDB原生Python驱动，实现数据库连接和操作
# ASCENDING：表示升序排序，用于MongoDB索引和查询排序
from pymongo import MongoClient, ASCENDING
# 导入bson的ObjectId：MongoDB默认的主键类型，用于唯一标识文档
from bson import ObjectId, json_util
# 导入dotenv模块：用于从.env文件加载环境变量，避免硬编码敏感配置（如MongoDB连接地址）
from dotenv import load_dotenv
from sympy.codegen.fnodes import use_rename
from app.core.tenant_scope import normalize_kb_id, normalize_tenant_id
from app.observability.events import emit_event
from app.observability.metrics import record_mongo_operation
from app.observability.replay_context import get_replay_context, is_dry_run

# 加载.env文件中的环境变量，使os.getenv能读取到配置
load_dotenv()


class HistoryMongoTool:
    """
    MongoDB 历史对话记录读写工具类 (基于原生 PyMongo 实现)
    核心功能：封装MongoDB的连接、集合初始化、索引创建，为上层提供统一的数据库操作入口
    扩展功能：支持与LangChain消息对象的格式转换（原代码预留能力）
    """

    def __init__(self):
        """
        类初始化方法：完成MongoDB的连接、数据库/集合获取、索引创建
        初始化失败会抛出异常并记录错误日志，确保程序感知连接问题
        """
        try:
            # 从环境变量读取MongoDB连接地址（敏感配置，不硬编码）
            self.mongo_url = os.getenv("MONGO_URL")
            # 从环境变量读取要使用的数据库名称
            self.db_name = os.getenv("MONGO_DB_NAME")

            self.username = os.getenv("MONGO_DB_USERNAME")
            self.password = os.getenv("MONGO_DB_PASSWORD")


            # 创建MongoDB客户端实例，建立与数据库的连接
            self.client = MongoClient(self.mongo_url, username=self.username, password=self.password, authSource="admin")
            self.client.admin.command("ping")
            # 获取指定名称的数据库对象
            self.db = self.client[self.db_name]
            # 获取对话记录的集合（相当于关系型数据库的表），集合名：chat_message
            self.chat_message = self.db["chat_message"]

            # 为chat_message集合创建复合索引，提升查询性能
            # 索引规则：session_id升序 + ts降序，适配"按会话查最新记录"的核心查询场景
            # create_index自带幂等性：索引已存在时不会重复创建，无需额外判断
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])
            self.chat_message.create_index([("tenant_id", 1), ("kb_id", 1), ("session_id", 1), ("ts", -1)])

            # 记录成功日志，确认数据库连接和初始化完成
            logging.info(f"Successfully connected to MongoDB: {self.db_name}")
        except Exception as e:
            # 捕获所有初始化异常，记录详细错误日志
            logging.error(f"Failed to connect to MongoDB: {e}")
            # 重新抛出异常，让调用方感知初始化失败，避免使用未初始化的实例
            raise


# 定义全局变量：存储HistoryMongoTool的单例实例
# 作用：避免多次创建HistoryMongoTool实例，从而避免重复建立MongoDB连接
_history_mongo_tool = None
# 模块加载时尝试初始化单例实例，实现预加载
# 目的：将数据库连接的初始化提前到模块加载阶段，避免第一次调用接口时才建立连接（提升首次响应速度）
try:
    _history_mongo_tool = HistoryMongoTool()
except Exception as e:
    # 初始化失败时仅记录警告日志，不抛出异常
    # 原因：模块加载阶段的异常可能导致整个程序启动失败，此处保留懒加载兜底（get_history_mongo_tool会再次尝试创建）
    logging.warning(f"Could not initialize HistoryMongoTool on module load: {e}")


def _record_chat_message_operation(operation: str, status: str, start: float) -> None:
    """记录 chat_message 集合的 MongoDB 操作指标。

    Args:
        operation: 操作名称。
        status: 操作状态。
        start: 操作开始时的 perf_counter。

    Returns:
        None.
    """
    record_mongo_operation(operation, "chat_message", status, time.perf_counter() - start)


def get_history_mongo_tool() -> HistoryMongoTool:
    """获取 HistoryMongoTool 的单例实例。

    Returns:
        HistoryMongoTool 的单例实例。
    """
    # 声明使用全局变量，避免函数内视为局部变量
    global _history_mongo_tool
    # 懒加载：仅当全局实例为空时，才创建新的实例
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    # 返回单例实例
    return _history_mongo_tool


def clear_history(session_id: str, tenant_id: str = "default", kb_id: str = "default") -> int:
    """清空指定会话的所有历史对话记录。

    Args:
        session_id: 会话唯一标识。
        tenant_id: 租户标识。
        kb_id: 知识库标识。

    Returns:
        实际删除的文档数量，删除失败返回 0。
    """
    if is_dry_run():
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        replay_context = get_replay_context()
        emit_event(
            "dry_run.mongo.clear_history_skipped",
            graph="query",
            session_id=session_id,
            status="ok",
            summary={"tenant_id": tenant_id, "kb_id": kb_id, "replay_id": replay_context.get("replay_id")},
            tags=["dry_run", "mongo", "clear_history"],
        )
        return 0

    # 获取全局的HistoryMongoTool实例，使用单例模式避免重复创建数据库连接
    mongo_tool = get_history_mongo_tool()
    tenant_id = normalize_tenant_id(tenant_id)
    kb_id = normalize_kb_id(kb_id)
    start = time.perf_counter()
    status = "ok"
    try:
        # 执行批量删除操作：删除所有session_id匹配的文档
        # 历史删除必须带租户和知识库范围，避免同名 session 跨租户误删。
        result = mongo_tool.chat_message.delete_many(
            {"session_id": session_id, "tenant_id": tenant_id, "kb_id": kb_id}
        )
        # 记录删除成功日志，包含删除数量和会话ID，便于问题排查
        logging.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        # 返回实际删除的数量（delete_many的返回对象包含deleted_count属性）
        return result.deleted_count
    except Exception as e:
        status = "error"
        # 捕获删除异常，记录错误日志，包含会话ID
        logging.error(f"Error clearing history for session {session_id}: {e}")
        # 异常时返回0，标识删除失败
        return 0
    finally:
        _record_chat_message_operation("delete_many", status, start)


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        image_urls: List[str] = None,
        message_id: str = None,
        tenant_id: str = "default",
        kb_id: str = "default",
        user_id: str = "anonymous",
) -> str:
    """写入或更新单条会话记录。

    Args:
        session_id: 会话唯一标识。
        role: 消息角色，固定值为 user 或 assistant。
        text: 对话核心内容。
        rewritten_query: 重写后的查询语句。
        item_names: 关联的商品名称列表。
        image_urls: 关联的图片 URL 列表。
        message_id: 记录主键 ID；有值时更新已有记录。
        tenant_id: 租户标识。
        kb_id: 知识库标识。
        user_id: 用户标识。

    Returns:
        插入或更新的记录唯一标识。

    Raises:
        Exception: MongoDB 写入或更新失败时抛出原始异常。
    """
    if is_dry_run():
        tenant_id = normalize_tenant_id(tenant_id)
        kb_id = normalize_kb_id(kb_id)
        replay_context = get_replay_context()
        emit_event(
            "dry_run.mongo.save_chat_message_skipped",
            graph="query",
            session_id=session_id,
            status="ok",
            summary={
                "role": role,
                "text_chars": len(text or ""),
                "rewritten_query_chars": len(rewritten_query or ""),
                "item_names_count": len(item_names or []),
                "image_urls_count": len(image_urls or []),
                "message_id": message_id or "",
                "tenant_id": tenant_id,
                "kb_id": kb_id,
                "user_id": user_id,
                "replay_id": replay_context.get("replay_id"),
            },
            tags=["dry_run", "mongo", "save_chat_message"],
        )
        return message_id or "dry_run"

    # 生成当前时间的时间戳（秒级），用于记录消息的创建时间，后续用于排序和查询
    ts = datetime.now().timestamp()
    tenant_id = normalize_tenant_id(tenant_id)
    kb_id = normalize_kb_id(kb_id)

    # 构造要插入/更新的文档数据（MongoDB的基本数据单元是文档，类似Python字典）
    document = {
        "session_id": session_id,  # 会话ID，关联维度
        "tenant_id": tenant_id,  # 租户ID，隔离维度
        "kb_id": kb_id,  # 知识库ID，隔离维度
        "user_id": str(user_id or "anonymous"),  # 用户ID，审计维度
        "role": role,  # 消息角色
        "text": text,  # 消息内容
        "rewritten_query": rewritten_query or "",  # 重写查询，空值处理为空字符串
        "item_names": item_names,  # 关联商品名称列表
        "image_urls": image_urls,  # 关联图片URL列表
        "ts": ts  # 时间戳，排序和时间筛选维度
    }

    # 获取全局的HistoryMongoTool实例，使用单例模式
    mongo_tool = get_history_mongo_tool()
    # 判断是否传入主键ID，区分更新/新增逻辑
    start = time.perf_counter()
    status = "ok"
    operation = "update_one" if message_id else "insert_one"
    try:
        if message_id:
        # 有message_id：执行更新操作（根据主键更新）
            result = mongo_tool.chat_message.update_one(
                {"_id": ObjectId(message_id)},  # 更新条件：主键匹配（需将字符串转为ObjectId类型）
                {"$set": document}  # 更新操作：$set表示只更新指定字段，保留其他字段
            )
            # 更新操作返回传入的message_id作为标识
            return message_id
        # 无message_id：执行新增操作
        result = mongo_tool.chat_message.insert_one(document)
        # 新增操作返回插入的ObjectId并转为字符串，便于上层使用（避免直接返回ObjectId对象）
        return str(result.inserted_id)
    except Exception:
        status = "error"
        raise
    finally:
        _record_chat_message_operation(operation, status, start)


def update_message_item_names(
        ids: List[str],
        item_names: List[str],
        tenant_id: str = "default",
        kb_id: str = "default",
) -> int:
    """批量更新历史会话记录的关联商品名称。

    Args:
        ids: 要更新的记录主键 ID 列表。
        item_names: 要设置的新商品名称列表。
        tenant_id: 租户标识。
        kb_id: 知识库标识。

    Returns:
        实际更新的文档数量，更新失败返回 0。
    """
    if is_dry_run():
        replay_context = get_replay_context()
        emit_event(
            "dry_run.mongo.update_item_names_skipped",
            graph="query",
            status="ok",
            summary={
                "ids_count": len(ids or []),
                "item_names_count": len(item_names or []),
                "replay_id": replay_context.get("replay_id"),
            },
            tags=["dry_run", "mongo", "update_item_names"],
        )
        return 0

    # 获取全局的HistoryMongoTool实例，使用单例模式
    mongo_tool = get_history_mongo_tool()
    tenant_id = normalize_tenant_id(tenant_id)
    kb_id = normalize_kb_id(kb_id)
    start = time.perf_counter()
    status = "ok"
    try:
        # 将字符串类型的主键列表转为MongoDB的ObjectId类型（数据库中主键是ObjectId类型）
        object_ids = [ObjectId(i) for i in ids]
        # 执行批量更新操作
        result = mongo_tool.chat_message.update_many(
            # 更新条件：复合条件，同时满足
            {
                "_id": {"$in": object_ids},  # 主键在指定的ID列表中（批量筛选）
                "tenant_id": tenant_id,
                "kb_id": kb_id,
            },
            {"$set": {"item_names": item_names}}  # 更新操作：设置新的商品名称列表
        )
        # 记录更新成功日志，包含更新数量和新的商品名称
        logging.info(f"Updated {result.modified_count} records to item_names: {item_names}")
        # 返回实际更新的数量（modified_count：真正被修改的文档数，区别于matched_count）
        return result.modified_count
    except Exception as e:
        status = "error"
        # 捕获批量更新异常，记录错误日志
        logging.error(f"Error updating history item_names: {e}")
        # 异常时返回0，标识更新失败
        return 0
    finally:
        _record_chat_message_operation("update_many", status, start)


def get_recent_messages(
        session_id: str,
        limit: int = 10,
        tenant_id: str = "default",
        kb_id: str = "default",
) -> List[Dict[str, Any]]:
    """查询指定会话的最近对话记录。

    Args:
        session_id: 会话唯一标识。
        limit: 返回条数上限。
        tenant_id: 租户标识。
        kb_id: 知识库标识。

    Returns:
        对话记录列表，查询失败返回空列表。
    """
    # 获取全局的HistoryMongoTool实例，使用单例模式
    mongo_tool = get_history_mongo_tool()
    tenant_id = normalize_tenant_id(tenant_id)
    kb_id = normalize_kb_id(kb_id)
    start = time.perf_counter()
    status = "ok"
    try:
        # 构造查询条件：仅查询指定session_id的记录
        query = {"session_id": session_id, "tenant_id": tenant_id, "kb_id": kb_id}

        # 执行查询：按时间戳升序排序，限制返回条数
        # find(query)：获取符合条件的游标（惰性加载，不立即查询）
        # sort("ts", ASCENDING)：按ts字段升序（从旧到新），适配LLM上下文顺序
        # limit(limit)：限制返回的最大条数
        cursor = mongo_tool.chat_message.find(query).sort("ts", ASCENDING).limit(limit)
        # 将游标转为列表，触发实际数据库查询，获取所有符合条件的文档
        # return list(cursor)
        json_str = json_util.dumps(list(cursor))
        return json.loads(json_str)
    except Exception as e:
        status = "error"
        # 捕获查询异常，记录错误日志
        logging.error(f"Error getting recent messages: {e}")
        # 异常时返回空列表，避免上层处理None报错
        return []
    finally:
        _record_chat_message_operation("find", status, start)


# 主程序入口：仅当直接运行该脚本时执行，用于简单的功能测试
if __name__ == "__main__":
    # 简单测试代码：验证数据库的写入和查询功能是否正常
    # 测试会话ID，用于标识测试的对话记录
    sid = "000015_hybrid"
    # 1. 写入用户消息（手动指定ts=1000，便于测试排序）
    save_chat_message(sid, "user", "你好 (Hybrid)")
    # 2. 写入助手回复（手动指定ts=1001，按时间顺序紧跟用户消息）
    save_chat_message(sid, "assistant", "你好！我是基于原生 Mongo + LangChain 对象的助手。")
    # 3. 写入带关联商品的用户消息（手动指定ts=1002，测试item_names字段）
    save_chat_message(sid, "user", "这个万用表怎么换电池？", item_names=["混合万用表"])

    # 4. 查询指定会话的最近5条记录，验证查询功能
    print("--- 查询 LangChain 对象记录 ---")
    messages = get_recent_messages(sid, limit=5)
    # 打印查询到的记录数量
    print(f"查询到的记录数: {len(messages)}")
    # 遍历打印每条记录的详细内容
    for m in messages:
        print(f" {m}  ")
