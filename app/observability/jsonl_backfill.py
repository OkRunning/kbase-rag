"""提供 observability JSONL fallback 回灌 MongoDB 的工具。"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection

from app.core.logger import logger
from app.observability.metrics import record_event_store_write, record_mongo_operation
from app.observability.sanitizers import sanitize_summary
from app.utils.path_util import PROJECT_ROOT

load_dotenv()

_COLLECTION_NAME = "observability_events"
_DEFAULT_JSONL_PATH = PROJECT_ROOT / "logs" / "observability_events.jsonl"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """逐行读取 JSONL 事件。

    Args:
        path: JSONL 文件路径。

    Yields:
        事件文档。
    """
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("跳过无法解析的 observability JSONL 行")
                continue
            if isinstance(payload, dict):
                yield payload


def _mongo_collection() -> Collection:
    """创建 MongoDB 事件集合连接。

    Returns:
        MongoDB collection。

    Raises:
        RuntimeError: Mongo 配置缺失时抛出。
    """
    mongo_url = os.getenv("MONGO_URL")
    db_name = os.getenv("MONGO_DB_NAME")
    if not mongo_url or not db_name:
        raise RuntimeError("MONGO_URL and MONGO_DB_NAME are required for observability event backfill")
    client = MongoClient(
        mongo_url,
        username=os.getenv("MONGO_DB_USERNAME") or None,
        password=os.getenv("MONGO_DB_PASSWORD") or None,
        authSource="admin",
        serverSelectionTimeoutMS=2000,
    )
    client.admin.command("ping")
    return client[db_name][_COLLECTION_NAME]


def _normalize_event(document: Dict[str, Any]) -> Dict[str, Any]:
    """规范化待回灌事件。

    Args:
        document: 原始事件文档。

    Returns:
        可写入 MongoDB 的事件文档。
    """
    payload = dict(document)
    if payload.get("summary") is not None:
        payload["summary"] = sanitize_summary(payload.get("summary"))
    if payload.get("tags") is not None:
        payload["tags"] = sanitize_summary(payload.get("tags"))
    return payload


def backfill_events_jsonl_to_mongo(
    path: Optional[Path] = None,
    dry_run: bool = True,
    batch_size: int = 500,
    limit: Optional[int] = None,
) -> Dict[str, int]:
    """将本地 observability JSONL fallback 事件回灌 MongoDB。

    Args:
        path: JSONL 文件路径，默认读取项目 logs 下的 fallback 文件。
        dry_run: 是否只统计不写入。
        batch_size: Mongo 批量 upsert 大小。
        limit: 最多处理的事件条数。

    Returns:
        回灌统计信息。

    Raises:
        FileNotFoundError: JSONL 文件不存在时抛出。
        RuntimeError: Mongo 配置缺失或不可用时抛出。
    """
    jsonl_path = path or _DEFAULT_JSONL_PATH
    if not jsonl_path.exists():
        raise FileNotFoundError(f"observability JSONL fallback not found: {jsonl_path}")

    collection = None if dry_run else _mongo_collection()
    processed = 0
    written = 0
    skipped = 0
    operations = []
    start = time.perf_counter()
    status = "ok"
    try:
        for document in _iter_jsonl(jsonl_path):
            if limit is not None and processed >= limit:
                break
            processed += 1
            payload = _normalize_event(document)
            event_id = payload.get("_id")
            if not event_id:
                skipped += 1
                continue
            if dry_run:
                continue
            operations.append(UpdateOne({"_id": event_id}, {"$setOnInsert": payload}, upsert=True))
            if len(operations) >= batch_size:
                result = collection.bulk_write(operations, ordered=False)
                written += result.upserted_count
                operations = []
        if operations and collection is not None:
            result = collection.bulk_write(operations, ordered=False)
            written += result.upserted_count
        record_event_store_write("mongo_backfill", "dry_run" if dry_run else "ok")
    except Exception:
        status = "error"
        record_event_store_write("mongo_backfill", "error")
        raise
    finally:
        if not dry_run:
            record_mongo_operation("bulk_write", _COLLECTION_NAME, status, time.perf_counter() - start)

    return {
        "processed": processed,
        "written": written,
        "skipped": skipped,
        "dry_run": int(dry_run),
    }


def main() -> None:
    """执行 JSONL fallback 回灌命令行工具。

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Backfill observability JSONL fallback events into MongoDB.")
    parser.add_argument("--path", default=str(_DEFAULT_JSONL_PATH), help="JSONL fallback file path.")
    parser.add_argument("--write", action="store_true", help="Actually write to MongoDB. Default is dry-run.")
    parser.add_argument("--batch-size", type=int, default=500, help="Mongo bulk write batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of events to process.")
    args = parser.parse_args()
    result = backfill_events_jsonl_to_mongo(
        path=Path(args.path),
        dry_run=not args.write,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
