
# kbase-rag

基于 RAG（检索增强生成）技术的智能问答系统，支持文档导入、向量化存储和智能问答功能。

## 项目简介

本项目是一个完整的 AI 问答系统，集成了文档处理、向量检索、知识图谱查询和大语言模型回答生成等功能。主要包括两大核心模块：

- **导入模块 (import_process)**：支持 PDF 文档解析、Markdown 处理、图像提取、向量 embedding 并存储到 Milvus 向量数据库
- **查询模块 (query_process)**：支持混合检索（向量检索 + 网络搜索）、重排序、知识图谱查询和 LLM 回答生成

## 技术栈

- **后端框架**：FastAPI
- **向量数据库**：Milvus
- **文件存储**：MinIO
- **历史记录**：MongoDB
- **知识图谱**：Neo4j
- **Embedding 模型**：BGE-M3
- **重排序模型**：BGE-Reranker
- **LLM**：支持多种大语言模型接入

## 项目结构

```
app/
├── clients/              # 各种客户端工具
│   ├── milvus_utils.py  # Milvus 向量数据库工具
│   ├── minio_utils.py   # MinIO 文件存储工具
│   ├── mongo_history_utils.py    # MongoDB 历史记录工具
│   └── neo4j_utils.py   # Neo4j 知识图谱工具
├── conf/                # 配置模块
│   ├── bailian_mcp_config.py
│   ├── embedding_config.py
│   ├── lm_config.py
│   ├── milvus_config.py
│   ├── mineru_config.py
│   ├── minio_config.py
│   └── reranker_config.py
├── core/                # 核心功能
│   ├── load_prompt.py   # 提示词加载
│   ├── logger.py        # 日志模块
│   └── tracing.py       # 链路追踪
├── import_process/      # 文档导入模块
│   ├── agent/           # 导入 Agent
│   │   ├── nodes/       # 各个处理节点
│   │   │   ├── node_bge_embedding.py
│   │   │   ├── node_document_split.py
│   │   │   ├── node_entry.py
│   │   │   ├── node_import_milvus.py
│   │   │   ├── node_item_name_recognition.py
│   │   │   ├── node_md_img.py
│   │   │   └── node_pdf_to_md.py
│   │   └── state.py     # 导入状态定义
│   └── api/             # 导入 API
├── lm/                  # 语言模型工具
│   ├── embedding_utils.py
│   ├── lm_utils.py
│   └── reranker_utils.py
├── query_process/       # 查询问答模块
│   ├── agent/           # 查询 Agent
│   │   ├── nodes/       # 各个处理节点
│   │   │   ├── node_answer_output.py
│   │   │   ├── node_item_name_confirm.py
│   │   │   ├── node_query_kg.py
│   │   │   ├── node_rerank.py
│   │   │   ├── node_rrf.py
│   │   │   ├── node_search_embedding.py
│   │   │   └── node_search_embedding_hyde.py
│   │   └── state.py     # 查询状态定义
│   ├── api/             # 查询 API
│   └── sse/             # SSE 流式响应
└── utils/               # 工具模块
    ├── sse_utils.py     # SSE 工具
    └── task_utils.py    # 任务工具
```

## 功能特性

### 文档导入

- PDF 文档解析转换为 Markdown
- 文档智能分块
- BGE-M3 向量化（支持稠密向量和稀疏向量）
- 商品名称识别
- 图片提取和描述生成
- Milvus 向量数据库存储

### 智能问答

- 混合检索（RRF 融合向量检索和网络搜索）
- 商品名称确认
- HyDE 检索增强
- 知识图谱查询
- 重排序优化
- 流式响应输出
- 对话历史管理

## 快速开始

### 环境要求

- Python 3.10+
- Milvus 向量数据库
- MongoDB
- MinIO
- Neo4j（可选）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

创建 `.env` 文件，配置以下环境变量：

```env
# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# MongoDB 配置
MONGODB_URL=mongodb://localhost:27017

# MinIO 配置
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=ai-project

# LLM 配置
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1

# 其他配置
...
```

### 生产环境 LangGraph Checkpoint

生产环境建议使用 Postgres 持久化 LangGraph checkpoint，避免服务重启后丢失查询、导入和回放状态。

```env
LANGGRAPH_CHECKPOINT_BACKEND=postgres
LANGGRAPH_CHECKPOINT_NAMESPACE=kbase-rag
LANGGRAPH_CHECKPOINT_POSTGRES_DSN=postgresql://dataset_rag:your-password@postgres:5432/dataset_rag_observability
LANGGRAPH_CHECKPOINT_POSTGRES_SETUP=true
LANGGRAPH_CHECKPOINT_POSTGRES_POOL_MIN=1
LANGGRAPH_CHECKPOINT_POSTGRES_POOL_MAX=10
```

本地开发可以继续使用内存后端：

```env
LANGGRAPH_CHECKPOINT_BACKEND=memory
```

生产环境如果配置为 `postgres`，Postgres 初始化失败会阻止服务启动，避免误以为 checkpoint 已持久化。

### 生产回放安全配置

```env
OBSERVABILITY_SNAPSHOT_LEVEL=safe
OBSERVABILITY_REPLAY_ALLOW_LIVE=false
LANGGRAPH_CHECKPOINT_RETENTION_DAYS=30
```

快照级别：

- `safe`：默认生产策略，长文本脱敏/截断，向量省略。
- `debug`：受控调试环境保存更多文本，便于复现。
- `full`：完整保存白名单字段，仅限隔离环境。
- `off`：只写事件，不保存快照 payload。

常用调试接口：

```text
POST /debug/replay/failed
POST /debug/snapshots/diff
GET /debug/checkpoints/{thread_id}/audit
```

排障时建议按这个顺序走：

1. 从告警里的 `dashboard_url` 打开 Grafana，看 `Dataset RAG Overview` 是否已经指向查询、导入或依赖链路。
2. 用 `trace_id`、`session_id` 或 `task_id` 打开对应的 `/debug/.../timeline`。
3. 定位到异常节点后，使用 `/debug/replay/node`、`/debug/replay/failed` 或 `/debug/replay/latest-failed` 复现。
4. 如果需要比较回放前后的输入输出，继续看 `/debug/snapshots/diff` 和 `/debug/events/{event_id}`。

稳定性自动化：

```env
OBSERVABILITY_AUTOMATION_ENABLED=true
OBSERVABILITY_AUTOMATION_INTERVAL_SECONDS=60
OBSERVABILITY_QUERY_STUCK_SECONDS=600
OBSERVABILITY_IMPORT_STUCK_SECONDS=1800
OBSERVABILITY_CAPACITY_WARNING_PERCENT=85
OBSERVABILITY_CAPACITY_CRITICAL_PERCENT=95
```

JSONL fallback 回灌 Mongo：

```bash
python -m app.observability.jsonl_backfill
python -m app.observability.jsonl_backfill --write
```

告警规则校验：

```bash
python deploy/observability/validate_prometheus_rules.py
```

### 启动服务

```bash
python main.py
```

服务启动后访问：
- 导入页面：`http://localhost:8000/import`
- 问答页面：`http://localhost:8000/chat`

## API 接口

### 导入接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/import` | GET | 导入页面 |
| `/upload` | POST | 上传文件 |
| `/status/{task_id}` | GET | 任务状态查询 |

### 查询接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/chat` | GET | 问答页面 |
| `/query` | POST | 提交问答请求 |
| `/stream/{session_id}` | GET | 流式获取回答 |
| `/history/{session_id}` | GET | 获取历史记录 |
| `/history/{session_id}` | DELETE | 删除历史记录 |

## 开发说明

### 添加新的导入节点

1. 在 `app/import_process/agent/nodes/` 目录下创建新的节点文件
2. 实现节点函数，接收 `ImportGraphState` 并返回更新后的状态
3. 在 `app/import_process/agent/main_graph.py` 中注册节点

### 添加新的查询节点

1. 在 `app/query_process/agent/nodes/` 目录下创建新的节点文件
2. 实现节点函数，接收 `QueryGraphState` 并返回更新后的状态
3. 在 `app/query_process/agent/main_graph.py` 中注册节点

## 许可证

MIT License
