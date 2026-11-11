# AI Project

An intelligent Q&A system based on RAG (Retrieval-Augmented Generation) technology, supporting document import, vectorized storage, and intelligent question answering.

## Project Overview

This project is a complete AI Q&A system integrating document processing, vector retrieval, knowledge graph querying, and large language model (LLM) response generation. It primarily consists of two core modules:

- **Import Module (import_process)**: Supports PDF document parsing, Markdown processing, image extraction, vector embedding, and storage into the Milvus vector database.
- **Query Module (query_process)**: Supports hybrid retrieval (vector search + web search), re-ranking, knowledge graph querying, and LLM response generation.

## Technology Stack

- **Backend Framework**: FastAPI
- **Vector Database**: Milvus
- **File Storage**: MinIO
- **History Storage**: MongoDB
- **Knowledge Graph**: Neo4j
- **Embedding Model**: BGE-M3
- **Re-ranking Model**: BGE-Reranker
- **LLM**: Supports integration with multiple large language models

## Project Structure

```
app/
├── clients/              # Various client utilities
│   ├── milvus_utils.py  # Milvus vector database utilities
│   ├── minio_utils.py   # MinIO file storage utilities
│   ├── mongo_history_utils.py    # MongoDB history utilities
│   └── neo4j_utils.py   # Neo4j knowledge graph utilities
├── conf/                # Configuration module
│   ├── bailian_mcp_config.py
│   ├── embedding_config.py
│   ├── lm_config.py
│   ├── milvus_config.py
│   ├── mineru_config.py
│   ├── minio_config.py
│   └── reranker_config.py
├── core/                # Core functionality
│   ├── load_prompt.py   # Prompt loading
│   ├── logger.py        # Logging module
│   └── tracing.py       # Tracing
├── import_process/      # Document import module
│   ├── agent/           # Import Agent
│   │   ├── nodes/       # Processing nodes
│   │   │   ├── node_bge_embedding.py
│   │   │   ├── node_document_split.py
│   │   │   ├── node_entry.py
│   │   │   ├── node_import_milvus.py
│   │   │   ├── node_item_name_recognition.py
│   │   │   ├── node_md_img.py
│   │   │   └── node_pdf_to_md.py
│   │   └── state.py     # Import state definition
│   └── api/             # Import API
├── lm/                  # Language model utilities
│   ├── embedding_utils.py
│   ├── lm_utils.py
│   └── reranker_utils.py
├── query_process/       # Query and answering module
│   ├── agent/           # Query Agent
│   │   ├── nodes/       # Processing nodes
│   │   │   ├── node_answer_output.py
│   │   │   ├── node_item_name_confirm.py
│   │   │   ├── node_query_kg.py
│   │   │   ├── node_rerank.py
│   │   │   ├── node_rrf.py
│   │   │   ├── node_search_embedding.py
│   │   │   └── node_search_embedding_hyde.py
│   │   └── state.py     # Query state definition
│   ├── api/             # Query API
│   └── sse/             # SSE streaming responses
└── utils/               # Utility modules
    ├── sse_utils.py     # SSE utilities
    └── task_utils.py    # Task utilities
```

## Features

### Document Import

- Parse PDF documents and convert to Markdown
- Intelligent document chunking
- Vectorization with BGE-M3 (supports dense and sparse vectors)
- Product name recognition
- Image extraction and description generation
- Storage in Milvus vector database

### Intelligent Q&A

- Hybrid retrieval (RRF fusion of vector search and web search)
- Product name confirmation
- HyDE retrieval enhancement
- Knowledge graph querying
- Re-ranking optimization
- Streaming response output
- Conversation history management

## Quick Start

### Prerequisites

- Python 3.10+
- Milvus vector database
- MongoDB
- MinIO
- Neo4j (optional)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file and configure the following environment variables:

```env
# Milvus configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530

# MongoDB configuration
MONGODB_URL=mongodb://localhost:27017

# MinIO configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=ai-project

# LLM configuration
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1

# Other configurations
...
```

### Production LangGraph Checkpoint

Use Postgres for LangGraph checkpoints in production so query, import, and replay states survive service restarts.

```env
LANGGRAPH_CHECKPOINT_BACKEND=postgres
LANGGRAPH_CHECKPOINT_NAMESPACE=kbase-rag
LANGGRAPH_CHECKPOINT_POSTGRES_DSN=postgresql://dataset_rag:your-password@postgres:5432/dataset_rag_observability
LANGGRAPH_CHECKPOINT_POSTGRES_SETUP=true
LANGGRAPH_CHECKPOINT_POSTGRES_POOL_MIN=1
LANGGRAPH_CHECKPOINT_POSTGRES_POOL_MAX=10
```

For local development, keep the in-memory backend:

```env
LANGGRAPH_CHECKPOINT_BACKEND=memory
```

When `postgres` is configured in production, checkpoint initialization failures stop service startup instead of silently falling back to memory.

### Production Replay Safety

```env
OBSERVABILITY_SNAPSHOT_LEVEL=safe
OBSERVABILITY_REPLAY_ALLOW_LIVE=false
LANGGRAPH_CHECKPOINT_RETENTION_DAYS=30
```

Snapshot levels:

- `safe`: Production default. Long text is sanitized/truncated and vectors are omitted.
- `debug`: Stores more text for controlled debugging.
- `full`: Stores full whitelisted fields. Use only in isolated environments.
- `off`: Writes events only and skips snapshot payloads.

Useful debug APIs:

```text
POST /debug/replay/failed
POST /debug/snapshots/diff
GET /debug/checkpoints/{thread_id}/audit
```

Recommended triage order:

1. Open the Grafana dashboard from the alert `dashboard_url`.
2. Use `trace_id`, `session_id`, or `task_id` to open the matching `/debug/.../timeline`.
3. Reproduce deterministic failures with `/debug/replay/node`, `/debug/replay/failed`, or `/debug/replay/latest-failed`.
4. Compare before/after payloads with `/debug/snapshots/diff` and inspect single events with `/debug/events/{event_id}`.

Stability automation:

```env
OBSERVABILITY_AUTOMATION_ENABLED=true
OBSERVABILITY_AUTOMATION_INTERVAL_SECONDS=60
OBSERVABILITY_QUERY_STUCK_SECONDS=600
OBSERVABILITY_IMPORT_STUCK_SECONDS=1800
OBSERVABILITY_CAPACITY_WARNING_PERCENT=85
OBSERVABILITY_CAPACITY_CRITICAL_PERCENT=95
```

Backfill JSONL fallback events into MongoDB:

```bash
python -m app.observability.jsonl_backfill
python -m app.observability.jsonl_backfill --write
```

Validate alert rules:

```bash
python deploy/observability/validate_prometheus_rules.py
```

### Start the Service

```bash
python main.py
```

After startup, access:
- Import page: `http://localhost:8000/import`
- Q&A page: `http://localhost:8000/chat`

## API Endpoints

### Import Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/import` | GET | Import page |
| `/upload` | POST | Upload file |
| `/status/{task_id}` | GET | Query task status |

### Query Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | GET | Q&A page |
| `/query` | POST | Submit Q&A request |
| `/stream/{session_id}` | GET | Stream response |
| `/history/{session_id}` | GET | Retrieve history |
| `/history/{session_id}` | DELETE | Delete history |

## Development Notes

### Adding a New Import Node

1. Create a new node file under `app/import_process/agent/nodes/`
2. Implement the node function to accept `ImportGraphState` and return the updated state
3. Register the node in `app/import_process/agent/main_graph.py`

### Adding a New Query Node

1. Create a new node file under `app/query_process/agent/nodes/`
2. Implement the node function to accept `QueryGraphState` and return the updated state
3. Register the node in `app/query_process/agent/main_graph.py`

## License

MIT License
