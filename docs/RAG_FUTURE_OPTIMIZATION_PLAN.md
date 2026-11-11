# kbase-rag 后续优化方案：效果评估、生产化能力、成本与稳定性

## 1. 背景

当前项目已经完成了基于 FastAPI + LangGraph 的 RAG 查询链路和文档导入链路，并围绕 OpenTelemetry、Prometheus、Alertmanager、Grafana、Loki、Tempo、事件时间线、节点快照和回放能力完成了可观测与告警改造。

系统已经具备从“能跑通 RAG 流程”到“能定位复杂链路问题”的基础。但如果继续向生产级大模型应用演进，还需要补齐三个核心方向：

1. 效果评估：建立可持续的 RAG 评测体系，避免只靠人工主观判断效果。
2. 生产化能力：支持权限隔离、多租户知识库、增量更新、版本管理等真实业务场景。
3. 成本和稳定性优化：控制 LLM/token/embedding 成本，增强降级、缓存、异步队列和 Runbook。

本文基于当前代码结构提出后续优化方案，尽量保持小范围演进，不做无关重构。

## 2. 总体目标

### 2.1 产品目标

- 能量化回答质量、召回质量和引用质量。
- 支持不同用户、团队、知识库之间的数据隔离。
- 支持文档增量导入、更新、删除和版本回溯。
- 支持 Prompt、检索策略、模型配置的可控发布和回滚。
- 在成本、延迟、可用性之间建立可观测、可调优的平衡。

### 2.2 工程目标

- 保持现有查询和导入 API 尽量兼容。
- 保持 LangGraph 编排模式，将新增能力拆成独立节点或工具。
- 复用现有 MongoDB、Milvus、MinIO、Postgres checkpoint 和 observability 模块。
- 所有新增指标遵循低基数标签原则。
- 所有新增配置可通过环境变量或配置表控制，支持灰度和回滚。

## 3. 目标架构概览

```text
文档导入
  -> 文档解析
  -> Chunk 生成
  -> Embedding
  -> Milvus 写入
  -> 文档/Chunk/向量版本记录
  -> 评测样本候选生成

用户查询
  -> Query Understanding
  -> Retrieval Planning
  -> 多路召回
  -> RRF / Rerank
  -> Evidence Gate
  -> Answer Generation
  -> 引用来源生成
  -> 质量事件与成本指标记录

离线评测
  -> 评测集管理
  -> 批量运行 RAG
  -> 召回率 / 答案准确率 / 引用准确性计算
  -> 版本对比
  -> 发布门禁
```

## 4. 方向一：RAG 效果评估体系

### 4.1 当前问题

当前项目已经具备多路召回、RRF、Rerank 和 LLM 生成能力，但效果判断主要依赖人工体验，存在以下问题：

- 检索是否召回正确文档不可量化。
- Rerank 是否真的提升结果质量不可量化。
- LLM 答案是否基于证据、是否存在幻觉不可量化。
- Prompt、模型、chunk 参数调整后缺少前后对比。
- Web Search、HyDE、普通向量检索各自贡献度不清晰。
- 缺少上线前自动化质量门禁。

### 4.2 评测目标

建立一套覆盖检索、重排、答案、引用、稳定性的评测体系。

| 评测层级 | 目标 | 示例指标 |
| --- | --- | --- |
| Query Understanding | 判断意图、实体、查询改写是否正确 | intent accuracy、entity F1 |
| Retrieval | 判断是否召回目标证据 | Recall@K、Hit@K、MRR、NDCG |
| Rerank | 判断排序是否更合理 | MRR@K、NDCG@K、Top1 accuracy |
| Answer | 判断答案是否正确、完整、无幻觉 | correctness、faithfulness、completeness |
| Citation | 判断引用是否支撑答案 | citation precision、citation recall |
| Regression | 判断改动是否引入退化 | pass rate、关键样本失败数 |

### 4.3 评测集设计

建议新增 MongoDB 集合或本地 JSONL 文件管理评测集。

#### 4.3.1 评测样本结构

```json
{
  "case_id": "qa_0001",
  "tenant_id": "default",
  "kb_id": "product_manual",
  "question": "HAK 180 烫金机如何设置局部烫金区域？",
  "query_type": "procedure",
  "expected_answer": "需要先进入区域设置页面，选择局部烫金模式，然后配置起止位置...",
  "expected_doc_ids": ["doc_hak_180_manual"],
  "expected_chunk_ids": ["chunk_001", "chunk_002"],
  "required_keywords": ["局部烫金", "起止位置", "区域设置"],
  "forbidden_keywords": ["无法确定", "请联系销售"],
  "difficulty": "medium",
  "tags": ["product", "procedure", "local_retrieval"],
  "created_at": "2026-07-24T00:00:00+08:00"
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `case_id` | 评测样本唯一标识。 |
| `tenant_id` | 多租户场景下的数据隔离标识。 |
| `kb_id` | 知识库标识。 |
| `question` | 用户问题。 |
| `query_type` | 问题类型，例如事实、步骤、对比、故障诊断。 |
| `expected_answer` | 参考答案，不要求模型逐字一致。 |
| `expected_doc_ids` | 必须召回的文档。 |
| `expected_chunk_ids` | 必须召回的 chunk。 |
| `required_keywords` | 答案中期望覆盖的关键概念。 |
| `forbidden_keywords` | 答案中不应出现的错误内容。 |
| `difficulty` | 难度分层，用于观察复杂问题退化。 |
| `tags` | 样本标签，用于分组评估。 |

#### 4.3.2 样本来源

1. 从真实业务问题中脱敏沉淀。
2. 从产品手册、FAQ、操作文档中人工构造。
3. 从失败查询事件中转化，例如召回为空、低置信度、用户追问。
4. 使用 LLM 辅助生成候选问题，但必须人工审核标准答案和证据。

### 4.4 检索评估

#### 4.4.1 Recall@K

判断 Top K 召回结果中是否包含标准证据。

```text
Recall@K = 命中的 expected_chunk_ids 数量 / expected_chunk_ids 总数
```

适用场景：

- 判断 embedding、HyDE、Web Search、RRF 的召回能力。
- 对比 chunk size、overlap、embedding 模型、Milvus 检索参数。

#### 4.4.2 Hit@K

判断 Top K 中是否至少命中一个目标 chunk。

```text
Hit@K = Top K 中命中任意 expected_chunk_id 的样本数 / 总样本数
```

适用场景：

- 快速观察检索链路是否可用。
- 作为上线前最低门槛。

#### 4.4.3 MRR

关注第一个正确证据出现的位置。

```text
MRR = 平均 1 / 第一个正确 chunk 的排名
```

适用场景：

- 评估 Rerank 是否把正确证据排到前面。
- 评估最终喂给 LLM 的上下文质量。

### 4.5 答案评估

答案评估建议采用“规则评估 + LLM-as-Judge + 人工抽检”的组合。

#### 4.5.1 规则评估

规则评估适合稳定、低成本、可自动化的场景。

| 规则 | 说明 |
| --- | --- |
| required keyword 命中 | 答案必须覆盖关键概念。 |
| forbidden keyword 检测 | 答案不应包含明确错误说法。 |
| 引用数量检查 | 答案必须包含至少一个 source。 |
| 拒答检查 | 证据不足时应拒答，而不是编造。 |
| 格式检查 | JSON 输出、引用结构、字段完整性。 |

#### 4.5.2 LLM-as-Judge

对开放式问答，使用评审模型从多个维度打分。

建议评分维度：

| 维度 | 说明 |
| --- | --- |
| correctness | 答案与参考答案是否一致。 |
| faithfulness | 答案是否被检索证据支撑。 |
| completeness | 是否覆盖关键步骤和限制条件。 |
| helpfulness | 回答是否对用户可执行。 |
| conciseness | 是否简洁且没有无关扩展。 |

评审输入建议包含：

```json
{
  "question": "用户问题",
  "reference_answer": "参考答案",
  "retrieved_contexts": ["上下文1", "上下文2"],
  "model_answer": "模型回答",
  "citations": ["chunk_001", "chunk_002"]
}
```

评审输出建议结构化：

```json
{
  "correctness": 4,
  "faithfulness": 5,
  "completeness": 4,
  "citation_supported": true,
  "hallucination": false,
  "reason": "答案覆盖了局部烫金区域设置步骤，引用证据能够支撑主要结论。"
}
```

### 4.6 引用准确性评估

引用不是“有 source 就行”，还要判断 source 是否真正支撑答案。

#### 4.6.1 Citation Precision

```text
Citation Precision = 有效引用数量 / 返回引用总数
```

用于发现模型乱引用、引用弱相关文档。

#### 4.6.2 Citation Recall

```text
Citation Recall = 命中的 expected_chunk_ids 数量 / expected_chunk_ids 总数
```

用于发现答案虽然正确，但没有引用关键证据。

#### 4.6.3 Evidence Coverage

针对步骤类、对比类、故障诊断类问题，检查答案中的关键结论是否都能在引用 chunk 中找到依据。

### 4.7 评测运行流程

建议新增离线评测命令：

```bash
python -m app.evaluation.run_rag_eval --dataset data/eval/rag_cases.jsonl --profile baseline
```

运行流程：

1. 加载评测集。
2. 按样本调用查询链路，可选择 mock LLM 或真实 LLM。
3. 保存每个 case 的检索结果、Rerank 结果、答案、引用、trace_id。
4. 计算检索指标、答案指标、引用指标。
5. 和上一个 baseline 版本对比。
6. 输出 HTML/Markdown/JSON 报告。

### 4.8 评测报告结构

```text
RAG Evaluation Report
  - Summary
    - total_cases
    - pass_rate
    - recall_at_5
    - mrr_at_5
    - answer_correctness_avg
    - citation_precision
  - Breakdown
    - by query_type
    - by difficulty
    - by retrieval_strategy
  - Regression
    - newly_failed_cases
    - newly_passed_cases
    - metric_deltas
  - Failed Cases
    - question
    - expected evidence
    - actual retrieval
    - model answer
    - judge reason
```

### 4.9 与现有观测体系结合

建议新增指标：

| 指标 | 类型 | 标签 | 用途 |
| --- | --- | --- | --- |
| `rag_eval_cases_total` | Counter | `profile`, `status`, `query_type` | 评测样本通过率。 |
| `rag_eval_recall_at_k` | Histogram | `profile`, `k` | 召回质量趋势。 |
| `rag_eval_answer_score` | Histogram | `profile`, `dimension` | 答案评分趋势。 |
| `rag_eval_citation_score` | Histogram | `profile`, `dimension` | 引用质量趋势。 |

注意：`case_id` 不建议作为 Prometheus label，可写入评测报告和事件系统。

### 4.10 阶段计划

| 阶段 | 内容 | 产出 |
| --- | --- | --- |
| P0 | 建立 30-50 条核心评测样本 | `rag_cases.jsonl`、基础报告 |
| P1 | 实现检索指标 Recall@K、Hit@K、MRR | 检索评测脚本 |
| P1 | 实现答案规则评估 | keyword、引用、格式检查 |
| P2 | 接入 LLM-as-Judge | 结构化评分和失败原因 |
| P2 | 建立 baseline 对比 | 每次改 Prompt/检索策略前后对比 |
| P3 | 接入 CI 或发布门禁 | 低于阈值时阻止发布 |

### 4.11 验收标准

- 每次调整 Prompt、chunk、embedding、rerank 参数后，都能输出量化对比报告。
- 核心问题集 Recall@5 达到预设阈值，例如 85% 以上。
- 答案 faithfulness 平均分达到预设阈值，例如 4/5 以上。
- 新版本相对 baseline 的关键指标下降超过阈值时能被发现。

## 5. 方向二：生产化能力

### 5.1 当前问题

当前项目已经能支撑单知识库、单业务场景的 RAG 流程，但距离真实生产环境还缺少以下能力：

- 用户、团队、知识库之间缺少权限隔离。
- Milvus collection、Mongo 历史、MinIO 文件和事件数据缺少统一租户模型。
- 文档更新后，旧 chunk 和旧向量如何处理不清晰。
- Prompt、模型参数、检索参数没有版本管理。
- 知识库发布、回滚、灰度能力不足。
- 数据删除和权限变更后，检索结果可能残留旧数据。

### 5.2 租户与权限模型

#### 5.2.1 核心对象

| 对象 | 说明 |
| --- | --- |
| `tenant` | 租户，可以对应公司、部门或团队。 |
| `user` | 用户身份。 |
| `role` | 角色，例如 admin、editor、viewer。 |
| `kb` | 知识库。 |
| `document` | 文档。 |
| `chunk` | 文档切片。 |
| `collection` | Milvus collection 或 partition。 |
| `prompt_version` | Prompt 版本。 |
| `retrieval_profile` | 检索策略版本。 |

#### 5.2.2 权限原则

1. 所有查询必须携带 `tenant_id` 和 `user_id`。
2. 所有知识库资源必须绑定 `tenant_id`。
3. 用户只能查询自己有权限的 `kb_id`。
4. Milvus 检索必须带 metadata filter，避免跨租户召回。
5. Mongo 历史、事件、快照、评测数据都必须带租户字段。
6. 管理接口需要按角色控制导入、删除、发布、回滚能力。

#### 5.2.3 建议数据结构

```json
{
  "tenant_id": "tenant_a",
  "kb_id": "product_manual",
  "doc_id": "doc_001",
  "doc_version": 3,
  "chunk_id": "doc_001_v3_chunk_0001",
  "visibility": "private",
  "acl": {
    "users": ["u001", "u002"],
    "roles": ["viewer"]
  },
  "metadata": {
    "file_name": "HAK180_manual.pdf",
    "file_type": "pdf",
    "source": "upload",
    "created_by": "u001"
  }
}
```

### 5.3 多租户知识库设计

#### 5.3.1 Milvus 隔离方案

可选方案：

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 每租户一个 collection | 隔离强，删除简单 | collection 数量多，管理复杂 | 租户少、隔离要求高 |
| 共享 collection + tenant_id filter | 管理简单，资源利用高 | 必须确保 filter 正确 | 租户多、中小知识库 |
| 每知识库 partition | 查询过滤清晰 | partition 数量受限制 | 知识库数量可控 |

建议第一阶段采用：共享 collection + `tenant_id`、`kb_id` metadata filter。后续对高价值租户支持独立 collection。

#### 5.3.2 MongoDB 集合调整

建议所有业务集合增加字段：

```text
tenant_id
kb_id
user_id
doc_id
doc_version
created_at
updated_at
deleted_at
```

涉及范围：

- 对话历史。
- 导入任务。
- 事件时间线。
- 快照索引。
- 评测样本。
- Prompt 版本。
- 检索配置版本。

### 5.4 文档增量更新

#### 5.4.1 当前挑战

文档更新不是简单重新导入，因为旧向量可能仍然被召回，导致答案使用过期知识。

需要解决：

- 如何判断文档是否变化。
- 如何只更新变化的 chunk。
- 如何删除旧 chunk 和旧向量。
- 如何保证查询不会读到半更新状态。
- 如何支持版本回滚。

#### 5.4.2 文档版本模型

建议引入文档版本表：

```json
{
  "doc_id": "doc_001",
  "tenant_id": "tenant_a",
  "kb_id": "product_manual",
  "active_version": 3,
  "versions": [
    {
      "version": 2,
      "status": "archived",
      "content_hash": "sha256_old",
      "created_at": "2026-07-20T10:00:00+08:00"
    },
    {
      "version": 3,
      "status": "active",
      "content_hash": "sha256_new",
      "created_at": "2026-07-24T10:00:00+08:00"
    }
  ]
}
```

#### 5.4.3 增量更新流程

```text
1. 上传新文档
2. 计算文件 hash 和文本 hash
3. 如果 hash 未变化，跳过导入
4. 解析文档并生成新 chunk
5. 对 chunk 计算 chunk_hash
6. 与上一版本 chunk 对比
7. 只对新增和变化 chunk 生成 embedding
8. 写入 Milvus，标记 doc_version=new_version
9. 更新 Mongo 文档版本状态
10. 将旧版本标记为 archived 或 soft_deleted
11. 查询侧只检索 active_version
```

#### 5.4.4 删除策略

建议采用软删除 + 延迟物理删除：

- 查询侧立即过滤 `deleted_at is null` 和 `status=active`。
- Milvus 中旧向量先标记不可见。
- 后台清理任务定期物理删除旧向量和旧快照。
- 删除前保留短期回滚窗口，例如 7 天。

### 5.5 向量版本管理

向量版本不仅和文档版本有关，也和 embedding 模型、chunk 参数有关。

建议向量 metadata 增加：

```text
embedding_model
embedding_version
chunk_strategy
chunk_size
chunk_overlap
schema_version
doc_version
vector_status
```

当 embedding 模型或 chunk 策略变化时，不应该直接覆盖旧向量，而是创建新的 `index_version` 或 `vector_version`。

查询侧通过配置选择当前生效版本：

```json
{
  "kb_id": "product_manual",
  "active_vector_version": "bge_m3_v2_chunk_800_100",
  "fallback_vector_version": "bge_m3_v1_chunk_500_80"
}
```

### 5.6 Prompt 版本管理

#### 5.6.1 当前问题

Prompt 文件位于 `prompts/` 目录，适合开发阶段，但生产中还需要：

- 记录每次回答使用的 Prompt 版本。
- 支持灰度发布新 Prompt。
- 支持按问题类型选择不同 Prompt。
- 出现效果退化时可以快速回滚。

#### 5.6.2 Prompt 版本结构

```json
{
  "prompt_id": "answer_out",
  "version": "2026-07-24-001",
  "status": "active",
  "content": "Prompt 内容",
  "variables": ["question", "contexts", "history"],
  "model": "qwen3-32b",
  "temperature": 0.2,
  "created_by": "u001",
  "created_at": "2026-07-24T10:00:00+08:00",
  "eval_report_id": "eval_20260724_001"
}
```

#### 5.6.3 发布流程

```text
1. 创建新 Prompt 草稿
2. 在评测集上运行离线评测
3. 指标达到阈值后进入灰度
4. 按 tenant/kb/query_type/percent 生效
5. 监控线上效果和错误率
6. 全量发布或回滚
```

### 5.7 检索策略版本管理

建议将检索参数抽象为 `retrieval_profile`：

```json
{
  "profile_id": "default_v1",
  "kb_id": "product_manual",
  "top_k_embedding": 20,
  "top_k_hyde": 20,
  "top_k_web": 5,
  "rerank_top_k": 8,
  "rrf_k": 60,
  "enable_hyde": true,
  "enable_web": false,
  "evidence_threshold": 0.65,
  "active": true
}
```

这样可以做到：

- 不同知识库使用不同召回策略。
- Web Search 默认关闭，仅对需要外部信息的问题开启。
- 检索参数变化可评测、可灰度、可回滚。

### 5.8 接口建议

| 接口 | 说明 |
| --- | --- |
| `POST /kb` | 创建知识库。 |
| `POST /kb/{kb_id}/documents` | 上传文档。 |
| `POST /kb/{kb_id}/documents/{doc_id}/versions` | 上传新版本。 |
| `DELETE /kb/{kb_id}/documents/{doc_id}` | 软删除文档。 |
| `POST /kb/{kb_id}/publish` | 发布知识库版本。 |
| `POST /prompts/{prompt_id}/versions` | 创建 Prompt 版本。 |
| `POST /prompts/{prompt_id}/rollback` | 回滚 Prompt。 |
| `POST /retrieval-profiles` | 创建检索策略。 |
| `POST /retrieval-profiles/{profile_id}/activate` | 激活检索策略。 |

### 5.9 阶段计划

| 阶段 | 内容 | 产出 |
| --- | --- | --- |
| P0 | 为查询、导入、Milvus metadata 补齐 `tenant_id`、`kb_id` | 基础隔离能力 |
| P0 | 查询侧强制 metadata filter | 防止跨知识库召回 |
| P1 | 文档版本表和 active version 机制 | 支持文档更新 |
| P1 | 增量更新和软删除 | 避免旧向量污染 |
| P2 | Prompt 版本管理 | 可评测、可灰度、可回滚 |
| P2 | Retrieval profile 管理 | 检索策略版本化 |
| P3 | 多租户权限 API | admin/editor/viewer 权限 |
| P3 | 知识库发布流水线 | 评测通过后发布 |

### 5.10 验收标准

- 不同 `tenant_id` 的数据无法互相召回。
- 文档更新后，查询只命中新版本 active chunk。
- 文档删除后，查询结果中不再出现被删除内容。
- Prompt 和检索策略能记录版本，并能回滚。
- 每次回答能追踪到 `prompt_version`、`retrieval_profile`、`vector_version`。

## 6. 方向三：成本与稳定性优化

### 6.1 当前问题

RAG 系统的成本和稳定性主要集中在以下几个点：

- LLM 调用成本不可控，token 消耗缺少预算和趋势分析。
- HyDE、Query Understanding、Answer Generation 都可能调用 LLM，链路成本叠加。
- Embedding 对大文档导入成本和耗时影响明显。
- 同类问题重复查询没有缓存。
- 导入任务同步化程度较高，高峰期容易堆积。
- LLM、Milvus、MongoDB 等依赖异常时，降级策略需要更精细。
- 告警已经有规则，但处置 Runbook 还可以进一步标准化。

### 6.2 Token 成本统计

#### 6.2.1 统计维度

建议记录以下字段：

| 字段 | 说明 |
| --- | --- |
| `tenant_id` | 租户。 |
| `kb_id` | 知识库。 |
| `session_id` | 会话。 |
| `task_id` | 查询或导入任务。 |
| `node_name` | 触发 LLM 的节点。 |
| `model` | 模型名称。 |
| `prompt_version` | Prompt 版本。 |
| `input_tokens` | 输入 token。 |
| `output_tokens` | 输出 token。 |
| `total_tokens` | 总 token。 |
| `estimated_cost` | 估算成本。 |
| `status` | ok、error、timeout、degraded。 |

#### 6.2.2 成本指标

建议新增或完善：

| 指标 | 类型 | 标签 | 用途 |
| --- | --- | --- | --- |
| `rag_llm_tokens_total` | Counter | `model`, `type` | token 总量趋势。 |
| `rag_llm_cost_total` | Counter | `model`, `tenant_tier` | 成本趋势。 |
| `rag_llm_budget_remaining` | Gauge | `tenant_tier` | 预算余量。 |
| `rag_llm_calls_blocked_total` | Counter | `reason` | 超预算或限流次数。 |

注意：`tenant_id` 可能数量较多，不建议直接作为 Prometheus label。精确租户账单应写 MongoDB 或成本明细表，Prometheus 只保留聚合维度。

#### 6.2.3 成本看板

Grafana 建议增加：

- 每日 token 趋势。
- 按模型统计调用量和 token。
- 平均每次查询 token。
- HyDE / Answer / Query Understanding 各节点 token 占比。
- fallback 模型调用占比。
- 超长 prompt 和超长 answer 样本列表。

### 6.3 LLM fallback 策略

#### 6.3.1 fallback 场景

| 场景 | 策略 |
| --- | --- |
| 主模型超时 | 切换备用模型或返回降级答案。 |
| 主模型 5xx | 短时间重试，仍失败后 fallback。 |
| JSON 输出解析失败 | 自动修复或使用更稳定模型重试。 |
| token 超限 | 压缩上下文后重试。 |
| 成本预算不足 | 使用低成本模型或拒绝高成本任务。 |

#### 6.3.2 fallback 分层

```text
Level 0: 主模型正常回答
Level 1: 主模型重试一次
Level 2: 切换备用模型
Level 3: 使用压缩上下文回答
Level 4: 返回基于检索摘要的降级答案
Level 5: 明确告知当前无法回答
```

#### 6.3.3 与熔断结合

如果某个模型连续失败：

1. `record_failure` 增加连续失败计数。
2. 达到阈值后打开熔断。
3. 查询链路跳过该模型，直接走 fallback。
4. 半开状态下只放少量探测请求。
5. 成功达到阈值后关闭熔断。

需要导出的指标：

- `rag_circuit_state{name="llm_primary", state="open"}`
- `rag_consecutive_failures{name="llm_primary"}`
- `rag_llm_requests_total{fallback="true"}`

### 6.4 缓存策略

#### 6.4.1 缓存层级

| 缓存对象 | Key | Value | TTL |
| --- | --- | --- | --- |
| Query Understanding | normalized question + history hash | 意图、实体、改写结果 | 1-6 小时 |
| Embedding | text hash + model version | embedding vector | 长期 |
| Retrieval | query embedding hash + kb version | chunk ids | 5-30 分钟 |
| Rerank | question hash + candidate chunk ids | rerank scores | 5-30 分钟 |
| Answer | normalized question + evidence hash + prompt version | answer + sources | 5-60 分钟 |

#### 6.4.2 缓存失效

缓存必须绑定版本，避免返回旧答案。

Key 中建议包含：

```text
tenant_id
kb_id
kb_version
vector_version
prompt_version
retrieval_profile
model
normalized_question_hash
evidence_hash
```

当文档版本、Prompt 版本、检索策略版本变化时，缓存自然失效。

#### 6.4.3 缓存风险

- 用户权限不同，不能共用包含私有数据的答案缓存。
- 对多轮对话，必须考虑 history hash。
- 对时效性问题，不能使用长 TTL。
- 对 Web Search 结果，要设置更短 TTL。

### 6.5 批量 Embedding 优化

#### 6.5.1 当前问题

大文档导入时，embedding 是主要耗时和成本来源之一。若每个 chunk 单独调用，会造成吞吐低、成本高、失败重试复杂。

#### 6.5.2 优化策略

1. 按模型最大 batch size 进行批量请求。
2. 对文本 hash 去重，重复 chunk 直接复用 embedding。
3. 对失败 batch 做二分重试，定位异常文本。
4. 对超长 chunk 提前截断或重新切分。
5. 记录 batch size、耗时、失败率，持续调优。

#### 6.5.3 指标

| 指标 | 用途 |
| --- | --- |
| `rag_embedding_duration_seconds` | embedding 耗时。 |
| `rag_embedding_batch_size` | batch size 分布。 |
| `rag_embedding_cache_hits_total` | embedding 缓存命中。 |
| `rag_embedding_failed_chunks_total` | 失败 chunk 数量。 |

### 6.6 异步导入队列

#### 6.6.1 为什么需要

文档导入属于长任务，涉及文件解析、图片处理、embedding 和 Milvus 写入。生产环境中不适合完全依赖 API 请求生命周期。

建议将导入改为：

```text
API 接收上传
  -> 保存原始文件
  -> 创建导入任务
  -> 投递队列
  -> Worker 执行 LangGraph 导入流程
  -> 更新任务状态
  -> SSE 或轮询通知前端
```

#### 6.6.2 队列选型

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| Redis Queue / RQ | 简单，容易落地 | 复杂调度能力有限 |
| Celery + Redis | 成熟，支持重试和并发 | 组件稍重 |
| Dramatiq | 简洁，适合 Python | 生态小一些 |
| Kafka | 高吞吐，可追溯 | 引入成本高 |

第一阶段建议使用 Redis/Celery 或 Redis Queue，和项目已有 Redis 配置衔接更自然。

#### 6.6.3 任务状态

```text
pending
running
completed
failed
cancelled
retrying
stuck
```

状态流转需要写入事件时间线，并记录 `task_age_seconds`、`task_inflight`、队列深度。

#### 6.6.4 重试策略

| 节点 | 是否重试 | 原因 |
| --- | --- | --- |
| 文件保存 | 可重试 | 临时 IO 异常。 |
| PDF 解析 | 谨慎重试 | 文件内容问题重试无意义。 |
| 图片处理 | 可重试 | 网络或对象存储异常。 |
| Embedding | 可重试 | 模型服务临时失败。 |
| Milvus 写入 | 可重试 | 依赖抖动。 |
| Mongo 状态写入 | 可重试并降级 | 避免任务状态丢失。 |

### 6.7 查询链路稳定性优化

#### 6.7.1 超时预算

建议为一次查询设置整体 deadline，并拆到各节点。

示例：

| 节点 | 超时 |
| --- | --- |
| Query Understanding | 2s |
| Embedding Retrieval | 2s |
| HyDE | 4s |
| Web Search | 3s |
| Rerank | 3s |
| Answer Generation | 8s |
| Overall | 15s |

当某一路召回超时，不一定让整条链路失败，可以标记 `degraded`，继续使用其他证据回答。

#### 6.7.2 Evidence Gate

在生成答案前增加证据门控：

```text
如果 Top evidence 分数不足
  -> 拒答或追问
如果 evidence 冲突
  -> 给出不确定性说明
如果 evidence 足够
  -> 进入答案生成
```

这样可以减少“没有证据也硬答”的幻觉风险。

#### 6.7.3 降级策略

| 故障点 | 降级方式 |
| --- | --- |
| Web Search 失败 | 仅使用本地知识库。 |
| HyDE 失败 | 使用原始问题向量检索。 |
| Rerank 失败 | 使用 RRF 排序结果。 |
| LLM 失败 | fallback 模型或返回检索摘要。 |
| Mongo 历史失败 | 不阻塞查询，写事件降级 JSONL。 |
| MinIO 快照失败 | 降级本地快照或只写事件。 |

### 6.8 告警 Runbook 完善

当前告警规则已经覆盖主要异常，后续需要补充可执行 Runbook。

#### 6.8.1 Runbook 模板

每个告警建议包含：

```text
告警名称
影响范围
常见原因
第一步检查
Grafana 面板
PromQL 查询
日志查询
Trace 查询
临时止血
根因修复
恢复验证
复盘关注点
```

#### 6.8.2 示例：QueryHighErrorRate

```text
影响范围：
  查询接口错误率升高，用户可能无法获得答案。

第一步检查：
  1. 打开 Dataset RAG Overview Dashboard。
  2. 查看错误是否集中在 query graph。
  3. 查看 rag_node_duration_seconds 是否集中在某个节点。
  4. 根据 trace_id 查询 Loki 和 Tempo。

常见原因：
  - LLM 服务异常。
  - Milvus 检索异常。
  - Prompt 输出格式不符合解析逻辑。
  - Mongo 历史写入异常导致上下文读取失败。

临时止血：
  - 关闭 Web Search。
  - 切换备用模型。
  - 降低 rerank_top_k。
  - 对异常租户或知识库限流。

恢复验证：
  - QueryHighErrorRate resolved。
  - 查询成功率恢复。
  - P95 延迟恢复。
  - 抽样查询答案正常。
```

### 6.9 阶段计划

| 阶段 | 内容 | 产出 |
| --- | --- | --- |
| P0 | 完善 token 统计和成本明细 | 成本报表、成本指标 |
| P0 | LLM fallback 和熔断联动 | 模型异常自动降级 |
| P1 | Embedding 缓存和批处理优化 | 导入耗时下降 |
| P1 | 查询缓存 | 重复问题成本下降 |
| P1 | 查询整体 timeout 和节点 timeout | 延迟可控 |
| P2 | 异步导入队列 | 高峰期导入更稳定 |
| P2 | Evidence Gate | 幻觉风险降低 |
| P3 | 完整 Runbook | 告警可处置 |
| P3 | 成本预算和租户限额 | 防止成本失控 |

### 6.10 验收标准

- 单次查询平均 token 成本可统计、可分节点分析。
- 主模型异常时 fallback 能自动生效，并产生告警。
- 高频重复问题命中缓存后，平均响应时间和 token 消耗下降。
- 大文档导入支持异步排队、失败重试和状态追踪。
- 所有 critical 告警都有 Runbook，并能指导一线排障。

## 7. 三个方向的联动关系

这三个方向不是彼此独立的。

| 能力 | 依赖关系 |
| --- | --- |
| Prompt 版本发布 | 必须依赖评测集判断效果。 |
| 检索策略调整 | 必须通过 Recall@K、MRR 验证。 |
| 向量版本升级 | 必须通过评测集和灰度验证。 |
| 缓存策略 | 必须绑定 kb_version、prompt_version、vector_version。 |
| 成本优化 | 必须结合质量评估，不能只追求低成本。 |
| fallback 策略 | 必须观察答案质量，避免低质量模型污染回答。 |
| Runbook | 必须能链接到 Dashboard、Trace、事件时间线和回放 API。 |

推荐演进路径：

```text
先做评测集
  -> 再做 Prompt / 检索 / 向量版本管理
  -> 再做缓存、fallback、成本优化
  -> 最后接入发布门禁和自动化 Runbook
```

原因是：没有评测体系，后面的 Prompt 调整、模型切换、缓存命中、降级策略都很难判断是否真的变好。

## 8. 可执行计划拆分

建议将上述优化方案拆成 6 个可独立排期、独立验收的执行计划。整体顺序遵循“先量化效果，再做生产治理，最后做成本与韧性优化”的原则。

| 计划 | 名称 | 优先级 | 建议周期 | 核心产出 |
| --- | --- | --- | --- | --- |
| Plan 1 | RAG 质量评测基线 | P0 | 1-2 周 | 评测集、评测脚本、质量报告 |
| Plan 2 | 租户与知识库隔离 | P0 | 1-2 周 | `tenant_id` / `kb_id` 全链路隔离 |
| Plan 3 | 文档增量更新与版本治理 | P1 | 2-3 周 | 文档版本、chunk 版本、向量版本 |
| Plan 4 | Prompt 与检索策略发布治理 | P1 | 1-2 周 | Prompt version、retrieval profile、灰度回滚 |
| Plan 5 | 成本统计、缓存与降级 | P2 | 2-3 周 | token 成本、缓存、fallback、熔断 |
| Plan 6 | 异步导入队列与运维 Runbook | P2 | 2-3 周 | 导入队列、任务状态、告警处置手册 |

### 8.1 Plan 1：RAG 质量评测基线

#### 8.1.1 目标

建立最小可用的 RAG 离线评测体系，让后续 Prompt、检索策略、chunk 参数、embedding 模型调整都有量化对比依据。

#### 8.1.2 范围

本计划只覆盖离线评测，不改变线上查询主链路。

包含：

- 评测样本文件或集合。
- 检索指标计算。
- 答案规则评估。
- Markdown / JSON 报告输出。
- baseline 对比。

暂不包含：

- LLM-as-Judge 自动评分。
- CI 发布门禁。
- 线上用户反馈闭环。

#### 8.1.3 任务拆分

| 序号 | 任务 | 说明 | 产出 |
| --- | --- | --- | --- |
| 1 | 建立评测样本结构 | 采用 `case_id`、`tenant_id`、`kb_id`、`question`、`expected_chunk_ids` 等字段 | `data/eval/rag_cases.jsonl` |
| 2 | 准备核心样本 | 先沉淀 30-50 条高频、关键、容易退化的问题 | 初版评测集 |
| 3 | 实现评测 Runner | 批量调用现有查询链路，保存检索结果、答案、引用和 trace_id | `app/evaluation/run_rag_eval.py` |
| 4 | 实现检索指标 | 支持 Recall@K、Hit@K、MRR | 检索评分模块 |
| 5 | 实现规则评估 | 检查 required keywords、forbidden keywords、引用数量和格式 | 答案规则评分模块 |
| 6 | 生成评测报告 | 输出 Summary、Breakdown、Failed Cases、Regression | Markdown / JSON 报告 |
| 7 | 保存 baseline | 固化当前版本作为后续对比基线 | `data/eval/baselines/` |

#### 8.1.4 验收标准

- 可以通过一条命令完成评测运行，例如：

```bash
python -m app.evaluation.run_rag_eval --dataset data/eval/rag_cases.jsonl --profile baseline
```

- 报告至少包含 `total_cases`、`pass_rate`、`recall_at_5`、`hit_at_5`、`mrr_at_5`、失败样本列表。
- 任意 Prompt 或检索参数调整后，可以和 baseline 做指标对比。
- 核心评测样本不少于 30 条，且覆盖事实、步骤、对比、故障诊断等主要问题类型。

#### 8.1.5 风险与控制

| 风险 | 控制方式 |
| --- | --- |
| 评测样本过少导致指标失真 | 先覆盖核心业务问题，再持续从失败查询中补充。 |
| 规则评估无法判断开放式答案 | 第一阶段只做低成本规则评估，第二阶段再引入 LLM-as-Judge。 |
| 评测调用真实 LLM 成本过高 | 支持 mock LLM、抽样运行和 profile 分级。 |

### 8.2 Plan 2：租户与知识库隔离

#### 8.2.1 目标

补齐 `tenant_id`、`kb_id` 在查询、导入、存储和观测链路中的传递与过滤，确保不同租户和知识库之间的数据不会互相召回。

#### 8.2.2 范围

包含：

- 查询接口和导入接口补充租户、知识库字段。
- MongoDB 业务数据补齐租户维度。
- Milvus metadata 增加租户和知识库过滤字段。
- 查询侧强制 metadata filter。
- 事件、快照、评测数据补齐隔离字段。

暂不包含：

- 完整 RBAC 管理后台。
- 复杂组织架构权限继承。
- 每租户独立 collection。

#### 8.2.3 任务拆分

| 序号 | 任务 | 说明 | 产出 |
| --- | --- | --- | --- |
| 1 | 梳理数据入口 | 确认查询、导入、历史、事件、快照的入口参数 | 字段影响清单 |
| 2 | API 请求模型扩展 | 新增或透传 `tenant_id`、`kb_id`，保持默认值兼容老调用 | 更新后的 request schema |
| 3 | Mongo 数据模型扩展 | 历史、导入任务、事件、快照索引增加隔离字段 | 集合字段升级 |
| 4 | Milvus metadata 扩展 | 写入向量时带上 `tenant_id`、`kb_id`、`doc_id`、`chunk_id` | 向量 metadata |
| 5 | 查询过滤改造 | 检索时强制拼接 `tenant_id`、`kb_id` filter | 防跨租户召回 |
| 6 | 兼容默认租户 | 对已有数据采用 `tenant_id=default`、`kb_id=default` 迁移策略 | 兼容迁移脚本或说明 |
| 7 | 增加隔离测试 | 构造两个租户相同问题，验证不会互相召回 | 单元或集成测试 |

#### 8.2.4 验收标准

- 未携带 `tenant_id`、`kb_id` 的老请求仍可进入默认租户和默认知识库。
- 不同 `tenant_id` 的文档即使语义高度相似，也不会被跨租户召回。
- 不同 `kb_id` 的文档不会在未授权查询中混入结果。
- 每条查询事件、节点快照、对话历史都能追踪到对应租户和知识库。

#### 8.2.5 风险与控制

| 风险 | 控制方式 |
| --- | --- |
| 老数据缺少租户字段 | 使用默认租户兜底，并提供一次性迁移脚本。 |
| Milvus filter 漏加导致越权召回 | 将 filter 封装在统一检索入口，避免各节点自行拼装。 |
| Prometheus label 基数过高 | 指标不直接使用 `tenant_id`，只在事件和明细表中记录。 |

### 8.3 Plan 3：文档增量更新与版本治理

#### 8.3.1 目标

支持文档更新、删除、回滚和向量版本管理，避免旧 chunk、旧向量污染线上回答。

#### 8.3.2 范围

包含：

- 文档版本模型。
- chunk hash 对比。
- 增量 embedding。
- active version 查询过滤。
- 软删除和延迟物理删除。
- vector version / index version 元数据。

暂不包含：

- 多版本内容在线 diff 页面。
- 复杂审批流。
- 自动化知识库发布流水线。

#### 8.3.3 任务拆分

| 序号 | 任务 | 说明 | 产出 |
| --- | --- | --- | --- |
| 1 | 设计文档版本表 | 记录 `doc_id`、`active_version`、版本状态、content hash | 文档版本 schema |
| 2 | 扩展 chunk 元数据 | 增加 `doc_version`、`chunk_hash`、`chunk_status` | chunk 版本字段 |
| 3 | 扩展向量元数据 | 增加 `embedding_model`、`embedding_version`、`chunk_strategy`、`vector_status` | vector 版本字段 |
| 4 | 实现 hash 检测 | 文件 hash 未变化时跳过导入，chunk hash 未变化时复用 embedding | 去重逻辑 |
| 5 | 实现增量写入 | 只对新增和变化 chunk 生成 embedding 并写入 Milvus | 增量导入流程 |
| 6 | 实现 active 过滤 | 查询侧只检索 active 文档版本和 active 向量版本 | 查询过滤条件 |
| 7 | 实现软删除 | 删除后立即查询不可见，后台再清理旧向量 | 删除流程 |
| 8 | 实现回滚入口 | 将 `active_version` 切回旧版本，并刷新查询配置 | 基础回滚能力 |

#### 8.3.4 验收标准

- 同一文档重复上传且内容未变化时，不重复生成 embedding。
- 文档更新后，查询只命中新版本 active chunk。
- 文档删除后，查询结果中不再出现该文档内容。
- 可回滚到上一文档版本，且查询结果同步回到上一版本内容。
- 每次回答可以追踪到 `doc_version`、`vector_version`、`embedding_version`。

#### 8.3.5 风险与控制

| 风险 | 控制方式 |
| --- | --- |
| 更新过程中读到半成品数据 | 先写新版本，再原子切换 active version。 |
| 旧向量物理删除失败 | 查询侧先通过 `status` 过滤，物理删除异步补偿。 |
| chunk 策略变化导致版本难对齐 | 将 `chunk_strategy` 和 `schema_version` 纳入 vector version。 |

### 8.4 Plan 4：Prompt 与检索策略发布治理

#### 8.4.1 目标

将 Prompt、模型参数和检索参数从开发文件演进为可版本化、可评测、可灰度、可回滚的发布配置。

#### 8.4.2 范围

包含：

- Prompt version 数据结构。
- retrieval profile 数据结构。
- 配置加载与默认值兼容。
- 按 `tenant_id`、`kb_id`、`query_type`、灰度比例选择配置。
- 发布、激活、回滚接口。
- 与评测报告关联。

暂不包含：

- 可视化 Prompt 编辑器。
- 完整审批工作流。
- 自动发布门禁。

#### 8.4.3 任务拆分

| 序号 | 任务 | 说明 | 产出 |
| --- | --- | --- | --- |
| 1 | 设计 Prompt version schema | 记录版本、状态、变量、模型参数、评测报告 ID | Prompt 版本表 |
| 2 | 设计 retrieval profile schema | 记录 top_k、rerank、RRF、HyDE、Web Search、Evidence Gate 参数 | 检索策略表 |
| 3 | 实现配置读取层 | 优先读配置表，缺省回退到当前 prompts 文件和默认参数 | 兼容配置加载 |
| 4 | 接入查询链路 | 每次回答记录 `prompt_version`、`retrieval_profile` | Trace 和事件增强 |
| 5 | 实现激活与回滚 | 支持 active、draft、archived 状态流转 | 管理接口 |
| 6 | 实现灰度匹配 | 按租户、知识库、问题类型或比例选择版本 | 灰度策略 |
| 7 | 接入评测基线 | 新版本发布前关联 Plan 1 的评测报告 | 发布前验证 |

#### 8.4.4 验收标准

- 不修改代码即可切换 Prompt 版本和检索策略版本。
- 每次回答都能追踪使用的 Prompt、模型参数和 retrieval profile。
- 新 Prompt 或新检索策略可以先在指定租户或知识库灰度。
- 发现退化后可以回滚到上一 active 版本。
- Prompt 或检索策略发布前可以绑定一次评测报告。

#### 8.4.5 风险与控制

| 风险 | 控制方式 |
| --- | --- |
| 配置系统影响现有调用 | 保留文件 Prompt 和默认参数作为 fallback。 |
| 灰度选择不稳定导致排查困难 | 每次请求记录命中的版本和匹配原因。 |
| 参数组合过多难以治理 | retrieval profile 作为整体版本发布，不允许零散临时覆盖。 |

### 8.5 Plan 5：成本统计、缓存与降级

#### 8.5.1 目标

降低重复请求和大模型调用成本，并在 LLM、检索、重排等依赖异常时保持查询链路可降级、可观测。

#### 8.5.2 范围

包含：

- LLM token 和成本明细。
- 成本聚合指标和 Grafana 看板。
- Query Understanding、Embedding、Retrieval、Rerank、Answer 缓存。
- LLM fallback。
- 熔断。
- 查询整体 deadline 和节点 timeout。

暂不包含：

- 精细化租户账单系统。
- 自动购买或动态扩容策略。
- 复杂多模型路由优化。

#### 8.5.3 任务拆分

| 序号 | 任务 | 说明 | 产出 |
| --- | --- | --- | --- |
| 1 | 统一 LLM 调用记录 | 记录模型、节点、token、耗时、状态、估算成本 | 成本明细表或事件 |
| 2 | 增加成本指标 | 导出 token、cost、fallback、budget blocked 等聚合指标 | Prometheus 指标 |
| 3 | 建立成本看板 | 展示每日 token、模型占比、节点占比、超长请求 | Grafana Dashboard |
| 4 | 实现 embedding 缓存 | 以 text hash + model version 为 key 复用向量 | embedding cache |
| 5 | 实现查询缓存 | 对 Retrieval、Rerank、Answer 分层缓存，并绑定版本字段 | 查询缓存 |
| 6 | 实现节点 timeout | 为查询链路设置 overall deadline 和节点级超时 | timeout 配置 |
| 7 | 实现 LLM fallback | 主模型超时、5xx、解析失败、token 超限时走 fallback | 降级策略 |
| 8 | 实现熔断 | 连续失败后跳过异常模型，半开探测恢复 | circuit breaker |

#### 8.5.4 验收标准

- 单次查询可以看到各 LLM 节点的 token、耗时和估算成本。
- 高频重复问题命中缓存后，平均响应时间和 token 消耗明显下降。
- 主模型异常时 fallback 自动生效，并记录 `fallback=true`。
- 某模型连续失败达到阈值后会打开熔断，恢复后可以自动关闭。
- 查询链路局部超时不会必然导致整次请求失败，可产生 degraded 结果。

#### 8.5.5 风险与控制

| 风险 | 控制方式 |
| --- | --- |
| 缓存返回过期答案 | cache key 必须包含 `kb_version`、`prompt_version`、`vector_version`、`retrieval_profile`。 |
| 不同用户权限导致缓存泄露 | 私有知识库答案缓存必须包含 `tenant_id`、权限摘要或 user scope。 |
| fallback 模型质量下降 | fallback 结果标记 degraded，并纳入评测和线上观察。 |

### 8.6 Plan 6：异步导入队列与运维 Runbook

#### 8.6.1 目标

将长耗时文档导入从 API 请求生命周期中拆出，并完善告警处置流程，让导入和运维更适合生产环境。

#### 8.6.2 范围

包含：

- 导入任务表。
- 队列和 Worker。
- 导入状态流转。
- 失败重试和取消。
- SSE 或轮询查询进度。
- 队列和任务指标。
- critical 告警 Runbook。

暂不包含：

- 大规模分布式调度平台。
- 多优先级复杂队列。
- 自动根因分析系统。

#### 8.6.3 任务拆分

| 序号 | 任务 | 说明 | 产出 |
| --- | --- | --- | --- |
| 1 | 设计导入任务模型 | 支持 pending、running、completed、failed、cancelled、retrying、stuck | 任务状态表 |
| 2 | 引入队列组件 | 第一阶段优先 Redis Queue / Celery + Redis | 队列基础设施 |
| 3 | 改造上传接口 | API 保存文件和创建任务后立即返回 task_id | 异步导入入口 |
| 4 | 实现 Worker | Worker 执行现有 LangGraph 导入流程并更新状态 | 导入 Worker |
| 5 | 实现重试策略 | 对 Embedding、Milvus、MinIO、Mongo 等可恢复错误重试 | 重试配置 |
| 6 | 实现进度查询 | 前端或调用方通过 SSE / 轮询获取任务状态 | 任务查询接口 |
| 7 | 增加队列指标 | 导出队列深度、任务年龄、失败率、inflight 数量 | Prometheus 指标 |
| 8 | 编写 Runbook | 为 QueryHighErrorRate、ImportTaskStuck、LLMHighLatency 等告警补齐处置手册 | `docs/runbooks/` |

#### 8.6.4 验收标准

- 上传大文档后 API 不被长时间阻塞，可以快速返回 `task_id`。
- Worker 可以完成文档解析、chunk、embedding、Milvus 写入和状态更新。
- 失败任务可以按策略重试，并能在事件时间线中看到状态变化。
- 查询任务状态可以看到当前阶段、失败原因、重试次数和 trace_id。
- 所有 critical 告警都有 Runbook，包含影响范围、检查步骤、临时止血、恢复验证。

#### 8.6.5 风险与控制

| 风险 | 控制方式 |
| --- | --- |
| 队列引入后状态一致性变复杂 | 所有状态转换写入统一任务表和事件时间线。 |
| Worker 重试造成重复写入 | 导入节点按 `doc_id`、`doc_version`、`chunk_hash` 做幂等。 |
| Runbook 写完无人使用 | 每个 Runbook 必须链接 Grafana、Loki、Tempo 和相关命令。 |

### 8.7 推荐实施顺序

```text
Plan 1 RAG 质量评测基线
  -> Plan 2 租户与知识库隔离
  -> Plan 3 文档增量更新与版本治理
  -> Plan 4 Prompt 与检索策略发布治理
  -> Plan 5 成本统计、缓存与降级
  -> Plan 6 异步导入队列与运维 Runbook
```

Plan 1 应优先启动，因为它为后续所有改动提供质量基线。Plan 2 建议紧随其后，因为租户和知识库字段会影响后续版本、缓存、评测和成本统计的 key 设计。Plan 3 和 Plan 4 可以并行推进，但都应依赖 Plan 1 的评测报告做发布判断。Plan 5 和 Plan 6 更偏生产韧性，建议在核心数据模型稳定后实施。

### 8.8 里程碑组合建议

如果希望按项目迭代而不是单计划推进，可以组合成 3 个里程碑：

| 里程碑 | 包含计划 | 目标 | 可发布能力 |
| --- | --- | --- | --- |
| M1：质量可评估 | Plan 1 | 知道 RAG 改动是否变好或变差 | 离线评测报告和 baseline |
| M2：数据可治理 | Plan 2、Plan 3、Plan 4 | 支持隔离、版本、灰度、回滚 | 多知识库生产化基础 |
| M3：运行可持续 | Plan 5、Plan 6 | 降低成本并提升故障处置能力 | 缓存、fallback、异步导入、Runbook |

## 9. 简历和面试提炼

这三个方向可以在简历中提炼为：

```text
围绕 RAG 应用生产化演进，设计效果评估、权限隔离、知识库版本管理、Prompt 版本管理、成本统计、缓存、LLM fallback、异步导入队列和告警 Runbook 等优化方案，推动系统从功能验证型 Demo 向可评估、可隔离、可灰度、可回滚、可治理的大模型应用演进。
```

面试口述可以这样表达：

```text
我对这个 RAG 项目的后续优化主要分三块。第一是效果评估，建立评测集，用 Recall@K、MRR、答案准确性和引用准确性来量化效果，避免只靠主观体验。第二是生产化能力，包括 tenant、kb、doc、chunk、vector、prompt、retrieval profile 的版本和权限模型，解决多租户、增量更新和回滚问题。第三是成本和稳定性，比如 token 成本统计、缓存、批量 embedding、异步导入队列、LLM fallback、熔断和 Runbook。整体思路是把大模型应用当成一个真实后端系统来治理，而不是停留在调用模型 API。
```

## 10. 总结

RAG 项目继续向生产级演进时，最重要的不是继续堆更多模型能力，而是建立完整工程闭环：

- 用评测体系回答“效果有没有变好”。
- 用生产化能力回答“能不能安全服务真实用户和多知识库”。
- 用成本与稳定性治理回答“能不能长期稳定、可控成本地运行”。

这三个方向完成后，系统会从一个可运行的 RAG 应用，进一步升级为具备质量基线、版本治理、灰度回滚、成本控制和故障处置能力的大模型应用平台雏形。
