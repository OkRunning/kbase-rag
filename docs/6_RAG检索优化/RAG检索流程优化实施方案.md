# RAG 检索流程优化实施方案

## 1. 背景

当前查询链路：

```text
用户发起查询
  -> 确认问题所属 item_name 与用户问题重写
  -> 并行执行普通向量检索、HyDE 假设答案向量检索、MCP 网络查询
  -> 对普通向量检索与 HyDE 检索结果做 RRF
  -> 对 RRF 结果与网络检索结果做 Rerank
  -> 将 Rerank 结果交给大模型生成最终答案
  -> 返回用户
```

当前流程的优势是召回链路完整，能够同时利用本地知识库、HyDE 增强召回和 Web 搜索补充信息。但它也存在典型问题：

- 所有问题默认走最高成本路径，简单问题也会触发 HyDE 和 Web。
- Web 搜索延迟和稳定性不可控，容易拉高整体 P95/P99。
- HyDE 需要额外调用 LLM，成本和耗时较高。
- RRF、Rerank 的 topK、权重和阈值目前偏固定，缺少按问题类型动态调整。
- Rerank 结果缺少强置信度判断，低相关证据也可能进入生成阶段。
- 不同召回源的实际贡献度需要通过评测和观测继续量化。

本方案目标是在保持现有 LangGraph 节点结构基本稳定的前提下，分阶段提升响应速度、降低成本、提高答案质量与系统稳定性。

## 2. 优化目标

### 2.1 响应时间

- 降低查询平均耗时。
- 降低 P95/P99，重点控制 HyDE、Web Search、Rerank、最终 LLM 生成耗时。
- 缩短首 token 时间，让用户更早看到反馈。

### 2.2 成本

- 减少不必要的 LLM 调用，尤其是 HyDE 和最终大模型生成。
- 减少不必要的 Web Search 调用。
- 控制 Rerank 输入文档数量，降低本地模型或推理服务压力。
- 控制最终 Prompt 上下文长度，降低生成 token 消耗。

### 2.3 答案质量

- 提升检索 Recall@K、MRR、NDCG。
- 提升 Rerank 后证据排序质量。
- 降低幻觉率，确保答案被证据支撑。
- 对证据不足场景明确返回“资料不足”，避免强行生成。

### 2.4 稳定性

- 对 Milvus、MCP Web Search、Reranker、LLM 增加超时、熔断和降级。
- 单个外部依赖失败时，查询链路仍能给出可控结果。
- 对慢调用、空召回、低置信度结果建立监控和告警。

### 2.5 可维护性

- 检索策略配置化，避免策略散落在节点代码里。
- Prompt、topK、权重、阈值可版本化管理。
- 每次策略调整都可以通过评测集和观测数据验证收益。

## 3. 总体优化思路

核心思路：从“固定全量检索”升级为“查询理解驱动的动态检索计划”。

优化后推荐链路：

```text
用户发起查询
  -> Query Understanding
       - item_name 识别
       - query rewrite
       - query_type 分类
       - retrieval_plan 生成
       - 是否需要澄清
  -> 按 retrieval_plan 条件执行检索节点
       - local vector
       - HyDE
       - Web Search
       - 后续可扩展 BM25 / KG
  -> RRF / 分源融合 / 去重
  -> Rerank / 置信度判断 / 动态 topK
  -> Answer Output
       - 证据裁剪
       - 引用约束
       - 不足回答兜底
  -> 返回用户
```

## 4. 节点级优化方案

## 4.1 Query Understanding 节点

涉及文件：

- `app/query_process/agent/nodes/node_item_name_confirm.py`
- `app/query_process/agent/state.py`
- `app/query_process/agent/main_graph.py`
- `prompts/query_understanding.prompt`
- `prompts/rewritten_query_and_itemnames.prompt`

### 现状

当前节点主要完成：

- 从历史和原始问题中确认 `item_names`。
- 重写用户问题。
- 当无法确认 item 时提前返回提示。

### 问题

- 缺少问题类型识别，后续节点无法决定是否需要 HyDE 或 Web。
- 缺少检索计划，所有正常问题都默认并行执行三路召回。
- 对低置信度 item 识别只做较粗粒度判断。
- LLM 输出如果不稳定，会直接影响后续所有检索节点。

### 优化动作

新增 `query_type` 字段，建议枚举：

```text
fact             参数、定义、明确事实
procedure        操作步骤、使用流程
troubleshooting  故障排查
safety           安全注意事项
comparison       多商品或多方案对比
follow_up        追问
chitchat         闲聊或非知识库问题
realtime         明确需要实时信息
unknown          无法判断
```

新增 `retrieval_plan` 字段：

```json
{
  "use_local_vector": true,
  "use_hyde": false,
  "use_web": false,
  "local_top_k": 8,
  "hyde_top_k": 5,
  "web_top_k": 3,
  "rrf_top_k": 8,
  "rerank_top_k": 5,
  "dense_weight": 0.9,
  "sparse_weight": 0.1,
  "reason": "本地产品手册类事实问题，优先使用本地知识库"
}
```

推荐路由策略：

| 问题类型 | local vector | HyDE | Web | 说明 |
| --- | --- | --- | --- | --- |
| fact | 是 | 否 | 否 | 明确参数、型号、注意事项优先本地检索 |
| procedure | 是 | 可选 | 否 | 普通操作步骤先本地，召回不足再 HyDE |
| troubleshooting | 是 | 是 | 可选 | 故障描述模糊，HyDE 常有价值 |
| safety | 是 | 否 | 否 | 安全类应优先官方本地资料 |
| comparison | 是 | 否 | 否 | 按 item 分组检索，避免结果偏斜 |
| realtime | 可选 | 否 | 是 | 明确实时信息才启用 Web |
| chitchat | 否 | 否 | 否 | 不进入 RAG 检索 |
| unknown | 是 | 可选 | 否 | 保守走本地，低置信度时澄清 |

### 实施步骤

1. 在 `QueryGraphState` 中增加：

```python
query_type: str
retrieval_plan: dict
needs_clarification: bool
clarification_question: str
```

2. 调整 Query Understanding prompt，要求 LLM 输出结构化 JSON。

3. 对结构化输出增加解析和默认值兜底。

4. 在 `main_graph.py` 中把固定并行路由改成条件路由。

5. 将 `retrieval_plan` 写入观测事件和日志，便于后续评估策略命中情况。

### 验收标准

- `fact` 类问题默认不调用 HyDE 和 Web。
- `realtime` 类问题可以触发 Web。
- 无明确 item 时提前澄清，不执行后续检索。
- Langfuse / Trace 中可以看到 `query_type` 和 `retrieval_plan`。

## 4.2 普通向量检索节点

涉及文件：

- `app/query_process/agent/nodes/node_search_embedding.py`
- `app/clients/milvus_utils.py`
- `app/conf/milvus_config.py`

### 现状

当前节点对 `rewritten_query` 执行 dense + sparse 混合检索，并使用 `item_name in [...]` 过滤。

### 问题

- topK、dense/sparse 权重固定。
- 只按 item_name 过滤，缺少版本、文档状态、租户等更强约束。
- 对精确型号、错误码、参数类问题，sparse 权重可能偏低。
- embedding 结果没有缓存。

### 优化动作

1. 从 `retrieval_plan` 读取：

```text
local_top_k
dense_weight
sparse_weight
metadata_filters
```

2. 按 query_type 调整权重：

| 场景 | dense_weight | sparse_weight |
| --- | --- | --- |
| 语义描述、操作步骤 | 0.8 - 0.9 | 0.1 - 0.2 |
| 型号、参数、错误码 | 0.5 - 0.7 | 0.3 - 0.5 |
| 精确短语、标题检索 | 0.4 - 0.6 | 0.4 - 0.6 |

3. 增加 metadata filter：

```text
tenant_id
kb_id
item_name
doc_status = active
vector_version = active_vector_version
language
```

4. 增加 embedding 缓存：

缓存 key：

```text
embedding:{embedding_model}:{sha256(normalized_query)}
```

缓存内容：

```json
{
  "dense": [],
  "sparse": {},
  "created_at": "..."
}
```

5. 对检索结果增加质量摘要：

```json
{
  "source": "local_vector",
  "count": 10,
  "top1_score": 0.82,
  "top3_avg_score": 0.77,
  "empty": false
}
```

### 验收标准

- 能按 `retrieval_plan.local_top_k` 控制候选数量。
- 能按问题类型调整 dense/sparse 权重。
- 相同 query 重复请求时 embedding 缓存命中。
- 指标中能看到 local vector 的召回数量和分数分布。

## 4.3 HyDE 检索节点

涉及文件：

- `app/query_process/agent/nodes/node_search_embedding_hyde.py`
- `prompts/hyde_prompt.prompt`
- `app/lm/lm_utils.py`

### 现状

HyDE 节点会先调用 LLM 生成假设答案，再对 `rewritten_query + hypothesis_answer` 做 embedding 和 Milvus 检索。

### 问题

- 每次调用都会增加一次 LLM 成本和延迟。
- 对明确事实问题，HyDE 可能收益低甚至引入噪声。
- 假设答案过长时会增加 embedding 噪声。
- HyDE 失败时需要更明确的降级策略。

### 优化动作

1. 按 `retrieval_plan.use_hyde` 决定是否执行。

2. 增加二阶段触发条件：

```text
如果普通向量检索 top1_score >= 0.75 且 top3_avg_score >= 0.65，则跳过 HyDE。
如果普通向量检索为空或 top1_score < 0.45，则触发 HyDE。
```

3. 限制假设答案长度：

```text
中文不超过 200 字
英文不超过 120 words
```

4. 使用低成本模型生成 HyDE。

5. 缓存 HyDE 假设答案：

```text
hyde:{model}:{sha256(rewritten_query)}
```

6. 记录 HyDE 贡献度：

```text
hyde_unique_chunks_count
hyde_overlap_with_local
hyde_docs_in_final_topk
```

### 验收标准

- fact、safety 类问题默认不触发 HyDE。
- troubleshooting、模糊描述类问题可以触发 HyDE。
- HyDE LLM 失败时不影响普通向量检索结果继续进入 RRF。
- 评测报告能看到 HyDE 对 Recall@K 的增益。

## 4.4 MCP Web Search 节点

涉及文件：

- `app/query_process/agent/nodes/node_web_search_mcp.py`
- `app/conf/bailian_mcp_config.py`

### 现状

当前 Web Search 默认与本地检索、HyDE 并行执行，并且 MCP 调用中存在较长 timeout。

### 问题

- Web Search 延迟和稳定性不可控。
- 产品手册类问题中，Web 结果可能不如本地资料可信。
- Web 结果质量参差不齐，可能被 reranker 排到前面。
- MCP 连接、工具列表、调用都在单次查询内完成，存在额外开销。

### 优化动作

1. 按 `retrieval_plan.use_web` 决定是否执行。

2. 默认只在以下场景启用 Web：

- 用户明确要求联网、最新、实时。
- query_type = realtime。
- 本地检索为空或低置信度，且问题允许外部资料补充。
- 后续产品策略允许 Web 作为补充来源。

3. 设置更严格超时：

```text
connect_timeout: 3s
tool_call_timeout: 5s - 8s
max_retry_attempts: 1 - 2
```

4. Web 结果做可信度过滤：

```text
domain_allowlist
domain_blocklist
title/snippet 去重
snippet 最小长度
广告/聚合页过滤
```

5. Web 结果加入来源权重：

```json
{
  "source": "web",
  "source_weight": 0.6,
  "trust_level": "medium"
}
```

6. 增加 Web 缓存：

```text
web_search:{sha256(normalized_query)}:{date_bucket}
```

对于实时性要求不强的问题，缓存 10 到 30 分钟。

7. 降级策略：

```text
MCP 超时 -> 返回空 web_search_docs，不中断主链路。
MCP 错误率持续升高 -> 熔断一段时间，只走本地检索。
```

### 验收标准

- 普通产品手册问题默认不调用 Web。
- Web 超时不会拖垮主链路。
- Trace 中能区分 Web 被跳过、成功、失败、超时、熔断。
- Web 进入最终 topK 的比例可观测。

## 4.5 RRF 融合节点

涉及文件：

- `app/query_process/agent/nodes/node_rrf.py`

### 现状

当前 RRF 只融合普通向量和 HyDE，权重均为 1.0，`top_k=5` 固定。

### 问题

- 固定 topK 对不同问题不够灵活。
- RRF score 没有保留到后续节点，不利于分析和二次融合。
- 多 item 对比问题可能出现某个 item 结果占满 topK。
- 缺少相邻 chunk 合并和重复内容压缩。

### 优化动作

1. 从 `retrieval_plan.rrf_top_k` 读取 topK。

2. 从 `retrieval_plan` 读取召回源权重：

```json
{
  "source_weights": {
    "local_vector": 1.0,
    "hyde": 0.7,
    "web": 0.5
  }
}
```

3. 保留 RRF score：

```json
{
  "chunk_id": "...",
  "entity": {},
  "rrf_score": 0.032,
  "source_hits": ["local_vector", "hyde"]
}
```

4. 增加去重策略：

- 按 `chunk_id` 去重。
- 按 `content_sha256` 去重。
- 对同一文档相邻 chunk 可进行窗口合并。

5. 多 item 问题按 item 分组：

```text
每个 item 至少保留 min_per_item 条候选，再进入全局 rerank。
```

### 验收标准

- RRF topK 可以按问题类型调整。
- 最终 rerank 输入中包含 `rrf_score` 和 `source_hits`。
- 多 item 对比问题不会只召回单一 item。
- 重复 chunk 明显减少。

## 4.6 Rerank 节点

涉及文件：

- `app/query_process/agent/nodes/node_rerank.py`
- `app/lm/reranker_utils.py`
- `app/conf/reranker_config.py`

### 现状

当前节点会合并 `rrf_chunks` 与 `web_search_docs`，调用 BGE Reranker 统一打分，然后通过 topK 和断崖规则裁剪。

### 问题

- Rerank 输入数量缺少前置硬限制。
- 本地结果和 Web 结果直接混排，可能产生来源偏差。
- Rerank score 缺少低置信度判断。
- Reranker 失败时应有明确降级。
- 断崖阈值需要通过评测集校准。

### 优化动作

1. 控制 Rerank 输入数量：

```text
local_candidates <= 12
web_candidates <= 3
total_candidates <= 15
```

2. 分源预处理：

- 本地证据保留 `rrf_score`、`chunk_id`、`item_name`。
- Web 证据保留 `url`、`domain`、`trust_level`。
- 过短、重复、低可信 Web 片段不进入 rerank。

3. 增加分数阈值：

```text
top1_score < 0.35 -> insufficient
top1_score >= 0.35 且 top1_score < 0.5 -> low_confidence
top1_score >= 0.5 -> answerable
```

具体阈值需要通过评测集调参。

4. 增加输出质量字段：

```json
{
  "rerank_quality": {
    "top1_score": 0.78,
    "top3_avg_score": 0.66,
    "score_gap": 0.21,
    "answerability": "answerable",
    "local_doc_count": 4,
    "web_doc_count": 1
  }
}
```

5. Reranker 降级策略：

```text
Reranker 正常 -> 使用 rerank 排序结果
Reranker 超时 -> 使用 RRF 排序结果
Reranker OOM/加载失败 -> 熔断 reranker，短时间内跳过 rerank
```

6. 批处理和模型常驻：

- 确认 `get_reranker_model()` 不在每次请求重复加载模型。
- 对并发请求增加队列或 batch 推理，防止模型被打爆。

7. 断崖规则评测校准：

当前常量：

```text
RERANK_GAP_RATIO = 0.25
RERANK_GAP_ABS = 0.5
```

建议用 `data/eval` 样本分别测试多组参数：

```text
gap_ratio: 0.15 / 0.25 / 0.35
gap_abs: 0.2 / 0.35 / 0.5
max_topk: 5 / 8 / 10
```

以答案指标和 token 成本共同决定默认值。

### 验收标准

- Rerank 输入数量稳定可控。
- Reranker 失败时链路不报错，可以降级返回 RRF 证据。
- 低置信度问题不会强行生成确定性答案。
- 指标中能看到 rerank 输入数量、输出数量、top1 score、answerability 分布。

## 4.7 Answer Output 节点

涉及文件：

- `app/query_process/agent/nodes/node_answer_output.py`
- `prompts/answer_out.prompt`

### 现状

当前节点将 `reranked_docs` 拼接后交给 LLM 生成最终答案。

### 问题

- 如果证据相关性低，LLM 可能仍然生成看似确定的答案。
- 上下文长度未必按相关片段裁剪，可能增加 token 成本。
- 不同问题类型使用同一回答风格，结构化程度不足。
- 引用和证据约束还可以进一步强化。

### 优化动作

1. 根据 `rerank_quality.answerability` 决定输出策略：

```text
answerable -> 正常生成
low_confidence -> 生成谨慎答案，并提示资料可能不足
insufficient -> 不生成编造答案，返回资料不足和建议澄清问题
```

2. 按问题类型使用回答模板：

| query_type | 输出结构 |
| --- | --- |
| fact | 直接答案 + 来源 |
| procedure | 步骤列表 + 注意事项 |
| troubleshooting | 可能原因 + 检查步骤 + 安全提醒 |
| safety | 安全注意事项 + 禁止操作 |
| comparison | 按 item 分组对比表 |

3. 证据裁剪：

- 每个文档只截取与 query 最相关的片段。
- 单条证据控制在 300 到 500 中文字符。
- 最终上下文控制在模型窗口的安全范围内。

4. 引用约束：

Prompt 中明确要求：

```text
只能基于给定资料回答。
没有资料支撑时必须说明资料不足。
不要编造参数、步骤、型号、法规、网址。
```

5. 简单问题小模型化：

对于高置信度 fact 问题，可以考虑使用低成本模型生成，或使用抽取式模板返回。

### 验收标准

- 无证据或低置信度时不产生确定性幻觉。
- 答案包含可追溯来源。
- 输出 token 数下降。
- procedure、troubleshooting 类答案结构更稳定。

## 5. 策略级改造

## 5.1 新增 Retrieval Planner

建议新增文件：

```text
app/query_process/agent/nodes/node_retrieval_planner.py
prompts/retrieval_planner.prompt
```

也可以第一阶段先不新增节点，而是在 `node_item_name_confirm` 中直接产出 `retrieval_plan`。长期建议拆出独立节点，职责更清晰。

### Planner 输入

```json
{
  "original_query": "...",
  "rewritten_query": "...",
  "history": [],
  "item_names": [],
  "query_type": "fact",
  "tenant_id": "default",
  "kb_id": "default"
}
```

### Planner 输出

```json
{
  "needs_clarification": false,
  "clarification_question": "",
  "retrieval_plan": {
    "use_local_vector": true,
    "use_hyde": false,
    "use_web": false,
    "local_top_k": 8,
    "hyde_top_k": 5,
    "web_top_k": 3,
    "rrf_top_k": 8,
    "rerank_top_k": 5,
    "dense_weight": 0.9,
    "sparse_weight": 0.1,
    "source_weights": {
      "local_vector": 1.0,
      "hyde": 0.7,
      "web": 0.5
    },
    "reason": "..."
  }
}
```

## 5.2 条件边改造

当前 `main_graph.py` 中 `node_item_name_confirm` 后固定返回：

```python
return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"
```

建议改为按 `retrieval_plan` 动态路由：

```text
if needs_clarification:
    -> node_answer_output
else:
    if use_local_vector -> node_search_embedding
    if use_hyde -> node_search_embedding_hyde
    if use_web -> node_web_search_mcp
```

如果没有任何检索节点被启用：

```text
chitchat -> node_answer_output
无法处理 -> node_answer_output 返回澄清或拒答
```

注意：LangGraph 并行分支汇聚时，需要确保未执行的节点在 state 中有默认空列表，例如：

```python
embedding_chunks = []
hyde_embedding_chunks = []
web_search_docs = []
```

## 5.3 缓存策略

建议分四类缓存：

| 缓存对象 | Key | TTL | 说明 |
| --- | --- | --- | --- |
| Query Understanding | `qu:{tenant}:{sha256(query+history摘要)}` | 5-30 min | 减少重复 LLM 理解 |
| Embedding | `emb:{model}:{sha256(text)}` | 7-30 days | 高复用、低风险 |
| HyDE | `hyde:{model}:{sha256(query)}` | 1-7 days | 降低 HyDE LLM 成本 |
| Web Search | `web:{sha256(query)}:{date_bucket}` | 10-30 min | 控制实时性 |

缓存命中也要写入指标：

```text
rag_cache_hits_total{cache_type="embedding"}
rag_cache_misses_total{cache_type="embedding"}
```

## 5.4 降级策略

| 依赖 | 异常 | 降级策略 |
| --- | --- | --- |
| Query LLM | 超时/错误 | 使用原始问题，不做 rewrite；必要时提示稍后再试 |
| Milvus | 查询失败 | 可选降级 BM25；否则返回资料检索失败 |
| HyDE LLM | 超时/错误 | 跳过 HyDE，只使用普通向量结果 |
| MCP Web | 超时/错误 | 返回空 Web 结果，不中断主链路 |
| Reranker | OOM/超时 | 跳过 Rerank，使用 RRF 排序 |
| Answer LLM | 超时/错误 | 返回检索摘要或错误提示 |

## 6. 指标与评测

## 6.1 响应时间指标

建议按节点记录：

```text
rag_node_duration_seconds{node_name="node_item_name_confirm"}
rag_node_duration_seconds{node_name="node_search_embedding"}
rag_node_duration_seconds{node_name="node_search_embedding_hyde"}
rag_node_duration_seconds{node_name="node_web_search_mcp"}
rag_node_duration_seconds{node_name="node_rrf"}
rag_node_duration_seconds{node_name="node_rerank"}
rag_node_duration_seconds{node_name="node_answer_output"}
```

重点看：

- 平均耗时
- P50
- P95
- P99
- 首 token 时间
- 超时率

## 6.2 成本指标

建议记录：

```text
rag_llm_requests_total{node_name, model}
rag_llm_tokens_total{node_name, model, type="input|output"}
rag_web_search_requests_total{status}
rag_rerank_docs_total{stage="input|output"}
rag_embedding_requests_total{model}
rag_cache_hits_total{cache_type}
```

需要回答的问题：

- 每个 query 平均调用几次 LLM？
- HyDE 占总 LLM 成本比例是多少？
- Web Search 调用率是多少？
- Rerank 平均输入多少条候选？
- 最终 Prompt 平均 token 数是多少？

## 6.3 检索质量指标

已有评测文档中提到的指标可以继续使用：

- `retrieval_recall_at_5`
- `retrieval_hit_at_5`
- `retrieval_mrr_at_5`
- `recall_at_5`
- `hit_at_5`
- `mrr_at_5`

建议新增分源贡献指标：

```text
local_recall_at_k
hyde_recall_at_k
web_final_topk_ratio
hyde_unique_hit_ratio
rerank_top1_score_avg
low_confidence_ratio
insufficient_ratio
```

## 6.4 答案质量指标

建议评估：

- `faithfulness`：答案是否被证据支撑。
- `answer_relevance`：答案是否回应用户问题。
- `context_precision`：给 LLM 的上下文是否都是有用证据。
- `context_recall`：必要证据是否进入上下文。
- `citation_accuracy`：引用是否准确。
- `refusal_accuracy`：资料不足时是否正确拒答。

## 6.5 A/B 对比

每次策略调整应至少对比：

```text
baseline: 当前固定三路召回
candidate: 动态 retrieval_plan
```

对比维度：

- 平均耗时、P95、P99。
- 平均 LLM 调用次数。
- 平均输入/输出 token。
- Web 调用率。
- HyDE 调用率。
- Recall@K、MRR@K。
- 答案规则命中率。
- 人工抽样满意度。

## 7. 分阶段实施计划

## 7.1 第一阶段：低风险降本提速

周期：1 到 2 周。

目标：不大改图结构，先减少不必要调用。

实施内容：

1. 在 Query Understanding 输出中增加 `query_type` 和 `retrieval_plan`。
2. fact、safety 类问题默认跳过 HyDE 和 Web。
3. Web Search 增加更短 timeout，失败返回空列表。
4. Rerank 输入增加硬上限。
5. Answer Output 增加低置信度拒答逻辑。
6. 指标记录 HyDE 调用率、Web 调用率、Rerank 输入数量。

预期收益：

- 简单问题延迟明显下降。
- LLM 调用次数下降。
- Web Search 对 P95 的影响下降。

风险：

- 跳过 HyDE 后，部分模糊问题召回可能下降。

控制方式：

- 只对高置信 fact/safety 跳过 HyDE。
- 保留配置开关，可快速回滚。

## 7.2 第二阶段：动态检索计划

周期：2 到 4 周。

目标：从固定并行检索升级为条件路由。

实施内容：

1. 新增或独立 `node_retrieval_planner`。
2. `main_graph.py` 按 `retrieval_plan` 条件执行检索节点。
3. 普通向量检索支持动态 topK 和 dense/sparse 权重。
4. RRF 支持动态 topK、source weight、保留 RRF score。
5. HyDE 增加普通向量结果质量触发条件。
6. Web 增加可信域过滤和缓存。

预期收益：

- 不同问题使用不同成本路径。
- 召回质量更可控。
- 策略可解释、可观测。

风险：

- LangGraph 并行和汇聚状态处理复杂度上升。

控制方式：

- 给未执行节点设置默认空结果。
- 为每种 `retrieval_plan` 增加单元测试和集成测试。

## 7.3 第三阶段：质量闭环和策略调参

周期：持续迭代。

目标：用评测数据驱动 topK、权重、阈值调整。

实施内容：

1. 扩充 `data/eval` 样本，覆盖不同 query_type。
2. 建立 baseline 报告，所有策略调整都跑评测。
3. 调参 RRF weight、Rerank gap、answerability 阈值。
4. 增加分源贡献报表。
5. 在 Grafana 中展示 query_type、retrieval_plan、召回贡献、低置信比例。

预期收益：

- 优化从经验驱动变成数据驱动。
- 能定位“召回问题、排序问题、生成问题”分别发生在哪里。

## 7.4 第四阶段：高级能力扩展

周期：按业务需要推进。

可选方向：

- 增加 BM25 或稀疏检索独立节点。
- 增加知识图谱 KG 检索节点。
- 支持多 item 分组检索和对比回答。
- 支持 chunk 邻接扩展，把命中 chunk 的上下文窗口一起带入。
- 支持 Prompt、模型、检索策略灰度发布和回滚。

## 8. 推荐配置

## 8.1 默认 retrieval_plan

```json
{
  "use_local_vector": true,
  "use_hyde": false,
  "use_web": false,
  "local_top_k": 8,
  "hyde_top_k": 5,
  "web_top_k": 3,
  "rrf_top_k": 8,
  "rerank_top_k": 5,
  "dense_weight": 0.85,
  "sparse_weight": 0.15,
  "source_weights": {
    "local_vector": 1.0,
    "hyde": 0.7,
    "web": 0.5
  }
}
```

## 8.2 troubleshooting retrieval_plan

```json
{
  "use_local_vector": true,
  "use_hyde": true,
  "use_web": false,
  "local_top_k": 10,
  "hyde_top_k": 8,
  "web_top_k": 0,
  "rrf_top_k": 10,
  "rerank_top_k": 6,
  "dense_weight": 0.85,
  "sparse_weight": 0.15,
  "source_weights": {
    "local_vector": 1.0,
    "hyde": 0.8
  }
}
```

## 8.3 realtime retrieval_plan

```json
{
  "use_local_vector": false,
  "use_hyde": false,
  "use_web": true,
  "local_top_k": 0,
  "hyde_top_k": 0,
  "web_top_k": 5,
  "rrf_top_k": 0,
  "rerank_top_k": 5,
  "source_weights": {
    "web": 1.0
  }
}
```

## 9. 测试方案

## 9.1 单元测试

建议新增或更新：

```text
test/test_query_main_graph.py
test/test_main_graph.py
test/test_main_graph_full_flow.py
```

覆盖场景：

- fact 问题只走 local vector。
- troubleshooting 问题走 local vector + HyDE。
- realtime 问题走 Web。
- 无 item_name 时提前澄清。
- Reranker 异常时降级 RRF。
- Web 超时时返回空 Web 结果。
- Rerank top1 低于阈值时输出 insufficient。

## 9.2 评测集

扩充样本类型：

| 类型 | 建议数量 |
| --- | --- |
| fact | 20+ |
| procedure | 20+ |
| troubleshooting | 20+ |
| safety | 10+ |
| comparison | 10+ |
| follow_up | 10+ |
| insufficient | 10+ |

每条样本至少包含：

```text
question
query_type
expected_answer
expected_chunk_ids
required_keywords
forbidden_keywords
```

## 9.3 压测

建议压测场景：

- 本地检索高并发。
- HyDE 高并发。
- Web Search 慢调用。
- Reranker 批量输入上限。
- LLM 生成慢调用。

核心观察：

- QPS
- P95/P99
- 错误率
- 超时率
- Reranker 队列堆积
- 缓存命中率

## 10. 发布与回滚

## 10.1 配置开关

建议增加配置：

```text
RAG_DYNAMIC_RETRIEVAL_PLAN_ENABLED=true
RAG_HYDE_DEFAULT_ENABLED=false
RAG_WEB_DEFAULT_ENABLED=false
RAG_RERANK_FALLBACK_ENABLED=true
RAG_LOW_CONFIDENCE_REFUSAL_ENABLED=true
```

## 10.2 灰度策略

1. 本地评测通过。
2. 测试环境全量开启。
3. 生产按租户或流量比例灰度：

```text
5% -> 20% -> 50% -> 100%
```

4. 每个阶段观察至少：

- 查询成功率。
- P95/P99。
- 平均成本。
- 低置信拒答比例。
- 人工抽样质量。

## 10.3 回滚策略

如果出现以下情况，回滚到固定三路召回：

- Recall@K 明显下降。
- 用户反馈答案缺失明显增加。
- 低置信拒答比例异常升高。
- P95/P99 未下降反而升高。
- Query Understanding 结构化输出错误率过高。

回滚方式：

```text
RAG_DYNAMIC_RETRIEVAL_PLAN_ENABLED=false
```

## 11. 优先级建议

| 优先级 | 事项 | 原因 |
| --- | --- | --- |
| P0 | Web Search 按需执行 + timeout 降低 | 直接降低 P95 和外部依赖风险 |
| P0 | HyDE 按需执行 | 直接减少 LLM 成本和延迟 |
| P0 | Rerank 输入上限 | 控制模型推理压力 |
| P1 | retrieval_plan 结构化输出 | 后续动态路由基础 |
| P1 | Rerank answerability 阈值 | 降低幻觉 |
| P1 | embedding / HyDE / Web 缓存 | 降低重复成本 |
| P2 | RRF 动态权重和保留 score | 提升融合可解释性 |
| P2 | 分源贡献评测 | 数据驱动调参 |
| P3 | BM25 / KG 扩展 | 增强复杂场景能力 |

## 12. 最小可落地版本

如果希望最快看到收益，建议先做一个最小版本：

1. 在 `state.py` 增加 `query_type` 和 `retrieval_plan`。
2. 在 `node_item_name_confirm.py` 中让 LLM 输出 `query_type`。
3. 用规则生成第一版 `retrieval_plan`：

```text
fact/safety -> local only
procedure -> local only
troubleshooting -> local + hyde
realtime -> web only 或 local + web
unknown -> local + hyde
```

4. 修改 `main_graph.py` 条件路由。
5. `node_web_search_mcp.py` 支持被跳过和超时降级。
6. `node_search_embedding_hyde.py` 支持被跳过。
7. `node_rerank.py` 增加输入数量上限和低置信度字段。
8. `node_answer_output.py` 根据低置信度字段决定是否生成拒答。
9. 跑 `data/eval`，与当前 baseline 对比。

这个版本改动范围可控，但能立刻验证核心收益：减少 HyDE/Web 调用、降低延迟和成本，同时保留本地检索主链路。
