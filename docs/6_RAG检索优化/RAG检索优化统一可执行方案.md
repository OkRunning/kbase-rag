# 	RAG 检索优化统一可执行方案

## 1. 方案定位

本方案整合以下两份文档的目标和实施细节：

- `RAG检索流程优化实施方案.md`：重点解决固定三路召回导致的延迟、成本、稳定性和检索质量问题。
- `查询流程标准RAG化改造方案.md`：重点将当前商品资料问答流程演进为标准生产级 RAG 流程。

统一后的目标不是一次性重构全部链路，而是在保留现有 LangGraph、接口和商品资料问答能力的前提下，分阶段落地：

1. 先降低 HyDE、Web Search、Rerank 的无效调用成本。
2. 再拆分查询理解、澄清、检索计划和证据门控。
3. 最后建立 Evidence Schema、Citation、评测和灰度闭环。

## 2. 当前问题归并

当前查询链路大致为：

```text
用户发起查询
  -> node_item_name_confirm
       - 商品名称确认
       - 用户问题改写
  -> 并行执行 node_search_embedding / node_search_embedding_hyde / node_web_search_mcp
  -> node_rrf 融合本地向量和 HyDE 结果
  -> node_rerank 合并 RRF 与 Web 结果并重排
  -> node_answer_output 生成答案
```

合并后的问题清单：

- 查询入口过度绑定商品名称，泛知识库问题也可能被“商品名未确认”卡住。
- `node_item_name_confirm` 职责过重，混合了历史读取、商品识别、问题改写、商品检索和流程决策。
- 检索策略固定，简单事实问题也默认调用 HyDE 和 Web Search。
- Web Search 默认参与，容易污染内部知识库答案，并拉高 P95/P99。
- HyDE 每次额外调用 LLM，成本和延迟较高，对明确事实问题收益有限。
- RRF、Rerank 的 topK、权重、阈值固定，缺少按问题类型的动态策略。
- 缺少统一 Evidence Schema，本地、HyDE、Web、后续 KG 结果结构不一致。
- 缺少证据质量门控，低相关证据也可能进入生成阶段。
- 答案缺少稳定的结构化引用来源。
- 历史对话缺少指代消解和压缩策略。
- 缺少可持续评测集、分源贡献指标和灰度回滚闭环。

## 3. 统一目标

### 3.1 产品目标

- 支持商品资料问答，同时支持不强绑定商品名称的通用知识库问答。
- 只有当问题必须绑定商品或实体且信息不足时，才向用户澄清。
- 对证据不足、证据冲突、低置信度场景明确拒答或谨慎回答。
- 答案可追溯到文档、chunk、标题、页码、URL 等来源。
- Web Search 从默认参与变为受控补充。

### 3.2 工程目标

- 保持现有接口尽量兼容，小步改造，避免一次性大重构。
- 新增能力全部带环境变量开关，可以灰度和回滚。
- 检索策略配置化，避免 topK、权重、阈值散落在节点代码中。
- 新增或修改的公开函数使用 Google Style Docstring。
- 复杂业务逻辑、状态转换、LangGraph State 更新增加中文注释。

### 3.3 指标目标

- 降低平均延迟、P95、P99 和首 token 时间。
- 降低 HyDE 调用率、Web Search 调用率、平均 LLM 调用次数和 token 消耗。
- 提升 Recall@K、MRR、nDCG、faithfulness、citation coverage。
- 降低 unsupported answer rate 和无依据确定性回答比例。

## 4. 目标架构

推荐最终链路：

```text
1. Query Intake
   请求校验、session、任务状态初始化、限流

2. Conversation Context
   最近历史提取、指代消解、必要时长历史摘要

3. Query Understanding
   意图识别、实体抽取、问题改写、query_type 判断

4. Entity Resolution
   仅在需要商品名或实体时执行匹配、消歧和置信度判断

5. Clarification Decision
   判断是否必须向用户追问

6. Retrieval Planning
   生成 retrieval_plan，决定 local / HyDE / Web / KG / filters / topK / weights

7. Retrieval
   按 retrieval_plan 条件执行召回节点

8. Evidence Fusion
   EvidenceDocument 标准化、去重、RRF、相邻 chunk 补全、分组保底

9. Rerank
   控制输入数量，重排候选证据，输出分数分布

10. Evidence Gate
    判断证据是否足够、是否冲突、是否只来自外部来源

11. Answer Generation
    按 query_type 和 evidence_status 生成答案或拒答

12. Post Processing
    构建 sources、image_urls、历史写入、观测事件记录
```

第一阶段不要求全部节点拆完，可以先在现有节点中产出兼容字段，再逐步拆分。

## 5. State 和核心数据结构

### 5.1 QueryGraphState 新增字段

文件：

```text
app/query_process/agent/state.py
```

建议新增字段：

```python
intent: str
entities: dict
need_item_name: bool
needs_clarification: bool
clarification_question: str
query_type: str
retrieval_plan: dict
retrieval_diagnostics: dict
rerank_quality: dict
evidence_status: str
evidence_gate: dict
sources: list
```

字段语义：

| 字段 | 说明 |
| --- | --- |
| `intent` | 用户意图，例如 `qa`、`compare`、`summary`、`troubleshooting`、`chitchat` |
| `entities` | 结构化实体，例如商品名、型号、部件、错误码 |
| `need_item_name` | 当前问题是否必须绑定商品名称 |
| `needs_clarification` | 是否需要追问用户 |
| `clarification_question` | 需要追问时的问题 |
| `query_type` | 检索问题类型 |
| `retrieval_plan` | 检索计划 |
| `retrieval_diagnostics` | 各召回源数量、分数、跳过原因、异常状态 |
| `rerank_quality` | Rerank 分数分布和可回答性判断 |
| `evidence_status` | `sufficient`、`low_confidence`、`insufficient`、`conflicting`、`external_only` |
| `evidence_gate` | 证据门控原因、缺失信息和置信度 |
| `sources` | 前端可展示的结构化来源 |

默认值必须完整，避免条件路由后未执行节点导致 KeyError：

```python
embedding_chunks = []
hyde_embedding_chunks = []
web_search_docs = []
rrf_chunks = []
reranked_docs = []
sources = []
retrieval_plan = {}
retrieval_diagnostics = {}
rerank_quality = {}
evidence_gate = {}
```

### 5.2 query_type 枚举

```text
fact             参数、定义、明确事实
procedure        操作步骤、使用流程
troubleshooting  故障排查、异常诊断
safety           安全注意事项、禁止操作
comparison       多商品或多方案对比
summary          文档或章节摘要
follow_up        依赖历史的追问
realtime         明确需要实时信息
chitchat         闲聊或非知识库问题
unknown          无法判断
```

### 5.3 Retrieval Plan Schema

第一版可以直接使用 `dict`，后续再收敛为 TypedDict 或 Pydantic Model。

```json
{
  "use_local_vector": true,
  "use_hyde": false,
  "use_web": false,
  "use_kg": false,
  "local_top_k": 8,
  "hyde_top_k": 5,
  "web_top_k": 3,
  "rrf_top_k": 8,
  "rerank_top_k": 5,
  "dense_weight": 0.85,
  "sparse_weight": 0.15,
  "filters": {
    "tenant_id": "default",
    "kb_id": "default",
    "item_name": [],
    "doc_status": "active",
    "vector_version": "active",
    "language": ""
  },
  "source_weights": {
    "local_vector": 1.0,
    "hyde": 0.7,
    "web": 0.5,
    "kg": 0.8
  },
  "reason": "本地产品手册类事实问题，优先使用本地知识库"
}
```

### 5.4 EvidenceDocument Schema

建议新增文件：

```text
app/query_process/agent/evidence.py
```

结构：

```python
class EvidenceDocument(TypedDict, total=False):
    """RAG 证据文档。

    Args:
        id: 证据唯一 ID。
        source_type: 来源类型，例如 local、hyde、web、kg。
        text: 证据文本。
        title: 标题。
        file_title: 文件标题。
        parent_title: 父级章节标题。
        item_name: 商品名称。
        chunk_id: Chunk ID。
        content_sha256: 证据文本哈希。
        page: 页码。
        url: 来源 URL。
        domain: Web 来源域名。
        retrieval_score: 召回分数。
        rrf_score: RRF 融合分数。
        rerank_score: 重排分数。
        source_hits: 命中的召回源列表。
        trust_level: 来源可信度。
        metadata: 扩展元数据。

    Returns:
        标准证据文档结构。
    """
```

数据规范：

| 来源 | 必填字段 | 说明 |
| --- | --- | --- |
| local | `id`、`source_type`、`text`、`chunk_id` | 尽量带 `file_title`、`title`、`parent_title`、`page` |
| hyde | `id`、`source_type`、`text`、`chunk_id` | 本质仍是本地 chunk，需保留 `source_hits` |
| web | `id`、`source_type`、`text`、`title`、`url` | 需带 `domain`、`trust_level` |
| kg | `id`、`source_type`、`text`、`metadata` | 后续扩展实体、关系、路径摘要 |

## 6. 检索策略

### 6.1 query_type 路由策略

| query_type | local vector | HyDE | Web | KG | 默认策略 |
| --- | --- | --- | --- | --- | --- |
| fact | 是 | 否 | 否 | 可选 | 明确参数、定义、型号优先本地 |
| procedure | 是 | 可选 | 否 | 否 | 操作步骤先本地，召回不足再 HyDE |
| troubleshooting | 是 | 是 | 可选 | 可选 | 故障描述模糊，HyDE 常有价值 |
| safety | 是 | 否 | 否 | 否 | 安全类优先官方本地资料 |
| comparison | 是 | 否 | 否 | 可选 | 按 item/entity 分组检索 |
| summary | 是 | 否 | 否 | 否 | 依赖指定文档或过滤条件 |
| realtime | 可选 | 否 | 是 | 否 | 明确实时信息才启用 Web |
| chitchat | 否 | 否 | 否 | 否 | 不进入 RAG 检索 |
| unknown | 是 | 可选 | 否 | 否 | 保守走本地，必要时澄清 |

### 6.2 dense/sparse 权重

| 场景 | dense_weight | sparse_weight |
| --- | --- | --- |
| 语义描述、操作步骤 | 0.8 - 0.9 | 0.1 - 0.2 |
| 型号、参数、错误码 | 0.5 - 0.7 | 0.3 - 0.5 |
| 精确短语、标题检索 | 0.4 - 0.6 | 0.4 - 0.6 |

### 6.3 HyDE 触发规则

默认按 `retrieval_plan.use_hyde` 执行，第二阶段增加二次门控：

```text
如果普通向量检索 top1_score >= 0.75 且 top3_avg_score >= 0.65，则跳过 HyDE。
如果普通向量检索为空或 top1_score < 0.45，则触发 HyDE。
```

HyDE 假设答案长度限制：

```text
中文不超过 200 字
英文不超过 120 words
```

HyDE 缓存 key：

```text
hyde:{model}:{sha256(rewritten_query)}
```

### 6.4 Web Search 使用原则

Web Search 默认关闭，满足以下条件之一才启用：

- 用户明确要求联网、最新、实时。
- `query_type = realtime`。
- 本地召回为空或低置信度，且问题允许外部资料补充。
- 管理端配置允许 Web 作为补充来源。

建议超时：

```text
connect_timeout: 3s
tool_call_timeout: 5s - 8s
max_retry_attempts: 1 - 2
```

Web 过滤：

```text
domain_allowlist
domain_blocklist
title/snippet 去重
snippet 最小长度
广告/聚合页过滤
```

Web 缓存 key：

```text
web:{sha256(normalized_query)}:{date_bucket}
```

非强实时场景 TTL：10 到 30 分钟。

### 6.5 RRF 融合规则

`node_rrf` 需要承担标准化、去重和融合：

- 从 `retrieval_plan.rrf_top_k` 读取 topK。
- 从 `retrieval_plan.source_weights` 读取来源权重。
- 按 `chunk_id`、`content_sha256`、`url`、`kg_path` 去重。
- 保留 `rrf_score` 和 `source_hits`。
- 对多 item/entity 问题，每个 item/entity 至少保留 `min_per_item` 条候选。
- 对同一文档相邻 chunk 可做窗口补全。

RRF 输出示例：

```json
{
  "id": "chunk_001",
  "source_type": "local",
  "chunk_id": "chunk_001",
  "text": "...",
  "rrf_score": 0.032,
  "source_hits": ["local_vector", "hyde"],
  "metadata": {}
}
```

### 6.6 Rerank 规则

建议硬上限：

```text
local_candidates <= 12
web_candidates <= 3
total_candidates <= 15
```

answerability 初始阈值：

```text
top1_score < 0.35 -> insufficient
top1_score >= 0.35 且 top1_score < 0.5 -> low_confidence
top1_score >= 0.5 -> answerable
```

输出质量字段：

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

Reranker 降级：

```text
Reranker 正常 -> 使用 rerank 排序结果
Reranker 超时 -> 使用 RRF 排序结果
Reranker OOM/加载失败 -> 熔断 reranker，短时间内跳过 rerank
```

### 6.7 Evidence Gate 规则

第一版使用规则门控，不额外引入 LLM：

```text
reranked_docs 为空 -> insufficient
top rerank score 低于阈值 -> insufficient
top 文档来源全是 Web 且本地知识库为空 -> external_only
对比类问题缺少任一实体证据 -> insufficient
多个高分证据结论冲突 -> conflicting
```

输出：

```json
{
  "evidence_status": "sufficient",
  "confidence": 0.82,
  "reason": "Top 文档覆盖了操作区域设置步骤",
  "missing_info": []
}
```

## 7. 分阶段执行计划

### 阶段 0：基线和保护开关

周期：1 到 2 天。

目标：先建立可回滚、可观测的改造边界。

涉及文件：

```text
app/conf/*
app/query_process/agent/state.py
app/query_process/agent/main_graph.py
app/query_process/agent/nodes/*
test/test_query_main_graph.py
test/test_main_graph.py
```

实施动作：

1. 新增配置开关，默认保持旧流程行为：

```text
QUERY_USE_STANDARD_UNDERSTANDING=false
QUERY_USE_RETRIEVAL_PLANNER=false
QUERY_WEB_SEARCH_DEFAULT=false
QUERY_WEB_SEARCH_ON_LOW_CONFIDENCE=true
QUERY_ENABLE_EVIDENCE_GATE=false
QUERY_ENABLE_CITATIONS=false
QUERY_RERANK_FALLBACK_ENABLED=true
QUERY_LOW_CONFIDENCE_REFUSAL_ENABLED=false
```

2. 扩展 `QueryGraphState` 默认字段，但不改变旧节点输出。
3. 给现有节点增加最小诊断信息：

```text
node_name
duration_ms
input_count
output_count
skip_reason
error_type
```

4. 固化当前 baseline 用例和指标。

验收标准：

- 所有开关关闭时，旧流程行为不变。
- 现有测试通过。
- Trace 或日志中能看到各节点耗时和候选数量。

回滚方式：

```text
关闭所有新增开关。
```

### 阶段 1：低风险降本提速

周期：1 到 2 周。

目标：不大改图结构，先减少无效 HyDE、Web 和 Rerank 成本。

涉及文件：

```text
app/query_process/agent/state.py
app/query_process/agent/nodes/node_item_name_confirm.py
app/query_process/agent/nodes/node_search_embedding_hyde.py
app/query_process/agent/nodes/node_web_search_mcp.py
app/query_process/agent/nodes/node_rerank.py
app/query_process/agent/nodes/node_answer_output.py
app/query_process/agent/main_graph.py
prompts/query_understanding.prompt
prompts/rewritten_query_and_itemnames.prompt
prompts/answer_out.prompt
test/test_query_main_graph.py
test/test_main_graph_full_flow.py
```

实施动作：

1. 在 `node_item_name_confirm.py` 中先产出 `query_type` 和第一版 `retrieval_plan`。
2. 第一版 plan 使用规则生成，不强依赖新增 LLM prompt：

```text
fact/safety -> local only
procedure -> local only
troubleshooting -> local + HyDE
realtime -> web only 或 local + web
comparison -> local only，并按 item 分组
unknown -> local，低置信时可 HyDE
chitchat -> 不检索
```

3. `node_search_embedding_hyde.py` 支持按 `retrieval_plan.use_hyde=false` 跳过，返回空列表和 `skip_reason`。
4. `node_web_search_mcp.py` 支持按 `retrieval_plan.use_web=false` 跳过，并降低 timeout。
5. `node_rerank.py` 增加输入数量硬上限和 `rerank_quality`。
6. `node_answer_output.py` 支持 `rerank_quality.answerability`，低置信时谨慎回答或资料不足。
7. 记录 HyDE 调用率、Web 调用率、Rerank 输入数量、低置信比例。

验收标准：

- `fact`、`safety` 类问题默认不调用 HyDE 和 Web。
- `troubleshooting`、模糊描述类问题可以触发 HyDE。
- `realtime` 类问题可以触发 Web。
- Web 超时不会中断主链路。
- Reranker 失败时可降级为 RRF 排序。
- 低置信度问题不会生成确定性无依据答案。

风险：

- Planner 规则过度关闭 HyDE，导致部分模糊问题召回下降。

控制：

- 只对高置信 `fact/safety` 跳过 HyDE。
- 保留 `QUERY_USE_RETRIEVAL_PLANNER=false` 回滚旧固定三路召回。

### 阶段 2：标准查询理解和澄清拆分

周期：2 到 3 周。

目标：将商品名确认从入口硬约束改为按需实体解析，支持通用知识库问答。

新增文件：

```text
app/query_process/agent/nodes/node_query_understanding.py
app/query_process/agent/nodes/node_entity_resolution.py
app/query_process/agent/nodes/node_clarification_decision.py
```

可复用文件：

```text
app/query_process/agent/nodes/query_understanding_utils.py
app/query_process/agent/nodes/node_item_name_confirm.py
prompts/query_understanding.prompt
```

实施动作：

1. 新增 `node_query_understanding`，负责意图识别、实体抽取、指代消解、问题改写和 `need_item_name` 判断。
2. 将旧 `node_item_name_confirm.py` 中商品向量匹配、消歧逻辑迁移或包装到 `node_entity_resolution`。
3. 新增 `node_clarification_decision`，根据实体匹配结果和 `need_item_name` 决定是否追问。
4. 调整 `main_graph.py` 路由：

```text
node_query_understanding
  -> node_entity_resolution
  -> node_clarification_decision
  -> needs_clarification=True: node_answer_output
  -> needs_clarification=False: node_retrieval_planner 或旧检索链路
```

5. 历史指代先做轻量版：最近 3 轮历史只进入 Query Understanding，不直接污染最终证据上下文。

LLM 输出结构：

```json
{
  "intent": "qa",
  "query_type": "procedure",
  "rewritten_query": "HAK 180 烫金机局部烫金区域如何设置",
  "entities": {
    "item_names": ["HAK 180 烫金机"],
    "model_numbers": ["HAK 180"],
    "parts": [],
    "error_codes": []
  },
  "need_item_name": true,
  "reason": "用户询问设备操作步骤，需要限定具体设备型号"
}
```

验收标准：

- 不需要商品名的问题不会因为缺少商品名提前结束。
- 需要商品名但用户没说清时，返回澄清问题，不进入检索生成。
- 历史中已有商品名时，追问能补全为完整查询。
- 原有商品问答路径保持可用。

回滚方式：

```text
QUERY_USE_STANDARD_UNDERSTANDING=false
```

### 阶段 3：Retrieval Planner 和条件路由

周期：2 到 4 周。

目标：从固定并行检索升级为“查询理解驱动的动态检索计划”。

新增文件：

```text
app/query_process/agent/nodes/node_retrieval_planner.py
prompts/retrieval_planner.prompt
```

修改文件：

```text
app/query_process/agent/main_graph.py
app/query_process/agent/nodes/node_search_embedding.py
app/query_process/agent/nodes/node_search_embedding_hyde.py
app/query_process/agent/nodes/node_web_search_mcp.py
```

实施动作：

1. `node_retrieval_planner` 输入：

```json
{
  "original_query": "...",
  "rewritten_query": "...",
  "history": [],
  "intent": "qa",
  "query_type": "fact",
  "entities": {},
  "item_names": [],
  "tenant_id": "default",
  "kb_id": "default"
}
```

2. `node_retrieval_planner` 输出标准 `retrieval_plan`。
3. `main_graph.py` 按 plan 动态路由：

```text
if needs_clarification:
    -> node_answer_output
else:
    if use_local_vector -> node_search_embedding
    if use_hyde -> node_search_embedding_hyde
    if use_web -> node_web_search_mcp
    if use_kg -> node_query_kg
```

4. 没有任何召回源时，直接进入 `node_answer_output`，处理闲聊、澄清或拒答。
5. `node_search_embedding.py` 从 plan 读取 `local_top_k`、`dense_weight`、`sparse_weight`、`filters`。
6. `node_search_embedding_hyde.py` 从 plan 读取 `hyde_top_k`，并加入二次触发条件。
7. `node_web_search_mcp.py` 从 plan 读取 `web_top_k`，加入可信域过滤和缓存。
8. 所有跳过、成功、失败、超时、熔断都写入 `retrieval_diagnostics`。

验收标准：

- 简单参数查询只走 local vector。
- 故障排查问题可以走 local vector + HyDE。
- 明确实时问题可以走 Web。
- 对比问题能为多个实体生成检索计划。
- Trace 中能看到 `query_type`、`retrieval_plan` 和跳过原因。

风险：

- LangGraph 并行和汇聚状态处理复杂度上升。

控制：

- 未执行节点统一写默认空列表。
- 为每种 plan 增加单元测试和集成测试。

### 阶段 4：Evidence Schema、融合、Rerank 和证据门控

周期：2 到 4 周。

目标：统一多源证据结构，先判断证据质量，再决定是否回答。

新增文件：

```text
app/query_process/agent/evidence.py
app/query_process/agent/nodes/node_evidence_gate.py
```

修改文件：

```text
app/query_process/agent/nodes/node_rrf.py
app/query_process/agent/nodes/node_rerank.py
app/query_process/agent/main_graph.py
app/lm/reranker_utils.py
app/conf/reranker_config.py
```

实施动作：

1. 定义 `EvidenceDocument`，并为 local、HyDE、Web 增加标准化函数。
2. 重构 `node_rrf.py`：

- 标准化多源证据。
- 按 `chunk_id/content_sha256/url` 去重。
- 支持 `source_weights` 和动态 `rrf_top_k`。
- 保留 `rrf_score`、`source_hits`。
- 支持多 item/entity 分组保底。

3. 增加相邻 chunk 补全：

```text
top chunk -> previous chunk + current chunk + next chunk
```

配置：

```text
QUERY_CONTEXT_EXPAND_NEIGHBORS=true
QUERY_CONTEXT_NEIGHBOR_SIZE=1
```

4. 重构 `node_rerank.py`：

- 输入统一为 `EvidenceDocument`。
- 控制输入上限。
- 输出 `rerank_quality`。
- Reranker 超时或 OOM 时降级为 RRF 排序。

5. 新增 `node_evidence_gate.py`：

- 判断 `sufficient`、`low_confidence`、`insufficient`、`conflicting`、`external_only`。
- 将原因写入 `evidence_gate` 和观测事件。

配置：

```text
QUERY_EVIDENCE_MIN_RERANK_SCORE=0.45
QUERY_EVIDENCE_MIN_DOCS=1
QUERY_ALLOW_EXTERNAL_ONLY=false
```

验收标准：

- Rerank 输入统一为 EvidenceDocument。
- 同一个 chunk 不重复进入 Prompt。
- 多 item 对比不会只保留单一 item 的证据。
- Reranker 失败时链路不报错。
- 无检索结果或低分召回时不生成编造答案。
- Evidence Gate 结果进入可观测事件。

### 阶段 5：答案生成、Citation 和接口返回

周期：1 到 2 周。

目标：答案只基于证据生成，并返回结构化来源。

修改文件：

```text
app/query_process/agent/nodes/node_answer_output.py
app/query_process/api/query__server.py
prompts/answer_out.prompt
```

建议新增函数：

```python
def build_answer_sources(reranked_docs: list[dict]) -> list[dict]:
    """构建答案引用来源。

    Args:
        reranked_docs: 重排后的证据文档列表。

    Returns:
        前端可展示的结构化引用来源。
    """
```

Prompt 约束：

```text
只能依据给定资料回答。
如果资料不足，明确说明不足。
关键结论后标注引用编号，例如 [1]、[2]。
不要使用未出现在资料中的参数、步骤或结论。
不要编造型号、法规、网址或操作步骤。
```

Prompt Context 格式：

```text
[1]
source_type: local
file_title: HAK 180 操作手册.pdf
section: 操作面板 > 局部烫金设置
page: 12
score: 0.87
text: ...
```

按 `query_type` 的答案结构：

| query_type | 输出结构 |
| --- | --- |
| fact | 直接答案 + 来源 |
| procedure | 步骤列表 + 注意事项 |
| troubleshooting | 可能原因 + 检查步骤 + 安全提醒 |
| safety | 安全注意事项 + 禁止操作 |
| comparison | 按 item/entity 分组对比表 |
| summary | 摘要 + 关键条目 + 来源 |

同步接口返回：

```json
{
  "answer": "...",
  "session_id": "...",
  "sources": [
    {
      "ref_id": 1,
      "source_type": "local",
      "file_title": "HAK 180 操作手册.pdf",
      "title": "局部烫金设置",
      "chunk_id": "xxx",
      "page": 12,
      "score": 0.87,
      "url": ""
    }
  ],
  "image_urls": [],
  "message": "本次查询处理完毕！"
}
```

流式 `FINAL` 返回：

```json
{
  "answer": "...",
  "status": "completed",
  "sources": [],
  "image_urls": []
}
```

验收标准：

- 每次 RAG 回答都返回 `sources`。
- 答案中引用编号能对应 sources。
- Web 来源和本地来源可区分。
- `evidence_status=insufficient` 时 sources 为空或仅返回参考证据，answer 明确说明资料不足。
- 输出 token 数可观测并下降。

### 阶段 6：缓存、评测和策略调参

周期：持续迭代。

目标：让优化从经验驱动变为数据驱动。

新增目录和文件：

```text
test/eval/query_cases.jsonl
test/eval/run_query_eval.py
output/eval/query_eval_YYYYMMDD.json
```

缓存策略：

| 缓存对象 | Key | TTL | 说明 |
| --- | --- | --- | --- |
| Query Understanding | `qu:{tenant}:{sha256(query+history摘要)}` | 5-30 min | 减少重复 LLM 理解 |
| Embedding | `emb:{model}:{sha256(text)}` | 7-30 days | 高复用、低风险 |
| HyDE | `hyde:{model}:{sha256(query)}` | 1-7 days | 降低 HyDE LLM 成本 |
| Web Search | `web:{sha256(query)}:{date_bucket}` | 10-30 min | 控制实时性 |

评测样本结构：

```json
{
  "case_id": "q001",
  "query": "HAK 180 如何设置局部烫金区域？",
  "session_history": [],
  "query_type": "procedure",
  "expected_item_names": ["HAK 180 烫金机"],
  "expected_chunk_ids": ["chunk_001", "chunk_002"],
  "expected_answer_points": ["设置起始位置", "设置结束位置"],
  "required_keywords": ["局部", "区域"],
  "forbidden_keywords": ["无法确认但仍编造的参数"],
  "should_answer": true
}
```

样本覆盖：

| 类型 | 建议数量 |
| --- | --- |
| fact | 20+ |
| procedure | 20+ |
| troubleshooting | 20+ |
| safety | 10+ |
| comparison | 10+ |
| follow_up | 10+ |
| summary | 10+ |
| insufficient | 10+ |

指标：

| 类别 | 指标 |
| --- | --- |
| 响应时间 | 平均耗时、P50、P95、P99、首 token 时间、超时率 |
| 成本 | LLM 调用次数、input/output token、HyDE 调用率、Web 调用率、Rerank 输入数量 |
| 检索 | Recall@K、MRR、nDCG、context precision、local_recall_at_k、hyde_unique_hit_ratio |
| 排序 | rerank_top1_score_avg、score_gap、low_confidence_ratio、insufficient_ratio |
| 答案 | faithfulness、answer relevance、citation coverage、unsupported answer rate、no-answer precision |

A/B 对比：

```text
baseline: 当前固定三路召回
candidate: 动态 retrieval_plan + evidence gate
```

每次策略调整必须对比：

- 平均耗时、P95、P99。
- 平均 LLM 调用次数。
- 平均输入/输出 token。
- Web Search 调用率。
- HyDE 调用率。
- Recall@K、MRR@K。
- 低置信拒答比例。
- 人工抽样满意度。

验收标准：

- 每次检索策略调整前后可以比较指标。
- 能定位是 Query Understanding、Retrieval、Rerank 还是 Generation 出问题。
- 至少覆盖 30 条核心业务问题后再进入生产灰度。

## 8. 代码改造清单

### 8.1 State 层

```text
app/query_process/agent/state.py
```

改造：

- 增加查询理解字段。
- 增加检索计划字段。
- 增加诊断、证据门控和 sources 字段。
- 为所有条件分支可能跳过的节点提供默认空值。

### 8.2 节点层

新增：

```text
app/query_process/agent/nodes/node_query_understanding.py
app/query_process/agent/nodes/node_entity_resolution.py
app/query_process/agent/nodes/node_clarification_decision.py
app/query_process/agent/nodes/node_retrieval_planner.py
app/query_process/agent/nodes/node_evidence_gate.py
app/query_process/agent/evidence.py
```

重构：

```text
app/query_process/agent/nodes/node_item_name_confirm.py
app/query_process/agent/nodes/node_search_embedding.py
app/query_process/agent/nodes/node_search_embedding_hyde.py
app/query_process/agent/nodes/node_web_search_mcp.py
app/query_process/agent/nodes/node_rrf.py
app/query_process/agent/nodes/node_rerank.py
app/query_process/agent/nodes/node_answer_output.py
```

### 8.3 Graph 层

```text
app/query_process/agent/main_graph.py
```

改造：

- 阶段 1：保留旧入口，支持 plan 控制节点跳过。
- 阶段 2：新增标准 Query Understanding 链路。
- 阶段 3：新增 Retrieval Planner 条件路由。
- 阶段 4：Rerank 后新增 Evidence Gate。

### 8.4 Prompt 层

新增或修改：

```text
prompts/query_understanding.prompt
prompts/retrieval_planner.prompt
prompts/answer_out.prompt
prompts/hyde_prompt.prompt
```

### 8.5 API 层

```text
app/query_process/api/query__server.py
```

改造：

- 同步返回增加 `sources` 和 `image_urls`。
- 流式 `FINAL` 增加 `sources` 和 `image_urls`。
- 查询状态接口可返回 `evidence_status`、`query_type`、`retrieval_plan`。

### 8.6 配置层

建议按现有配置风格新增：

```text
QUERY_USE_STANDARD_UNDERSTANDING=false
QUERY_USE_RETRIEVAL_PLANNER=false
QUERY_WEB_SEARCH_DEFAULT=false
QUERY_WEB_SEARCH_ON_LOW_CONFIDENCE=true
QUERY_WEB_SEARCH_MIN_LOCAL_SCORE=0.55
QUERY_ENABLE_EVIDENCE_GATE=false
QUERY_ENABLE_CITATIONS=false
QUERY_RERANK_FALLBACK_ENABLED=true
QUERY_LOW_CONFIDENCE_REFUSAL_ENABLED=false
QUERY_CONTEXT_EXPAND_NEIGHBORS=false
QUERY_CONTEXT_NEIGHBOR_SIZE=1
QUERY_EVIDENCE_MIN_RERANK_SCORE=0.45
QUERY_EVIDENCE_MIN_DOCS=1
QUERY_ALLOW_EXTERNAL_ONLY=false
```

## 9. 测试计划

### 9.1 单元测试

建议新增或更新：

```text
test/test_query_main_graph.py
test/test_main_graph.py
test/test_main_graph_full_flow.py
```

覆盖：

- `fact` 问题只走 local vector。
- `safety` 问题只走 local vector。
- `troubleshooting` 问题走 local vector + HyDE。
- `realtime` 问题走 Web。
- `chitchat` 不进入 RAG 检索。
- 无 item_name 但问题不需要商品名时继续通用 RAG。
- 需要 item_name 但用户未说明时提前澄清。
- 历史指代问题能补全 rewritten_query。
- Web 超时时返回空 Web 结果。
- Reranker 异常时降级 RRF。
- Rerank top1 低于阈值时输出 insufficient。
- Citation 编号和 sources 一一对应。

### 9.2 集成测试

覆盖完整链路：

- 旧流程开关关闭时保持兼容。
- 新 Query Understanding 链路开启后商品问答可用。
- Retrieval Planner 开启后条件路由正确。
- Evidence Gate 开启后低置信拒答正确。
- 流式返回 `FINAL` 包含 `sources`。

### 9.3 压测

场景：

- 本地检索高并发。
- HyDE 高并发。
- Web Search 慢调用和超时。
- Reranker 批量输入上限。
- Answer LLM 慢调用。

观察：

- QPS。
- P95/P99。
- 错误率和超时率。
- Reranker 队列堆积。
- 缓存命中率。
- HyDE/Web 熔断是否生效。

## 10. 验收用例

### 10.1 不需要商品名的问题

输入：

```text
知识库里有哪些类型的设备文档？
```

预期：

- 不返回“没有找到对应的商品名称”。
- 可以走 summary 或 general QA。

### 10.2 需要商品名但用户没说清

输入：

```text
这个机器怎么设置局部区域？
```

历史为空。

预期：

- 返回澄清问题。
- 不进入检索生成。

### 10.3 历史指代问题

历史：

```text
用户：HAK 180 烫金机怎么设置温度？
助手：...
```

输入：

```text
那局部区域怎么设置？
```

预期：

- `rewritten_query` 补全为 HAK 180 烫金机局部区域设置。
- 正常进入检索。

### 10.4 本地知识库足够

输入：

```text
HAK 180 如何设置顶部 50mm 到 170mm 的烫金区域？
```

预期：

- 默认不启用 Web Search。
- 返回答案和本地 sources。

### 10.5 故障排查

输入：

```text
HAK 180 局部烫金位置总是偏移，应该检查什么？
```

预期：

- 触发 local vector + HyDE。
- 答案按可能原因、检查步骤、安全提醒组织。

### 10.6 明确实时问题

输入：

```text
请联网查询 HAK 180 今年最新固件是否发布。
```

预期：

- 触发 Web Search。
- Web sources 标记为外部来源。
- 如果本地证据不足且外部来源不可信，谨慎回答。

### 10.7 证据不足

输入：

```text
HAK 180 支不支持 2026 年最新云端远程控制？
```

预期：

- 本地证据不足时不编造。
- 未启用 Web 时明确说明知识库无依据。

## 11. 发布和回滚

### 11.1 灰度顺序

1. 本地单元测试和集成测试通过。
2. 跑 eval baseline 和 candidate 对比。
3. 测试环境全量开启。
4. 生产按租户或流量比例灰度：

```text
5% -> 20% -> 50% -> 100%
```

每个阶段至少观察：

- 查询成功率。
- 平均耗时、P95、P99。
- 平均成本。
- HyDE 调用率。
- Web Search 调用率。
- 低置信拒答比例。
- 人工抽样质量。

### 11.2 回滚条件

出现以下任一情况，回滚到上一阶段：

- Recall@K 明显下降。
- 用户反馈答案缺失明显增加。
- 低置信拒答比例异常升高。
- P95/P99 未下降反而升高。
- Query Understanding 结构化输出错误率过高。
- State 字段缺省导致线上 KeyError。

### 11.3 回滚方式

按阶段关闭：

```text
QUERY_ENABLE_CITATIONS=false
QUERY_ENABLE_EVIDENCE_GATE=false
QUERY_USE_RETRIEVAL_PLANNER=false
QUERY_USE_STANDARD_UNDERSTANDING=false
```

必要时恢复固定三路召回：

```text
QUERY_USE_RETRIEVAL_PLANNER=false
QUERY_WEB_SEARCH_DEFAULT=true
```

## 12. 优先级

| 优先级 | 事项 | 原因 |
| --- | --- | --- |
| P0 | 新增开关和 State 默认值 | 保证可回滚，避免条件路由 KeyError |
| P0 | Web Search 按需执行 + timeout 降低 | 直接降低 P95 和外部依赖风险 |
| P0 | HyDE 按需执行 | 直接减少 LLM 成本和延迟 |
| P0 | Rerank 输入上限和降级 | 控制模型推理压力，保证链路稳定 |
| P1 | `query_type` 和第一版 `retrieval_plan` | 动态路由基础 |
| P1 | Query Understanding / Clarification 拆分 | 支持标准 RAG 和泛知识库问答 |
| P1 | Evidence Gate | 降低幻觉和无依据回答 |
| P1 | Citation 和 sources 返回 | 提升答案可追溯性 |
| P2 | EvidenceDocument 标准化 | 支撑多源融合和长期扩展 |
| P2 | embedding / HyDE / Web 缓存 | 降低重复成本 |
| P2 | 分源贡献评测 | 数据驱动调参 |
| P3 | BM25 / KG / 相邻 chunk 扩展 | 增强复杂场景能力 |

## 13. 最小可落地版本

如果希望最快看到收益，先做以下 9 个动作：

1. 在 `state.py` 增加 `query_type`、`retrieval_plan`、`retrieval_diagnostics`、`rerank_quality`、`evidence_status`，并设置默认值。
2. 在 `node_item_name_confirm.py` 中输出 `query_type`。
3. 用规则生成第一版 `retrieval_plan`：

```text
fact/safety -> local only
procedure -> local only
troubleshooting -> local + HyDE
realtime -> web only 或 local + web
unknown -> local + HyDE
```

4. 修改 `node_search_embedding_hyde.py`，支持按 plan 跳过。
5. 修改 `node_web_search_mcp.py`，支持按 plan 跳过、缩短 timeout、失败返回空列表。
6. 修改 `node_rerank.py`，增加输入数量上限、降级和 `rerank_quality`。
7. 修改 `node_answer_output.py`，根据低置信字段输出资料不足或谨慎答案。
8. 增加单元测试覆盖 fact、troubleshooting、realtime、无 item 澄清、Web 超时、Reranker 降级。
9. 跑现有 `test/test_query_main_graph.py`、`test/test_main_graph.py`、`test/test_main_graph_full_flow.py`，再与 baseline 对比 HyDE/Web 调用率和 P95。

这个版本改动范围可控，但能立刻验证核心收益：减少 HyDE/Web 调用、降低延迟和成本，同时保留本地检索主链路。

## 14. 完成定义

统一方案完成时应满足：

- 查询不再固定绑定商品名，能按需澄清。
- 检索不再固定三路并行，能按 `retrieval_plan` 动态执行。
- Web Search 默认受控，不污染内部知识库答案。
- HyDE 只在模糊、故障或低召回场景启用。
- RRF、Rerank、Evidence Gate 均有可观测指标。
- 答案基于证据生成，并返回结构化 sources。
- 低置信或证据不足场景不会编造确定性答案。
- 每次策略调整都有 eval 对比、灰度开关和回滚路径。

## 15. 具体执行步骤

本节把前面的方案拆成可直接排期的任务。建议按顺序推进，每一步都应能单独提交、测试和回滚。

### Step 0：建立基线和开关

目标：保证后续所有优化都可以灰度和回滚。

任务：

1. 在配置层新增以下开关，并默认关闭新流程：

```text
QUERY_USE_STANDARD_UNDERSTANDING=false
QUERY_USE_RETRIEVAL_PLANNER=false
QUERY_WEB_SEARCH_DEFAULT=false
QUERY_ENABLE_EVIDENCE_GATE=false
QUERY_ENABLE_CITATIONS=false
QUERY_RERANK_FALLBACK_ENABLED=true
```

2. 在 `app/query_process/agent/state.py` 增加新字段默认值：

```text
query_type
retrieval_plan
retrieval_diagnostics
rerank_quality
evidence_status
evidence_gate
sources
```

3. 给现有节点补充最小观测日志：

```text
node_name
duration_ms
input_count
output_count
skip_reason
error_type
```

完成标准：

- 开关全部关闭时，旧链路行为不变。
- 现有测试通过。
- 每个节点至少能看到耗时和输出数量。

### Step 1：先做最小版 retrieval_plan

目标：不用先拆大节点，也能让 HyDE 和 Web Search 按需执行。

任务：

1. 在 `node_item_name_confirm.py` 中基于 LLM 输出或规则补出 `query_type`。
2. 在同一节点生成第一版 `retrieval_plan`。
3. 第一版规则：

```text
fact -> local only
safety -> local only
procedure -> local only
troubleshooting -> local + HyDE
comparison -> local only
realtime -> Web 或 local + Web
chitchat -> no retrieval
unknown -> local，必要时 HyDE
```

4. 将 `query_type` 和 `retrieval_plan` 写入 state 和观测事件。

完成标准：

- fact/safety 问题的 state 中 `use_hyde=false`、`use_web=false`。
- troubleshooting 问题的 state 中 `use_hyde=true`。
- realtime 问题的 state 中 `use_web=true`。
- Trace 中能看到 plan 的生成原因。

### Step 2：让 HyDE 节点可跳过

目标：减少不必要的 LLM 调用。

任务：

1. 修改 `node_search_embedding_hyde.py`，读取 `retrieval_plan.use_hyde`。
2. 当 `use_hyde=false` 时，直接返回：

```text
hyde_embedding_chunks=[]
retrieval_diagnostics.hyde.status=skipped
retrieval_diagnostics.hyde.skip_reason=disabled_by_plan
```

3. 当 HyDE LLM 失败时，不抛断主链路，返回空结果并记录错误。
4. 限制 HyDE 假设答案长度。

完成标准：

- fact/safety 问题不会调用 HyDE LLM。
- HyDE 超时或失败时，local vector 结果仍能继续进入后续链路。

### Step 3：让 Web Search 节点可跳过和降级

目标：降低 Web Search 对 P95/P99 和答案可信度的影响。

任务：

1. 修改 `node_web_search_mcp.py`，读取 `retrieval_plan.use_web`。
2. 当 `use_web=false` 时，直接返回：

```text
web_search_docs=[]
retrieval_diagnostics.web.status=skipped
retrieval_diagnostics.web.skip_reason=disabled_by_plan
```

3. 设置更短 timeout：

```text
connect_timeout=3s
tool_call_timeout=5s-8s
max_retry_attempts=1-2
```

4. Web 超时或异常时返回空列表，不中断主链路。
5. 给 Web 结果增加 `source_type=web`、`url`、`domain`、`trust_level`。

完成标准：

- 普通知识库问题默认不调用 Web。
- Web 超时不会拖慢或打断完整查询。
- 最终证据中能区分本地来源和 Web 来源。

### Step 4：限制 Rerank 输入并增加降级

目标：稳定 Reranker 成本和失败表现。

任务：

1. 修改 `node_rerank.py`，在调用模型前限制输入数量：

```text
local_candidates <= 12
web_candidates <= 3
total_candidates <= 15
```

2. 输出 `rerank_quality`：

```text
top1_score
top3_avg_score
score_gap
answerability
local_doc_count
web_doc_count
```

3. 增加 Reranker 降级：

```text
timeout -> 使用 RRF 或原始召回排序
OOM/model_load_error -> 熔断一段时间并跳过 rerank
```

完成标准：

- Rerank 输入数量稳定可控。
- Reranker 异常时查询不报错。
- state 中能看到 `rerank_quality.answerability`。

### Step 5：让 Answer Output 支持低置信拒答

目标：先解决“证据不足还强答”的问题。

任务：

1. 修改 `node_answer_output.py`，读取 `rerank_quality.answerability`。
2. 输出策略：

```text
answerable -> 正常生成
low_confidence -> 谨慎回答，并提示资料可能不足
insufficient -> 不编造，返回资料不足和可补充信息
```

3. 修改 `prompts/answer_out.prompt`，增加证据约束：

```text
只能依据给定资料回答。
没有资料支撑时必须说明资料不足。
不要编造参数、步骤、型号、法规、网址。
```

完成标准：

- 无证据或低分证据时不生成确定性答案。
- 普通高置信问题仍能正常回答。

### Step 6：拆分标准 Query Understanding 链路

目标：从“商品名确认入口”升级为标准查询理解入口。

任务：

1. 新增 `node_query_understanding.py`：

```text
意图识别
实体抽取
问题改写
query_type 判断
need_item_name 判断
```

2. 新增 `node_entity_resolution.py`，复用旧商品名匹配和消歧逻辑。
3. 新增 `node_clarification_decision.py`，集中判断是否需要追问。
4. 修改 `main_graph.py`，通过 `QUERY_USE_STANDARD_UNDERSTANDING` 切换新旧入口。

完成标准：

- 泛知识库问题不会被商品名确认卡住。
- 需要商品名但未提供时返回澄清。
- 历史指代问题能补全 rewritten_query。
- 关闭开关后能回到旧入口。

### Step 7：新增独立 Retrieval Planner 和条件路由

目标：把 plan 生成从旧入口节点中拆出来，形成独立职责。

任务：

1. 新增 `node_retrieval_planner.py`。
2. 新增 `prompts/retrieval_planner.prompt`。
3. 修改 `main_graph.py`，在 planner 后按 plan 条件进入检索节点：

```text
use_local_vector -> node_search_embedding
use_hyde -> node_search_embedding_hyde
use_web -> node_web_search_mcp
```

4. 没有任何检索源时，直接进入 `node_answer_output`。
5. 未执行节点必须写默认空列表，避免汇聚时报错。

完成标准：

- 不同 query_type 进入不同检索路径。
- LangGraph 并行汇聚稳定。
- plan 可以通过配置快速回滚。

### Step 8：标准化 EvidenceDocument 和融合节点

目标：让 RRF、Rerank、Prompt 和 Citation 使用同一证据结构。

任务：

1. 新增 `app/query_process/agent/evidence.py`。
2. 定义 `EvidenceDocument`。
3. 增加 local、HyDE、Web 到 EvidenceDocument 的转换函数。
4. 重构 `node_rrf.py` 为 Evidence Fusion：

```text
标准化
去重
RRF
source_hits 合并
rrf_score 保留
多 item/entity 分组保底
```

5. 只有单源少量候选时，可以只做标准化和去重，不必真正计算 RRF。

完成标准：

- Rerank 输入统一为 EvidenceDocument。
- 重复 chunk 不再重复进入 Prompt。
- 多源命中能通过 `source_hits` 追踪。

### Step 9：新增 Evidence Gate

目标：在生成答案前统一判断证据是否足够。

任务：

1. 新增 `node_evidence_gate.py`。
2. 第一版使用规则门控：

```text
reranked_docs 为空 -> insufficient
top rerank score 低于阈值 -> insufficient
只命中外部 Web 且不允许 external_only -> external_only
对比问题缺少任一实体证据 -> insufficient
高分证据结论冲突 -> conflicting
```

3. 将结果写入：

```text
evidence_status
evidence_gate
```

4. 修改 `main_graph.py`，让 `node_rerank` 后进入 `node_evidence_gate`。

完成标准：

- 证据不足时不进入普通生成路径。
- 对比问题缺少一方证据时能明确指出。
- Evidence Gate 结果进入 Trace。

### Step 10：增加 Citation 和 sources 返回

目标：让答案可追溯。

任务：

1. 在 `node_answer_output.py` 中新增 `build_answer_sources`。
2. Prompt context 使用编号格式：

```text
[1]
source_type:
file_title:
section:
page:
score:
text:
```

3. 修改 `app/query_process/api/query__server.py`：

```text
同步接口返回 sources
流式 FINAL 返回 sources
```

4. 答案中关键结论后标注 `[1]`、`[2]`。

完成标准：

- 每次 RAG 回答都有 sources。
- 引用编号和 sources 能对应。
- 无证据拒答时 sources 为空或明确标记为参考证据。

### Step 11：补充测试和评测集

目标：让策略调整可验证。

任务：

1. 更新单元测试：

```text
test/test_query_main_graph.py
test/test_main_graph.py
test/test_main_graph_full_flow.py
```

2. 新增评测数据：

```text
test/eval/query_cases.jsonl
```

3. 新增评测脚本：

```text
test/eval/run_query_eval.py
```

4. 每条 case 至少包含：

```text
query
query_type
expected_item_names
expected_chunk_ids
expected_answer_points
should_answer
```

完成标准：

- 覆盖 fact、procedure、troubleshooting、comparison、follow_up、insufficient。
- 能输出 baseline 和 candidate 对比报告。
- 每次策略调整能看到 Recall@K、MRR、P95、HyDE/Web 调用率变化。

### Step 12：灰度发布

目标：小流量验证收益和风险。

任务：

1. 测试环境开启：

```text
QUERY_USE_RETRIEVAL_PLANNER=true
QUERY_ENABLE_EVIDENCE_GATE=true
```

2. 生产灰度：

```text
5% -> 20% -> 50% -> 100%
```

3. 每个阶段观察：

```text
查询成功率
P95/P99
HyDE 调用率
Web 调用率
低置信拒答比例
人工抽样质量
```

完成标准：

- P95/P99 下降或不劣化。
- HyDE/Web 调用率下降。
- Recall@K 和人工质量不明显下降。
- 出现异常时能按开关回滚到上一阶段。
