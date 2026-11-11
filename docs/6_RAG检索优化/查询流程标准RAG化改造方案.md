# 查询流程标准 RAG 化改造方案

## 1. 背景

当前查询流程已经具备 LangGraph 编排、多路召回、RRF、Rerank、流式输出、历史记录和可观测事件等能力，但从标准 RAG 项目角度看，流程仍然偏向“商品资料问答 Demo”。

核心问题不是单个节点不可用，而是查询链路的业务假设过强：

- 所有问题都先进入商品名称确认。
- 商品名称不明确时直接结束，而不是判断是否真的需要商品名称。
- 检索策略固定为本地向量、HyDE、Web Search 并行。
- Web Search 默认参与，容易污染企业内部知识库问答。
- 缺少证据质量门控，无法稳定判断“是否应该回答”。
- 答案缺少结构化引用来源。
- 历史对话直接拼入 Prompt，缺少指代消解和记忆压缩。
- 导入侧和查询侧的 Chunk Schema 契约不够清晰。

本方案目标是将当前查询流程改造成更接近生产级标准 RAG 的结构，同时保留当前商品资料问答能力。

## 2. 改造目标

### 2.1 产品目标

- 支持不强绑定商品名称的通用知识库问答。
- 对需要商品名称的问题再触发澄清。
- 对证据不足的问题明确拒答或要求补充信息。
- 答案可追溯到具体文档、chunk、标题、来源。
- 不让外部 Web Search 默认污染内部知识库答案。

### 2.2 工程目标

- 保持现有接口尽量兼容。
- 保持 LangGraph 编排方式。
- 小步拆分节点，避免一次性大重构。
- 每阶段都有可验收结果和回滚方案。
- 新增或修改的公开函数遵守 Google Style Docstring。
- 复杂业务逻辑增加中文注释。

## 3. 推荐目标架构

目标查询链路如下：

```text
1. Query Intake
   请求校验、session、限流、任务状态初始化

2. Query Understanding
   意图识别、实体抽取、指代消解、查询改写

3. Clarification Decision
   判断是否必须向用户追问

4. Retrieval Planning
   生成检索计划，决定 local / hyde / web / kg / filters / top_k

5. Retrieval
   按计划执行多路召回

6. Evidence Fusion
   去重、RRF、metadata 合并、相邻 chunk 补全

7. Rerank
   基于问题对候选证据重排

8. Evidence Gate
   判断证据是否足够支撑回答

9. Answer Generation
   基于证据生成答案，并要求引用来源

10. Post Processing
   结构化 sources、图片、历史写入、质量事件记录
```

## 4. 分阶段执行计划

## 阶段一：查询理解和澄清决策拆分

### 4.1 目标

将当前 `node_item_name_confirm` 拆成更标准的查询理解模块，避免所有问题都被商品名称确认卡住。

### 4.2 当前问题

当前入口节点承担了过多职责：

- 读取历史。
- 调用 LLM 抽取商品名。
- 重写用户问题。
- 搜索商品名向量库。
- 按分数确认商品名。
- 决定是否提前结束。
- 写入用户历史。

这导致流程隐含规则变成：没有商品名就不能继续 RAG。

### 4.3 新增 State 字段

建议在 `QueryGraphState` 中新增字段：

```python
intent: str
entities: dict
need_item_name: bool
needs_clarification: bool
clarification_question: str
query_type: str
retrieval_plan: dict
evidence_status: str
sources: list
```

字段语义：

- `intent`: 用户意图，例如 `qa`、`compare`、`summary`、`troubleshooting`、`chitchat`。
- `entities`: 结构化实体，例如商品名、型号、部件、错误码。
- `need_item_name`: 当前问题是否必须绑定商品名称。
- `needs_clarification`: 是否需要追问用户。
- `clarification_question`: 需要追问时给用户的问题。
- `query_type`: 检索问题类型，例如 `fact`、`procedure`、`comparison`、`diagnosis`。
- `retrieval_plan`: 后续检索计划。
- `evidence_status`: 证据门控结果，例如 `sufficient`、`insufficient`、`conflicting`。
- `sources`: 最终返回给前端的结构化来源。

### 4.4 新节点设计

新增节点：

```text
node_query_understanding
node_entity_resolution
node_clarification_decision
```

职责拆分：

- `node_query_understanding`: 意图识别、实体抽取、查询改写、是否需要商品名。
- `node_entity_resolution`: 仅在需要商品名时进行商品名向量匹配和消歧。
- `node_clarification_decision`: 根据实体、意图和匹配结果决定是否追问。

### 4.5 LLM 输出结构

`node_query_understanding` 应使用结构化 JSON 输出：

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

### 4.6 路由规则

替换当前 `route_after_node_item_name_confirm`：

```text
node_query_understanding
  -> node_entity_resolution
  -> node_clarification_decision
  -> 如果 needs_clarification=True: node_answer_output
  -> 如果 needs_clarification=False: node_retrieval_planner
```

### 4.7 验收标准

- 用户问泛知识问题时，不因缺少商品名直接结束。
- 用户问具体商品操作但商品名模糊时，仍能追问。
- 历史中已有商品名时，能通过指代消解补全。
- 原有商品问答路径保持可用。

### 4.8 回滚方案

保留旧 `node_item_name_confirm`，通过环境变量切换：

```text
QUERY_USE_STANDARD_UNDERSTANDING=false
```

出现问题时切回旧流程。

## 阶段二：检索计划 Retrieval Planner

### 5.1 目标

将固定多路召回改成按问题动态选择检索策略，降低成本、延迟和错误召回。

### 5.2 当前问题

当前流程只要商品名确认，就固定并行：

```text
node_search_embedding
node_search_embedding_hyde
node_web_search_mcp
```

这会导致：

- 简单问题也调用 HyDE 和 Web Search。
- 外部搜索默认参与，答案可信度下降。
- 对比类问题没有按实体分别检索。
- 故障类问题没有优先使用错误码、FAQ 或 KG。

### 5.3 新增节点

```text
node_retrieval_planner
```

输出：

```json
{
  "use_local_vector": true,
  "use_hyde": false,
  "use_web": false,
  "use_kg": false,
  "filters": {
    "item_name": ["HAK 180 烫金机"],
    "doc_type": [],
    "section_type": ["operation"]
  },
  "top_k": 20,
  "rerank_top_k": 8,
  "reason": "操作步骤类问题，本地说明书优先，暂不需要外部搜索"
}
```

### 5.4 推荐策略表

| query_type | local_vector | hyde | web | kg | 说明 |
| --- | --- | --- | --- | --- | --- |
| fact | 是 | 否 | 否 | 可选 | 参数、定义类问题 |
| procedure | 是 | 可选 | 否 | 否 | 操作步骤类问题 |
| troubleshooting | 是 | 可选 | 否 | 是 | 故障排查类问题 |
| comparison | 是 | 否 | 否 | 可选 | 多实体对比，需要按实体分组检索 |
| summary | 是 | 否 | 否 | 否 | 文档或章节摘要 |
| realtime | 可选 | 否 | 是 | 否 | 明确需要外部实时信息 |
| chitchat | 否 | 否 | 否 | 否 | 闲聊或非知识库问题 |

### 5.5 Web Search 使用原则

Web Search 默认关闭，满足以下条件之一才启用：

- 用户明确要求联网。
- 用户询问实时信息。
- 本地召回证据不足，且问题允许外部补充。
- 管理端配置允许外部搜索。

新增配置：

```text
QUERY_WEB_SEARCH_DEFAULT=false
QUERY_WEB_SEARCH_ON_LOW_CONFIDENCE=true
QUERY_WEB_SEARCH_MIN_LOCAL_SCORE=0.55
```

### 5.6 LangGraph 路由调整

`node_retrieval_planner` 后使用条件边：

```text
if use_local_vector: node_search_embedding
if use_hyde: node_search_embedding_hyde
if use_web: node_web_search_mcp
if use_kg: node_query_kg
```

如果没有任何检索源：

```text
node_answer_output
```

用于处理闲聊、澄清、拒答等非 RAG 场景。

### 5.7 验收标准

- 简单参数查询不默认触发 Web Search。
- 明确实时问题能触发 Web Search。
- 对比问题能够为多个实体生成检索计划。
- 检索计划写入观测事件，便于排障。

## 阶段三：证据融合和 Chunk Schema 标准化

### 6.1 目标

统一本地文档、HyDE 结果、Web 结果、KG 结果的数据结构，使 RRF、Rerank、Prompt 构造和 Citation 都基于同一证据模型。

### 6.2 标准 Evidence Schema

建议定义 `EvidenceDocument`：

```python
class EvidenceDocument(TypedDict, total=False):
    """RAG 证据文档。

    Args:
        id: 证据唯一 ID。
        source_type: 来源类型，例如 local、web、kg。
        text: 证据文本。
        title: 标题。
        file_title: 文件标题。
        parent_title: 父级章节标题。
        item_name: 商品名称。
        chunk_id: Chunk ID。
        page: 页码。
        url: 来源 URL。
        retrieval_score: 召回分数。
        rerank_score: 重排分数。
        metadata: 扩展元数据。

    Returns:
        标准证据文档结构。
    """
```

### 6.3 数据规范

本地向量结果：

- 必须包含 `chunk_id`
- 必须包含 `text`
- 尽量包含 `file_title`、`title`、`parent_title`、`page`

Web 结果：

- 必须包含 `url`
- 必须包含 `title`
- 必须包含 `snippet` 或 `text`
- `source_type` 固定为 `web`

KG 结果：

- 必须包含实体、关系、路径摘要
- `source_type` 固定为 `kg`

### 6.4 Evidence Fusion 规则

新增或重构 `node_rrf`，让它承担：

- 多源证据标准化。
- 按 `chunk_id/url/kg_path` 去重。
- RRF 融合排序。
- 合并 metadata。
- 标记来源类型。

### 6.5 相邻 Chunk 补全

对高分本地 chunk，补充前后相邻 chunk：

```text
top chunk -> previous chunk + current chunk + next chunk
```

适用于：

- 操作步骤被切断。
- 表格上下文跨 chunk。
- 图片说明和正文分离。

新增配置：

```text
QUERY_CONTEXT_EXPAND_NEIGHBORS=true
QUERY_CONTEXT_NEIGHBOR_SIZE=1
```

### 6.6 验收标准

- Rerank 输入统一为 EvidenceDocument。
- Answer 输出可以从 EvidenceDocument 生成 sources。
- 同一个 chunk 不重复进入 Prompt。
- 本地和 Web 来源在最终答案中可区分。

## 阶段四：证据质量门控 Evidence Gate

### 7.1 目标

让系统知道什么时候不应该回答，降低幻觉和无依据回答。

### 7.2 新增节点

```text
node_evidence_gate
```

输入：

- `reranked_docs`
- `rewritten_query`
- `intent`
- `retrieval_plan`

输出：

```json
{
  "evidence_status": "sufficient",
  "confidence": 0.82,
  "reason": "Top 文档覆盖了操作区域设置步骤",
  "missing_info": []
}
```

### 7.3 判断规则

第一版先使用规则门控：

- `reranked_docs` 为空：`insufficient`
- top rerank score 低于阈值：`insufficient`
- top 文档来源全是 Web 且本地知识库为空：`external_only`
- 对比类问题缺少任一实体证据：`insufficient`
- 多个高分证据结论冲突：`conflicting`

配置：

```text
QUERY_EVIDENCE_MIN_RERANK_SCORE=0.45
QUERY_EVIDENCE_MIN_DOCS=1
QUERY_ALLOW_EXTERNAL_ONLY=false
```

### 7.4 不足证据回答策略

当 `evidence_status=insufficient`：

```text
当前知识库没有找到足够依据回答这个问题。你可以补充具体商品名称、型号，或导入相关说明文档后再查询。
```

当 `evidence_status=conflicting`：

```text
检索到的资料存在不一致信息，建议以最新版本说明书或官方文档为准。以下是冲突来源...
```

### 7.5 验收标准

- 无检索结果时不生成编造答案。
- 低分召回时返回证据不足。
- 对比问题缺少某个实体资料时明确指出缺少哪一方。
- Evidence Gate 结果进入可观测事件。

## 阶段五：答案生成和 Citation

### 8.1 目标

答案必须基于证据生成，并返回结构化来源。

### 8.2 Prompt 改造

Prompt 增加约束：

```text
只能依据给定资料回答。
如果资料不足，明确说明不足。
关键结论后标注引用编号，例如 [1]、[2]。
不要使用未出现在资料中的参数、步骤或结论。
```

### 8.3 Prompt Context 格式

```text
[1]
source_type: local
file_title: HAK 180 操作手册.pdf
section: 操作面板 > 局部烫金设置
page: 12
score: 0.87
text: ...

[2]
source_type: local
file_title: HAK 180 操作手册.pdf
section: 参数说明
page: 18
score: 0.81
text: ...
```

### 8.4 接口返回结构

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

### 8.5 Sources 生成

新增函数：

```python
def build_answer_sources(reranked_docs: list[dict]) -> list[dict]:
    """构建答案引用来源。

    Args:
        reranked_docs: 重排后的证据文档列表。

    Returns:
        前端可展示的结构化引用来源。
    """
```

### 8.6 验收标准

- 每次 RAG 回答都返回 `sources`。
- Web 来源和本地来源可区分。
- 答案中引用编号能对应 sources。
- 无证据回答时 sources 为空，且 answer 明确说明证据不足。

## 阶段六：历史对话和指代消解

### 9.1 目标

让历史对话服务于当前问题，而不是简单堆进 Prompt。

### 9.2 历史处理策略

新增节点或函数：

```text
node_conversation_context
```

职责：

- 取最近 N 轮历史。
- 判断当前问题是否依赖历史。
- 提取当前问题相关的上下文。
- 做指代消解。

示例：

```text
用户上一轮：HAK 180 烫金机怎么设置温度？
用户当前：那局部区域呢？

改写后：HAK 180 烫金机局部烫金区域如何设置？
```

### 9.3 长历史压缩

后续可引入会话摘要：

```text
conversation_summary:{session_id}
```

策略：

- 最近 3 轮原文保留。
- 更早历史压缩为摘要。
- 摘要只用于 query understanding，不直接污染最终证据上下文。

### 9.4 验收标准

- 省略指代类问题可以正确补全上下文。
- 用户切换话题时，旧商品名不强行污染当前问题。
- Prompt 中历史占用不会挤掉核心证据。

## 阶段七：RAG 评估闭环

### 10.1 目标

建立可持续评估体系，而不是只靠人工体验判断 RAG 效果。

### 10.2 评估数据集

新增目录建议：

```text
test/eval/query_cases.jsonl
```

注意：不放在 `/doc` 下。

每条 case：

```json
{
  "case_id": "q001",
  "query": "HAK 180 如何设置局部烫金区域？",
  "session_history": [],
  "expected_item_names": ["HAK 180 烫金机"],
  "expected_chunk_ids": ["chunk_001", "chunk_002"],
  "expected_answer_points": ["设置起始位置", "设置结束位置"],
  "should_answer": true
}
```

### 10.3 指标

检索指标：

- Recall@K
- MRR
- nDCG
- context precision

生成指标：

- answer relevance
- faithfulness
- citation coverage
- unsupported answer rate
- no-answer precision

### 10.4 Eval Runner

新增脚本建议：

```text
test/eval/run_query_eval.py
```

输出：

```text
output/eval/query_eval_YYYYMMDD.json
```

### 10.5 验收标准

- 每次检索策略调整前后可以比较指标。
- 能定位是 query understanding、retrieval、rerank 还是 generation 出问题。
- 至少覆盖 30 条核心业务问题。

## 11. 代码改造清单

### 11.1 State 层

文件：

```text
app/query_process/agent/state.py
```

改造：

- 增加 `intent`
- 增加 `entities`
- 增加 `need_item_name`
- 增加 `needs_clarification`
- 增加 `clarification_question`
- 增加 `query_type`
- 增加 `retrieval_plan`
- 增加 `evidence_status`
- 增加 `sources`

### 11.2 节点层

新增：

```text
app/query_process/agent/nodes/node_query_understanding.py
app/query_process/agent/nodes/node_entity_resolution.py
app/query_process/agent/nodes/node_clarification_decision.py
app/query_process/agent/nodes/node_retrieval_planner.py
app/query_process/agent/nodes/node_evidence_gate.py
```

重构：

```text
app/query_process/agent/nodes/node_item_name_confirm.py
app/query_process/agent/nodes/node_rrf.py
app/query_process/agent/nodes/node_rerank.py
app/query_process/agent/nodes/node_answer_output.py
```

### 11.3 Graph 层

文件：

```text
app/query_process/agent/main_graph.py
```

改造：

- 替换入口节点。
- 新增条件路由。
- 按 retrieval_plan 动态进入检索节点。
- 新增 evidence gate。

### 11.4 Prompt 层

新增或修改：

```text
prompts/query_understanding.txt
prompts/retrieval_planner.txt
prompts/answer_out.txt
prompts/evidence_gate.txt
```

### 11.5 API 层

文件：

```text
app/query_process/api/query__server.py
```

改造：

- 同步返回增加 `sources` 和 `image_urls`。
- 流式 `FINAL` 增加 `sources`。
- 查询状态接口可返回 `evidence_status`。

## 12. 推荐实施顺序

### 12.1 第一批任务

1. 扩展 `QueryGraphState`。
2. 新增 `node_query_understanding`。
3. 新增 `node_clarification_decision`。
4. 保留旧商品名确认逻辑，移动到 `node_entity_resolution`。
5. 修改 `main_graph.py` 路由。
6. 增加基础测试用例。

### 12.2 第二批任务

1. 新增 `node_retrieval_planner`。
2. 修改 Web Search 默认关闭。
3. 检索节点按 plan 执行。
4. 记录 retrieval_plan 观测事件。

### 12.3 第三批任务

1. 统一 EvidenceDocument schema。
2. 重构 RRF 输入输出。
3. 增加相邻 chunk 补全。
4. 修改 Rerank 输入。

### 12.4 第四批任务

1. 新增 `node_evidence_gate`。
2. 增加证据不足和冲突处理。
3. 改造 answer prompt。
4. 增加 sources 输出。

### 12.5 第五批任务

1. 增加 query eval 数据集。
2. 增加 eval runner。
3. 建立基线指标。
4. 每次策略调整输出对比报告。

## 13. 风险和控制

### 13.1 主要风险

- Query Understanding 误判导致原本能答的问题被澄清。
- Retrieval Planner 过度关闭 HyDE 或 Web，导致召回下降。
- Evidence Gate 阈值过高，导致拒答率上升。
- Citation 要求增加 Prompt 复杂度，可能影响答案自然度。
- State 字段增加后，部分节点未适配导致 KeyError。

### 13.2 控制手段

- 所有新节点先通过环境变量开关灰度。
- 保留旧流程作为 fallback。
- 每阶段只改一类行为。
- 每阶段增加最小测试集。
- 通过可观测事件记录决策结果和原因。

建议配置：

```text
QUERY_USE_STANDARD_UNDERSTANDING=false
QUERY_USE_RETRIEVAL_PLANNER=false
QUERY_ENABLE_EVIDENCE_GATE=false
QUERY_ENABLE_CITATIONS=false
```

## 14. 验收用例

### 14.1 不需要商品名的问题

输入：

```text
知识库里有哪些类型的设备文档？
```

预期：

- 不应返回“没有找到对应的商品名称”。
- 可以走 summary 或 general QA。

### 14.2 需要商品名但用户没说清

输入：

```text
这个机器怎么设置局部区域？
```

历史为空。

预期：

- 返回澄清问题。
- 不进入检索生成。

### 14.3 历史指代问题

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

- rewritten_query 补全为 HAK 180 烫金机局部区域设置。
- 正常进入检索。

### 14.4 本地知识库足够

输入：

```text
HAK 180 如何设置顶部 50mm 到 170mm 的烫金区域？
```

预期：

- 默认不启用 Web Search。
- 返回答案和本地 sources。

### 14.5 证据不足

输入：

```text
HAK 180 支不支持 2026 年最新云端远程控制？
```

预期：

- 本地证据不足时不编造。
- 如果未启用 Web，明确说明知识库无依据。

## 15. 最终效果

改造完成后，查询流程会从固定商品问答链路变成标准 RAG 查询链路：

- 先理解问题，再决定是否需要商品名。
- 先生成检索计划，再决定召回源。
- 先判断证据质量，再决定是否回答。
- 答案带结构化来源。
- Web Search 从默认参与变成受控补充。
- 质量评估从人工感受变成可量化指标。

这能让项目同时保留当前商品资料问答能力，并向更通用、更可靠的企业知识库 RAG 形态演进。
