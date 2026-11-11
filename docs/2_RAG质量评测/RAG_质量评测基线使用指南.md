# RAG 质量评测基线使用指南

## 1. 评测样本标注

评测样本使用 JSONL，一行一条样本。核心字段：

```json
{"case_id":"safety_0001","tenant_id":"default","kb_id":"default","question":"华为显示器 B3-211H 关于儿童健康应该注意什么？","query_type":"safety","expected_answer":"本设备及其配件可能包含一些小零件，应放置在儿童接触不到的地方，避免儿童吞下小零件导致窒息或其他危险；儿童应在成人监护下使用。","expected_doc_ids":["华为显示器 B3-211H 用户指南-(NSN-21BZ,02,zh-cn)"],"expected_chunk_ids":["467991773213313697"],"required_keywords":["小零件","儿童","吞下","窒息","成人监护"],"forbidden_keywords":["无法确定","请联系销售"],"difficulty":"easy","tags":["product","safety","local_retrieval"],"metadata":{"source":"manual_label","label_method":"from_rerank_snapshot"}}
```

标注原则：

- `expected_chunk_ids` 只写真正能支撑答案的本地知识库 chunk。
- `required_keywords` 选择答案关键概念，不要选择“设备”“注意”等泛词。
- `forbidden_keywords` 用于拦截明显错误话术或不合理拒答。
- 每条样本建议保留 `source_session_id`，方便回看原始查询 timeline。

## 2. 本地运行评测

```bash
python3 -m app.evaluation.run_rag_eval \
  --dataset data/eval/rag_cases.jsonl \
  --profile baseline
```

只跑前 N 条：

```bash
python3 -m app.evaluation.run_rag_eval \
  --dataset data/eval/rag_cases.jsonl \
  --profile label_check \
  --limit 5
```

## 3. Langfuse 上报

确保环境变量可用：

```text
LANGFUSE_BASE_URL
LANGFUSE_PUBLIC_KEY
LANGFUSE_SECRET_KEY
```

运行：

```bash
python3 -m app.evaluation.run_rag_eval \
  --dataset data/eval/rag_cases.jsonl \
  --profile baseline \
  --langfuse
```

在 Langfuse Traces 页面搜索：

```text
tag = rag_eval
user_id = rag_eval
session_id = eval:<profile>:<case_id>
```

## 4. Baseline 对比

先生成一份 baseline JSON 报告，然后后续版本使用：

```bash
python3 -m app.evaluation.run_rag_eval \
  --dataset data/eval/rag_cases.jsonl \
  --profile prompt_v2 \
  --baseline-report reports/evaluation/rag_eval_baseline_YYYYMMDDTHHMMSSZ.json
```

报告会新增 `Baseline Comparison`，展示当前指标与 baseline 的 delta。

## 5. 阈值门禁

默认阈值文件：

```text
data/eval/thresholds.default.json
```

使用自定义阈值：

```bash
python3 -m app.evaluation.run_rag_eval \
  --dataset data/eval/rag_cases.jsonl \
  --profile release_check \
  --thresholds data/eval/thresholds.default.json \
  --fail-on-threshold
```

当任一指标低于阈值时，CLI 退出码为 `2`，可用于 CI 或发布门禁。

## 6. 完成标准

Plan 1 质量基线建议满足：

- 核心人工样本不少于 30 条。
- 覆盖 fact、procedure、troubleshooting、safety、follow_up 等问题类型。
- 每条样本都有人工确认的 `expected_chunk_ids`。
- 每次 Prompt、检索策略、chunk 参数调整后都能生成报告。
- 报告能展示 baseline delta 和 threshold violations。
- Langfuse 能看到 `rag_eval` trace 和 scores。