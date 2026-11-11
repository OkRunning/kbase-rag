# 1. 可观测体系

基于 OpenTelemetry 搭建完整可观测体系，统一采集 Trace、Metric、JSON 结构化日志；依托开源技术栈：OTel Collector、Promtail、Prometheus、Alertmanager、Loki、Tempo、Grafana，构建**采集 - 存储 - 可视化 - 告警**一体化观测流水线。

```
口语版：
基于 OpenTelemetry 构建可观测平台，统一收拢链路追踪、指标、结构化日志三类信号。
技术栈采用 OTel Collector + Promtail 做采集；指标存在 Prometheus，日志存储到 Loki，Trace 使用 Tempo；最后借助 Grafana 统一可视化，搭配 Alertmanager 实现告警，打通了从采集、存储、展示到告警的完整观测流水线。
```

1. `OpenTelemetry SDK`
   嵌入在应用代码里，负责生成 trace 和 metric。比如一次 RAG 查询经过 query API、LangGraph 节点、Milvus、LLM、MongoDB，每一段都可以生成 span；同时记录查询耗时、错误率、节点耗时、LLM 调用次数等指标。

2. `OTel Collector`
   相当于观测数据的中转站。应用不直接写 Tempo 或 Prometheus，而是先把 OTLP 数据发给 Collector。Collector 再统一转发：trace 发给 Tempo，metric 暴露给 Prometheus 抓取。这样以后更换后端存储或增加处理逻辑，不需要大量改业务代码。

3. `Prometheus`
   负责存储和查询指标。比如：

   - `rag_query_total`：查询总数和错误率
   - `rag_query_duration_seconds`：查询耗时
   - `rag_node_duration_seconds`：LangGraph 节点耗时
   - `rag_llm_requests_total`：LLM 调用成功/失败
   - `rag_milvus_operations_total`：Milvus 操作错误率

   Prometheus 也会加载告警规则，比如“查询错误率 5 分钟超过 5%”或“LLM 错误率超过 10%”。

4. `Alertmanager`
   Prometheus 发现规则触发后，把告警交给 Alertmanager。Alertmanager 负责告警分组、去重、抑制和路由。

   比如：

   - `critical` 告警发强提醒
   - `warning` 告警发普通提醒
   - `OtelCollectorDown` 触发时，抑制一部分派生告警，避免刷屏
   - 同类告警在一段时间内聚合发送

5. `Grafana`
   负责展示。它会连接 Prometheus、Loki、Tempo、Alertmanager，把指标、日志、链路和告警统一展示出来。

   比如 Dashboard 可以看到：

   - 查询成功率
   - 查询 P95/P99
   - 每个 LangGraph 节点耗时
   - LLM 错误率和 token 趋势
   - Milvus / MongoDB / MinIO 依赖状态
   - 当前告警列表

6. `Tempo`
   负责存储 trace。它记录一次请求完整经过了哪些节点、每个节点耗时多久、哪个节点报错。比如一次查询慢了，可以从 Grafana 点进 Tempo，看是 `node_search_embedding_hyde` 慢，还是 `node_answer_output` 的 LLM 慢。

7. `Loki`
   负责存储日志。应用输出 JSON 日志，里面带 `trace_id`、`span_id`、`session_id`、`task_id`。Loki 可以按这些字段查日志。

8. `Promtail`
   负责采集日志。它监听本地 `logs/app_YYYYMMDD.jsonl` 这类文件，把日志推送到 Loki。



## **排障时怎么配合**

比如线上出现 `QueryHighErrorRate`：

1. Prometheus 根据 `rag_query_total{status="error"}` 发现查询错误率过高。
2. Prometheus 触发告警，把告警发送给 Alertmanager。
3. Alertmanager 根据 severity 路由通知。
4. 开发打开 Grafana 的 `Dataset RAG Overview`。
5. 在 Dashboard 里看到错误集中在查询链路。
6. 继续看节点耗时和错误率，发现是 `node_answer_output` 异常。
7. 点进 Tempo 查看具体 trace，确认是 LLM 调用超时。
8. 用同一个 `trace_id` 去 Loki 查日志，看到具体异常堆栈。
9. 如果项目事件时间线记录了 `task_id`，还能进入 debug API 看节点事件、快照并做失败回放。

> Prometheus 看“有没有异常”，Grafana 看“异常在哪里”，Tempo 看“一次请求怎么走的”，Loki 看“具体日志是什么”，Alertmanager 负责“该通知谁”，OTel Collector 负责“统一接收和转发观测数据”，Promtail 负责“把日志送进 Loki”。



# 2. 节点快照和回放

设计并实现 LangGraph 节点级快照与回放能力，持久化关键节点输入/输出快照和事件时间线，支持按 `trace_id`、`session_id`、`task_id` 定位失败节点，并通过节点回放/流程回放复现问题、对比历史与回放结果，提升 RAG 多节点链路的排障效率和可维护性。

## Event事件

Event 主要是**可观测时间线 + 回放索引**，它记录“发生了什么、在哪个 trace/session/task/node 里发生、输入输出快照在哪里、错误是什么”。

1. 每条event包含的内容

   > `_id`：事件 ID，回放时会拿它当入口。
   >
   > `ts`、`sequence`：时间和同一 session/task/trace 下的顺序。
   >
   > `trace_id`、`span_id`、`langfuse_trace_id`：和 Trace/Langfuse 关联。
   >
   > `session_id` / `task_id` / `request_id`：查询、导入、HTTP 请求维度。
   >
   > `graph`、`node_name`：属于哪个 LangGraph 图和节点。
   >
   > `event_type`：如 `workflow.started`、`node.started`、`node.completed`、`node.failed`。
   >
   > `summary`：脱敏后的摘要。
   >
   > `input_ref` / `output_ref`：指向快照，不直接把完整 State 放进 event。
   >
   > `error`、`tags`：错误和标签。

   

2. Event存储位置

   入口是 `emit_event()`，写入 MongoDB 的 `observability_events`，Mongo 不可用时降级到本地 `logs/observability_events.jsonl`

   

3. 记录的事件类型

   - node级：

     `node.started`：节点开始，保存输入快照 `input_ref`。

     `node.completed`：节点成功，保存输出快照 `output_ref`。

     `node.failed`：节点异常，保存错误信息和输入快照。

   - workflow 级：

     `workflow.started/completed/failed`：（导入/查询 ）流程 开始/完成/失败

   - 任务进度：`task.status_changed`、`node.running`、`node.done`



## Snapshot快照

日志不能记录完整 state，trace 也不适合保存大对象。调试回放需要知道节点输入和输出，因此需要将大对象快照保存到 MinIO，只在 Event 中保存引用。

- 存储：**优先 MinIO，失败或无配置时落本地文件**。

- 策略：

  - 快照级别，支持 off、safe、debug、full。
    - off：不保存快照
    - safe：对长文本，如提示词、用户请求、返回结果，进行脱敏摘要保存
    - debug：全量保存
    - full：全量保存

  | 原始 query          | **Event summary 保存脱敏文本，完整文本可加密保存** |
  | ------------------- | -------------------------------------------------- |
  | prompt              | 默认只保存 hash 和长度；调试环境可保存完整快照     |
  | answer              | 保存 hash、长度、前 200 字摘要；调试环境可保存完整 |
  | chunks              | 保存 `chunk_id/title/source/score`，不保存全文     |
  | node input state    | 按白名单字段保存                                   |
  | node output state   | 按白名单字段保存                                   |
  | exception traceback | 保存到 MinIO，event 中放 `stack_ref`               |
  |                     |                                                    |

  根据 快照内容 + 策略，决定是否保存完整内容，还是保存摘要。

  

## 节点回放

针对单独某个节点进行回放，节点级回放入口是 `replay_node(event_id, ...)`，它要求传入一个 `node.started` 事件 ID。

- 要求：

  - 要求传入一个 `node.started` 事件 ID
  - 必须有输入快照

  

- 回放模式

  | 模式                        | 说明                                         |
  | --------------------------- | -------------------------------------------- |
  | `timeline_only`             | 只展示事件时间线，不重跑                     |
  | `mock_external`             | 使用历史 LLM/Web/Milvus 响应，不调用外部依赖 |
  | `live_external`/`from_node` | 重新调用外部依赖，用于验证线上修复           |



- 回放流程
  1. 通过 `/debug/sessions/{session_id}/timeline` 找到某个 `node.started`。
  2. 用这个 event 的 `_id` 调 `/debug/replay/node`。
  3. 系统读取 event 里的 `input_ref`。
  4. 从 snapshot store 加载当时节点输入 State。
  5. 根据 `mode` 决定是否真正执行节点。
  6. 生成新的 `replay.started/replay.completed/replay.failed` event。
  7. 保存新的 replay output snapshot，并可和历史 output 做 diff。



## 全流程回放

**从某个历史 `node.started` 事件的输入快照开始，按工作流后续节点顺序一路回放到结束**。

使用**LangGraph 原生 checkpoint**，用于保存图执行状态。

用 event 找起点，用 snapshot 还原 State，用 timeline 决定历史参照，用 replay mode 决定模拟还是真跑，用 checkpointer 保存真实回放过程。



- 入口：POST /debug/replay/workflow 

- 回放流程：
  1. **根据event_id查询event内容，校验当前节点状态，并获取输入内容（还原输入state）。**
  2. 如有 `override_state`，临时覆盖 State，用于验证修复或构造边界场景。
  3. 根据`session_id/task_id/trace_id` 下的历史 timeline，用于比较回放前后的结果。
  4. **根据 `graph` 和 `start_node` 生成后续节点计划 `planned_nodes`。**
  5. 写入 `replay.workflow.started` event。
  6. **逐个节点执行或模拟。**
  7. 每个节点前后都保存 replay checkpoint snapshot。
  8. 每个节点写一条 `replay.checkpoint` event。
  9. 最后保存 `replay_workflow_final` 快照。
  10. 写入 `replay.workflow.completed` 或失败时写 `replay.workflow.failed`。
  
  

- 全流程回放为什么不用原图checkpoint？

  1. 不一定精确对应某个节点的输入

     我们排障时经常要复现的是：

     ```text
     node_rerank 当时拿到什么输入，重新执行这个节点会不会复现问题
     ```

     所以我们在 `node.started` 时显式保存 `input_ref`，在 `node.completed` 时保存 `output_ref`。这样能精准拿到某个节点的输入/输出快照。

     checkpoint 是图状态历史，粒度和语义更偏 LangGraph 内部，不一定刚好等于某个业务节点的输入边界。

  2. **可以进行节点级 diff 和审计**

     自己写 event/snapshot/replay 后，每次回放都能记录：

     ```
     replay.started
     replay.completed
     replay.failed
     replay.checkpoint
     ```

     并且能对比：

     ```
     历史 output_ref vs 回放 output_ref
     ```

     这对定位“代码改完后这个节点输出变了多少”很重要。原图 checkpoint 本身不提供这种业务语义化 diff。

  3. **需要 mock_external，避免真实调用外部依赖**

     RAG 链路里节点可能调用 LLM、Milvus、Web Search、Mongo、SSE 等。生产排障时不能随便重放一次就真的调用外部服务或写库。

     所以我们设计了 `mock_external`：不执行真实节点，直接复用历史 `node.completed.output_ref`。**这可以在不产生副作用的情况下复盘流程。**

     原图 checkpoint 回放默认是继续执行原图，走到外部依赖节点就会真实调用，副作用更难控制。

  4. **可以从任意节点精确切入**

     这让用户可以直接从失败节点或可疑节点开始复现，而不是一定要从图入口开始，也不用手动判断 checkpoint history 里哪个 state 对应哪个节点边界。

  5. **事件时间线更适合排障检索**

     线上排障时一般先拿到的是 `trace_id/session_id/task_id`。我们可以通过 event store 快速查时间线，定位失败节点，再跳到对应 input/output snapshot。

     checkpoint 更像状态存储，缺少 `event_type`、`error`、`tags`、`summary` 这些业务排障字段。

  6. 补充

     自己实现也有成本，尤其是流程级回放如果手写节点计划，原图拓扑变了就要同步维护。所以更合理的演进是：

     ```
     节点级诊断回放：继续用 event + snapshot + 单节点回放
     流程级端到端回放：可以考虑复用原图 checkpoint/time-travel
     ```

  **不用原图 checkpoint 直接回放，是因为我们更关注“可定位、可解释、可 diff、可控副作用”的节点级排障能力；原图 checkpoint 更适合状态恢复和端到端继续执行，两者解决的问题不完全一样。**



# 3. 健康检查与自动化任务

把系统健康检查拆成 liveness、readiness 和 observability 三层，同时在应用启动时拉起轻量级自动化后台任务，周期性做合成探测、卡住任务扫描和容量探测，把问题提前暴露到 Prometheus、Grafana、Alertmanager 和事件 timeline 中。



## 健康检查

接口设计

| 接口                    | 语义         | 检查内容                                                     | 用途                   |
| ----------------------- | ------------ | ------------------------------------------------------------ | ---------------------- |
| `/health/live`          | 进程存活     | 只返回进程是否能响应                                         | 容器/进程重启判断。    |
| `/health/ready`         | 服务可接流量 | MongoDB、Milvus、LLM base_url、Redis、Checkpoint Postgres、MinIO、OTel Collector 可达性 | 上线、负载均衡摘流。   |
| `/health/observability` | 观测能力     | event store、snapshot store、checkpoint backend、OTLP exporter 配置状态 | 防止排障能力静默失效。 |

`/health/live` 只判断进程是否存活，适合容器重启探针。

`/health/ready` 判断服务是否可以接流量，会检查 Mongo、Redis、Milvus、LLM 配置、OTel exporter 等依赖。这里如果核心依赖失败，就可以让负载均衡或 Kubernetes 暂时摘流。

`/health/observability` 专门检查排障能力，比如 event store、snapshot store、LangGraph checkpoint backend、OTLP exporter 是否可用。这个接口的价值是：即使业务还能跑，如果观测链路失效，也要尽早发现，否则故障发生后会“失明”。



## 自动化任务

在FastAPI 的 lifespan 阶段启动一个后台 daemon thread（守护线程）。在每隔一段时间做轻量稳定性巡检：

- 合成探测
- 卡住任务扫描
- 容量探测

### 合成探测

合成探测复用健康检查readiness 和 observability health的逻辑，如果探测返回 `error` 或 `degraded`，Prometheus 会触发 `SyntheticProbeFailed` 或 `SyntheticProbeDegraded` 告警。这样可以在真实用户请求失败前发现依赖或观测链路异常。



### 卡住任务扫描

卡住任务扫描是针对 RAG 这种长链路任务设计的。查询和导入都可能进入 `processing` 后长时间没有进展，所以我在任务状态里记录：

```reStructuredText
created_at
updated_at
graph
status
running_list
done_list
```

后台任务定期扫描，如果查询任务超过 10 分钟、导入任务超过 30 分钟仍处于 `processing`，就写入

```
task.stuck.detected
```

并通过 Prometheus 告警通知。事件里包含 task/session ID、当前运行节点和 idle 时间，便于通过 timeline 和 replay 定位



### 容量探测

记录本地日志、输出目录、以及可配置的数据目录的容量：

```
rag_capacity_headroom
rag_capacity_used_ratio
```

然后通过 `CapacityHeadroomLow` 提前告警，避免 Prometheus、Loki、Tempo、Mongo、MinIO、Postgres 或本地日志目录磁盘打满。


