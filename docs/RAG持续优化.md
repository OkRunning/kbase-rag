# 一、可用性类（最常见）
## 1. 连续异常 → 熔断降级（你说的这个）
监控：滑动窗口内 LLM 调用错误率 / 连续失败次数
降级：熔断后切备用模型、返回兜底话术、或只走检索不生成
你项目落点：lm_utils.py 加熔断器（如 pybreaker），get_llm_client 失败计数超阈值切到 VL_MODEL 或更小模型
 落点：query_app.invoke 加 timeout，超时降级为单路 node_search_embedding
## 3. 限流/配额耗尽（429）→ 退避重试 + 切换
监控：429 出现频率、各模型 quota 余量
降级：指数退避重试 → 仍失败切备用 key/模型
落点：lm_utils.py 里 ChatOpenAI 配 max_retries，加多 key 轮换
# 二、质量类（RAG 特有，比可用性更隐蔽）
## 4. 召回质量退化 → 告警
监控：rerank top1 分数均值、召回结果数为 0 的比例、平均召回数
降级/动作：top1 分持续走低 → 告警人工介入；召回 0 → 触发 web 搜索兜底
落点：node_rerank 把 top1 score 写进 Langfuse score_current_trace，node_rrf 记录召回数
## 5. 答案质量下降（幻觉/拒答）
监控：用另一个 LLM 做 LLM-as-judge 给答案打分（事实性、相关性）；拒答率；用户点踩率
降级：低分答案触发二次校验、加引用约束重生成
落点：node_answer_output 后加异步 judge 节点，分数写 Langfuse
## 6. 上下文超长 / 截断
监控：prompt token 数逼近模型上限的比例
降级：触发更激进的 chunk 裁剪、丢弃低分召回段
落点：召回后统计 token，超阈值告警 + 自动裁剪
# 三、资源/成本类
## 7. 成本突增 → 告警
监控：每小时 token 消耗、单次查询成本（Langfuse 自动记 cost）
降级：日预算超限切便宜模型、限制大用户 QPS
落点：Langfuse cost 指标 + 你前面路线 B 轮询告警
## 8. 嵌入/向量服务异常
监控：BGE-M3 embedding 调用错误率、Milvus 查询延迟/超时
降级：Milvus 不可用切降级到关键词检索（BM25/稀疏向量）
落点：node_search_embedding 异常时 fallback 到稀疏向量路
## 9. Reranker 模型 OOM/加载失败
监控：本地模型加载状态、GPU 显存
降级：reranker 挂了跳过 rerank 直接用 RRF 融合结果
落点：node_rerank try/except，失败透传 RRF 结果
# 四、链路/依赖类
## 10. 下游依赖雪崩（MCP web 搜索、MinIO、Mongo）
监控：各外部依赖错误率、连接池水位
降级：web 搜索挂了跳过该路召回；MinIO 图片挂了用占位图；Mongo 历史挂了跳过历史拼接
落点：每个外部调用节点独立 try/except + 降级路径，符合你 LangGraph 多路召回结构
## 11. 流式响应中断 / SSE 连接异常
监控：SSE 断连率、后台任务失败率（你 task_utils 里已存状态）
降级：流式中断自动转非流式返回
落点：sse_generator 断连检测 + update_task_status("failed")（你已有）
# 五、安全/滥用类
## 12. 异常流量/刷接口
监控：单 session_id QPS、单 IP 请求量、prompt 长度异常
降级：超阈值限流/封禁（你已有 rate_limit_utils.py）
落点：已具备，补监控指标即可
## 13. 敏感内容 / 越狱攻击
监控：prompt 中越狱模式命中率、敏感词触发率
降级：拦截 + 兜底回复 + 告警
落点：node_item_name_confirm 前加 guardrail，score 写 Langfuse

落地优先级
按项目现状和 ROI 排序：
优先级	    场景	                        理由
P0	        #1 连续异常熔断降级	        最基础，防雪崩
P0	        #10 下游依赖降级	            LangGraph 多路召回天然适合每路独立降级
P1	        #2 超时降级	                RAG 链路长，超时体验差
P1	        #4 召回质量监控	            RAG 核心指标，你 rerank 分数现成
P2	        #8/#9 本地模型降级	        BGE/Reranker 挂了影响召回
P2	        #5 LLM-as-judge	            质量深水区，成本较高


通用实现骨架：所有这些场景都遵循同一个模式——
监控指标(Langfuse score/metric) 
  → 阈值判断(熔断器/滑动窗口) 
    → 降级动作(切备/裁剪/兜底) 
      → 告警通知(路线A应用内推飞书) 
        → 记录(Langfuse trace 标记 + 日志)