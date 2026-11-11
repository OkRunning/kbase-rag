评测指标主要分两类：检索有没有找对资料，以及答案有没有满足基本规则。

# 检索指标
## `retrieval_recall_at_5`
大白话：前 5 个“召回候选资料”里，找回了多少应该找回的标准资料。

比如标准答案需要 2 个 chunk：chunk_a、chunk_b。
系统前 5 个候选里只找到了 chunk_a，那就是：
```
1 / 2 = 0.5
```
它回答的是：该找的资料找回来多少？
## `retrieval_hit_at_5`
大白话：前 5 个“召回候选资料”里，只要命中一个标准资料，就算命中。

还是标准资料是 chunk_a、chunk_b。
系统前 5 个候选里有 chunk_a，那就是 1.0。一个都没有就是 0.0。

它回答的是：有没有至少找对一个关键资料？

## `retrieval_mrr_at_5`
大白话：第一个正确资料排得越靠前，分越高。
比如标准资料是 chunk_a：
```
第 1 名就是 chunk_a -> 1 / 1 = 1.0
第 2 名才是 chunk_a -> 1 / 2 = 0.5
第 5 名才是 chunk_a -> 1 / 5 = 0.2
前 5 名都没有 -> 0.0
```
它回答的是：正确资料排得靠不靠前？

# Rerank 后指标
## `recall_at_5`
大白话：经过 Rerank 最终排序后，前 5 个资料里找回了多少标准资料。
它和 `retrieval_recall_at_5` 的区别是：
```
retrieval_recall_at_5 看的是召回阶段
recall_at_5 看的是 rerank 之后最终给 LLM 的资料
```
它回答的是：最终喂给大模型的资料，覆盖了多少正确证据？
## `hit_at_5`
大白话：Rerank 之后前 5 个资料里，有没有至少一个正确资料。

它回答的是：最终资料里有没有关键证据？
## `mrr_at_5`
大白话：Rerank 之后，第一个正确资料排第几名。越靠前越好。

它回答的是：Rerank 有没有把正确资料排到前面？

# 答案规则指标
## `has_answer`
大白话：有没有生成答案。
```
答案非空 -> 1.0
答案为空 -> 0.0
```
它只判断“有没有答”，不判断“答得对不对”。
## `has_local_citation`
大白话：最终有没有本地知识库证据。
现在代码里是看 `reranked_chunk_ids` 是否为空：
```
有 reranked chunk -> 1.0
没有 -> 0.0
```
它回答的是：这个答案有没有基于本地检索资料？
## `required_keywords_pass`
大白话：答案里必须出现的关键词，是不是都出现了。

比如样本要求：
```
"required_keywords": ["局部烫金", "起止位置"]
```
模型答案两个词都出现：
```
required_keywords_pass = 1.0
```
只出现一个或一个都没出现：
```
required_keywords_pass = 0.0
```
它回答的是：答案有没有覆盖关键概念？
## `required_keywords_coverage`
大白话：必须出现的关键词，出现了多少比例。

比如要求 4 个关键词，答案出现了 3 个：
```
3 / 4 = 0.75
```
它比 `required_keywords_pass` 更细一点：
```
required_keywords_pass 要全部出现才是 1
required_keywords_coverage 可以看到部分命中情况
```
## `forbidden_keywords_pass`
大白话：答案里有没有出现不该出现的词。

比如样本设置：
```
"forbidden_keywords": ["无法确定", "请联系销售"]
```
如果答案里没有这些词：
```
forbidden_keywords_pass = 1.0
```
如果出现了任意一个：
```
forbidden_keywords_pass = 0.0
```
它回答的是：答案有没有踩明显错误或不希望出现的话术？

# 一个整体例子
标准样本：
```
{
  "expected_chunk_ids": ["chunk_a", "chunk_b"],
  "required_keywords": ["局部烫金", "起止位置"],
  "forbidden_keywords": ["无法确定"]
}
```
系统最终 Rerank 后前 5 个资料是：
```
["chunk_x", "chunk_a", "chunk_y", "chunk_b", "chunk_z"]
```
答案是：
```
设置局部烫金时，需要配置起止位置。
```
那么：
 ```
recall_at_5 = 1.0        # chunk_a 和 chunk_b 都找到了
hit_at_5 = 1.0           # 至少找到了一个
mrr_at_5 = 0.5           # 第一个正确资料在第 2 名，所以是 1/2
has_answer = 1.0         # 有答案
has_local_citation = 1.0 # 有本地资料
required_keywords_pass = 1.0
required_keywords_coverage = 1.0
forbidden_keywords_pass = 1.0
```

一句话总结：
```
Recall 看“找全没”。
Hit 看“碰到没”。
MRR 看“排前没”。
required keywords 看“关键点答到没”。
forbidden keywords 看“明显不该说的有没有说”。
has_answer / has_local_citation 看“有没有答案、有没有证据”。
```