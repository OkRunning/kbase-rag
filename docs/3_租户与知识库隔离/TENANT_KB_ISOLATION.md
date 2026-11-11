# 租户与知识库隔离

## Scope 规则

查询和导入接口均支持以下字段：

- `tenant_id`：租户标识，默认值为 `default`。
- `kb_id`：知识库标识，默认值为 `default`。
- `user_id`：调用用户标识，默认值为 `anonymous`。

`tenant_id` 和 `kb_id` 仅允许字母、数字、点、下划线和中划线，最长 128 个字符。省略字段的旧调用会继续进入 `default/default`，保持向后兼容。

查询请求示例：

```json
{
  "query": "HAK 180 无法加热时应该检查什么？",
  "session_id": "session-001",
  "tenant_id": "tenant_a",
  "kb_id": "product_manual",
  "user_id": "u001",
  "is_stream": false
}
```

上传请求使用 multipart 表单字段 `files`、`tenant_id`、`kb_id` 和 `user_id`。

## 数据隔离

- 文档切片和商品名向量均写入 `tenant_id`、`kb_id`、`doc_id` metadata。
- 两条本地 Milvus 检索链路及商品名确认链路都会强制拼接 `tenant_id + kb_id` filter。
- MongoDB 对话历史按 `tenant_id + kb_id + session_id` 查询、删除和建立索引。
- 事件、节点快照和 LangGraph checkpoint thread ID 均保留 scope。

Prometheus 指标不带 `tenant_id` 或 `kb_id` label，避免高基数问题；精确范围信息保存在事件、快照和 Mongo 明细中。

## 旧数据迁移

强制 metadata filter 启用后，缺少 `tenant_id` 和 `kb_id` 的历史 Milvus 向量不会被召回。推荐按以下顺序迁移：

1. 使用原始文件重新调用导入接口，不传 scope 时系统会写入 `tenant_id=default`、`kb_id=default`。
2. 验证默认知识库可以正常召回后，再清理旧的无 scope 向量。
3. 对 Mongo 历史数据执行一次性回填：

```javascript
db.chat_message.updateMany(
  { tenant_id: { $exists: false } },
  { $set: { tenant_id: "default", kb_id: "default", user_id: "anonymous" } }
)
```

4. 对 `observability_events` 使用相同默认值回填；本地快照保留原文件即可，新快照会自动写入 scope。

生产迁移应先在备份或预发环境验证，尤其是在既有 collection 未启用 dynamic field 的部署中。若 collection 无法新增 metadata，请新建包含 `tenant_id`、`kb_id`、`doc_id` 的 collection，并通过重新导入完成切换。
