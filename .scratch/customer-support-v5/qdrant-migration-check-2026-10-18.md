# Qdrant/MCP 迁入验证（2026-10-18）

## 验收结果

1. 配置与工具发现：通过。客户端通过 stdio 启动项目内 kb/account/memory/qdrant 服务，Qdrant find/store 工具均可列出；未调用 store。
2. 真实只读检索：通过，但测试显式指定 QDRANT_URL=http://127.0.0.1:6333、COLLECTION_NAME=uda-hub-retrieval-eval。项目默认 COLLECTION_NAME 未设置，当前 schema 要求 collection_name，不传时返回 missing_argument。测试仅设置进程环境，未修改 .env。
3. 相关回归：通过，10 passed，1 warning，78.10 秒。范围：test_mcp_client、test_qdrant_tool、test_account_tool、test_kb_tool、test_checkpoint_capability、test_workflow_entry。使用 uda-hub，关闭第三方 pytest 插件自动加载并显式加载 pytest_asyncio.plugin；沙箱外运行以支持本地 MCP 子进程。未运行完整测试集、真实模型端到端及写入测试。

## 真实检索证据

Qdrant /collections 返回唯一集合 uda-hub-retrieval-eval。调用 qdrant_qdrant-find 检索 `How do I reserve a spot for an event with CultPass?`，解析出三条非空知识记录，首条为 `How to Reserve a Spot for an Event`，article_id 为 8a68f705-8ff5-4f26-88af-cd0539fbd85f。

实际 MCP 返回 str，经现有 parse_qdrant_find_result 解析。metadata 包含 article_id、account_id、title、tags、created_at、relevance_score；三条 account_id 均为 cultpass。重排分数约为 9.058、0.207、-2.995，不能当作 0–1 概率或旧 confidence。

完整结构证据：[qdrant-migration-evidence-2026-10-18.json](qdrant-migration-evidence-2026-10-18.json)。复跑探针：[probe_qdrant_migration.py](probe_qdrant_migration.py)。读取已有集合和公共知识，未执行入库、删除或业务写入。

## 遗留项

先前“Qdrant MCP 无法连接”的阻碍已解除。默认集合未配置仍需由调用方显式传参，或按项目关键配置留档规则配置默认集合；旧 probe_contracts.py 仅传 query，不能直接复跑证明完整契约验收。

01 号任务还有业务取消状态、活动时区、历史工单统计及最终审查等待完成项，此次迁移测试不等于整票完成，也不宣称跨账户隔离或完整检索质量已验收。FastMCP 日志弃用及 LangChain MCP beta 警告未阻断本次测试。
