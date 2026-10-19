# 01：建立可隔离的客服执行入口

Status: ready-for-agent
Blocked by: None

**What to build:** 将客服进入/恢复入口与可控模型、工具和真实临时存储接通，使后续 V5 行为可从公开边界验证；本票仅做必要的预重构。

**Blocked by:** 无，可立即开始。

**依据：** [V5 Spec](../../../agentic/design/客服工作流_v5_Spec_2026-10-14.md)；[共同实施约束](../README.md#共同实施约束)。

## 验收标准

- [ ] 先恢复 uda-hub 正常激活，查当前官方文档核对 create_agent、持久检查点、interrupt、中间件及模型原生结构化输出；能力不足立即报告，不静默降级。
- [ ] 核对真实业务 ID、工具参数、取消状态、活动时区、Qdrant 返回和现有工单数据；记录结论和阻碍，不发明状态值。
- [ ] 入口允许注入模型/工具替身及存储依赖，避免导入时强制连接真实外部服务；原有行为经适当回归仍可验证。
- [ ] 不修改 MCP/RAG 服务、导入及其他应用；不做与 V5 无关的大范围重构。

**Spec 验收对应：** 后续全部场景的可隔离执行前提；本票以入口回归和能力核对证据验收。

## Comments

### 2026-10-15：前置修复与入口预重构，等待真实工具验证

环境激活及 SQLite 检查点依赖已处理；用户补充 Responses API 文档后，Flash + create_agent + ProviderStrategy 原生结构化输出及中间件实测通过。入口离线导入、注入替身问答和 SQLite 重新加载测试 2 项通过。

真实 Qdrant MCP 连接失败，未获取 qdrant-find，故按关键验证失败停止规则暂停。实际取消状态字面值、活动业务时区、Qdrant 返回及历史工单统计仍须核验。未完成全套验收、审查或提交，不标记本票完成。详情见 [环境与验证记录](../environment-check-2026-10-15.md)。

### 2026-10-18：迁入后 Qdrant/MCP 补验

本地 stdio 工具发现及相关回归 10 项通过；显式传入现有集合后，真实检索返回三条可解析知识，首条命中预订指南。原连接失败阻碍解除。默认 COLLECTION_NAME 未设置，qdrant-find 当前要求 collection_name，旧探针须适配后再运行。此次未修改服务、未执行写入、未完成整票。见 [迁移验证记录](../qdrant-migration-check-2026-10-18.md)。

### 2026-10-19：入口恢复回归与业务契约核对

完成公开入口预重构：`build_workflow` 接受显式模型、工具加载器和 checkpoint saver；导入不再校验凭据或初始化外部服务；`run_ticket` 保留同步入口，新增 async 入口与同一 thread_id 的恢复入口。新增离线导入、注入替身端到端、真实 SQLite 跨图实例暂停/恢复、独立进程恢复测试。恢复入口另拒绝无 checkpoint 或已结束工单；终态拒绝有红灯复现。定向回归 `tests/test_workflow_entry.py tests/test_checkpoint_capability.py`：5 passed。

契约证据见 [只读核对 JSON](../contract-evidence-2026-10-15.json) 与 [Qdrant 迁移记录](../qdrant-migration-check-2026-10-18.md)：外部用户 ID、核心用户 ID、账户 ID 均为字符串且不可混用；账户工具参数为 `external_user_id: str`；核心库 11 张工单均为 open、都有 metadata，消息保存为 role/content；跨账户 external_user_id 重复数为 0。业务库 14 条预订状态全为 `reserved`，因此未观察到取消状态字面值；活动时间字段为无时区 DATETIME，业务时区仍待确认。订阅样本状态有 active/cancelled。Qdrant 需显式传 collection_name，stdio MCP 实际只读检索返回 3 条记录，首条为预订指南；没有调用 store。

框架能力记录已核对 create_agent、持久 checkpoint、interrupt/Command、middleware 及 ProviderStrategy；DeepSeek Flash Responses API 的结构化输出真实探测通过，v4-pro 未实测。未改提示词、StylingConfig 或默认模型配置。

本票入口与前置能力验收通过；未知的取消状态和活动时区是后续取消预订任务的明确阻碍，不据样本臆定。整仓测试、最终代码审查和提交仍待执行。

