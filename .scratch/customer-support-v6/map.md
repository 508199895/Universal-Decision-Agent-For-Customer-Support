# 客服 Agent V6 当前切片

## Notes

- 依据 `spec.md` 与根目录 `CONTEXT.md`。

## Decisions-so-far

- [01：修复 State 与 checkpoint 测试契约](issues/01-repair-state-and-checkpoint-test-contracts.md) 已完成：补强 TypedDict required keys 断言，checkpoint 恢复与 State 结构用例通过。
- [02：静态收集工具并绑定当前单 Agent](issues/02-static-tool-collection-and-agent-binding.md) 已完成：默认 CLI runtime 启动时收集并校验固定 Qdrant RAG/账户工具集，单 Agent 工厂绑定 allowlist；业务路由由后续场景票接入。
- [03：核对并补齐 V6 工单生命周期持久化](issues/03-v6-ticket-lifecycle-persistence.md) 已完成：CLI 支持继续最新未完成工单；终态和客户通知事件原子提交，临时 SQLite 生命周期验证通过。
- [04：从首条工单事件重建初始 checkpoint](issues/04-rebuild-pending-checkpoint-from-ticket-events.md) 已完成：覆盖 checkpoint 已持久化但初始化事件尚未写入时的新 graph 恢复与幂等性；目标测试 28 passed，全量测试 70 passed。

## Fog

- 本切片后续任务尚未开始。
