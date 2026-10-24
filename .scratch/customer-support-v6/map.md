# 客服 Agent V6 当前切片

## Notes

- 依据 `spec.md` 与根目录 `CONTEXT.md`。

## Decisions-so-far

- [01：修复 State 与 checkpoint 测试契约](issues/01-repair-state-and-checkpoint-test-contracts.md) 已完成：补强 TypedDict required keys 断言，checkpoint 恢复与 State 结构用例通过。
- [02：静态收集工具并绑定当前单 Agent](issues/02-static-tool-collection-and-agent-binding.md) 已完成：默认 CLI runtime 启动时收集并校验固定 Qdrant RAG/账户工具集，单 Agent 工厂绑定 allowlist；业务路由由后续场景票接入。

## Fog

- 本切片后续任务尚未开始。
