# 02: 静态收集工具并绑定当前单 Agent

**Estimated execution tokens:** 25k–40k

**What to build:** 应用启动后得到当前已接入的静态工具集，创建单 Agent 时只向它提供完成所选客服场景所需的工具；错误配置能在接收客户请求前被发现。

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] 应用启动时从当前配置和已接入能力中收集一次工具；必要配置缺失或工具名冲突时给出可定位的启动错误。
- [x] 当前单 Agent 可调用当前接入场景所需工具；legacy KB MCP 与 Qdrant 管理工具不可由 Agent 调用。
- [x] 工具集合不会按客户输入临时推断，也不会在每轮请求中重新收集或筛选。
- [x] 通过 CLI runtime 启动测试与受控模型、工具替身验证工具可用性及错误边界；不引入通用注册中心、动态发现或延迟加载。

## Answer

- CLI 默认运行时在接受客户输入前收集一次 MCP 工具、校验必需工具与重名，并构建固定工具集的单 Agent；连接或配置错误会在启动时失败。
- Agent 实际绑定了受控 allowlist 中的 Qdrant RAG 检索与账户资料/预约查询工具；legacy KB MCP、memory 与 Qdrant 管理工具不会传入。受控模型测试触发了真实 LangChain 工具调用。
- 当前工作流仍是 V5 交接图；单 Agent 的客服业务路由与调用将由后续场景票接入。现有 MCP 能力不含历史工单读取和取消写工具，后续工具接入时需扩展 allowlist。
- 验证：定向 `tests/test_cli_runtime_startup.py tests/test_support_toolset.py tests/test_mcp_client.py` 9 passed；全套 pytest 55 passed；compileall 与 `git diff --check` 通过。
