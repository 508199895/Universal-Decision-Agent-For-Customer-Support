# MCP 2 升级方案

## 1. 背景

项目原先通过 `langchain-mcp-adapters.MultiServerMCPClient` 将 MCP 工具转换为 LangChain `BaseTool`。当前运行环境中的依赖组合为：

| 组件 | 版本 | 状态 |
|---|---:|---|
| `langchain-mcp-adapters` | 0.3.1 | 按 MCP 1.x API 实现，导入 `RequestContext` |
| `mcp` | 2.2.0 | 已移除 MCP 1.x 的 `RequestContext` |
| `fastmcp` | 4.0.5 | 使用 MCP 2.x |
| `langchain` | 1.4.2 | 内置新的 `langchain.mcp.MCPAdapter` |

旧适配器在导入阶段失败：

```text
ImportError: cannot import name 'RequestContext' from 'mcp.shared.context'
```

`langchain-mcp-adapters` 最新版 0.3.2 仅通过约束 `mcp<2` 避免错误，并未支持 MCP 2。该独立项目已停止作为新版 MCP 集成入口，因此本项目迁移到 LangChain 核心提供的 `langchain.mcp.MCPAdapter`。

## 2. 升级目标

1. 保留 MCP 2.x 和 FastMCP 4.x。
2. 删除 `langchain-mcp-adapters` 依赖。
3. 保持现有业务层工具加载接口不变：
   - `aget_tools_for_servers()`
   - `get_tools_for_servers()`
4. 保持知识库、账户、记忆和 Qdrant 四类 MCP 工具可用。
5. 兼容现有业务代码所依赖的文本工具返回格式。

## 3. 依赖调整

`requirements.txt` 使用同一代 MCP 依赖：

```text
langchain[mcp]>=1.4.2,<2.0.0
fastmcp>=4.0.1,<5.0.0
mcp>=2.0.0,<3.0.0
langchain-core>=1.3.3,<2.0.0
```

删除：

```text
langchain-mcp-adapters>=0.1.9
```

## 4. 客户端迁移

主要修改文件：`agentic/tools/mcp_client.py`。

### 4.1 适配器替换

```python
from langchain.mcp import MCPAdapter
```

使用 `MCPAdapter` 替代 `MultiServerMCPClient`，并将工具发现方法由 `get_tools()` 改为 `list_tools()`。

### 4.2 标准多服务配置

连接配置使用标准 `mcpServers` 结构：

```python
{
    "mcpServers": {
        "kb": {...},
        "account": {...},
        "memory": {...},
        "qdrant": {...},
    }
}
```

本地 stdio 服务使用 `sys.executable` 启动，确保子进程与主程序使用同一个 Python 环境，避免系统 `python` 指向其他解释器。

Qdrant 保持 Streamable HTTP 连接：

```python
{
    "transport": "streamable-http",
    "url": QDRANT_MCP_URL,
}
```

### 4.3 工具命名空间

FastMCP 多服务客户端会将工具名称改为：

```text
<server>_<upstream_tool>
```

例如：

| 服务 | 原工具名 | 新公开名称 |
|---|---|---|
| KB | `kb_search` | `kb_kb_search` |
| Account | `account_get_user` | `account_account_get_user` |
| Memory | `memory_search` | `memory_memory_search` |
| Qdrant | `qdrant-find` | `qdrant_qdrant-find` |

项目按 `<server>_` 前缀过滤工具。上层客户端使用包含匹配，例如 `"kb_search" in tool.name`，因此不需要修改。

### 4.4 文本结果兼容

`MCPAdapter` 将文本结果表示为 LangChain 内容块：

```python
[{"type": "text", "text": "..."}]
```

现有工作流期望单个文本结果为字符串。统一封装层将“仅含一个文本块”的结果还原为字符串；图片、文件、多内容块等其他结果保持原格式。这避免在 `workflow.py` 以及四个业务客户端中分散添加兼容代码。

## 5. 文件影响范围

| 文件 | 处理 |
|---|---|
| `requirements.txt` | 更新 MCP/LangChain/FastMCP 版本约束，移除旧适配器 |
| `agentic/tools/mcp_client.py` | 切换到 `MCPAdapter`，更新配置、工具发现、命名空间过滤和文本结果兼容 |
| `tests/test_mcp_client.py` | 新增配置、命名空间过滤和文本结果兼容回归测试 |
| `agentic/tools/knowledge_client.py` | 无需修改，继续使用公共封装 |
| `agentic/tools/account_client.py` | 无需修改，继续使用公共封装 |
| `agentic/tools/memory_client.py` | 无需修改，继续使用公共封装 |
| `agentic/tools/qdrant_client.py` | 无需修改，继续使用公共封装 |
| `agentic/workflow.py` | 无需修改，继续接收兼容后的字符串结果 |

## 6. 验证

### 6.1 故障回归

迁移前运行：

```powershell
python -m pytest tests/test_mcp_client.py -q
```

测试收集阶段稳定复现 `RequestContext` 导入错误。

迁移后结果：

```text
2 passed
```

### 6.2 知识库工具

```powershell
python -m pytest tests/test_mcp_client.py tests/test_kb_tool.py -q
```

结果：

```text
3 passed
```

### 6.3 账户、记忆和 Qdrant 工具

```powershell
python -m pytest tests/test_account_tool.py tests/test_memory_tool.py tests/test_qdrant_tool.py -q
```

结果：

```text
5 passed
```

## 7. 已知风险

1. `langchain.mcp` 当前会发出 `LangChainBetaWarning`，其 API 后续可能变化。
2. 多服务工具名称增加服务端命名空间；新增调用方不能假设工具名称与服务端原名完全相同。
3. 文本兼容层只将单个文本块还原为字符串；多模态或多内容块结果仍按 LangChain 内容块返回。
4. 当前工具发现会访问配置中的全部 MCP 服务。任一服务不可用时，可能影响统一工具发现，后续可评估按服务创建适配器以隔离故障。

## 8. 后续建议

1. 在 CI 中固定并验证 `langchain`、`fastmcp` 和 `mcp` 的版本组合。
2. 为 MCP 服务不可用场景增加隔离和超时测试。
3. 在 `langchain.mcp` API 稳定后复查兼容层和命名空间逻辑。
4. 如开始使用图片、文件或结构化 MCP 输出，应让业务层原生处理内容块，逐步移除文本兼容层。
