# UDA-Hub 产品与架构设计（中文版）

## 1. 项目概述

UDA-Hub（Universal Decision Agent Hub，通用决策智能体中心）是一套端到端自动化客户支持工作流，主要由以下能力驱动：

- 使用 LangGraph 进行多智能体编排；
- 使用大语言模型智能体完成工单受理、分类、解决、升级和监督；
- 使用 MCP（Model Context Protocol，模型上下文协议）服务作为外部工具：
  - `kb`：知识库检索；
  - `account`：用户资料与预约信息查询；
  - `memory`：工单记忆与历史记录管理。

系统能够将一条原始客户工单处理为自动解决结果或人工升级结果，并保存上下文记忆，供未来工单处理和个性化回复使用。

## 2. 总体架构

系统采用基于 Supervisor 的分层多智能体架构：

- Supervisor Agent 负责中央路由和高级决策；
- Intake、Classifier、Resolver、Escalation 等专业智能体各自处理一项明确任务；
- 所有智能体共享由 LangGraph 管理的状态；
- Supervisor 根据不断更新的状态决定调用哪个智能体、何时结束以及何时转交人工。

这种架构便于：

- 增加新的专业智能体，例如情绪优先级智能体或退款智能体；
- 在不改变各专业智能体内部实现的情况下调整路由逻辑；
- 通过结构化状态和日志观察、解释及调试决策过程。

### 2.1 工作流结构

```text
开始
  ↓
Intake Agent（受理与标准化）
  ↓
Classifier Agent（分类）
  ↓
Supervisor Agent（监督与路由）
  ├─→ Resolver Agent（尝试自动解决）
  │      └─→ 返回 Supervisor 再次决策
  ├─→ Escalation Agent（升级人工）
  │      └─→ 结束
  └─→ 结束
```

对应的 LangGraph 流程节点为：

```text
__start__ → intake → classifier → supervisor
                                  ├→ resolver → supervisor
                                  ├→ escalation → __end__
                                  └→ __end__
```

### 2.2 LangGraph 编排器

编排器负责定义：

- 状态结构（TypedDict）；
- 工作流节点（各个智能体）；
- 节点之间的边；
- 自动解决、人工升级和流程结束等条件转换；
- 部分解决后重新进入 Resolver 的循环。

主要实现位于 `agentic/workflow.py`。

## 3. 智能体交互与职责

### 3.1 Intake Agent（受理智能体）

Intake Agent 接收原始工单内容及其元数据，并对工单进行标准化和补充。

输出内容包括：

- `summary`：工单摘要；
- `normalized_issue`：标准化后的问题描述；
- `sentiment`：客户情绪；
- `suspected_language`：推测的工单语言。

输出写入共享状态的 `intake` 字段。

### 3.2 Classifier Agent（分类智能体）

Classifier Agent 读取共享状态中的 `ticket` 和 `intake`，并生成：

- `issue_type`：问题类型，例如登录、账单、预约或其他；
- `urgency`：紧急程度，分为低、中、高；
- `complexity`：问题复杂度；
- `should_escalate_immediately`：是否应立即升级人工。

结果写入共享状态的 `classification` 字段。

### 3.3 Supervisor Agent（监督智能体）

Supervisor Agent 读取：

- 原始工单 `ticket`；
- 标准化结果 `intake`；
- 分类结果 `classification`；
- 最近一次解决结果 `resolution`（如果存在）。

它根据这些信息选择 `next_step`：

- `resolver`：尝试自动解决；
- `escalation`：转交人工客服；
- `end`：工单已经处理完成。

决策结果及理由写入共享状态的 `supervisor` 字段。其路由规则包括立即升级判断、Resolver 置信度阈值以及分类结果中的升级提示。

### 3.4 Resolver Agent（解决智能体）

当 `supervisor.next_step == "resolver"` 时，Resolver Agent 被调用。

它可以使用以下 MCP 工具：

- `kb_search`：检索相关知识库文章；
- `account_get_user`：查询用户资料；
- `account_get_user_reservations`：查询用户预约记录；
- `memory_search`：查询相似历史工单或处理结果；

Resolver 综合工具返回的信息，生成候选答案和置信度，并更新 `resolution`：

- `answer`：面向客户的建议回复；
- `status`：解决状态；
- `confidence`：0～1 的置信度；
- `used_kb_articles`：使用过的知识库文章；

完成后，Supervisor 再次读取解决结果，并决定：

- 结束工作流；
- 再次尝试解决；
- 升级给人工客服。

### 3.5 Escalation Agent（升级智能体）

当 Supervisor 决定升级，或分类结果要求立即升级时，Escalation Agent 被调用。

它读取工单、受理结果、分类结果和已有解决结果，并生成面向人工客服的结构化交接信息：

- `summary_for_human`：面向人工客服的问题摘要；
- `recommended_department`：建议转交的部门；
- `proposed_next_steps`：建议的后续处理步骤；
- Resolver 已生成的人工备注（如果存在）。

结果写入共享状态的 `escalation` 字段，并可通过 `memory_write` 保存相关记忆。

## 4. 工单输入与输出

### 4.1 输入格式

UDA-Hub 面向标准化工单对象，而不是仅接收一段无结构文本。输入示例如下：

```json
{
  "ticket_id": "工单编号",
  "content": "您好，我无法登录我的账户……",
  "owner_id": "外部用户编号",
  "owner_name": "用户姓名",
  "channel": "chat | email | web | other",
  "tags": "可选的逗号分隔标签"
}
```

### 4.2 共享状态

每个工作流步骤更新共享状态中的一部分：

```python
class TicketState(TypedDict, total=False):
    ticket: Dict[str, Any]
    intake: Dict[str, Any]
    classification: Dict[str, Any]
    resolution: Dict[str, Any]
    escalation: Dict[str, Any]
    supervisor: Dict[str, Any]
```

各字段含义如下：

- `ticket`：原始工单及元数据；
- `intake`：标准化问题、摘要和情绪等信息；
- `classification`：问题类型、紧急度、复杂度和升级建议；
- `resolution`：自动解决状态、答案、置信度和使用过的知识；
- `escalation`：人工交接信息；
- `supervisor`：最近一次路由决定及其依据。

### 4.3 最终输出

每张工单的最终状态可能包含：

- 原始请求元数据；
- 标准化后的问题和情绪信息；
- 问题分类、紧急度和复杂度；
- 自动解决结果：
  - 状态；
  - 面向客户的答案；
  - 置信度；
  - 使用的知识库文章；
- 可选的人工升级结果：
  - 人工处理摘要；
  - 推荐部门；
  - 建议的后续步骤。

系统对外可观察的最终结果为以下二者之一：

1. 自动生成的客户回复；
2. 提交给人工客服的结构化升级包。

## 5. 不同工单类型的处理方式

### 5.1 登录与访问问题

- Classifier 将问题类型标记为 `login`，通常会赋予较高紧急度；
- Resolver 优先检索带有登录、密码、访问等标签的知识文章；
- 如果客户未收到重置邮件且解决置信度较低，Supervisor 倾向于将工单升级至技术支持部门。

### 5.2 账单与订阅问题

- 问题类型标记为 `billing` 或订阅相关类型；
- Resolver 检索账单相关知识，并根据需要查询用户资料；
- 如果知识库中的政策清晰且匹配度较高，则更可能自动解决。

### 5.3 预约与体验问题

- 问题类型标记为 `reservation` 或体验相关类型；
- Resolver 查询用户的具体预约记录；
- 同时检索取消、未到场或活动规则等知识库内容。

### 5.4 一般或未知问题

- 问题类型标记为 `other`；
- Resolver 使用标准化问题进行尽力而为的知识库检索；
- 如果没有匹配结果或置信度较低，Supervisor 将工单转交人工处理。

## 6. MCP 外部工具服务

### 6.1 知识库服务

位置：`mcp_services/kb/server.py`

职责：

- 加载知识库文章数据；
- 根据工单问题检索相关文章；
- 返回候选文章及正文内容。

工具：

- `kb_search`；
- `kb_get`（设计文档中列出）。

### 6.2 账户服务

位置：`mcp_services/account/server.py`

职责：

- 查询用户资料；
- 查询用户预约历史。

工具：

- `account_get_user`；
- `account_get_user_reservations`。

### 6.3 记忆服务

位置：`mcp_services/memory/server.py`

用于保存和检索工单历史，包括：

- 工单编号；
- 工单内容；
- 元数据；
- 时间戳。

工具：

- `memory_write`；
- `memory_search`；
- `memory_get_all`。

## 7. 日志与可观测性

UDA-Hub 使用端到端结构化 JSON 日志，完整跟踪：

```text
Intake → Classification → Supervisor → Resolver → Escalation → Completion
```

所有日志通过共享日志组件输出，主要覆盖 `agentic/workflow.py` 和 `03_agentic_app.py`。

### 7.1 日志输出位置

- 标准输出：便于本地开发和调试；
- JSONL 文件：`logs/uda_hub.jsonl`，每行一个 JSON 日志对象。

日志示例：

```json
{
  "timestamp": "2025-12-02T09:42:10.123456Z",
  "level": "INFO",
  "message": "supervisor_decision",
  "extra": {
    "ticket_id": "DEMO-TICKET-001",
    "thread_id": "DEMO-TICKET-001",
    "issue_type": "login",
    "urgency": "high",
    "complexity": "medium",
    "next_step": "escalation",
    "resolver_status": "needs_escalation",
    "resolver_confidence": 0.2
  }
}
```

### 7.2 节点生命周期事件

每个工作流节点都会记录开始和结束事件，例如：

- `node_start_intake` / `node_end_intake`；
- `node_start_classifier` / `node_end_classifier`；
- `node_start_supervisor` / `node_end_supervisor`；
- `node_start_resolver` / `node_end_resolver`；
- `node_start_escalation` / `node_end_escalation`。

日志包含 `ticket_id` 和 `thread_id`。

### 7.3 Supervisor 决策日志

事件名为 `supervisor_decision`，记录：

- 工单编号与线程编号；
- 问题类型、紧急度和复杂度；
- Resolver 状态和置信度；
- 下一步动作；
- 决策原因。

### 7.4 工具调用日志

各 MCP 工具记录调用开始和结束事件，包括：

- 知识库检索；
- 用户资料查询；
- 用户预约查询；
- 长期记忆检索。

记录内容包括：

- 工单编号、线程编号和工具名；
- 经过清理的输入参数；
- 返回结果数量；
- 记忆和预约数量；
- 知识检索的最高得分、词汇重叠、显著词重叠及命中信息。

### 7.5 Resolver 结果日志

系统记录 Resolver 的知识库置信度和最终决定，包括：

- 知识库结果数量；
- 最高匹配分；
- 词汇重叠和显著词重叠；
- 最终置信度；
- 自动解决或需要升级的状态。

### 7.6 工作流完成日志

事件名为 `workflow_completed`，记录：

- 工单编号；
- 线程编号；
- 最终状态：自动解决或需要升级；
- 最终置信度。

这些日志使系统具备完整的工单追踪、决策可见性、知识库质量调试、工具调用审计以及运行监控能力。

## 8. 使用的技术

- Python 3.11；
- LangChain；
- LangGraph；
- MCP / FastMCP；
- OpenAI 模型；
- SQLite 核心数据库和外部数据库；
- JSONL 知识数据；
- pytest 与异步测试支持；
- Mermaid 和 LangGraph 图导出。

## 9. 部署与运行

### 9.1 启动 MCP 服务

```bash
python mcp_services/kb/server.py
python mcp_services/account/server.py
python mcp_services/memory/server.py
```

### 9.2 运行应用

演示模式：

```bash
python 03_agentic_app.py --mode demo
```

### 9.3 运行测试

```bash
pytest -q
```

## 10. 总结

UDA-Hub 提供了一套模块化、可扩展且可测试的自动化客户支持设计，核心能力包括：

- 多智能体大语言模型推理；
- 外部工具和业务数据集成；
- 短期及长期上下文记忆；
- 由 Supervisor 驱动的确定性工作流路由；
- 自动解决与人工升级；
- 完整的结构化日志及审计链路。

该架构可以用于客户支持自动化，也可以继续扩展到企业工单和服务决策场景。

## 11. 来源文档

本文不参考 `README.md`，仅根据项目当前仓库中的以下设计文件翻译整理：

1. `agentic/design/architecture-design.md`：唯一文字内容来源，包含系统概述、架构、智能体职责、输入输出、MCP 工具、状态、技术栈和运行方式。
2. `uda_hub_graph.mmd`：用于核对 LangGraph 工作流节点及连接关系。
3. `agentic/design/Graph.jpeg`：原始架构图的图片版本。

> 说明：本文保留了 `architecture-design.md` 的术语、章节含义和设计内容，对重复的日志描述进行了合并，并使用流程图文件核对节点关系。本文未采用 `README.md` 中的任何补充说明。
