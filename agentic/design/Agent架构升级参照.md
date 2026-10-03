# Agent 架构升级参照

以下项目按照“业务结构相似”和“微软技术栈相似”两个方向筛选，重点关注 Orchestrator、专业 Agent、RAG、业务工具和人工升级等能力。

| 项目 | 相似程度 | Agent／组件 | 工作流特点 | 适合参考的部分 |
|---|---:|---|---|---|
| [AWS Multi-Agent Customer Service Platform](https://github.com/aws-samples/sample-multi-agent-customer-service-for-bedrock) | 很高 | Supervisor、Billing Specialist、Scheduling Specialist、Escalation Specialist | Supervisor 先分类，将专业 Agent 当作工具调用；Billing 和 Escalation 使用 RAG，Scheduling 执行业务操作，Escalation 创建工单并转人工 | 与 Microsoft Customer Chatbot Solution Accelerator 最接近的“Orchestrator + 专业 Agent + RAG + 人工升级”完整实现；代码目录也比较清晰 |
| [OpenAI Customer Service Agents Demo](https://github.com/openai/openai-cs-agents-demo) | 很高 | Triage、Flight Information、Booking & Cancellation、Seat & Special Services、FAQ、Refunds & Compensation | Triage 负责初始分流；专业 Agent 可以相互 Handoff；工具执行改签、取消、座位和补偿等业务操作 | 适合参考专业 Agent 的职责拆分、Handoff、Guardrail 和前后端交互 |
| [Azure Language + OpenAI Conversational Agent Accelerator](https://github.com/Azure-Samples/Azure-Language-OpenAI-Conversational-Agent-Accelerator) | 高 | Orchestrator、Intent Routing、CQA、Function Calling、RAG Fallback | Orchestrator 根据配置选择意图路由、问答、函数调用或直接 RAG；无法匹配时进入兜底流程 | 与 Microsoft Customer Chatbot 的“路由 + 企业知识检索”最接近，适合研究混合路由策略 |
| [Microsoft Moneta Agents](https://github.com/Azure-Samples/moneta-agents) | 高 | Coordinator、CRM Agent、Policies Agent、Funds Agent、CIO Agent、News Agent | Coordinator 使用 `HandoffBuilder` 路由；各专业 Agent 访问 CRM、Azure AI Search、RSS 等数据源；Cosmos DB 保存对话 | 虽然是银行／保险场景，但和目标项目使用相同的 Microsoft Agent Framework、Azure AI Search、Cosmos DB 与 Handoff 模式 |
| [Microsoft Agents for Enhanced Customer Care](https://github.com/microsoft/Agents-for-Enhanced-Customer-Care-Solution-Accelerator) | 中高 | Copilot Studio 客服 Agent、知识与动作、人工客服 Copilot | 语音客户进入 AI Agent → 识别意图 → 动态编排知识、Topic 和 Action → 无法解决时转 Dynamics 365 人工客服，并生成会话摘要 | 适合参考语音客服、CRM 集成、人工转接和坐席辅助，但代码定制更依赖 Microsoft 商业平台 |

