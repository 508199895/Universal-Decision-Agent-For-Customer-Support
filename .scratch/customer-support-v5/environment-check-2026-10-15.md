# 01 前置环境检查（2026-10-15）

本记录仅说明前置能力，不代表 01 号任务或 V5 验收已完成。

## Conda 激活

默认 PowerShell 的 conda 命令解析为 `D:\miniconda3\Library\bin\conda.bat`，并非 Conda PowerShell 函数。直接调用 `conda activate uda-hub` 即使退出码为 0，Python 仍可能指向 base。

在项目目录执行 `. .\Activate-UdaHub.ps1` 加载 Conda PowerShell 模块及 Python dispatcher，之后可正常执行 `conda activate uda-hub`。脚本现在检查实际 Python 可执行文件与 sys.prefix，避免仅凭退出码或 CONDA_PREFIX 判断成功。

沙箱外验证通过：sys.executable 为 `E:\007.agent\007.project\RAG\ENV\uda-hub\python.exe`。沙箱内 base Python 仍可能出现 `Failed to find real location` 警告，不能据此宣称全局 Conda 已修复；未修改用户 PowerShell 配置或 Conda 安装。

## SQLite 持久检查点

在 uda-hub 安装 langgraph-checkpoint-sqlite 3.1.2、aiosqlite 0.22.1、sqlite-vec 0.1.9；依赖解析将 langgraph-checkpoint 从 4.2.0 更新到 4.3.0。requirements.txt 增加 SQLite saver 依赖。pip check 返回 No broken requirements found。

create_agent、before_model、ProviderStrategy、AsyncSqliteSaver、interrupt、Command 导入通过。导入不代表中间件或真实模型行为已验收。

测试：`python -m pytest tests/test_checkpoint_capability.py -q -p no:cacheprovider --basetemp=.scratch/checkpoint-capability-elevated`，关闭第三方 pytest 插件自动加载后，沙箱外 1 passed（2.22 秒）。测试使用两个独立进程及真实临时 SQLite 文件，验证 interrupt 暂停和 Command 恢复保留问题及回复。沙箱内子进程测试超时，需在允许的执行环境运行。该测试不涉及真实客户数据、不证明核心工单事务或 V5 业务恢复已实现。

## 模型能力阻碍

当前 workflow 使用 ChatOpenAI 接入 DeepSeek，专用 langchain-deepseek 缺失不构成现有接法的依赖故障，未为此安装新集成。

Context7 获取的官方文档：

- https://docs.langchain.com/oss/python/langchain/structured-output ：ProviderStrategy 使用提供商原生结构化输出；自动策略可能转为 ToolStrategy。
- https://docs.langchain.com/oss/python/langchain/models ：json_mode 保证 JSON，schema 仍须通过提示词描述。
- https://api-docs.deepseek.com/api/create-chat-completion ：response_format 列出 text/json_object；strict schema 为工具调用 Beta 能力。
- https://docs.langchain.com/oss/python/langgraph/interrupts ：恢复使用相同 thread_id 与 Command(resume=...)，暂停节点重新从头执行。

现有 DeepSeek JSON 输出不能作为 V5 ProviderStrategy 原生 schema 能力的证明。等待用户选择支持该能力的模型及配置方式，或明确批准调整规格；未修改提示词、模型配置或 V5 规格，未静默降级。真实提供商契约尚未请求验证。

## 后续更正：Responses API 实测通过

用户补充官方 Responses API 文档后，核对 https://api-docs.deepseek.com/zh-cn/api/create-response/ ：该接口列出 deepseek-flash/deepseek-v4-pro，并支持 text.format 的 json_schema。此前检查仅覆盖 Chat Completions，不能据此断言 DeepSeek 提供商整体不支持原生结构化输出。保留上述历史记录，在此更正，不需调整 V5 规格。

执行 `probe_responses.py`，使用项目现有凭据和 deepseek-flash，ChatOpenAI(use_responses_api=True, use_previous_response_id=False)、create_agent、显式 ProviderStrategy(严格 Pydantic 模型) 通过真实调用。结果为 result=capability_ok、count=7；before_model 执行一次，原生消息无工具调用。该探测无业务工具，不接触客户数据。deepseek-v4-pro 未实测，不泛化为所有复杂 schema 已验证。

DeepSeek Responses 接口无服务端会话存储，后续应回传完整历史，不依赖 previous_response_id。当前票未修改既有模型默认配置或提示词，未来 V5 节点需显式选择 Responses 接口。

## 入口预重构与新阻碍

新增 WorkflowDependencies、可注入 checkpointer 的 build_workflow、arun_ticket/aresume_ticket。run_ticket 保持原有调用参数兼容；默认模型及服务延迟至实际使用时初始化。模型、查询加载器及账户/记忆工具均由实例持有，禁用工具不会自动连接默认服务。缺检查点时恢复拒绝重新创建工单。

红灯复现：无凭据且禁用网络后 import agentic.workflow 因导入期凭据校验失败。重构后入口回归 `tests/test_workflow_entry.py` 的离线导入、替身问答与真实 SQLite 重新加载两项通过（2 passed，3.21 秒）。持久暂停恢复能力仍由独立进程能力测试覆盖，不声称已实现 V5 两类业务暂停和核心工单事务。

只读契约核对发现：业务 ID 字段为字符串，外部 user_id、核心 user_id、account_id 不能混用；现有账户查询入参为 external_user_id；活动 when 为未标注时区的 DateTime，需在预订取消票前确定业务时区；取消状态列均为自由字符串，尚未从实际数据取得取消字面值。旧工单消息存储为 role/content，不能直接证明支持原生完整消息。现有账户 profile serializer 读取 name，而业务模型字段为 full_name，记录供后续受控适配使用，本票未修改服务。

实际 MCP/Qdrant 契约探测失败：配置的远端连接报告 All connection attempts failed，未取得 qdrant-find，返回结构未核验。probe_contracts.py 未完成，未生成成功的契约证据 JSON；不能把其存在当成验收通过。依项目“关键验证失败时停止”规则暂停实现，等待恢复已有 Qdrant MCP 服务后补验。未运行完整测试集、未作最终 code-review、未提交、01 未标记完成。数据库读取使用 mode=ro，未修改业务数据库或 MCP/RAG 服务。
