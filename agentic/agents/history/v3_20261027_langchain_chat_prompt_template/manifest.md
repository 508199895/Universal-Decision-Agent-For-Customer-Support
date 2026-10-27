# Prompt configuration snapshot

- Version: `v3-2026-10-27-langchain-chat-prompt-template`
- Captured: 2026-10-27
- Reason: migrate system-prompt assembly to the official LangChain `ChatPromptTemplate.from_messages()` API and reserve an optional `app_config` argument for later global settings.
- Configs: each role directory contains an exact copy of its `config.yaml`.
- Generator: `prompt.py` is the exact source used to load and render the saved role configs.
- Generated copy: `rendered_prompts.json` contains one assembled system prompt per current Agent.
- Correspondence check: each prompt was rendered from the saved YAML with the saved `apply_prompt_template(agent_name)`; rendering with an empty optional `app_config` produced the same text because global config handling is deferred.
