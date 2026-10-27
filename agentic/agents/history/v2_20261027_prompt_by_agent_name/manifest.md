# Prompt configuration snapshot

- Version: `v2-2026-10-27-prompt-by-agent-name`
- Captured: 2026-10-27
- Reason: record the user-confirmed API where `apply_prompt_template(agent_name)` loads role prompt data through `get_agent_soul(agent_name)`.
- Configs: each role directory contains an exact copy of its `config.yaml`.
- Generator: `prompt.py` is the exact source used to read and render the saved role configs.
- Generated copy: `rendered_prompts.json` contains one assembled system prompt per current Agent.
- Correspondence check: both functions were loaded from the saved `prompt.py`; all four outputs were generated and re-read using only the saved YAML files.
