# Prompt configuration snapshot

- Version: `v4-2026-10-27-before-agent-soul-move`
- Captured: 2026-10-27
- Reason: preserve the current role YAML, loader/template source, and rendered prompts before moving YAML loading into `agent.py`.
- Configs: each role directory contains an exact copy of its `config.yaml`.
- Generator: `prompt.py` and `agent.py` are exact copies from the pre-change workspace.
- Generated copy: `rendered_prompts.json` contains each role prompt rendered from the saved YAML by the current implementation.
- Correspondence check: prompt rendering completed for all roles listed by `AGENT_NAMES`; saved YAML and code are byte-for-byte copies of the current source.
