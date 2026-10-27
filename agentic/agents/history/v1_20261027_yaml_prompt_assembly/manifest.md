# Prompt configuration snapshot

- Version: `v1-2026-10-27-yaml-prompt-assembly`
- Captured: 2026-10-27
- Reason: preserve the first YAML-backed system prompt configuration and its generated prompt text after unifying Agent roles and prompt assembly.
- Configs: `configs/*.yaml` are exact copies of the four current Agent YAML files.
- Generator: `prompt.py` is the exact prompt assembly source used for this snapshot.
- Generated copy: `rendered_prompts.json` contains one assembled system prompt per Agent.
- Correspondence check: each saved YAML file was loaded and rendered with the saved `apply_prompt_template()`; each generated value was checked against a second render from the same saved files.
