# Contributing

## Adding a new prompt
- Edit `queryGym/prompt_bank.yaml`
- Add an entry with fields: `id`, `method_family`, `version`, `introduced_by`, `license`, `authors`, `tags`, `template:{system,user}`, `notes`.

## Adding a new method
- Create a class under `queryGym/methods/*.py`
- Subclass `BaseReformulator`, annotate `VERSION`, and register with `@register_method("name")`.
- Pull templates via `PromptBank.render(prompt_id, query=...)`.
