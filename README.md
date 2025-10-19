# queryGym

A lightweight, reproducible toolkit for **LLM-based query reformulation**.

- Single **Prompt Bank** (YAML) with metadata.
- **Unified dataloader**: Local / MS MARCO / BEIR (file or HF) → `qid<TAB>query`.
- **OpenAI-compatible** LLM client (works with any OpenAI API–compatible endpoint).
- **Pyserini** optional: either pass contexts (JSONL) or pass a retriever instance to build contexts.
- Export-only: emits reformulated queries; optionally generates a **bash** script for Pyserini + `trec_eval`.

## Quickstart

```bash
pip install -e .[hf,beir,dev]
export OPENAI_API_KEY=sk-...

# Convert dataset → standard TSV (choose one)
queryGym data-to-tsv --backend msMarco --source file   --msmarco-queries-tsv /data/msmarco/queries.dev.small.tsv   --out out/msmarco.dev.tsv

# Run a method (e.g., genqr_ensemble)
queryGym run --method genqr_ensemble   --queries-tsv out/msmarco.dev.tsv   --output-tsv out/genqr_ens.tsv   --cfg-path queryGym/config/defaults.yaml
```
