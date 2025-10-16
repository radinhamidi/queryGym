from __future__ import annotations
import typer, csv, json
from pathlib import Path
from typing import Optional

from .core.base import MethodConfig
from .core.runner import run_method
from .core.prompts import PromptBank
from .data.dataloader import UnifiedQuerySource
app = typer.Typer(help="QueryReform Toolkit CLI")

def build_script_lines(index_path:str, topics:str, run:str, qrels:Optional[str]=None,
                       bm25:bool=True, extra:str="") -> list[str]:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Pyserini search",
        "python -m pyserini.search.lucene \\",
        f"  --index {index_path} \\",
        f"  --topics {topics} \\",
        f"  --output {run} \\",
        "  --bm25" if bm25 else "  --qld",
    ]
    if extra:
        lines.append(f"  {extra}")
    lines.append("")
    if qrels:
        lines += [
            "# trec_eval",
            f"trec_eval -m map -m P.10 -m ndcg_cut.10 {qrels} {run} | tee {run}.eval.txt"
        ]
    return lines

@app.command()
def run(method: str = typer.Option(...),
        queries_tsv: Path = typer.Option(..., "--queries-tsv", exists=True),
        output_tsv: Path = typer.Option(..., "--output-tsv"),
        cfg_path: Optional[Path] = typer.Option(None, "--cfg-path"),
        prompt_bank: Path = typer.Option(Path(__file__).with_name("prompt_bank.yaml"), "--prompt-bank"),
        ctx_jsonl: Optional[Path] = typer.Option(None, "--ctx-jsonl", help="Optional contexts JSONL"),
):
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path else {}
    mc = MethodConfig(name=method, params=cfg.get("params",{}), llm=cfg["llm"],
                      seed=cfg.get("seed",42), retries=cfg.get("retries",2))
    src = UnifiedQuerySource(backend="local", format="tsv", path=queries_tsv)
    queries = list(src.iter())
    ctx_map = None
    if ctx_jsonl:
        from .data.dataloader import UnifiedContextSource
        ctx_src = UnifiedContextSource(mode="file", path=ctx_jsonl)
        ctx_map = ctx_src.load(queries)
    results = run_method(method_name=method, cfg=mc, queries=queries,
                         prompt_bank_path=str(prompt_bank), ctx_map=ctx_map)
    with open(output_tsv, "w", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for r in results:
            w.writerow([r.qid, r.reformulated])
    typer.echo(f"Wrote {output_tsv}")

@app.command("data-to-tsv")
def data_to_tsv(backend: str = typer.Option(...),
                source: str = typer.Option(...),
                out: Path = typer.Option(..., "--out"),
                path: Optional[Path] = typer.Option(None, help="Local TSV/JSONL path"),
                format: Optional[str] = typer.Option(None, help="tsv|jsonl"),
                tsv_qid_col: int = 0, tsv_query_col: int = 1,
                jsonl_qid_key: str = "qid", jsonl_query_key: str = "query",
                msmarco_queries_tsv: Optional[Path] = None,
                hf_name: Optional[str] = None, hf_config: Optional[str] = None,
                split: str = "dev", hf_qid_key: str = "query_id", hf_query_key: str = "query",
                beir_root: Optional[Path] = None, beir_name: Optional[str] = None,
):
    src = UnifiedQuerySource(
        backend=backend if backend in ("local","msmarco","beir") else None,
        source=source if source in ("file","hf","beir") else None,
        path=path, format=format, tsv_qid_col=tsv_qid_col, tsv_query_col=tsv_query_col,
        jsonl_qid_key=jsonl_qid_key, jsonl_query_key=jsonl_query_key,
        msmarco_queries_tsv=msmarco_queries_tsv,
        hf_name=hf_name, hf_config=hf_config, split=split,
        hf_qid_key=hf_qid_key, hf_query_key=hf_query_key,
        beir_root=beir_root, beir_name=beir_name,
    )
    UnifiedQuerySource.export_to_tsv(src.iter(), out)
    typer.echo(f"Wrote {out}")

@app.command("prompts-list")
def prompts_list(prompt_bank: Path = typer.Option(Path(__file__).with_name("prompt_bank.yaml"))):
    pb = PromptBank(prompt_bank)
    for pid in pb.list():
        typer.echo(pid)

@app.command("prompts-show")
def prompts_show(prompt_id: str,
                 prompt_bank: Path = typer.Option(Path(__file__).with_name("prompt_bank.yaml"))):
    pb = PromptBank(prompt_bank)
    meta = pb.get_meta(prompt_id)
    typer.echo(json.dumps(meta, indent=2))

@app.command("script-gen")
def script_gen(index_path: Path = typer.Option(..., exists=False),
               topics: Path = typer.Option(..., help="TSV queries"),
               run: Path = typer.Option(..., help="Output run file"),
               output_bash: Path = typer.Option(Path("run_retrieval.sh")),
               qrels: Optional[Path] = None,
               bm25: bool = True,
               extra: str = typer.Option("", help="Extra Pyserini CLI flags, concatenated")
):
    lines = build_script_lines(str(index_path), str(topics), str(run),
                               str(qrels) if qrels else None, bm25=bm25, extra=extra)
    output_bash.write_text("\n".join(lines))
    typer.echo(f"Generated {output_bash}")

if __name__ == "__main__":
    app()
