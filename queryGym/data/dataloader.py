from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Literal, Union
import csv, json

from ..core.base import QueryItem

Backend = Literal["local", "msmarco", "beir"]
Source = Literal["file", "hf", "beir"]

@dataclass
class UnifiedQuerySource:
    backend: Backend
    source: Optional[Source] = None
    split: str = "dev"

    # Local
    path: Optional[Path] = None
    format: Optional[Literal["tsv","jsonl"]] = None
    tsv_qid_col: int = 0
    tsv_query_col: int = 1
    jsonl_qid_key: str = "qid"
    jsonl_query_key: str = "query"

    # MS MARCO
    msmarco_queries_tsv: Optional[Path] = None
    hf_name: Optional[str] = None
    hf_config: Optional[str] = None
    hf_qid_key: str = "query_id"
    hf_query_key: str = "query"

    # BEIR
    beir_root: Optional[Path] = None
    beir_name: Optional[str] = None

    def iter(self) -> Iterable[QueryItem]:
        if self.backend == "local":
            yield from self._iter_local()
        elif self.backend == "msmarco":
            if self.source == "file":
                yield from self._iter_msmarco_file()
            elif self.source == "hf":
                yield from self._iter_msmarco_hf()
            else:
                raise ValueError("msmarco requires source='file' or 'hf'")
        elif self.backend == "beir":
            if self.source == "beir":
                yield from self._iter_beir_official()
            elif self.source == "hf":
                yield from self._iter_beir_hf()
            else:
                raise ValueError("beir requires source='beir' or 'hf'")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # -------- Local --------
    def _iter_local(self) -> Iterable[QueryItem]:
        if self.path is None or self.format not in ("tsv","jsonl"):
            raise ValueError("backend=local needs path and format=tsv|jsonl")
        if self.format == "tsv":
            with open(self.path, "r", encoding="utf-8") as f:
                r = csv.reader(f, delimiter="\t")
                for row in r:
                    if len(row) <= max(self.tsv_qid_col, self.tsv_query_col): continue
                    qid = str(row[self.tsv_qid_col]).strip()
                    q = str(row[self.tsv_query_col]).strip()
                    if q: yield QueryItem(qid=qid, text=q)
        else:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    obj = json.loads(line)
                    qid = str(obj[self.jsonl_qid_key])
                    q = str(obj[self.jsonl_query_key]).strip()
                    if q: yield QueryItem(qid=qid, text=q)

    # -------- MS MARCO --------
    def _iter_msmarco_file(self) -> Iterable[QueryItem]:
        if not self.msmarco_queries_tsv:
            raise ValueError("msmarco file mode requires --msmarco-queries-tsv")
        with open(self.msmarco_queries_tsv, "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            for row in r:
                if len(row) < 2: continue
                yield QueryItem(qid=str(row[0]).strip(), text=str(row[1]).strip())

    def _iter_msmarco_hf(self) -> Iterable[QueryItem]:
        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError("pip install datasets") from e
        if not self.hf_name:
            raise ValueError("msmarco hf mode requires hf_name (e.g., 'ms_marco')")
        cfg = self.hf_config or "passage"
        ds = load_dataset(self.hf_name, cfg, split=self.split)
        for ex in ds:
            qid = str(ex[self.hf_qid_key])
            q = str(ex[self.hf_query_key]).strip()
            if q: yield QueryItem(qid=qid, text=q)

    # -------- BEIR --------
    def _iter_beir_official(self) -> Iterable[QueryItem]:
        try:
            from beir.datasets.data_loader import GenericDataLoader
        except Exception as e:
            raise RuntimeError("pip install beir") from e
        if not self.beir_root:
            raise ValueError("beir official mode requires --beir-root")
        loader = GenericDataLoader(data_folder=str(self.beir_root))
        try:
            _corpus, queries, _qrels = loader.load(split=self.split)
        except TypeError:
            _corpus, queries, _qrels = loader.load()
        for qid, q in queries.items():
            q = (q or "").strip()
            if q: yield QueryItem(qid=str(qid), text=q)

    def _iter_beir_hf(self) -> Iterable[QueryItem]:
        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError("pip install datasets") from e
        if not self.hf_name:
            raise ValueError("beir hf mode requires hf_name (e.g., 'beir/fiqa')")
        ds = load_dataset(self.hf_name, split=self.split)
        qid_key = "qid" if "qid" in ds.column_names else "query_id"
        query_key = "query" if "query" in ds.column_names else "text"
        for ex in ds:
            qid = str(ex[qid_key])
            q = str(ex[query_key]).strip()
            if q: yield QueryItem(qid=qid, text=q)

    @staticmethod
    def export_to_tsv(items: Iterable[QueryItem], out_path: Union[str, Path]) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            for it in items:
                w.writerow([it.qid, it.text])

@dataclass
class UnifiedContextSource:
    mode: Literal["file","pyserini"]
    path: Optional[Path] = None
    qid_key: str = "qid"
    ctx_key: str = "contexts"
    retriever: Optional[object] = None
    k: int = 10

    def load(self, queries: List[QueryItem]) -> Dict[str, List[str]]:
        if self.mode == "file":
            if not self.path:
                raise ValueError("context file mode requires JSONL path")
            out: Dict[str, List[str]] = {}
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    obj = json.loads(line)
                    out[str(obj[self.qid_key])] = list(obj[self.ctx_key])
            return out
        elif self.mode == "pyserini":
            if self.retriever is None:
                raise ValueError("pyserini mode requires a retriever instance")
            out: Dict[str, List[str]] = {}
            for q in queries:
                hits = self.retriever.search(q.text, self.k)
                ctxs: List[str] = []
                for h in hits:
                    if hasattr(h,"raw") and h.raw:
                        ctxs.append(h.raw)
                    elif hasattr(h,"contents"):
                        ctxs.append(h.contents)
                out[q.qid] = ctxs
            return out
        else:
            raise ValueError("Unknown context mode")
