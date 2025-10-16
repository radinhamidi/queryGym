from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class QueryItem:
    qid: str
    text: str

@dataclass
class ReformulationResult:
    qid: str
    original: str
    reformulated: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MethodConfig:
    name: str
    params: Dict[str, Any]
    llm: Dict[str, Any]
    seed: int = 42
    retries: int = 2

class BaseReformulator:
    VERSION = "0.1"
    REQUIRES_CONTEXT = False

    def __init__(self, cfg: MethodConfig, llm_client, prompt_resolver):
        self.cfg = cfg
        self.llm = llm_client
        self.prompts = prompt_resolver

    def reformulate(self, q: QueryItem, contexts: Optional[List[str]] = None) -> ReformulationResult:
        raise NotImplementedError

    def reformulate_batch(self, queries: List[QueryItem], ctx_map: Optional[Dict[str, List[str]]] = None):
        out: List[ReformulationResult] = []
        for q in queries:
            ctx = (ctx_map or {}).get(q.qid)
            out.append(self.reformulate(q, ctx))
        return out
