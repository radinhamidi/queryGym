from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

@register_method("lamer")
class LameR(BaseReformulator):
    VERSION = "0.1"
    REQUIRES_CONTEXT = True

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        ctxs = contexts or []
        combined = q.text + " " + " ".join(ctxs[: self.cfg.params.get("k_ctx", 3)])
        return ReformulationResult(q.qid, q.text, combined, metadata={"used_ctx": len(ctxs)})
