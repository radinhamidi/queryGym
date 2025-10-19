from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

@register_method("lamer")
class LameR(BaseReformulator):
    VERSION = "0.1"
    REQUIRES_CONTEXT = True
    CONCATENATION_STRATEGY = "interleaved_query_content"  # q + a1 + q + a2 + q + a3 + q + a4 + q + a5

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        ctxs = contexts or []
        # Take only the first k contexts
        k_ctx = self.cfg.params.get("k_ctx", 3)
        selected_contexts = ctxs[:k_ctx]
        
        # Use the enhanced concatenate_result method with multiple content pieces
        reformulated = self.concatenate_result(q.text, selected_contexts)
        
        return ReformulationResult(q.qid, q.text, reformulated, metadata={
            "used_ctx": len(selected_contexts),
            "total_ctx": len(ctxs)
        })
