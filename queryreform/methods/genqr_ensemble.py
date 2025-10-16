from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

VARIANT_IDS = [
  "genqr.keywords.v1",
  "genqr.keywords.variant.v1",
]

@register_method("genqr_ensemble")
class GenQREnsemble(BaseReformulator):
    VERSION = "1.0"

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        reps = int(self.cfg.params.get("repeat_query_weight",3))
        all_terms = []
        for pid in VARIANT_IDS:
            msgs = self.prompts.render(pid, query=q.text)
            out = self.llm.chat(msgs, temperature=self.cfg.llm.get("temperature",0.9),
                                      max_tokens=self.cfg.llm.get("max_tokens",256))
            all_terms += [t.strip() for t in out.replace("\n"," ").split(",") if t.strip()]
        expanded = " ".join(([q.text]*reps) + all_terms)
        return ReformulationResult(q.qid, q.text, expanded, metadata={"keywords": all_terms})
