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
    CONCATENATION_STRATEGY = "query_repeat_plus_generated"  # (Q × 5) + K1 + K2 + ... + K10
    DEFAULT_QUERY_REPEATS = 5  # GenQREnsemble uses 5 repetitions

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        all_terms = []
        for pid in VARIANT_IDS:
            msgs = self.prompts.render(pid, query=q.text)
            out = self.llm.chat(msgs, temperature=self.cfg.llm.get("temperature",0.9),
                                      max_tokens=self.cfg.llm.get("max_tokens",256))
            all_terms += [t.strip() for t in out.replace("\n"," ").split(",") if t.strip()]
        generated_content = " ".join(all_terms)
        reformulated = self.concatenate_result(q.text, generated_content)
        return ReformulationResult(q.qid, q.text, reformulated, metadata={"keywords": all_terms})
