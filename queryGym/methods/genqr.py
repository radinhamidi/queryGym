from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

@register_method("genqr")
class GENQR(BaseReformulator):
    VERSION = "1.0"
    CONCATENATION_STRATEGY = "query_repeat_plus_generated"  # (Q × 5) + expansion_text
    DEFAULT_QUERY_REPEATS = 5  # GenQR uses 5 repetitions

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        msgs = self.prompts.render("genqr.keywords.v1", query=q.text)
        out = self.llm.chat(msgs, temperature=self.cfg.llm.get("temperature",0.8),
                                  max_tokens=self.cfg.llm.get("max_tokens",256))
        terms = [t.strip() for t in out.replace("\n"," ").split(",") if t.strip()]
        generated_content = " ".join(terms)
        reformulated = self.concatenate_result(q.text, generated_content)
        return ReformulationResult(q.qid, q.text, reformulated, metadata={"keywords": terms})
