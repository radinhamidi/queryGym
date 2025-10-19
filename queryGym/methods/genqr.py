from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

@register_method("genqr")
class GENQR(BaseReformulator):
    VERSION = "1.0"

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        msgs = self.prompts.render("genqr.keywords.v1", query=q.text)
        out = self.llm.chat(msgs, temperature=self.cfg.llm.get("temperature",0.8),
                                  max_tokens=self.cfg.llm.get("max_tokens",256))
        terms = [t.strip() for t in out.replace("\n"," ").split(",") if t.strip()]
        expanded = " ".join(([q.text] * int(self.cfg.params.get("repeat_query_weight",3))) + terms)
        return ReformulationResult(q.qid, q.text, expanded, metadata={"keywords": terms})
