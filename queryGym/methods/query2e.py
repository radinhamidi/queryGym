from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

@register_method("query2e")
class Query2E(BaseReformulator):
    VERSION = "1.0"

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        msgs = self.prompts.render("genqr.keywords.v1", query=q.text)
        out = self.llm.chat(msgs, temperature=0.3, max_tokens=self.cfg.llm.get("max_tokens",256))
        terms = [t.strip() for t in out.replace("\n"," ").split(",") if t.strip()]
        return ReformulationResult(q.qid, q.text, " ".join([q.text] + terms), metadata={"keywords": terms})
