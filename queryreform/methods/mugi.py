from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

@register_method("mugi")
class MuGI(BaseReformulator):
    VERSION = "0.1"
    REQUIRES_CONTEXT = False

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        msgs = self.prompts.render("query2doc.zeroshot.v1", query=q.text)
        pseudo = self.llm.chat(msgs, temperature=0.7, max_tokens=self.cfg.llm.get("max_tokens",256))
        combined = f"{q.text} {pseudo}"
        return ReformulationResult(q.qid, q.text, combined, metadata={"pseudo": pseudo})
