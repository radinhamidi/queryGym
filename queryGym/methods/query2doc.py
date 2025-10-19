from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

@register_method("query2doc")
class Query2Doc(BaseReformulator):
    VERSION = "1.0"

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        mode = str(self.cfg.params.get("mode","zs"))
        pid = "query2doc.zeroshot.v1" if mode == "zs" else "query2doc.cot.v1"
        msgs = self.prompts.render(pid, query=q.text)
        doc = self.llm.chat(msgs, temperature=self.cfg.llm.get("temperature",0.7),
                                   max_tokens=self.cfg.llm.get("max_tokens",256))
        return ReformulationResult(q.qid, q.text, doc, metadata={"mode": mode})
