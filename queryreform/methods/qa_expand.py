from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

@register_method("qa_expand")
class QAExpand(BaseReformulator):
    VERSION = "1.0"
    REQUIRES_CONTEXT = True

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        ctx_blob = "\n".join(contexts or [])
        msgs = self.prompts.render("qa_expand.subq.v1", query=q.text)
        subqs = self.llm.chat(msgs, temperature=0.2, max_tokens=self.cfg.llm.get("max_tokens",256))
        msgs = self.prompts.render("qa_expand.answer.v1", query=q.text, contexts=ctx_blob)
        answers = self.llm.chat(msgs, temperature=0.2, max_tokens=self.cfg.llm.get("max_tokens",256))
        msgs = self.prompts.render("qa_expand.refine.v1", query=q.text, subqs=subqs, answers=answers)
        final_q = self.llm.chat(msgs, temperature=0.3, max_tokens=self.cfg.llm.get("max_tokens",256))
        return ReformulationResult(q.qid, q.text, final_q, metadata={"subqs": subqs, "answers": answers})
