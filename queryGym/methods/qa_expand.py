from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

@register_method("qa_expand")
class QAExpand(BaseReformulator):
    VERSION = "1.0"
    REQUIRES_CONTEXT = True
    CONCATENATION_STRATEGY = "query_repeat_plus_generated"  # (Q × 3) + refined_answers
    DEFAULT_QUERY_REPEATS = 3  # QA-Expand uses 3 repetitions

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        ctx_blob = "\n".join(contexts or [])
        msgs = self.prompts.render("qa_expand.subq.v1", query=q.text)
        subqs = self.llm.chat(msgs, temperature=0.2, max_tokens=self.cfg.llm.get("max_tokens",256))
        msgs = self.prompts.render("qa_expand.answer.v1", query=q.text, contexts=ctx_blob)
        answers = self.llm.chat(msgs, temperature=0.2, max_tokens=self.cfg.llm.get("max_tokens",256))
        msgs = self.prompts.render("qa_expand.refine.v1", query=q.text, subqs=subqs, answers=answers)
        final_q = self.llm.chat(msgs, temperature=0.3, max_tokens=self.cfg.llm.get("max_tokens",256))
        
        # For QAExpand, we use the final refined query as the generated content
        reformulated = self.concatenate_result(q.text, final_q)
        
        return ReformulationResult(q.qid, q.text, reformulated, metadata={
            "subqs": subqs, 
            "answers": answers,
            "final_q": final_q
        })
