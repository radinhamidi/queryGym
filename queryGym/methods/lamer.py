from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method
from typing import List, Optional, Dict, Any

@register_method("lamer")
class LameR(BaseReformulator):
    VERSION = "0.1"
    REQUIRES_CONTEXT = True
    CONCATENATION_STRATEGY = "interleaved_query_content"  # q + a1 + q + a2 + q + a3 + q + a4 + q + a5

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        # Use provided contexts (from batch retrieval)
        ctxs = contexts or []

        # Prepare contexts for prompting (use top-10 by default)
        retrieval_k = int(self.cfg.params.get("retrieval_k", 10))
        prompt_ctxs = ctxs[:retrieval_k]
        contexts_blob = "\n".join([f"{i+1}. {psg}" for i, psg in enumerate(prompt_ctxs)])

        # Prompt LLM to generate multiple passages given contexts
        gen_passages: List[str] = []
        num_generations = int(self.cfg.params.get("gen_passages", 5))
        max_tokens = int(self.cfg.llm.get("max_tokens", 128))
        temperature = float(self.cfg.llm.get("temperature", 1.0))

        # Use prompt template defined in prompt_bank.yaml (lamer.msmarco.v1)
        msgs = self.prompts.render("lamer.msmarco.v1", query=q.text, contexts=contexts_blob)
        for _ in range(num_generations):
            passage = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            # Clean up quotes from the generated passage
            passage = passage.strip().strip('"').strip("'")
            gen_passages.append(passage)

        # Construct final expanded query using interleaved_query_content
        reformulated = self.concatenate_result(q.text, gen_passages)

        return ReformulationResult(
            q.qid,
            q.text,
            reformulated,
            metadata={
                "generated_passages": len(gen_passages),
                "used_ctx": len(prompt_ctxs),
            },
        )

    def _get_retrieval_params(self) -> Optional[Dict[str, Any]]:
        """Get LameR-specific retrieval parameters for batch retrieval."""
        index = self.cfg.params.get("index")
        if not index:
            return None
            
        return {
            "index": index,
            "k": int(self.cfg.params.get("retrieval_k", 10)),  # LameR retrieves top-10 by default
            "bm25": True,  # LameR uses BM25 by default
            "impact": False,
            "rm3": bool(self.cfg.params.get("rm3", False)),
            "rocchio": bool(self.cfg.params.get("rocchio", False)),
            "rocchio_use_negative": bool(self.cfg.params.get("rocchio_use_negative", False)),
            "disable_bm25_param": True,
            "k1": self.cfg.params.get("k1"),
            "b": self.cfg.params.get("b"),
            "batch_size": int(self.cfg.params.get("batch_size", 128)),
            "threads": int(self.cfg.params.get("threads", 16)),
        }
