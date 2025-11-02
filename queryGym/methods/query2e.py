from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

@register_method("query2e")
class Query2E(BaseReformulator):
    """
    Query2E: Query to keyword expansion.
    
    Modes:
        - zs (zero-shot): Simple keyword generation
        - fs (few-shot): Uses 4 examples for keyword generation [default]
    
    Formula: (query × 5) + keywords
    """
    VERSION = "1.0"
    CONCATENATION_STRATEGY = "query_repeat_plus_generated"
    DEFAULT_QUERY_REPEATS = 5

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        # Get mode parameter
        mode = str(self.cfg.params.get("mode", "zs"))  # Default to zero-shot
        temperature = float(self.cfg.llm.get("temperature", 0.3))
        max_tokens = int(self.cfg.llm.get("max_tokens", 256))
        
        # Select prompt based on mode
        if mode in ["fs", "fewshot"]:
            prompt_id = "q2e.fs.v1"
        elif mode in ["zs", "zeroshot"]:
            prompt_id = "q2e.zs.v1"
        else:
            raise ValueError(f"Invalid mode '{mode}' for Query2E. Must be 'zs' (zero-shot) or 'fs' (few-shot).")
        
        # Generate keywords
        msgs = self.prompts.render(prompt_id, query=q.text)
        out = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
        terms = [t.strip() for t in out.replace("\n", " ").split(",") if t.strip()]
        
        # Limit to max 20 keywords
        max_keywords = int(self.cfg.params.get("max_keywords", 20))
        if len(terms) > max_keywords:
            terms = terms[:max_keywords]
        
        generated_content = " ".join(terms)
        
        # Concatenate: query + keywords
        reformulated = self.concatenate_result(q.text, generated_content)
        
        return ReformulationResult(
            q.qid, 
            q.text, 
            reformulated,
            metadata={
                "mode": mode,
                "prompt_id": prompt_id,
                "keywords": terms,
                "temperature": temperature
            }
        )
