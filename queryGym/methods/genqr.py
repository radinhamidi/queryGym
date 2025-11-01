from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method
from typing import List

@register_method("genqr")
class GENQR(BaseReformulator):
    """
    GenQR: Generate keywords 5 times and concatenate.
    
    Pipeline:
        1. Call LLM 5 times to generate keywords
        2. Concatenate: query + keywords1 + keywords2 + keywords3 + keywords4 + keywords5
    """
    VERSION = "1.0"

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        # Get parameters
        n_generations = int(self.cfg.params.get("n_generations", 5))
        temperature = float(self.cfg.llm.get("temperature", 0.8))
        max_tokens = int(self.cfg.llm.get("max_tokens", 256))
        
        # Generate keywords N times (5 by default)
        all_keywords_lists: List[List[str]] = []
        
        for _ in range(n_generations):
            msgs = self.prompts.render("genqr.keywords.v1", query=q.text)
            out = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            terms = [t.strip() for t in out.replace("\n", " ").split(",") if t.strip()]
            all_keywords_lists.append(terms)
        
        # Concatenate: query + keywords1 + keywords2 + ... + keywordsN
        parts = [q.text]
        for keywords in all_keywords_lists:
            parts.append(" ".join(keywords))
        
        reformulated = " ".join(parts)
        
        # Flatten all keywords for metadata
        all_keywords = [kw for keywords_list in all_keywords_lists for kw in keywords_list]
        
        return ReformulationResult(
            q.qid, 
            q.text, 
            reformulated,
            metadata={
                "n_generations": n_generations,
                "all_keywords_lists": all_keywords_lists,
                "total_keywords": len(all_keywords),
                "keywords": all_keywords
            }
        )
