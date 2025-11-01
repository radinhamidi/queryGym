from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method
from typing import List, Tuple
import random
import os

@register_method("query2doc")
class Query2Doc(BaseReformulator):
    """
    Query2Doc: Generate pseudo-documents from LLM knowledge.
    
    Modes:
        - "zs" (zero-shot): Generate passage directly [default]
        - "cot" (chain-of-thought): Zero-shot with reasoning
        - "fs" (few-shot): Uses MS MARCO training examples (dynamic)
    
    Few-Shot Config (via params or env vars):
        - msmarco_collection / MSMARCO_COLLECTION
        - msmarco_train_queries / MSMARCO_TRAIN_QUERIES
        - msmarco_train_qrels / MSMARCO_TRAIN_QRELS
        - num_examples: Number of few-shot examples (default: 4)
    """
    VERSION = "1.0"
    CONCATENATION_STRATEGY = "query_repeat_plus_generated"
    
    def __init__(self, cfg, llm_client, prompt_resolver):
        super().__init__(cfg, llm_client, prompt_resolver)
        self._msmarco_data = None 
    
    def _load_msmarco_data(self):
        """Lazy load MS MARCO data for few-shot mode using existing loaders."""
        if self._msmarco_data is not None:
            return self._msmarco_data
        
        try:
            from ..loaders import msmarco
            
            # Get paths from config params or environment variables
            collection_path = self.cfg.params.get("msmarco_collection", os.getenv("MSMARCO_COLLECTION"))
            train_queries_path = self.cfg.params.get("msmarco_train_queries", os.getenv("MSMARCO_TRAIN_QUERIES"))
            train_qrels_path = self.cfg.params.get("msmarco_train_qrels", os.getenv("MSMARCO_TRAIN_QRELS"))
            
            if not all([collection_path, train_queries_path, train_qrels_path]):
                raise RuntimeError(
                    "Few-shot mode requires MS MARCO paths (via config params or env vars):\n"
                    "  - msmarco_collection / MSMARCO_COLLECTION\n"
                    "  - msmarco_train_queries / MSMARCO_TRAIN_QUERIES\n"
                    "  - msmarco_train_qrels / MSMARCO_TRAIN_QRELS"
                )
            
            # Load using msmarco loaders
            collection = msmarco.load_collection(collection_path)
            train_queries_list = msmarco.load_queries(train_queries_path)
            train_qrels = msmarco.load_qrels(train_qrels_path)
            
            # Convert queries to dict
            train_queries_dict = {q.qid: q.text for q in train_queries_list}
            
            self._msmarco_data = {
                "collection": collection,
                "train_queries": train_queries_dict,
                "train_qrels": train_qrels
            }
            
            return self._msmarco_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MS MARCO data: {e}")
    
    def _select_few_shot_examples(self, num_examples: int = 4) -> List[Tuple[str, str]]:
        """Randomly sample relevant query-passage pairs from MS MARCO training data."""
        try:
            data = self._load_msmarco_data()
            collection = data["collection"]
            train_queries = data["train_queries"]
            train_qrels = data["train_qrels"]
            
            # Collect all relevant query-doc pairs
            relevant_pairs = []
            for qid, doc_rels in train_qrels.items():
                for docid, relevance in doc_rels.items():
                    if relevance > 0:  # Only relevant
                        relevant_pairs.append((qid, docid))
            
            if not relevant_pairs:
                raise RuntimeError("No relevant query-document pairs found")
            
            # Sample pairs and fetch texts
            sample_size = min(num_examples * 10, len(relevant_pairs))
            sampled_pairs = random.sample(relevant_pairs, sample_size)
            
            examples = []
            for qid, docid in sampled_pairs:
                if len(examples) >= num_examples:
                    break
                
                query_text = train_queries.get(qid)
                doc_text = collection.get(docid)
                
                if query_text and doc_text:
                    examples.append((query_text, doc_text))
            
            if not examples:
                raise RuntimeError("Could not find valid examples with matching IDs")
            
            if len(examples) < num_examples:
                print(f"Warning: Only found {len(examples)}/{num_examples} examples")
            
            return examples
            
        except Exception as e:
            raise RuntimeError(f"Failed to select few-shot examples: {e}")
    
    def _format_examples(self, examples: List[Tuple[str, str]]) -> str:
        """Format examples for prompt template."""
        examples_text = ""
        for query, passage in examples:
            examples_text += f"Query: {query}\nPassage: {passage}\n\n"
        return examples_text

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        """Generate pseudo-document for query using LLM."""
        # Get parameters
        mode = str(self.cfg.params.get("mode", "zs"))
        temperature = float(self.cfg.llm.get("temperature", 0.7))
        max_tokens = int(self.cfg.llm.get("max_tokens", 256))
        
        metadata = {"mode": mode}
        
        try:
            # Select prompt based on mode
            if mode == "fs" or mode == "fewshot":
                # Few-shot: dynamic MS MARCO examples
                num_examples = int(self.cfg.params.get("num_examples", 4))
                examples = self._select_few_shot_examples(num_examples)
                examples_text = self._format_examples(examples)
                
                msgs = self.prompts.render("query2doc.fewshot.v1", query=q.text, examples=examples_text)
                metadata["prompt_id"] = "query2doc.fewshot.v1"
                metadata["num_examples"] = len(examples)
            else:
                # Zero-shot or CoT
                prompt_id = "query2doc.cot.v1" if mode == "cot" else "query2doc.zeroshot.v1"
                msgs = self.prompts.render(prompt_id, query=q.text)
                metadata["prompt_id"] = prompt_id
            
            # Generate pseudo-document using LLM
            pseudo_doc = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            
            if not pseudo_doc or not pseudo_doc.strip():
                raise RuntimeError("LLM returned empty response")
            
            # Use base class concatenation (query_repeat_plus_generated)
            reformulated = self.concatenate_result(q.text, pseudo_doc)
            
            metadata.update({
                "pseudo_doc": pseudo_doc,
                "temperature": temperature,
                "max_tokens": max_tokens
            })
            
            return ReformulationResult(q.qid, q.text, reformulated, metadata=metadata)
            
        except Exception as e:
            # Graceful error handling - fallback to original query
            error_msg = f"Query2Doc failed: {e}"
            print(f"Error for qid={q.qid}: {error_msg}")
            
            return ReformulationResult(
                q.qid, q.text, q.text,
                metadata={"mode": mode, "error": error_msg, "fallback": True}
            )
