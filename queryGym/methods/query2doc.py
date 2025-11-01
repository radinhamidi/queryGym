from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import random
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

@register_method("query2doc")
class Query2Doc(BaseReformulator):
    """
    Query2Doc: Generate pseudo-documents from queries.
    
    Modes:
        - "zs" (zero-shot): Generate passage directly from query
        - "cot" (chain-of-thought): Zero-shot with reasoning
        - "fs" (few-shot): Generate using MS MARCO training examples
    
    Few-Shot Parameters:
        - collection_path: Path to MS MARCO collection.tsv
        - train_queries_path: Path to MS MARCO train queries
        - train_qrels_path: Path to MS MARCO train qrels
        - num_examples: Number of few-shot examples (default: 4)
    """
    VERSION = "1.0"
    CONCATENATION_STRATEGY = "query_repeat_plus_generated"
    
    def __init__(self, cfg, llm_client, prompt_resolver):
        super().__init__(cfg, llm_client, prompt_resolver)
        self._msmarco_data = None  # Lazy loading
    
    def _load_msmarco_data(self):
        """Lazy load MS MARCO data for few-shot mode."""
        if self._msmarco_data is not None:
            return self._msmarco_data
        
        try:
            from ..loaders import msmarco
            
            collection_path = self.cfg.params.get("collection_path")
            train_queries_path = self.cfg.params.get("train_queries_path")
            train_qrels_path = self.cfg.params.get("train_qrels_path")
            
            if not all([collection_path, train_queries_path, train_qrels_path]):
                raise ValueError(
                    "Few-shot mode requires: collection_path, train_queries_path, train_qrels_path"
                )
            
            # Load MS MARCO data using loaders
            collection = msmarco.load_collection(collection_path)
            train_queries_list = msmarco.load_queries(train_queries_path)
            train_qrels = msmarco.load_qrels(train_qrels_path)
            
            # Convert queries list to dict for faster lookup
            train_queries_dict = {q.qid: q.text for q in train_queries_list}
            
            self._msmarco_data = {
                "collection": collection,
                "train_queries": train_queries_dict,
                "train_qrels": train_qrels
            }
            
            return self._msmarco_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MS MARCO data for few-shot mode: {e}")
    
    def _select_few_shot_examples(self, num_examples: int = 4) -> List[Tuple[str, str]]:
        """
        Select few-shot examples by randomly sampling relevant query-passage pairs.
        
        Args:
            num_examples: Number of examples to select
            
        Returns:
            List of (query_text, passage_text) tuples
            
        Raises:
            RuntimeError: If insufficient examples found
        """
        try:
            data = self._load_msmarco_data()
            collection = data["collection"]
            train_queries = data["train_queries"]
            train_qrels = data["train_qrels"]
            
            # Collect all relevant query-doc pairs
            relevant_pairs = []
            for qid, doc_rels in train_qrels.items():
                for docid, relevance in doc_rels.items():
                    if relevance > 0:  # Only relevant documents
                        relevant_pairs.append((qid, docid))
            
            if len(relevant_pairs) == 0:
                raise RuntimeError("No relevant query-document pairs found in training data")
            
            # Randomly sample pairs and fetch texts
            sample_size = min(num_examples * 10, len(relevant_pairs))
            sampled_pairs = random.sample(relevant_pairs, sample_size)
            
            examples = []
            for qid, docid in sampled_pairs:
                if len(examples) >= num_examples:
                    break
                
                # Get query text
                query_text = train_queries.get(qid)
                if not query_text:
                    continue
                
                # Get document text
                doc_text = collection.get(docid)
                if not doc_text:
                    continue
                
                examples.append((query_text, doc_text))
            
            if len(examples) == 0:
                raise RuntimeError(
                    f"Could not find any valid examples from {sample_size} sampled pairs. "
                    "Check if collection and queries have matching IDs."
                )
            
            if len(examples) < num_examples:
                # Warning but continue with what we have
                print(f"Warning: Only found {len(examples)} examples (requested {num_examples})")
            
            return examples
            
        except Exception as e:
            raise RuntimeError(f"Failed to select few-shot examples: {e}")
    
    def _format_examples(self, examples: List[Tuple[str, str]]) -> str:
        """
        Format few-shot examples for prompt.
        
        Args:
            examples: List of (query_text, passage_text) tuples
            
        Returns:
            Formatted examples string
        """
        examples_text = ""
        for query, passage in examples:
            examples_text += f"Query: {query}\nPassage: {passage}\n\n"
        
        return examples_text

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        # Get mode and parameters
        mode = str(self.cfg.params.get("mode", "zs"))
        temperature = float(self.cfg.llm.get("temperature", 0.7))
        max_tokens = int(self.cfg.llm.get("max_tokens", 256))
        
        metadata = {"mode": mode}
        
        try:
            if mode == "fs" or mode == "fewshot":
                # Few-shot mode: Use MS MARCO examples
                num_examples = int(self.cfg.params.get("num_examples", 4))
                
                # Select examples
                examples = self._select_few_shot_examples(num_examples)
                metadata["num_examples"] = len(examples)
                
                # Format examples for prompt
                examples_text = self._format_examples(examples)
                
                # Use prompt bank with examples variable
                msgs = self.prompts.render("query2doc.fewshot.v1", query=q.text, examples=examples_text)
                metadata["prompt_id"] = "query2doc.fewshot.v1"
                
            else:
                # Zero-shot or CoT mode: Use prompt bank
                if mode == "cot":
                    prompt_id = "query2doc.cot.v1"
                else:  # Default to zero-shot
                    prompt_id = "query2doc.zeroshot.v1"
                
                msgs = self.prompts.render(prompt_id, query=q.text)
                metadata["prompt_id"] = prompt_id
            
            # Generate pseudo-document
            pseudo_doc = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            
            if not pseudo_doc or not pseudo_doc.strip():
                raise RuntimeError("LLM returned empty pseudo-document")
            
            # Use base class concatenation
            reformulated = self.concatenate_result(q.text, pseudo_doc)
            
            # Store metadata
            metadata["pseudo_doc"] = pseudo_doc
            metadata["temperature"] = temperature
            metadata["max_tokens"] = max_tokens
            
            return ReformulationResult(q.qid, q.text, reformulated, metadata=metadata)
            
        except Exception as e:
            # Handle errors gracefully - return original query with error metadata
            error_msg = f"Query2Doc reformulation failed: {e}"
            print(f"Error for query {q.qid}: {error_msg}")
            
            return ReformulationResult(
                q.qid,
                q.text,
                q.text,  # Fallback to original query
                metadata={
                    "mode": mode,
                    "error": error_msg,
                    "fallback": True
                }
            )
