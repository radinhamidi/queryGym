#!/usr/bin/env python3
"""
Context Retrieval Script for queryGym

Retrieves top-k documents for queries to provide context for query reformulation methods.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher
# Add queryGym to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import queryGym as qg



class Retriever:
    
    def __init__(self, args):
        # Initialize searcher based on arguments
        if args.impact:
            if os.path.exists(args.index):
                self.searcher = LuceneImpactSearcher(args.index, args.encoder, args.min_idf)
            else:
                self.searcher = LuceneImpactSearcher.from_prebuilt_index(args.index, args.encoder, args.min_idf)
        else:
            # Default to BM25
            if os.path.exists(args.index):
                self.searcher = LuceneSearcher(args.index)
            else:
                self.searcher = LuceneSearcher.from_prebuilt_index(args.index)
            
            # Set BM25 parameters if specified
            if args.bm25 and not args.disable_bm25_param:
                self.set_bm25_parameters(args.index, args.k1, args.b)
            
            # Add RM3 if requested
            if args.rm3:
                self.searcher.set_rm3()
            
            # Add Rocchio if requested
            if args.rocchio:
                if args.rocchio_use_negative:
                    self.searcher.set_rocchio(gamma=0.15, use_negative=True)
                else:
                    self.searcher.set_rocchio()
        
        self.args = args
    
    def set_bm25_parameters(self, index, k1=None, b=None):
        """Set BM25 parameters based on index type."""
        if k1 is not None and b is not None:
            self.searcher.set_bm25(k1, b)
        else:
            # Auto-set based on known indices
            if 'msmarco-passage' in index:
                self.searcher.set_bm25(0.82, 0.68)
            elif 'msmarco-doc' in index:
                self.searcher.set_bm25(4.46, 0.82)
    
    def retrieve(self, query: str, k: int) -> List[tuple]:
        """Retrieve top-k documents for a query."""
        hits = self.searcher.search(query, k)
        return self._process_hits(hits)
    
    def retrieve_batch(self, queries: List[str], k: int, num_threads: int = 1) -> List[List[tuple]]:
        """Retrieve top-k documents for a batch of queries."""
        if self.args.impact:
            # For Impact search, use batch_search
            pseudo_batch_topic_ids = [str(idx) for idx, _ in enumerate(queries)]
            results = self.searcher.batch_search(
                queries, pseudo_batch_topic_ids, k, num_threads
            )
            # Convert results to list in original order
            batch_results = [results[id_] for id_ in pseudo_batch_topic_ids]
        else:
            # For BM25, use batch_search
            pseudo_batch_topic_ids = [str(idx) for idx, _ in enumerate(queries)]
            results = self.searcher.batch_search(
                queries, pseudo_batch_topic_ids, k, num_threads
            )
            # Convert results to list in original order
            batch_results = [results[id_] for id_ in pseudo_batch_topic_ids]
        
        # Process each result
        processed_results = []
        for hits in batch_results:
            processed_results.append(self._process_hits(hits))
        
        return processed_results
    
    def _process_hits(self, hits) -> List[tuple]:
        """Process hits and extract content."""
        contexts = []
        
        for hit in hits:
            # Extract content from hit
            if hasattr(hit, 'raw') and hit.raw:
                try:
                    doc_data = json.loads(hit.raw)
                    # Look for content fields
                    content = doc_data.get('contents', doc_data.get('text', hit.raw))
                    contexts.append((hit.docid, content))
                except:
                    contexts.append((hit.docid, hit.raw))
            elif hasattr(hit, 'contents'):
                contexts.append((hit.docid, hit.contents))
            else:
                contexts.append((hit.docid, str(hit.docid)))
        
        return contexts


def load_queries(file_path: str) -> List[qg.QueryItem]:
    """Load queries from TSV file."""
    queries = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    qid, query_text = parts[0], parts[1]
                    queries.append(qg.QueryItem(qid, query_text))
    return queries


def save_contexts(contexts: Dict[str, List[tuple]], output_path: str, trec_format: bool = False):
    """Save contexts to JSONL file or TREC run file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if trec_format:
        # Save as TREC run format
        with open(output_path, 'w') as f:
            for qid, ctx_list in contexts.items():
                for rank, (docid, content) in enumerate(ctx_list, 1):
                    # TREC format: qid Q0 docid rank score runname
                    score = 1.0 / rank  # Decreasing scores
                    runname = "querygym_context"
                    f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {runname}\n")
    else:
        # Save as JSONL file
        with open(output_path, 'w') as f:
            for qid, ctx_list in contexts.items():
                # Extract just the content for JSONL format
                content_list = [content for docid, content in ctx_list]
                f.write(json.dumps({
                    "qid": qid,
                    "contexts": content_list,
                    "num_contexts": len(content_list)
                }) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Retrieve context documents for queryGym')
    
    # Required arguments
    parser.add_argument('--queries', type=str, required=True, help='Path to queries TSV file')
    parser.add_argument('--index', type=str, required=True, help='Index path or prebuilt index name')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file path')
    parser.add_argument('--k', type=int, default=10, help='Number of documents to retrieve')
    
    # Retrieval methods
    parser.add_argument('--bm25', action='store_true', default=True, help='Use BM25 (default)')
    parser.add_argument('--impact', action='store_true', help='Use Impact search')
    parser.add_argument('--rm3', action='store_true', help='Use RM3')
    parser.add_argument('--rocchio', action='store_true', help='Use Rocchio')
    parser.add_argument('--rocchio-use-negative', action='store_true', help='Use negative feedback in Rocchio')
    
    # BM25 parameters
    parser.add_argument('--k1', type=float, help='BM25 k1 parameter')
    parser.add_argument('--b', type=float, help='BM25 b parameter')
    parser.add_argument('--disable_bm25_param', action='store_true', default=True, help='Disable auto BM25 params')
    
    # Impact parameters
    parser.add_argument('--encoder', type=str, default=None, help="encoder name")
    parser.add_argument('--min-idf', type=int, default=0, help="minimum idf")
    
    # Output format
    parser.add_argument('--trec-format', action='store_true', help='Output in TREC run format instead of JSONL')
    
    # Batch processing
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for concurrent retrieval')
    parser.add_argument('--threads', type=int, default=16, help='Number of threads for batch retrieval')
    
    args = parser.parse_args()
    
    # Load queries and initialize retriever
    queries = load_queries(args.queries)

    retriever = Retriever(args)
    
    contexts = {}
    
    # Process queries in batches for efficiency
    if args.batch_size <= 1 and args.threads <= 1:
        # Single query processing
        for query in tqdm(queries, desc="Retrieving"):
            contexts[query.qid] = retriever.retrieve(query.text, args.k)
    else:
        # Batch processing
        batch_queries = []
        batch_qids = []
        
        for index, query in enumerate(tqdm(queries, desc="Retrieving")):
            batch_queries.append(query.text)
            batch_qids.append(query.qid)
            
            # Process batch when it reaches batch_size or at the end
            if (index + 1) % args.batch_size == 0 or index == len(queries) - 1:
                # Retrieve for the current batch
                batch_results = retriever.retrieve_batch(batch_queries, args.k, args.threads)
                
                # Store results
                for qid, result in zip(batch_qids, batch_results):
                    contexts[qid] = result
                
                # Clear batch for next iteration
                batch_queries.clear()
                batch_qids.clear()
    
    # Save contexts
    save_contexts(contexts, args.output, args.trec_format)
    
if __name__ == "__main__":
    main()
