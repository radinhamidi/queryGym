#!/usr/bin/env python3
"""
Context Retrieval Module for queryGym

Retrieves top-k documents for queries to provide context for query reformulation methods.
"""

import argparse
import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher


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
        # Set answer key for content extraction (default to "contents" like csqe.py)
        self.answer_key = getattr(args, 'answer_key', 'contents')
    
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
        """Process hits and extract content following csqe.py pattern."""
        contexts = []
        
        for hit in hits:
            # Extract content using hit.lucene_document.get('raw') as suggested
            raw_content = hit.lucene_document.get('raw')
            raw_df = json.loads(raw_content)
            text_list = [raw_df[k] for k in self.answer_key.split("|") if raw_df[k]]
            content = "\t".join(text_list)
            contexts.append((hit.docid, content))
        
        return contexts


def main():
    """Main function for standalone usage."""
    parser = argparse.ArgumentParser(description="Retrieve contexts for queries")
    parser.add_argument("--index", required=True, help="Path to Lucene index")
    parser.add_argument("--queries-tsv", required=True, help="Path to queries TSV file")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--k", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--bm25", action="store_true", help="Use BM25")
    parser.add_argument("--impact", action="store_true", help="Use Impact search")
    parser.add_argument("--rm3", action="store_true", help="Use RM3")
    parser.add_argument("--rocchio", action="store_true", help="Use Rocchio")
    parser.add_argument("--rocchio-use-negative", action="store_true", help="Use negative feedback in Rocchio")
    parser.add_argument("--disable-bm25-param", action="store_true", help="Disable BM25 parameter tuning")
    parser.add_argument("--k1", type=float, help="BM25 k1 parameter")
    parser.add_argument("--b", type=float, help="BM25 b parameter")
    parser.add_argument("--encoder", help="Encoder for Impact search")
    parser.add_argument("--min-idf", type=float, default=0, help="Minimum IDF for Impact search")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for processing")
    parser.add_argument("--threads", type=int, default=16, help="Number of threads")
    parser.add_argument("--trec-format", action="store_true", help="Output in TREC run format")
    parser.add_argument("--answer-key", type=str, default="contents", help="Field name(s) to extract content from (pipe-separated)")
    
    args = parser.parse_args()
    
    # Create retriever
    retriever = Retriever(args)
    
    # Load queries
    queries = []
    with open(args.queries_tsv, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    qid, query = parts[0], parts[1]
                    queries.append((qid, query))
    
    # Perform retrieval
    if args.batch_size > 1 or args.threads > 1:
        # Batch retrieval
        query_texts = [q[1] for q in queries]
        query_ids = [q[0] for q in queries]
        
        batch_results = retriever.retrieve_batch(query_texts, args.k, args.threads)
        
        # Save results
        with open(args.output, 'w') as f:
            for qid, results in zip(query_ids, batch_results):
                for rank, (docid, content) in enumerate(results, 1):
                    if args.trec_format:
                        f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{1.0/rank:.6f}\tretrieval\n")
                    else:
                        f.write(f"{qid}\t{docid}\t{content}\n")
    else:
        # Single query retrieval
        with open(args.output, 'w') as f:
            for qid, query in tqdm(queries, desc="Retrieving contexts"):
                results = retriever.retrieve(query, args.k)
                for rank, (docid, content) in enumerate(results, 1):
                    if args.trec_format:
                        f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{1.0/rank:.6f}\tretrieval\n")
                    else:
                        f.write(f"{qid}\t{docid}\t{content}\n")


if __name__ == "__main__":
    main()