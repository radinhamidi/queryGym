#!/usr/bin/env python3
"""
Complete example showing queryGym integration with Pyserini for retrieval.

Workflow:
1. Load queries
2. Initial retrieval with Pyserini
3. Query reformulation with queryGym
4. Re-retrieval with reformulated queries
5. Compare results
"""

import os
from typing import List, Dict, Any
import queryGym as qg


def initial_retrieval(queries: List[qg.QueryItem], index_path: str, k: int = 100) -> Dict[str, Any]:
    """
    Perform initial retrieval with original queries.
    
    Args:
        queries: List of QueryItem objects
        index_path: Path to Pyserini index or prebuilt index name
        k: Number of documents to retrieve
    
    Returns:
        Dictionary mapping qid to list of hits
    """
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError:
        print("⚠️  Pyserini not installed. Install with: pip install pyserini")
        return {}
    
    print(f"🔍 Performing initial retrieval with {len(queries)} queries...")
    
    searcher = LuceneSearcher.from_prebuilt_index(index_path)
    results = {}
    
    for query in queries:
        hits = searcher.search(query.text, k=k)
        results[query.qid] = hits
        print(f"  {query.qid}: Retrieved {len(hits)} documents")
    
    return results


def reformulate_queries(
    queries: List[qg.QueryItem],
    method: str = "genqr_ensemble",
    model: str = "gpt-4"
) -> List[qg.ReformulationResult]:
    """
    Reformulate queries using queryGym.
    
    Args:
        queries: List of QueryItem objects
        method: Reformulation method name
        model: LLM model to use
    
    Returns:
        List of ReformulationResult objects
    """
    print(f"\n🔄 Reformulating queries with {method}...")
    
    reformulator = qg.create_reformulator(
        method_name=method,
        model=model,
        params={"repeat_query_weight": 3},
        llm_config={"temperature": 0.8, "max_tokens": 256}
    )
    
    results = reformulator.reformulate_batch(queries)
    
    for result in results:
        print(f"  {result.qid}:")
        print(f"    Original:     {result.original}")
        print(f"    Reformulated: {result.reformulated[:80]}...")
    
    return results


def re_retrieval(
    reformulated_results: List[qg.ReformulationResult],
    index_path: str,
    k: int = 100
) -> Dict[str, Any]:
    """
    Perform retrieval with reformulated queries.
    
    Args:
        reformulated_results: List of ReformulationResult objects
        index_path: Path to Pyserini index or prebuilt index name
        k: Number of documents to retrieve
    
    Returns:
        Dictionary mapping qid to list of hits
    """
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError:
        print("⚠️  Pyserini not installed.")
        return {}
    
    print(f"\n🔍 Performing re-retrieval with reformulated queries...")
    
    searcher = LuceneSearcher.from_prebuilt_index(index_path)
    results = {}
    
    for result in reformulated_results:
        hits = searcher.search(result.reformulated, k=k)
        results[result.qid] = hits
        print(f"  {result.qid}: Retrieved {len(hits)} documents")
    
    return results


def compare_results(
    original_results: Dict[str, Any],
    reformulated_results: Dict[str, Any],
    qrels: Dict[str, Dict[str, int]] = None
) -> None:
    """
    Compare retrieval results before and after reformulation.
    
    Args:
        original_results: Results from original queries
        reformulated_results: Results from reformulated queries
        qrels: Optional relevance judgments for evaluation
    """
    print("\n📊 Comparing Results:")
    print("-" * 60)
    
    for qid in original_results.keys():
        orig_hits = original_results[qid]
        reform_hits = reformulated_results.get(qid, [])
        
        if not orig_hits or not reform_hits:
            continue
        
        print(f"\n{qid}:")
        print(f"  Original top score:     {orig_hits[0].score:.4f}")
        print(f"  Reformulated top score: {reform_hits[0].score:.4f}")
        print(f"  Score change:           {reform_hits[0].score - orig_hits[0].score:+.4f}")
        
        # Compare document overlap
        orig_docids = {hit.docid for hit in orig_hits[:10]}
        reform_docids = {hit.docid for hit in reform_hits[:10]}
        overlap = len(orig_docids & reform_docids)
        print(f"  Top-10 overlap:         {overlap}/10 documents")
        
        # If qrels provided, compute metrics
        if qrels and qid in qrels:
            orig_relevant = sum(1 for hit in orig_hits[:10] if hit.docid in qrels[qid])
            reform_relevant = sum(1 for hit in reform_hits[:10] if hit.docid in qrels[qid])
            print(f"  Original P@10:          {orig_relevant}/10")
            print(f"  Reformulated P@10:      {reform_relevant}/10")


def save_reformulated_queries(
    results: List[qg.ReformulationResult],
    output_path: str
) -> None:
    """
    Save reformulated queries to TSV file for later use.
    
    Args:
        results: List of ReformulationResult objects
        output_path: Path to output TSV file
    """
    import csv
    
    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        for result in results:
            writer.writerow([result.qid, result.reformulated])
    
    print(f"\n💾 Saved reformulated queries to: {output_path}")


def main():
    """Main workflow demonstrating queryGym + Pyserini integration."""
    
    print("=" * 60)
    print("queryGym + Pyserini Integration Example")
    print("=" * 60)
    
    # Check for API key
    if "OPENAI_API_KEY" not in os.environ:
        print("\n⚠️  Error: OPENAI_API_KEY not set.")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        return
    
    # Configuration
    INDEX_PATH = "msmarco-v1-passage"  # Pyserini prebuilt index
    METHOD = "genqr_ensemble"
    MODEL = "gpt-4"
    K = 100  # Number of documents to retrieve
    
    # Step 1: Load queries
    print("\n📥 Loading queries...")
    queries = [
        qg.QueryItem("q1", "what causes diabetes"),
        qg.QueryItem("q2", "python programming best practices"),
        qg.QueryItem("q3", "climate change effects"),
    ]
    print(f"Loaded {len(queries)} queries")
    
    # Alternative: Load from file
    # queries = qg.load_queries("local", path="queries.tsv", format="tsv")
    
    # Step 2: Initial retrieval
    try:
        original_results = initial_retrieval(queries, INDEX_PATH, k=K)
    except Exception as e:
        print(f"⚠️  Initial retrieval failed: {e}")
        print("Note: This example requires Pyserini and a valid index.")
        original_results = {}
    
    # Step 3: Query reformulation
    try:
        reformulated = reformulate_queries(queries, method=METHOD, model=MODEL)
    except Exception as e:
        print(f"⚠️  Reformulation failed: {e}")
        return
    
    # Step 4: Re-retrieval
    if original_results:
        try:
            reformulated_results = re_retrieval(reformulated, INDEX_PATH, k=K)
        except Exception as e:
            print(f"⚠️  Re-retrieval failed: {e}")
            reformulated_results = {}
        
        # Step 5: Compare results
        if reformulated_results:
            compare_results(original_results, reformulated_results)
    
    # Step 6: Save reformulated queries
    save_reformulated_queries(reformulated, "reformulated_queries.tsv")
    
    print("\n" + "=" * 60)
    print("✅ Workflow completed!")
    print("=" * 60)
    
    print("\n💡 Next steps:")
    print("  1. Use reformulated_queries.tsv with Pyserini:")
    print("     python -m pyserini.search.lucene \\")
    print("       --index msmarco-v1-passage \\")
    print("       --topics reformulated_queries.tsv \\")
    print("       --output run.reformulated.txt")
    print("\n  2. Evaluate with trec_eval:")
    print("     trec_eval -m map -m ndcg_cut.10 qrels.txt run.reformulated.txt")


if __name__ == "__main__":
    main()
