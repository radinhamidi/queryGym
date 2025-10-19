#!/usr/bin/env python3
"""
Complete workflow example: Pyserini retrieval → queryGym reformulation → Re-retrieval

This demonstrates the typical use case where a researcher:
1. Performs initial retrieval with Pyserini
2. Uses queryGym to reformulate queries
3. Re-retrieves with reformulated queries
4. Compares results

This is the main use case you mentioned: "user uses pyserini to retrieve and then 
uses querygym to make query better and retrieve again"
"""

import os
from typing import List, Dict, Tuple
import queryGym as qg


def example_without_pyserini():
    """
    Example workflow WITHOUT Pyserini (for testing the API).
    Shows how clean the new import structure is.
    """
    print("=" * 70)
    print("Example: queryGym API (without Pyserini)")
    print("=" * 70)
    
    # Step 1: Create queries
    print("\n📝 Step 1: Create queries")
    queries = [
        qg.QueryItem("q1", "what causes diabetes"),
        qg.QueryItem("q2", "python programming best practices"),
        qg.QueryItem("q3", "climate change effects on agriculture"),
    ]
    print(f"Created {len(queries)} queries")
    
    # Step 2: Create reformulator (simple one-liner!)
    print("\n🔧 Step 2: Create reformulator")
    reformulator = qg.create_reformulator(
        method_name="genqr_ensemble",
        model="gpt-4",
        params={"repeat_query_weight": 3},
        llm_config={"temperature": 0.8, "max_tokens": 256}
    )
    print(f"Created: {type(reformulator).__name__}")
    
    # Step 3: Reformulate queries
    print("\n🔄 Step 3: Reformulate queries")
    print("Note: This would call the LLM API (requires OPENAI_API_KEY)")
    
    if "OPENAI_API_KEY" in os.environ:
        results = reformulator.reformulate_batch(queries)
        
        for result in results:
            print(f"\n{result.qid}:")
            print(f"  Original:     {result.original}")
            print(f"  Reformulated: {result.reformulated[:80]}...")
    else:
        print("⚠️  OPENAI_API_KEY not set, skipping LLM calls")
        print("Set with: export OPENAI_API_KEY='sk-...'")
    
    # Step 4: Save reformulated queries
    print("\n💾 Step 4: Save reformulated queries")
    print("You can save to TSV and use with Pyserini:")
    print("  queryGym run --method genqr_ensemble \\")
    print("    --queries-tsv queries.tsv \\")
    print("    --output-tsv reformulated.tsv")


def example_with_pyserini():
    """
    Complete example WITH Pyserini integration.
    This is the real-world workflow.
    """
    print("\n\n" + "=" * 70)
    print("Example: Complete Workflow with Pyserini")
    print("=" * 70)
    
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError:
        print("\n⚠️  Pyserini not installed.")
        print("Install with: pip install pyserini")
        print("\nShowing workflow structure only...")
        show_workflow_structure()
        return
    
    # Configuration
    INDEX = "msmarco-v1-passage"
    K = 10
    
    # Step 1: Create queries
    print("\n📝 Step 1: Create queries")
    queries = [
        qg.QueryItem("q1", "what causes diabetes"),
        qg.QueryItem("q2", "python programming best practices"),
    ]
    print(f"Created {len(queries)} queries")
    
    # Step 2: Initial retrieval
    print(f"\n🔍 Step 2: Initial retrieval (top-{K})")
    try:
        searcher = LuceneSearcher.from_prebuilt_index(INDEX)
        
        initial_results = {}
        for query in queries:
            hits = searcher.search(query.text, k=K)
            initial_results[query.qid] = hits
            print(f"  {query.qid}: top score = {hits[0].score:.4f}")
    except Exception as e:
        print(f"⚠️  Retrieval failed: {e}")
        return
    
    # Step 3: Query reformulation
    print("\n🔄 Step 3: Reformulate queries with queryGym")
    
    if "OPENAI_API_KEY" not in os.environ:
        print("⚠️  OPENAI_API_KEY not set, cannot reformulate")
        return
    
    reformulator = qg.create_reformulator("genqr_ensemble", model="gpt-4")
    reformulated = reformulator.reformulate_batch(queries)
    
    for result in reformulated:
        print(f"\n  {result.qid}:")
        print(f"    Original:     {result.original}")
        print(f"    Reformulated: {result.reformulated[:60]}...")
    
    # Step 4: Re-retrieval with reformulated queries
    print(f"\n🔍 Step 4: Re-retrieval with reformulated queries (top-{K})")
    
    reformulated_results = {}
    for result in reformulated:
        hits = searcher.search(result.reformulated, k=K)
        reformulated_results[result.qid] = hits
        print(f"  {result.qid}: top score = {hits[0].score:.4f}")
    
    # Step 5: Compare results
    print("\n📊 Step 5: Compare results")
    
    for qid in initial_results.keys():
        orig_score = initial_results[qid][0].score
        reform_score = reformulated_results[qid][0].score
        improvement = reform_score - orig_score
        
        print(f"\n  {qid}:")
        print(f"    Original score:     {orig_score:.4f}")
        print(f"    Reformulated score: {reform_score:.4f}")
        print(f"    Improvement:        {improvement:+.4f}")
        
        # Document overlap
        orig_docs = {hit.docid for hit in initial_results[qid]}
        reform_docs = {hit.docid for hit in reformulated_results[qid]}
        overlap = len(orig_docs & reform_docs)
        print(f"    Top-{K} overlap:    {overlap}/{K} documents")


def show_workflow_structure():
    """Show the workflow structure as code."""
    print("""
# Complete Workflow Structure:

import queryGym as qg
from pyserini.search.lucene import LuceneSearcher

# 1. Setup
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
queries = [qg.QueryItem("q1", "what causes diabetes")]

# 2. Initial retrieval
initial_hits = searcher.search(queries[0].text, k=100)
print(f"Initial top score: {initial_hits[0].score}")

# 3. Reformulate with queryGym (one line!)
reformulator = qg.create_reformulator("genqr_ensemble", model="gpt-4")
result = reformulator.reformulate(queries[0])

# 4. Re-retrieve
reformulated_hits = searcher.search(result.reformulated, k=100)
print(f"Reformulated top score: {reformulated_hits[0].score}")

# 5. Compare
improvement = reformulated_hits[0].score - initial_hits[0].score
print(f"Score improvement: {improvement:+.4f}")
    """)


def show_api_comparison():
    """Show old vs new API."""
    print("\n\n" + "=" * 70)
    print("API Comparison: Old vs New")
    print("=" * 70)
    
    print("\n❌ OLD WAY (not user-friendly):")
    print("""
from queryGym.methods.genqr_ensemble import GenQREnsemble
from queryGym.data.dataloader import UnifiedQuerySource
from queryGym.core.base import QueryItem, MethodConfig
from queryGym.core.llm import OpenAICompatibleClient
from queryGym.core.prompts import PromptBank

# Complex setup...
config = MethodConfig(
    name="genqr_ensemble",
    params={"repeat_query_weight": 3},
    llm={"model": "gpt-4", "temperature": 0.8},
    seed=42, retries=2
)
llm = OpenAICompatibleClient(model="gpt-4")
pb = PromptBank("queryGym/prompt_bank.yaml")
reformulator = GenQREnsemble(config, llm, pb)

# Load queries
src = UnifiedQuerySource(backend="local", format="tsv", path="queries.tsv")
queries = list(src.iter())

# Reformulate
results = reformulator.reformulate_batch(queries)
    """)
    
    print("\n✅ NEW WAY (user-friendly):")
    print("""
import queryGym as qg

# Simple one-liners!
reformulator = qg.create_reformulator("genqr_ensemble", model="gpt-4")
queries = qg.load_queries("local", path="queries.tsv", format="tsv")
results = reformulator.reformulate_batch(queries)
    """)
    
    print("\n🎯 Much cleaner and more intuitive!")


def main():
    """Run all examples."""
    
    # Example 1: API without Pyserini
    example_without_pyserini()
    
    # Example 2: Complete workflow with Pyserini
    example_with_pyserini()
    
    # Example 3: API comparison
    show_api_comparison()
    
    print("\n\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
The new queryGym API makes it easy to:

1. Import everything from one place:
   import queryGym as qg

2. Create reformulators with one line:
   reformulator = qg.create_reformulator("genqr_ensemble", model="gpt-4")

3. Load queries easily:
   queries = qg.load_queries("local", path="queries.tsv", format="tsv")

4. Integrate with Pyserini seamlessly:
   - Initial retrieval → Reformulate → Re-retrieve
   - All with clean, intuitive code

5. Access all classes at top level:
   qg.QueryItem, qg.GENQR, qg.PromptBank, etc.

This is perfect for the use case you described:
"user uses pyserini to retrieve and then uses querygym to make query 
better and retrieve again"
    """)


if __name__ == "__main__":
    main()
