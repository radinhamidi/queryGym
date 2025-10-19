#!/usr/bin/env python3
"""
Simple usage example demonstrating the user-friendly queryGym API.

This shows how to use queryGym in a typical workflow:
1. Initial retrieval with Pyserini
2. Query reformulation with queryGym
3. Re-retrieval with reformulated queries
"""

import os
import queryGym as qg

# Set up API key (or use environment variable OPENAI_API_KEY)
# os.environ["OPENAI_API_KEY"] = "sk-..."


def main():
    print("=" * 60)
    print("queryGym Simple Usage Example")
    print("=" * 60)
    
    # ========================================
    # Example 1: Basic Usage
    # ========================================
    print("\n📝 Example 1: Basic Query Reformulation")
    print("-" * 60)
    
    # Create a reformulator with simple API
    reformulator = qg.create_reformulator(
        method_name="genqr",
        model="gpt-4",
        params={"repeat_query_weight": 3},
        llm_config={"temperature": 0.8, "max_tokens": 256}
    )
    
    # Create a query
    query = qg.QueryItem(qid="q1", text="what causes diabetes")
    
    # Reformulate
    result = reformulator.reformulate(query)
    
    print(f"Original:     {result.original}")
    print(f"Reformulated: {result.reformulated}")
    print(f"Metadata:     {result.metadata}")
    
    # ========================================
    # Example 2: Batch Processing
    # ========================================
    print("\n\n📦 Example 2: Batch Query Reformulation")
    print("-" * 60)
    
    # Load queries from file
    queries = qg.load_queries(
        source="local",
        path="examples/tiny_queries.tsv",
        format="tsv"
    )
    
    print(f"Loaded {len(queries)} queries")
    
    # Reformulate all queries
    reformulator = qg.create_reformulator("genqr_ensemble", model="gpt-4")
    results = reformulator.reformulate_batch(queries)
    
    for result in results[:3]:  # Show first 3
        print(f"\n{result.qid}:")
        print(f"  Original:     {result.original}")
        print(f"  Reformulated: {result.reformulated[:100]}...")
    
    # ========================================
    # Example 3: Different Methods
    # ========================================
    print("\n\n🔧 Example 3: Trying Different Methods")
    print("-" * 60)
    
    query = qg.QueryItem(qid="q1", text="python programming best practices")
    
    methods = ["genqr", "query2doc", "genqr_ensemble"]
    
    for method_name in methods:
        reformulator = qg.create_reformulator(method_name, model="gpt-4")
        result = reformulator.reformulate(query)
        print(f"\n{method_name}:")
        print(f"  {result.reformulated[:80]}...")
    
    # ========================================
    # Example 4: Integration with Pyserini
    # ========================================
    print("\n\n🔍 Example 4: Typical Pyserini Workflow")
    print("-" * 60)
    
    print("""
# Step 1: Initial retrieval with Pyserini
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
original_query = "what causes diabetes"
hits = searcher.search(original_query, k=10)

print(f"Original query: {original_query}")
print(f"Top result score: {hits[0].score}")

# Step 2: Reformulate query with queryGym
import queryGym as qg

reformulator = qg.create_reformulator("genqr_ensemble", model="gpt-4")
query_item = qg.QueryItem(qid="q1", text=original_query)
result = reformulator.reformulate(query_item)

reformulated_query = result.reformulated
print(f"Reformulated query: {reformulated_query}")

# Step 3: Re-retrieve with reformulated query
hits_reformulated = searcher.search(reformulated_query, k=10)
print(f"Top result score after reformulation: {hits_reformulated[0].score}")

# Compare results
print(f"Score improvement: {hits_reformulated[0].score - hits[0].score}")
    """)
    
    # ========================================
    # Example 5: All Available Methods
    # ========================================
    print("\n\n📚 Example 5: Available Methods")
    print("-" * 60)
    
    print("Available reformulation methods:")
    for method_name in qg.METHODS.keys():
        print(f"  - {method_name}")
    
    # ========================================
    # Example 6: Custom Configuration
    # ========================================
    print("\n\n⚙️  Example 6: Custom Configuration")
    print("-" * 60)
    
    # Using MethodConfig directly for more control
    config = qg.MethodConfig(
        name="query2doc",
        params={"mode": "cot"},  # Chain-of-thought mode
        llm={
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 512,
            # "base_url": "https://api.openai.com/v1",  # Custom endpoint
            # "api_key": "sk-...",  # Custom API key
        },
        seed=42,
        retries=3
    )
    
    # Build components manually
    llm = qg.build_llm(config)
    prompt_bank = qg.PromptBank("queryGym/prompt_bank.yaml")
    
    # Create reformulator
    reformulator = qg.Query2Doc(config, llm, prompt_bank)
    
    query = qg.QueryItem(qid="q1", text="machine learning algorithms")
    result = reformulator.reformulate(query)
    
    print(f"Query: {result.original}")
    print(f"Generated pseudo-document:\n{result.reformulated}")
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Note: This script requires OPENAI_API_KEY to be set
    if "OPENAI_API_KEY" not in os.environ:
        print("⚠️  Warning: OPENAI_API_KEY not set. Examples will fail.")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        print("\nShowing example structure only (not executing LLM calls)...\n")
    
    main()
