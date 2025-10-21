#!/usr/bin/env python3
"""
Example: Using queryGym with Custom Searchers

This example demonstrates how to use queryGym with different searcher libraries
(Pyserini, PyTerrier) through the new searcher interface.
"""

import os
import queryGym as qg
from queryGym.core.searcher import BaseSearcher, SearchHit, SearcherRegistry


def example_with_pyserini():
    """Example using Pyserini searcher."""
    print("=" * 60)
    print("Example: Using Pyserini Searcher")
    print("=" * 60)
    
    # Method 1: Using searcher_type parameter
    print("\n📝 Method 1: Using searcher_type parameter")
    
    # Create reformulator with Pyserini searcher
    reformulator = qg.create_reformulator(
        method_name="lamer",
        model="gpt-4",
        params={
            "searcher_type": "pyserini",
            "searcher_kwargs": {
                "index": "msmarco-v1-passage",
                "searcher_type": "bm25",
                "k1": 0.9,
                "b": 0.4,
                "rm3": False,
                "rocchio": False,
            },
            "retrieval_k": 10,
            "gen_passages": 5,
        }
    )
    
    # Create a query
    query = qg.QueryItem(qid="q1", text="what causes diabetes")
    
    print(f"Query: {query.text}")
    print("Reformulating with LameR + Pyserini...")
    
    # This will automatically retrieve contexts using Pyserini
    result = reformulator.reformulate(query)
    
    print(f"Reformulated: {result.reformulated[:100]}...")
    print(f"Generated passages: {result.metadata.get('generated_passages_count', 0)}")
    
    # Method 2: Using custom searcher instance
    print("\n📝 Method 2: Using custom searcher instance")
    
    # Create a custom Pyserini searcher
    custom_searcher = qg.create_searcher(
        "pyserini",
        index="msmarco-v1-passage",
        searcher_type="bm25",
        k1=0.9,
        b=0.4,
        rm3=True  # Enable RM3
    )
    
    # Create retriever with custom searcher
    retriever = qg.Retriever(searcher=custom_searcher)
    
    # Use retriever directly
    results = retriever.retrieve("machine learning algorithms", k=5)
    
    print(f"Retrieved {len(results)} documents:")
    for i, (docid, content) in enumerate(results[:2], 1):
        print(f"  {i}. {docid}: {content[:50]}...")


def example_with_pyterrier():
    """Example using PyTerrier searcher."""
    print("\n\n" + "=" * 60)
    print("Example: Using PyTerrier Searcher")
    print("=" * 60)
    
    try:
        # Method 1: Using searcher_type parameter
        print("\n📝 Method 1: Using searcher_type parameter")
        
        # Create reformulator with PyTerrier searcher
        reformulator = qg.create_reformulator(
            method_name="lamer",
            model="gpt-4",
            params={
                "searcher_type": "pyterrier",
                "searcher_kwargs": {
                    "index_name": "msmarco_passage",  # Prebuilt PyTerrier index
                    "searcher_type": "bm25",
                },
                "retrieval_k": 10,
                "gen_passages": 5,
            }
        )
        
        # Create a query
        query = qg.QueryItem(qid="q1", text="climate change effects")
        
        print(f"Query: {query.text}")
        print("Reformulating with LameR + PyTerrier...")
        
        # This will automatically retrieve contexts using PyTerrier
        result = reformulator.reformulate(query)
        
        print(f"Reformulated: {result.reformulated[:100]}...")
        print(f"Generated passages: {result.metadata.get('generated_passages_count', 0)}")
        
        # Method 2: Using custom searcher instance
        print("\n📝 Method 2: Using custom searcher instance")
        
        # Create a custom PyTerrier searcher
        custom_searcher = qg.create_searcher(
            "pyterrier",
            index_name="msmarco_passage",
            searcher_type="bm25"
        )
        
        # Create retriever with custom searcher
        retriever = qg.Retriever(searcher=custom_searcher)
        
        # Use retriever directly
        results = retriever.retrieve("artificial intelligence", k=5)
        
        print(f"Retrieved {len(results)} documents:")
        for i, (docid, content) in enumerate(results[:2], 1):
            print(f"  {i}. {docid}: {content[:50]}...")
            
    except ImportError:
        print("⚠️  PyTerrier not installed. Install with: pip install python-terrier")
    except Exception as e:
        print(f"⚠️  PyTerrier example failed: {e}")


def example_custom_searcher():
    """Example of creating a custom searcher implementation."""
    print("\n\n" + "=" * 60)
    print("Example: Custom Searcher Implementation")
    print("=" * 60)
    
    class MockSearcher(BaseSearcher):
        """A mock searcher for demonstration purposes."""
        
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._searcher_info = {
                "name": "MockSearcher",
                "type": "mock",
                **kwargs
            }
        
        def search(self, query: str, k: int = 10, **kwargs) -> List[SearchHit]:
            """Mock search that returns fake results."""
            hits = []
            for i in range(min(k, 3)):  # Return max 3 fake results
                hit = SearchHit(
                    docid=f"mock_doc_{i+1}",
                    score=1.0 - (i * 0.1),
                    content=f"This is mock content for query '{query}' - result {i+1}",
                    metadata={"mock": True, "query": query}
                )
                hits.append(hit)
            return hits
        
        def batch_search(self, queries: List[str], k: int = 10, 
                        num_threads: int = 1, **kwargs) -> List[List[SearchHit]]:
            """Mock batch search."""
            return [self.search(query, k, **kwargs) for query in queries]
        
        def get_searcher_info(self) -> Dict[str, Any]:
            return self._searcher_info.copy()
    
    # Register the custom searcher
    SearcherRegistry.register("mock", MockSearcher)
    
    print("\n📝 Using custom MockSearcher")
    
    # Create retriever with custom searcher
    retriever = qg.Retriever(searcher_type="mock", searcher_kwargs={"custom_param": "value"})
    
    # Use retriever
    results = retriever.retrieve("test query", k=3)
    
    print(f"Retrieved {len(results)} documents:")
    for i, (docid, content) in enumerate(results, 1):
        print(f"  {i}. {docid}: {content}")
    
    # Get searcher info
    info = retriever.get_searcher_info()
    print(f"\nSearcher info: {info}")


def example_comparison():
    """Example comparing different searchers."""
    print("\n\n" + "=" * 60)
    print("Example: Comparing Different Searchers")
    print("=" * 60)
    
    query = "machine learning applications"
    k = 5
    
    print(f"Query: {query}")
    print(f"Retrieving top-{k} documents with different searchers...\n")
    
    # Compare Pyserini and PyTerrier (if available)
    searchers_to_test = ["pyserini"]
    
    try:
        import pyterrier as pt
        searchers_to_test.append("pyterrier")
    except ImportError:
        print("PyTerrier not available for comparison")
    
    for searcher_type in searchers_to_test:
        print(f"🔍 Testing {searcher_type}:")
        
        try:
            if searcher_type == "pyserini":
                searcher_kwargs = {
                    "index": "msmarco-v1-passage",
                    "searcher_type": "bm25",
                }
            elif searcher_type == "pyterrier":
                searcher_kwargs = {
                    "index_name": "msmarco_passage",
                    "searcher_type": "bm25",
                }
            
            retriever = qg.Retriever(searcher_type=searcher_type, searcher_kwargs=searcher_kwargs)
            results = retriever.retrieve(query, k)
            
            print(f"  Retrieved {len(results)} documents")
            for i, (docid, content) in enumerate(results[:2], 1):
                print(f"    {i}. {docid}: {content[:60]}...")
            
            # Get searcher info
            info = retriever.get_searcher_info()
            print(f"  Searcher: {info.get('name', 'Unknown')}")
            print()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}\n")


def main():
    """Run all examples."""
    
    # Check if API key is set
    if "OPENAI_API_KEY" not in os.environ:
        print("⚠️  Warning: OPENAI_API_KEY not set.")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        print("Some examples may fail without API key.\n")
    
    # Run examples
    example_with_pyserini()
    example_with_pyterrier()
    example_custom_searcher()
    example_comparison()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    
    print("""
Key Benefits of the New Searcher Interface:

1. 🔌 Library Agnostic: Works with any search library (Pyserini, PyTerrier, custom)
2. 🎯 Simple API: Just specify searcher_type and searcher_kwargs
3. 🔧 Flexible: Pass custom searcher instances for advanced use cases
4. 📊 Consistent: Same interface regardless of underlying search library
5. 🚀 Extensible: Easy to add new searcher implementations

Usage Patterns:
- Simple: retriever = qg.Retriever(searcher_type="pyserini", searcher_kwargs={...})
- Advanced: custom_searcher = qg.create_searcher("pyserini", ...); retriever = qg.Retriever(searcher=custom_searcher)
- Method integration: params={"searcher_type": "pyserini", "searcher_kwargs": {...}}
    """)


if __name__ == "__main__":
    main()
