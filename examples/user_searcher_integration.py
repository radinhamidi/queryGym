#!/usr/bin/env python3
"""
Example: Using User's Existing Pyserini and PyTerrier Code with queryGym

This example shows how users can take their existing Pyserini or PyTerrier code
and easily integrate it with queryGym using the wrapper functions.
"""

import queryGym as qg


def example_user_pyserini_code():
    """Example using user's existing Pyserini code."""
    print("=" * 70)
    print("Example: User's Existing Pyserini Code")
    print("=" * 70)
    
    try:
        from pyserini.search.lucene import LuceneSearcher
        
        # User's existing Pyserini code (unchanged)
        print("\n📝 User's existing Pyserini code:")
        print("lucene_bm25_searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')")
        print("lucene_bm25_searcher.set_bm25(k1=0.9, b=0.4)")
        
        # User creates their searcher (their existing code)
        lucene_bm25_searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
        lucene_bm25_searcher.set_bm25(k1=0.9, b=0.4)
        
        print("✅ User's Pyserini searcher created successfully")
        
        # User wraps it for queryGym (one line!)
        print("\n🔧 Wrapping for queryGym:")
        print("wrapped_searcher = qg.wrap_pyserini_searcher(lucene_bm25_searcher)")
        
        wrapped_searcher = qg.wrap_pyserini_searcher(lucene_bm25_searcher)
        
        # User uses it with queryGym
        print("\n🎯 Using with queryGym:")
        print("retriever = qg.Retriever(searcher=wrapped_searcher)")
        
        retriever = qg.Retriever(searcher=wrapped_searcher)
        
        # Test retrieval
        print("\n🔍 Testing retrieval:")
        results = retriever.retrieve("machine learning algorithms", k=3)
        
        print(f"✅ Retrieved {len(results)} documents:")
        for i, (docid, content) in enumerate(results, 1):
            print(f"  {i}. {docid}: {content[:60]}...")
        
        # Test batch retrieval
        print("\n📦 Testing batch retrieval:")
        queries = ["artificial intelligence", "deep learning", "neural networks"]
        batch_results = retriever.retrieve_batch(queries, k=2, num_threads=2)
        
        print(f"✅ Batch retrieved {len(batch_results)} query results:")
        for i, results in enumerate(batch_results):
            print(f"  Query {i+1}: {len(results)} documents")
        
        # Test with query reformulation
        print("\n🔄 Testing with query reformulation:")
        reformulator = qg.create_reformulator("genqr", model="gpt-4")
        query = qg.QueryItem("q1", "what causes diabetes")
        
        # This would use the user's searcher for context retrieval
        print("Note: LameR would use this searcher for context retrieval")
        
    except ImportError:
        print("❌ Pyserini not installed. Install with: pip install pyserini")
    except Exception as e:
        print(f"⚠️  Example failed (expected if no internet/index): {e}")


def example_user_pyterrier_code():
    """Example using user's existing PyTerrier code."""
    print("\n\n" + "=" * 70)
    print("Example: User's Existing PyTerrier Code")
    print("=" * 70)
    
    try:
        import pyterrier as pt
        
        # User's existing PyTerrier code (unchanged)
        print("\n📝 User's existing PyTerrier code:")
        print("pt.init()")
        print("topics = pt.io.read_topics(topicsFile)")
        print("qrels = pt.io.read_qrels(qrelsFile)")
        print("BM25_r = pt.terrier.Retriever(index, wmodel='BM25')")
        
        # User creates their setup (their existing code)
        pt.init()
        
        # For demo purposes, we'll use a mock index
        # In real usage, user would have their actual index
        print("✅ PyTerrier initialized")
        
        # User wraps their retriever for queryGym (one line!)
        print("\n🔧 Wrapping for queryGym:")
        print("wrapped_searcher = qg.wrap_pyterrier_retriever(BM25_r, index)")
        
        # Note: This would fail in demo without actual index, but shows the pattern
        print("Note: In real usage, user would wrap their actual retriever")
        
        # Example of how it would work:
        print("\n🎯 Usage pattern:")
        print("""
# User's existing code
pt.init()
BM25_r = pt.terrier.Retriever(index, wmodel="BM25")

# Wrap for queryGym
wrapped_searcher = qg.wrap_pyterrier_retriever(BM25_r, index)

# Use with queryGym
retriever = qg.Retriever(searcher=wrapped_searcher)
results = retriever.retrieve("test query", k=10)
        """)
        
    except ImportError:
        print("❌ PyTerrier not installed. Install with: pip install python-terrier")
    except Exception as e:
        print(f"⚠️  Example failed: {e}")


def example_custom_searcher():
    """Example using user's custom search function."""
    print("\n\n" + "=" * 70)
    print("Example: User's Custom Search Function")
    print("=" * 70)
    
    # User's custom search function
    def my_custom_search(query, k):
        """User's custom search implementation."""
        # Mock implementation - user would have their actual search logic
        results = []
        for i in range(min(k, 3)):
            results.append((
                f"custom_doc_{i+1}",
                1.0 - (i * 0.1),
                f"Custom content for query '{query}' - result {i+1}"
            ))
        return results
    
    def my_custom_batch_search(queries, k):
        """User's custom batch search implementation."""
        return [my_custom_search(query, k) for query in queries]
    
    print("\n📝 User's custom search functions:")
    print("def my_custom_search(query, k): ...")
    print("def my_custom_batch_search(queries, k): ...")
    
    # User wraps their functions for queryGym
    print("\n🔧 Wrapping for queryGym:")
    print("wrapped_searcher = qg.wrap_custom_searcher(my_custom_search, my_custom_batch_search)")
    
    wrapped_searcher = qg.wrap_custom_searcher(
        my_custom_search, 
        my_custom_batch_search,
        "MyCustomSearcher"
    )
    
    # User uses it with queryGym
    print("\n🎯 Using with queryGym:")
    retriever = qg.Retriever(searcher=wrapped_searcher)
    
    # Test retrieval
    print("\n🔍 Testing retrieval:")
    results = retriever.retrieve("test query", k=3)
    
    print(f"✅ Retrieved {len(results)} documents:")
    for i, (docid, content) in enumerate(results, 1):
        print(f"  {i}. {docid}: {content}")
    
    # Test batch retrieval
    print("\n📦 Testing batch retrieval:")
    queries = ["query 1", "query 2", "query 3"]
    batch_results = retriever.retrieve_batch(queries, k=2)
    
    print(f"✅ Batch retrieved {len(batch_results)} query results:")
    for i, results in enumerate(batch_results):
        print(f"  Query {i+1}: {len(results)} documents")


def example_advanced_usage():
    """Example showing advanced usage patterns."""
    print("\n\n" + "=" * 70)
    print("Example: Advanced Usage Patterns")
    print("=" * 70)
    
    print("\n📝 Advanced usage patterns:")
    
    print("\n1. 🔧 Custom Configuration:")
    print("""
# User's Pyserini with custom config
searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
searcher.set_bm25(k1=0.9, b=0.4)
searcher.set_rm3()  # Enable RM3
searcher.set_rocchio()  # Enable Rocchio

# Wrap with custom answer key
wrapped = qg.wrap_pyserini_searcher(searcher, answer_key="contents|title")
retriever = qg.Retriever(searcher=wrapped)
    """)
    
    print("\n2. 🔄 Integration with Query Reformulation:")
    print("""
# User's searcher
my_searcher = qg.wrap_pyserini_searcher(lucene_searcher)

# Use with LameR (requires context)
reformulator = qg.create_reformulator("lamer", model="gpt-4", params={
    "searcher_type": "user_provided",
    "searcher_instance": my_searcher,
    "retrieval_k": 10
})

# LameR will use the user's searcher for context retrieval
result = reformulator.reformulate(qg.QueryItem("q1", "test query"))
    """)
    
    print("\n3. 🎯 Multiple Searchers:")
    print("""
# User can have multiple searchers
pyserini_searcher = qg.wrap_pyserini_searcher(lucene_searcher)
pyterrier_searcher = qg.wrap_pyterrier_retriever(BM25_r, index)

# Compare results
pyserini_results = qg.Retriever(searcher=pyserini_searcher).retrieve(query, k=10)
pyterrier_results = qg.Retriever(searcher=pyterrier_searcher).retrieve(query, k=10)

# Use whichever works better
    """)


def main():
    """Run all examples."""
    print("🧪 Examples: Using User's Existing Code with queryGym")
    print("=" * 70)
    
    example_user_pyserini_code()
    example_user_pyterrier_code()
    example_custom_searcher()
    example_advanced_usage()
    
    print("\n\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
    
    print("""
Summary: User Integration Patterns

✅ Pyserini Users:
  - Keep existing code: lucene_searcher = LuceneSearcher.from_prebuilt_index(...)
  - Wrap in one line: wrapped = qg.wrap_pyserini_searcher(lucene_searcher)
  - Use with queryGym: retriever = qg.Retriever(searcher=wrapped)

✅ PyTerrier Users:
  - Keep existing code: BM25_r = pt.terrier.Retriever(index, wmodel="BM25")
  - Wrap in one line: wrapped = qg.wrap_pyterrier_retriever(BM25_r, index)
  - Use with queryGym: retriever = qg.Retriever(searcher=wrapped)

✅ Custom Searchers:
  - Define search function: def my_search(query, k): return [...]
  - Wrap in one line: wrapped = qg.wrap_custom_searcher(my_search)
  - Use with queryGym: retriever = qg.Retriever(searcher=wrapped)

✅ Key Benefits:
  - No need to change existing code
  - One-line integration with queryGym
  - Full control over searcher configuration
  - Works with all queryGym methods
    """)


if __name__ == "__main__":
    main()

