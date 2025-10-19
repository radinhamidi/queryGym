#!/usr/bin/env python3
"""
Test script to verify queryGym development setup is working correctly.

Run this after installing with: pip install -e ".[dev,all]"
"""

import sys
import importlib.util


def test_import():
    """Test that queryGym can be imported."""
    print("✓ Testing import...")
    try:
        import queryGym as qg
        print(f"  ✓ queryGym imported successfully")
        print(f"  ✓ Version: {qg.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import queryGym: {e}")
        return False


def test_top_level_imports():
    """Test that all key classes are available at top level."""
    print("\n✓ Testing top-level imports...")
    import queryGym as qg
    
    required_attrs = [
        "QueryItem",
        "ReformulationResult",
        "MethodConfig",
        "BaseReformulator",
        "OpenAICompatibleClient",
        "PromptBank",
        "UnifiedQuerySource",
        "GENQR",
        "GenQREnsemble",
        "Query2Doc",
        "QAExpand",
        "create_reformulator",
        "load_queries",
        "METHODS",
    ]
    
    all_ok = True
    for attr in required_attrs:
        if hasattr(qg, attr):
            print(f"  ✓ {attr}")
        else:
            print(f"  ✗ {attr} NOT FOUND")
            all_ok = False
    
    return all_ok


def test_methods_registry():
    """Test that all methods are registered."""
    print("\n✓ Testing methods registry...")
    import queryGym as qg
    
    expected_methods = [
        "genqr",
        "genqr_ensemble",
        "query2doc",
        "qa_expand",
        "mugi",
        "lamer",
        "query2e",
    ]
    
    all_ok = True
    for method in expected_methods:
        if method in qg.METHODS:
            print(f"  ✓ {method}")
        else:
            print(f"  ✗ {method} NOT REGISTERED")
            all_ok = False
    
    return all_ok


def test_create_reformulator():
    """Test the convenience factory function."""
    print("\n✓ Testing create_reformulator()...")
    import queryGym as qg
    
    try:
        # This won't make API calls, just creates the object
        reformulator = qg.create_reformulator("genqr", model="gpt-4")
        print(f"  ✓ Created reformulator: {type(reformulator).__name__}")
        
        # Check it's the right type
        if isinstance(reformulator, qg.GENQR):
            print(f"  ✓ Correct type: GENQR")
            return True
        else:
            print(f"  ✗ Wrong type: {type(reformulator)}")
            return False
    except Exception as e:
        print(f"  ✗ Failed to create reformulator: {e}")
        return False


def test_query_item():
    """Test QueryItem creation."""
    print("\n✓ Testing QueryItem...")
    import queryGym as qg
    
    try:
        query = qg.QueryItem(qid="q1", text="test query")
        if query.qid == "q1" and query.text == "test query":
            print(f"  ✓ QueryItem created correctly")
            return True
        else:
            print(f"  ✗ QueryItem has wrong values")
            return False
    except Exception as e:
        print(f"  ✗ Failed to create QueryItem: {e}")
        return False


def test_load_queries():
    """Test load_queries function."""
    print("\n✓ Testing load_queries()...")
    import queryGym as qg
    from pathlib import Path
    
    # Check if example file exists
    example_file = Path("examples/tiny_queries.tsv")
    if not example_file.exists():
        print(f"  ⚠ Skipping (examples/tiny_queries.tsv not found)")
        return True
    
    try:
        queries = qg.load_queries("local", path=str(example_file), format="tsv")
        print(f"  ✓ Loaded {len(queries)} queries")
        
        if len(queries) > 0:
            print(f"  ✓ First query: {queries[0].qid} - {queries[0].text[:50]}...")
            return True
        else:
            print(f"  ✗ No queries loaded")
            return False
    except Exception as e:
        print(f"  ✗ Failed to load queries: {e}")
        return False


def test_editable_install():
    """Test if package is installed in editable mode."""
    print("\n✓ Testing editable install...")
    import queryGym
    
    package_path = queryGym.__file__
    print(f"  Package location: {package_path}")
    
    # In editable mode, the path should point to source directory
    if "/site-packages/" in package_path:
        print(f"  ⚠ Package appears to be installed normally (not editable)")
        print(f"  ⚠ For development, use: pip install -e .")
        return False
    else:
        print(f"  ✓ Package is installed in editable mode")
        return True


def test_dependencies():
    """Test that key dependencies are installed."""
    print("\n✓ Testing dependencies...")
    
    dependencies = {
        "typer": "CLI framework",
        "yaml": "YAML parsing (pyyaml)",
        "openai": "OpenAI client",
    }
    
    all_ok = True
    for module, description in dependencies.items():
        spec = importlib.util.find_spec(module)
        if spec is not None:
            print(f"  ✓ {module} ({description})")
        else:
            print(f"  ✗ {module} NOT FOUND ({description})")
            all_ok = False
    
    return all_ok


def test_dev_dependencies():
    """Test that dev dependencies are installed."""
    print("\n✓ Testing dev dependencies...")
    
    dev_deps = {
        "pytest": "Testing framework",
        "black": "Code formatter",
        "ruff": "Linter",
    }
    
    all_ok = True
    for module, description in dev_deps.items():
        spec = importlib.util.find_spec(module)
        if spec is not None:
            print(f"  ✓ {module} ({description})")
        else:
            print(f"  ⚠ {module} NOT FOUND ({description})")
            print(f"    Install with: pip install -e \".[dev]\"")
            all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    print("=" * 60)
    print("queryGym Development Setup Test")
    print("=" * 60)
    
    tests = [
        ("Import", test_import),
        ("Top-level imports", test_top_level_imports),
        ("Methods registry", test_methods_registry),
        ("create_reformulator()", test_create_reformulator),
        ("QueryItem", test_query_item),
        ("load_queries()", test_load_queries),
        ("Editable install", test_editable_install),
        ("Dependencies", test_dependencies),
        ("Dev dependencies", test_dev_dependencies),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your development setup is ready.")
        print("\nNext steps:")
        print("  1. Make changes to code in queryGym/")
        print("  2. Test immediately with: pytest")
        print("  3. No need to reinstall!")
        return 0
    else:
        print("\n⚠ Some tests failed. Please check the output above.")
        print("\nCommon fixes:")
        print("  - Not installed: pip install -e .")
        print("  - Missing dev deps: pip install -e \".[dev,all]\"")
        print("  - Wrong environment: source venv/bin/activate")
        return 1


if __name__ == "__main__":
    sys.exit(main())
