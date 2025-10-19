# Development Guide

This guide covers best practices for developing and testing queryGym before publishing to PyPI.

## Table of Contents
- [Development Setup](#development-setup)
- [Editable Installation](#editable-installation)
- [Testing Your Changes](#testing-your-changes)
- [Code Quality](#code-quality)
- [Adding New Methods](#adding-new-methods)
- [Pre-Release Checklist](#pre-release-checklist)
- [Publishing to PyPI](#publishing-to-pypi)

---

## Development Setup

### 1. Clone and Create Virtual Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/queryGym.git
cd queryGym

# Create a virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 2. Install in Editable Mode

**Editable mode** (`-e` flag) is crucial for development. It creates a symlink to your source code, so any changes you make are immediately reflected without reinstalling.

```bash
# Install with all development dependencies
pip install -e ".[dev,all]"

# Or minimal install for basic development
pip install -e .

# Or with specific extras
pip install -e ".[dev,hf,beir]"
```

**What this does:**
- `-e` = editable mode (changes to code take effect immediately)
- `.[dev,all]` = install from current directory with `dev` and `all` extras
- Creates `queryGym.egg-info/` directory (don't commit this)

### 3. Verify Installation

```bash
# Check CLI works
queryGym --help

# Test Python import
python -c "import queryGym as qg; print(qg.__version__)"

# Check available methods
python -c "import queryGym as qg; print(list(qg.METHODS.keys()))"
```

---

## Editable Installation

### Why Editable Mode?

**Without editable mode:**
```bash
pip install .
# Make code changes
# Need to reinstall: pip install . --force-reinstall
```

**With editable mode:**
```bash
pip install -e .
# Make code changes
# Changes are immediately available! No reinstall needed.
```

### How It Works

Editable mode creates a `.pth` file in your site-packages that points to your source directory:

```
venv/lib/python3.x/site-packages/
├── queryGym.egg-info/          # Package metadata
└── __editable__.queryGym-0.1.0.pth  # Points to your source
```

When you `import queryGym`, Python follows this link to your actual source code.

### Testing Editable Install

```python
# test_editable.py
import queryGym as qg

# This should work immediately after any code changes
reformulator = qg.create_reformulator("genqr", model="gpt-4")
print(f"Loaded version: {qg.__version__}")
print(f"Available methods: {list(qg.METHODS.keys())}")
```

---

## Testing Your Changes

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=queryGym --cov-report=html

# Run specific test file
pytest tests/test_dataloader.py

# Run specific test function
pytest tests/test_dataloader.py::test_local_tsv

# Run with verbose output
pytest -v

# Run and stop at first failure
pytest -x
```

### Writing Tests

Create test files in `tests/` directory:

```python
# tests/test_my_feature.py
import pytest
import queryGym as qg

def test_create_reformulator():
    """Test the convenience factory function."""
    reformulator = qg.create_reformulator("genqr", model="gpt-4")
    assert isinstance(reformulator, qg.GENQR)
    assert reformulator.cfg.name == "genqr"

def test_query_item():
    """Test QueryItem creation."""
    query = qg.QueryItem("q1", "test query")
    assert query.qid == "q1"
    assert query.text == "test query"

@pytest.mark.parametrize("method_name", ["genqr", "query2doc", "genqr_ensemble"])
def test_all_methods_loadable(method_name):
    """Test that all methods can be instantiated."""
    reformulator = qg.create_reformulator(method_name, model="gpt-4")
    assert reformulator is not None
```

### Manual Testing Script

Create a test script to quickly verify changes:

```python
# dev_test.py
import os
os.environ["OPENAI_API_KEY"] = "sk-test-key"  # Use test key

import queryGym as qg

# Test 1: Create reformulator
print("Test 1: Creating reformulator...")
reformulator = qg.create_reformulator("genqr", model="gpt-4")
print(f"✓ Created {type(reformulator).__name__}")

# Test 2: Load queries
print("\nTest 2: Loading queries...")
queries = qg.load_queries("local", path="examples/tiny_queries.tsv", format="tsv")
print(f"✓ Loaded {len(queries)} queries")

# Test 3: Check imports
print("\nTest 3: Checking imports...")
assert hasattr(qg, "QueryItem")
assert hasattr(qg, "create_reformulator")
assert hasattr(qg, "GENQR")
print("✓ All imports available")

print("\n✅ All tests passed!")
```

Run it:
```bash
python dev_test.py
```

---

## Code Quality

### Formatting with Black

```bash
# Format all code
black queryGym/ tests/

# Check without modifying
black --check queryGym/

# Format specific file
black queryGym/core/base.py
```

### Linting with Ruff

```bash
# Lint all code
ruff check queryGym/ tests/

# Auto-fix issues
ruff check --fix queryGym/

# Check specific file
ruff check queryGym/core/base.py
```

### Type Checking with MyPy

```bash
# Type check the package
mypy queryGym/

# Check specific module
mypy queryGym/core/
```

### Pre-commit Hook (Optional)

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
echo "Running code quality checks..."

black --check queryGym/ tests/ || exit 1
ruff check queryGym/ tests/ || exit 1
pytest -x || exit 1

echo "✅ All checks passed!"
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

---

## Adding New Methods

### 1. Create Method Class

```python
# queryGym/methods/my_new_method.py
from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method

@register_method("my_method")
class MyNewMethod(BaseReformulator):
    VERSION = "1.0"
    REQUIRES_CONTEXT = False  # Set to True if needs contexts
    
    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        # Your reformulation logic here
        msgs = self.prompts.render("my_method.prompt.v1", query=q.text)
        result = self.llm.chat(msgs, temperature=0.7, max_tokens=256)
        
        return ReformulationResult(
            qid=q.qid,
            original=q.text,
            reformulated=result,
            metadata={"method": "my_method"}
        )
```

### 2. Add Prompt to prompt_bank.yaml

```yaml
- id: my_method.prompt.v1
  method_family: my_method
  version: 1
  introduced_by: "Your Name"
  license: "CC-BY-4.0"
  authors: ["Your Name"]
  tags: ["custom"]
  template:
    system: |
      You are a helpful assistant.
    user: |
      Reformulate this query: "{query}"
```

### 3. Export in methods/__init__.py

```python
# queryGym/methods/__init__.py
from .my_new_method import MyNewMethod

# Add to existing imports...
```

### 4. Add to main __init__.py

```python
# queryGym/__init__.py
from .methods import (
    # ... existing imports ...
    MyNewMethod,
)

__all__ = [
    # ... existing exports ...
    "MyNewMethod",
]
```

### 5. Test It

```python
# tests/test_my_method.py
import queryGym as qg

def test_my_method():
    reformulator = qg.create_reformulator("my_method", model="gpt-4")
    query = qg.QueryItem("q1", "test query")
    
    # Mock the LLM response for testing
    result = reformulator.reformulate(query)
    assert result.qid == "q1"
    assert result.original == "test query"
```

---

## Pre-Release Checklist

Before publishing to PyPI, ensure:

### Code Quality
- [ ] All tests pass: `pytest`
- [ ] Code is formatted: `black queryGym/ tests/`
- [ ] No linting errors: `ruff check queryGym/`
- [ ] Type hints are correct: `mypy queryGym/`

### Documentation
- [ ] README.md is up to date
- [ ] CHANGELOG.md lists all changes
- [ ] All docstrings are complete
- [ ] Examples work correctly

### Package Structure
- [ ] `__version__` is updated in `__init__.py` and `pyproject.toml`
- [ ] `prompt_bank.yaml` is included in package data
- [ ] All necessary files are in MANIFEST.in (if needed)
- [ ] `.gitignore` excludes build artifacts

### Testing
- [ ] Test in fresh virtual environment:
  ```bash
  python -m venv test_env
  source test_env/bin/activate
  pip install -e .
  python -c "import queryGym; print(queryGym.__version__)"
  ```
- [ ] Test CLI commands work
- [ ] Test all import styles work

### Build
- [ ] Build succeeds: `python -m build`
- [ ] Check dist/ contents
- [ ] Test installation from wheel:
  ```bash
  pip install dist/queryGym-0.1.0-py3-none-any.whl
  ```

---

## Publishing to PyPI

### 1. Install Build Tools

```bash
pip install build twine
```

### 2. Build Distribution

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build wheel and source distribution
python -m build
```

This creates:
- `dist/queryGym-0.1.0-py3-none-any.whl` (wheel)
- `dist/queryGym-0.1.0.tar.gz` (source)

### 3. Test on TestPyPI First

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ queryGym
```

### 4. Upload to PyPI

```bash
# Upload to real PyPI
twine upload dist/*

# Test installation
pip install queryGym
```

### 5. Create Git Tag

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

---

## Best Practices Summary

### ✅ Do's
- **Always use editable install** during development: `pip install -e .`
- **Write tests** for new features before implementing
- **Run tests frequently** to catch issues early
- **Use type hints** for better code documentation
- **Keep commits small** and focused
- **Update CHANGELOG.md** with every change
- **Test in clean environment** before releasing

### ❌ Don'ts
- **Don't commit** `*.egg-info/`, `build/`, `dist/`, `__pycache__/`
- **Don't skip tests** before committing
- **Don't hardcode** API keys or paths
- **Don't break** backward compatibility without major version bump
- **Don't publish** without testing on TestPyPI first

---

## Quick Reference

```bash
# Development workflow
pip install -e ".[dev,all]"      # Initial setup
pytest                           # Run tests
black queryGym/ tests/           # Format code
ruff check --fix queryGym/       # Lint and fix

# Before commit
pytest && black --check queryGym/ && ruff check queryGym/

# Build and publish
python -m build                  # Build package
twine upload --repository testpypi dist/*  # Test
twine upload dist/*              # Publish
```

---

## Getting Help

- **Issues**: https://github.com/yourusername/queryGym/issues
- **Discussions**: https://github.com/yourusername/queryGym/discussions
- **Contributing**: See CONTRIBUTING.md
