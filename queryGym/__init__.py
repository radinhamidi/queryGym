"""
queryGym: LLM-based Query Reformulation Toolkit

Simple usage:
    import queryGym as qg
    
    # Create a reformulator
    reformulator = qg.create_reformulator("genqr_ensemble")
    
    # Reformulate queries
    result = reformulator.reformulate(qg.QueryItem("q1", "what causes diabetes"))
"""

__version__ = "0.1.0"

# Core data structures
from .core.base import QueryItem, ReformulationResult, MethodConfig, BaseReformulator

# LLM and prompts
from .core.llm import OpenAICompatibleClient
from .core.prompts import PromptBank

# Data loaders
from .data.dataloader import UnifiedQuerySource

# All reformulation methods
from .methods import (
    GENQR,
    GenQREnsemble,
    Query2Doc,
    QAExpand,
    MuGI,
    LameR,
    Query2E,
)

# High-level runner
from .core.runner import run_method, build_llm
from .core.registry import METHODS, register_method

# Convenience factory function
def create_reformulator(
    method_name: str,
    model: str = "gpt-4",
    params: dict = None,
    llm_config: dict = None,
    prompt_bank_path: str = None,
    **kwargs
):
    """
    Create a reformulator instance with sensible defaults.
    
    Args:
        method_name: Name of the method (e.g., "genqr", "genqr_ensemble", "query2doc")
        model: LLM model name (default: "gpt-4")
        params: Method-specific parameters (default: {})
        llm_config: Additional LLM configuration (temperature, max_tokens, etc.)
        prompt_bank_path: Path to prompt bank YAML (default: bundled prompt_bank.yaml)
        **kwargs: Additional MethodConfig parameters (seed, retries)
    
    Returns:
        BaseReformulator instance
    
    Example:
        >>> import queryGym as qg
        >>> reformulator = qg.create_reformulator("genqr_ensemble", model="gpt-4")
        >>> result = reformulator.reformulate(qg.QueryItem("q1", "what causes diabetes"))
    """
    from pathlib import Path
    
    if params is None:
        params = {}
    
    if llm_config is None:
        llm_config = {}
    
    # Set up LLM config
    llm_cfg = {
        "model": model,
        "temperature": llm_config.get("temperature", 0.8),
        "max_tokens": llm_config.get("max_tokens", 256),
        **{k: v for k, v in llm_config.items() if k not in ["temperature", "max_tokens"]}
    }
    
    # Create config
    config = MethodConfig(
        name=method_name,
        params=params,
        llm=llm_cfg,
        seed=kwargs.get("seed", 42),
        retries=kwargs.get("retries", 2)
    )
    
    # Build LLM client
    llm = build_llm(config)
    
    # Load prompt bank
    if prompt_bank_path is None:
        prompt_bank_path = Path(__file__).parent / "prompt_bank.yaml"
    pb = PromptBank(prompt_bank_path)
    
    # Get method class and instantiate
    if method_name not in METHODS:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHODS.keys())}")
    
    MethodClass = METHODS[method_name]
    return MethodClass(config, llm, pb)


def load_queries(source: str, **kwargs):
    """
    Load queries from various sources.
    
    Args:
        source: "local", "msmarco", or "beir"
        **kwargs: Source-specific parameters (path, format, etc.)
    
    Returns:
        List of QueryItem objects
    
    Example:
        >>> queries = qg.load_queries("local", path="queries.tsv", format="tsv")
    """
    src = UnifiedQuerySource(backend=source, **kwargs)
    return list(src.iter())


__all__ = [
    # Version
    "__version__",
    
    # Core classes
    "QueryItem",
    "ReformulationResult",
    "MethodConfig",
    "BaseReformulator",
    
    # LLM & Prompts
    "OpenAICompatibleClient",
    "PromptBank",
    
    # Data
    "UnifiedQuerySource",
    
    # Methods
    "GENQR",
    "GenQREnsemble",
    "Query2Doc",
    "QAExpand",
    "MuGI",
    "LameR",
    "Query2E",
    
    # High-level API
    "run_method",
    "build_llm",
    "create_reformulator",
    "load_queries",
    
    # Registry
    "METHODS",
    "register_method",
]
