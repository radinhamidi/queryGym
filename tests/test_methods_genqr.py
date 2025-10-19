from queryGym.core.base import MethodConfig, QueryItem
from queryGym.core.prompts import PromptBank
from queryGym.methods.genqr_ensemble import GenQREnsemble
from pathlib import Path

class DummyLLM:
    def chat(self, messages, **kw): 
        return "alpha, beta, gamma"

def test_genqr_ensemble():
    cfg = MethodConfig(name="genqr_ensemble", params={"repeat_query_weight":2}, llm={"model":"dummy"})
    llm = DummyLLM()
    pb = PromptBank(Path(__file__).parents[1] / "queryGym" / "prompt_bank.yaml")
    meth = GenQREnsemble(cfg, llm, pb)
    res = meth.reformulate(QueryItem("Q1", "bertopic"))
    assert "bertopic bertopic" in res.reformulated
