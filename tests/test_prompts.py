from queryreform.core.prompts import PromptBank
from pathlib import Path

def test_prompt_bank_loads():
    pb = PromptBank(Path(__file__).parents[1] / "queryreform" / "prompt_bank.yaml")
    assert len(pb.list()) > 0
