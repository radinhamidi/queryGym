from __future__ import annotations
import os
from openai import OpenAI
from typing import List, Dict, Any

class OpenAICompatibleClient:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self.client = OpenAI(base_url=base_url or os.getenv("OPENAI_BASE_URL", None),
                             api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def chat(self, messages: List[Dict[str, str]], **kw: Any) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=messages, **kw)
        return resp.choices[0].message.content or ""
