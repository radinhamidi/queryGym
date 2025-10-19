from __future__ import annotations
from typing import List

class PyseriniBridge:
    def __init__(self, retriever=None, k:int=10):
        self.retriever = retriever
        self.k = k

    def retrieve(self, query: str) -> List[str]:
        if self.retriever is None:
            raise RuntimeError("No retriever set")
        hits = self.retriever.search(query, self.k)
        ctxs: List[str] = []
        for h in hits:
            if hasattr(h, "raw") and h.raw:
                ctxs.append(h.raw)
            elif hasattr(h, "contents"):
                ctxs.append(h.contents)
        return ctxs
