from __future__ import annotations
from typing import List

'''
TODO: This is most likely not the best way to do this. We should completely outsource any retrieval to user's preference.
They can use their own retirevers for building context for query reformulation or retrieval task itself.
Also, they can pass the retriever as a function to queryGym to use it to build the context for query reformulation.
'''
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
