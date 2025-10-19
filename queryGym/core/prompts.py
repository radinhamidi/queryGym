from __future__ import annotations
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

@dataclass
class PromptSpec:
    id: str
    method_family: str
    version: int
    template: Dict[str, str]
    meta: Dict[str, Any]

class PromptBank:
    def __init__(self, path: str | Path):
        items = yaml.safe_load(Path(path).read_text()) or []
        self._by_id: Dict[str, PromptSpec] = {}
        for x in items:
            self._by_id[x["id"]] = PromptSpec(
                id=x["id"],
                method_family=x.get("method_family",""),
                version=x.get("version",1),
                template=x["template"],
                meta={k:v for k,v in x.items() if k not in ["id","method_family","version","template"]}
            )

    def render(self, prompt_id: str, **vars) -> List[Dict[str, str]]:
        spec = self._by_id[prompt_id]
        messages = []
        
        # System message (if present)
        if "system" in spec.template:
            sys = spec.template.get("system","").format(**vars)
            messages.append({"role":"system","content":sys})
        
        # User message (if present)
        if "user" in spec.template:
            usr = spec.template.get("user","").format(**vars)
            messages.append({"role":"user","content":usr})
        
        # Assistant message (if present) - for priming responses
        if "assistant" in spec.template:
            asst = spec.template.get("assistant","").format(**vars)
            messages.append({"role":"assistant","content":asst})
        
        return messages

    def list(self) -> List[str]:
        return list(self._by_id.keys())

    def get_meta(self, prompt_id: str):
        return self._by_id[prompt_id].meta
