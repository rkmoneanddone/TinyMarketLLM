from __future__ import annotations
from pathlib import Path
import json

class ModelRepository:
    def save(self,model:dict,path:Path)->Path:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(model,indent=2,ensure_ascii=False),encoding="utf-8")
        return path
    def load(self,path:Path)->dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))
