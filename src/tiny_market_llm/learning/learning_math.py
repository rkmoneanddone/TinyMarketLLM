from __future__ import annotations

import pandas as pd
from .schemas import STATE_COLUMNS, EVIDENCE_COLUMNS

class LearningMath:
    @staticmethod
    def direction_probabilities(data: pd.DataFrame) -> dict:
        counts=data["label_direction"].value_counts(); total=len(data)
        if total==0: return {"UP":0.0,"DOWN":0.0,"FLAT":0.0}
        return {d:float(counts.get(d,0)/total) for d in ("UP","DOWN","FLAT")}

    @staticmethod
    def clean_value(value):
        if pd.isna(value): return None
        if hasattr(value,"item"):
            try: return value.item()
            except Exception: pass
        return value

    @staticmethod
    def safe_mean(data: pd.DataFrame, column: str) -> float:
        if column not in data.columns: return 0.0
        values=pd.to_numeric(data[column],errors="coerce")
        if values.dropna().empty: return 0.0
        return float(values.mean())

    @classmethod
    def state_key(cls,state:dict)->tuple:
        return tuple(cls.clean_value(state.get(c)) for c in STATE_COLUMNS)

    @classmethod
    def state_key_from_row(cls,row:pd.Series)->tuple:
        return tuple(cls.clean_value(row.get(c)) for c in STATE_COLUMNS)

    @classmethod
    def state_key_from_values(cls,state)->str:
        return "|".join(str(cls.clean_value(v)) for v in state)
