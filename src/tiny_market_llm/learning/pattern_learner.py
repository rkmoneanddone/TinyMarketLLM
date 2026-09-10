from __future__ import annotations

import pandas as pd
from .schemas import STATE_COLUMNS
from .learning_math import LearningMath

class PatternLearner:
    def __init__(self, minimum_samples=20, promotion_threshold=0.80):
        self.minimum_samples=minimum_samples
        self.promotion_threshold=promotion_threshold

    def train(self, data: pd.DataFrame)->dict:
        grouped=data.groupby(STATE_COLUMNS,dropna=False)
        patterns=[]
        for state,group in grouped:
            sample_count=len(group)
            if sample_count < self.minimum_samples: continue
            probs=LearningMath.direction_probabilities(group)
            predicted=max(probs,key=probs.get)
            selected=group[group["label_direction"]==predicted]
            if selected.empty: continue
            target_probability=float(selected["label_target_reached"].mean()) if "label_target_reached" in selected.columns else 0.0
            patterns.append({
                "state":{c:LearningMath.clean_value(v) for c,v in zip(STATE_COLUMNS,state)},
                "samples":sample_count,
                "predicted_direction":predicted,
                "direction_probability":probs,
                "expected_move_pct":LearningMath.safe_mean(selected,"label_move_pct"),
                "expected_favorable_move_pct":LearningMath.safe_mean(selected,"label_favorable_move_pct"),
                "expected_adverse_move_pct":LearningMath.safe_mean(selected,"label_adverse_move_pct"),
                "expected_duration":LearningMath.safe_mean(selected,"label_duration"),
                "sustain_probability":float(selected["label_sustained"].mean()),
                "target_probability":target_probability,
                "confidence":float(probs[predicted]),
            })
        return {
            "schema_version":"2.0",
            "model_type":"bucketed_historical_state_pattern",
            "state_columns":STATE_COLUMNS,
            "minimum_samples":self.minimum_samples,
            "promotion_threshold":self.promotion_threshold,
            "patterns":patterns,
        }
