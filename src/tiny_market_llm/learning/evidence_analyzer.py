from __future__ import annotations

import pandas as pd
from .schemas import STATE_COLUMNS, EVIDENCE_COLUMNS
from .learning_math import LearningMath

class EvidenceAnalyzer:
    def __init__(self, minimum_samples=20): self.minimum_samples=minimum_samples

    def analyze(self,data:pd.DataFrame,supporting_minimum_samples=20)->dict:
        if supporting_minimum_samples<1: raise ValueError("Supporting minimum samples must be >= 1.")
        evidence={}
        for state,state_group in data.groupby(STATE_COLUMNS,dropna=False):
            if len(state_group)<self.minimum_samples: continue
            state_key=LearningMath.state_key_from_values(state)
            base=LearningMath.direction_probabilities(state_group)
            result={"samples":len(state_group),"base_probability":base,"features":{}}
            for column in EVIDENCE_COLUMNS:
                feature_results={}
                for value,feature_group in state_group.groupby(column,dropna=False):
                    n=len(feature_group)
                    if n<supporting_minimum_samples: continue
                    probs=LearningMath.direction_probabilities(feature_group)
                    feature_results[LearningMath.clean_value(value)]={
                        "samples":n,"direction_probability":probs,
                        "delta":{d:probs[d]-base[d] for d in ("UP","DOWN","FLAT")},
                        "average_move_pct":LearningMath.safe_mean(feature_group,"label_move_pct"),
                        "average_favorable_move_pct":LearningMath.safe_mean(feature_group,"label_favorable_move_pct"),
                        "average_adverse_move_pct":LearningMath.safe_mean(feature_group,"label_adverse_move_pct"),
                        "sustain_probability":float(feature_group["label_sustained"].mean()),
                    }
                if feature_results: result["features"][column]=feature_results
            evidence[state_key]=result
        return evidence

    def train_with_evidence(self,data:pd.DataFrame,minimum_evidence_samples=20)->dict:
        if minimum_evidence_samples<1: raise ValueError("Minimum evidence samples must be >= 1.")
        patterns=[]
        for state,core_group in data.groupby(STATE_COLUMNS,dropna=False):
            if len(core_group)<self.minimum_samples: continue
            base=LearningMath.direction_probabilities(core_group); final=base.copy(); used=[]
            for column in EVIDENCE_COLUMNS:
                for value,evidence_group in core_group.groupby(column,dropna=False):
                    n=len(evidence_group)
                    if n<minimum_evidence_samples or n==len(core_group): continue
                    probs=LearningMath.direction_probabilities(evidence_group)
                    strength=min(1.0,n/(n+self.minimum_samples))
                    for d in ("UP","DOWN","FLAT"): final[d]+=(probs[d]-base[d])*strength
                    used.append({"feature":column,"value":LearningMath.clean_value(value),"samples":n,"strength":strength,"conditional_probability":probs})
            total=sum(final.values())
            if total>0: final={d:p/total for d,p in final.items()}
            predicted=max(final,key=final.get)
            patterns.append({
                "state":{c:LearningMath.clean_value(v) for c,v in zip(STATE_COLUMNS,state)},
                "samples":len(core_group),"base_probability":base,"final_probability":final,
                "predicted_direction":predicted,"confidence":final[predicted],"evidence_used":used,
            })
        return {"model_type":"bucketed_historical_state_with_evidence","state_columns":STATE_COLUMNS,"evidence_columns":EVIDENCE_COLUMNS,"minimum_samples":self.minimum_samples,"minimum_evidence_samples":minimum_evidence_samples,"patterns":patterns}
