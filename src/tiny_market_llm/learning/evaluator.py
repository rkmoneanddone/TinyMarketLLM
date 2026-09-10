from __future__ import annotations
import pandas as pd
from .learning_math import LearningMath

class Evaluator:
    def __init__(self,promotion_threshold=0.80): self.promotion_threshold=promotion_threshold
    def evaluate(self,model:dict,data:pd.DataFrame,mode="core")->dict:
        predictions=[]
        for _,row in data.iterrows():
            key=LearningMath.state_key_from_row(row)
            pattern=next((p for p in model.get("patterns",[]) if LearningMath.state_key(p["state"])==key),None)
            if pattern is None: continue
            predicted=pattern["predicted_direction"]; actual=row["label_direction"]
            predictions.append({"predicted":predicted,"actual":actual,"correct":predicted==actual,"confidence":pattern["confidence"],"samples":pattern["samples"]})
        if not predictions:
            return {"evaluated":0,"correct":0,"accuracy":0.0,"accuracy_pct":0.0,"coverage":0.0,"coverage_pct":0.0,"qualifies":False,"reason":"No known patterns in evaluation data."}
        evaluated=len(predictions); correct=sum(x["correct"] for x in predictions); total=len(data); accuracy=correct/evaluated; coverage=evaluated/total if total else 0.0
        return {"evaluated":evaluated,"correct":correct,"accuracy":accuracy,"accuracy_pct":accuracy*100,"coverage":coverage,"coverage_pct":coverage*100,"qualifies":accuracy>=self.promotion_threshold,"threshold":self.promotion_threshold}
