from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from dataclasses import dataclass
from typing import Dict


@dataclass
class PredictionResult:
    direction: str
    confidence: float
    probability: Dict[str, float]

    timeframe: str = "1D"
    horizon_candles: int = 1

    expected_move_pct: float = 0.0
    expected_favorable_move_pct: float = 0.0
    expected_adverse_move_pct: float = 0.0

    expected_duration: int = 0

    target_price: float | None = None
    target_low: float | None = None
    target_high: float | None = None

    sustain_probability: float = 0.0
    target_probability: float = 0.0

    remarks: list[str] | None = None

# Four-field V2 core state. Supporting evidence must NOT be added here.
STATE_COLUMNS = [
    "structure",
    "price_direction",
    "rsi_direction",
    "rsi_zone",
]

EVIDENCE_COLUMNS = [
    "move_3_bucket",
    "move_5_bucket",
    "range_5_bucket",
    "candle_body_bucket",
    "pullback_bucket",
    "recovery_bucket",
    "rsi_bucket",
    "rsi_change_3_bucket",
    "rsi_overbought",
    "rsi_oversold",
    "bullish_divergence_candidate",
    "bearish_divergence_candidate",

    # Market location evidence
    "below_previous_low",
    "previous_low_broken",
    "near_previous_low",
    "support_status",
    "resistance_status",
    "bounce_signal",
    "continuation_signal",
]

@dataclass
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
