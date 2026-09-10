from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LevelEvidence:
    current_price: float

    previous_high: Optional[float]
    previous_low: Optional[float]

    distance_to_high_pct: Optional[float]
    distance_to_low_pct: Optional[float]

    high_status: str
    low_status: str

    bounce_possible: bool
    continuation_possible: bool

    remarks: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    valid: bool
    direction: str
    ma_alignment: bool
    ma_alignment_type: str
    structure_valid: bool
    previous_level: Optional[float]
    previous_level_distance_pct: Optional[float]
    fib_1618: Optional[float]
    target: Optional[float]
    target_type: Optional[str]
    fib_target: Optional[float]
    reasons: List[str] = field(default_factory=list)
