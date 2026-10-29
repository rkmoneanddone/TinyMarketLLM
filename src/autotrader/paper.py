"""Simple in-memory paper execution. Does not connect to a broker."""
from dataclasses import dataclass
from decimal import Decimal
from .core import TradePlan

@dataclass(frozen=True)
class PaperFill:
    symbol: str
    side: str
    quantity: int
    price: Decimal
    mode: str = "PAPER"

class PaperBroker:
    def __init__(self):
        self.fills = []

    def enter(self, plan: TradePlan) -> PaperFill:
        fill = PaperFill(plan.symbol, plan.side, plan.quantity, plan.entry)
        self.fills.append(fill)
        return fill
