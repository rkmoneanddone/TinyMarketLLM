"""NON-NEGOTIABLE AutoTrader order gate.

All future order adapters MUST call validate_order before submitting an order.
Fail closed: unknown products/actions, missing broker reconciliation or invalid
quantities must never reach the broker.
"""
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping

D = lambda value: Decimal(str(value))

@dataclass(frozen=True)
class HardLimits:
    max_capital: Decimal = D("10000")
    max_daily_entries: int = 5
    max_positions: int = 3
    max_watched_stocks: int = 3
    max_risk_per_trade: Decimal = D("100")
    min_reward_risk: Decimal = D("2")
    product: str = "LONG"  # INTRADAY visible in configuration, disabled

@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str
    quantity: int
    price: Decimal
    product: str = "LONG"

@dataclass
class AccountState:
    # Must be reconciled against actual broker holdings, orders and fills before live use.
    confirmed_positions: Mapping[str, int] = field(default_factory=dict)
    pending_sell_qty: Mapping[str, int] = field(default_factory=dict)
    pending_buy_notional: Decimal = D("0")
    open_position_cost: Decimal = D("0")
    daily_entries: int = 0
    available_cash: Decimal = D("0")
    reconciled: bool = False
    trading_enabled: bool = True
    stop_loss_exit_pending: bool = False
    daily_loss_limit_reached: bool = False

class HardCheckRejected(ValueError):
    pass

def validate_order(order: OrderRequest, state: AccountState, limits: HardLimits = HardLimits()) -> None:
    """Raises on any violation; checks are independent of strategy signals."""
    if not state.reconciled:
        raise HardCheckRejected("Broker holdings, cash and pending orders not reconciled")
    if not state.trading_enabled and order.side != "SELL":
        raise HardCheckRejected("Trading halted: new BUY entries blocked")
    if not order.symbol or not order.symbol.strip():
        raise HardCheckRejected("Missing symbol")
    if not isinstance(order.quantity, int) or isinstance(order.quantity, bool) or order.quantity <= 0:
        raise HardCheckRejected("Quantity must be a positive whole number")
    price = D(order.price)
    if not price.is_finite() or price <= 0:
        raise HardCheckRejected("Invalid price")
    if order.product != "LONG" or limits.product != "LONG":
        raise HardCheckRejected("INTRADAY disabled: margin and product safeguards not implemented")
    if order.side not in ("BUY", "SELL"):
        raise HardCheckRejected("Unknown order side")
    if limits.max_capital <= 0 or limits.max_daily_entries < 1 or limits.max_positions < 1:
        raise HardCheckRejected("Invalid hard limits")
    if state.available_cash < 0 or state.open_position_cost < 0 or state.pending_buy_notional < 0:
        raise HardCheckRejected("Invalid account balances")
    if order.side == "SELL":
        held = state.confirmed_positions.get(order.symbol, 0)
        reserved = state.pending_sell_qty.get(order.symbol, 0)
        if held <= 0 or reserved < 0 or order.quantity > held - reserved:
            raise HardCheckRejected("SELL requires confirmed BUY holdings and unreserved quantity")
        return

    # BUY: stop-loss exits take priority; never re-enter to recover losses.
    if state.stop_loss_exit_pending:
        raise HardCheckRejected("Stop-loss exit pending: new BUY entries blocked")
    if state.daily_loss_limit_reached:
        raise HardCheckRejected("Daily loss limit reached: new BUY entries blocked")
    # BUY: no shorts, no leverage, no budget reuse for unsettled orders.
    notional = price * order.quantity
    if state.daily_entries >= limits.max_daily_entries:
        raise HardCheckRejected("Daily entry limit reached")
    if notional > limits.max_capital:
        raise HardCheckRejected("Single order exceeds capital ceiling")
    if state.open_position_cost + state.pending_buy_notional + notional > limits.max_capital:
        raise HardCheckRejected("Total positions and pending orders exceed capital ceiling")
    if notional > state.available_cash:
        raise HardCheckRejected("Insufficient unreserved cash; margin borrowing prohibited")
    held_symbols = {s for s, q in state.confirmed_positions.items() if q > 0}
    if order.symbol not in held_symbols and len(held_symbols) >= limits.max_positions:
        raise HardCheckRejected("Maximum concurrent positions reached")

def validate_watchlist(symbols, limits: HardLimits = HardLimits()):
    unique = tuple(dict.fromkeys(symbols))
    if not unique or len(unique) > limits.max_watched_stocks or any(not s for s in unique):
        raise HardCheckRejected("Watchlist must contain 1 to max_watched_stocks unique symbols")
    return unique

def reject_cash_transfer(*args, **kwargs):
    raise HardCheckRejected("Deposits and withdrawals are permanently outside AutoTrader permissions")
