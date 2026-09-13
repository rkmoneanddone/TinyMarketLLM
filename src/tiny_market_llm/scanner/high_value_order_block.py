from __future__ import annotations

import pandas as pd


def _swing_highs(data: pd.DataFrame) -> pd.Series:
    """Five-candle swing highs; a swing at i is known only at i+2."""
    return (
        (data["high"] > data["high"].shift(1))
        & (data["high"] > data["high"].shift(2))
        & (data["high"] >= data["high"].shift(-1))
        & (data["high"] >= data["high"].shift(-2))
    ).fillna(False)


def higher_timeframe_bullish_order_blocks(higher: pd.DataFrame) -> pd.DataFrame:
    """Return causal A-break-C-B structures from completed higher-TF candles.

    A is a previously confirmed swing high. C is the final bearish candle before
    the first close above A. B is the first later swing high above A. The zone
    becomes tradable only two higher-TF candles after B, when B is confirmed.
    """
    data = higher.copy().reset_index(drop=True)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing higher-timeframe columns: {sorted(missing)}")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    swings = _swing_highs(data)
    structures: list[dict] = []
    used_a: set[int] = set()

    for broken in range(4, len(data)):
        # At candle j, only swing highs at j-2 or earlier are confirmed.
        known_a = data.index[: broken - 1][swings.iloc[: broken - 1]]
        if not len(known_a):
            continue
        a = int(known_a[-1])
        level_a = float(data.at[a, "high"])
        first_break = data.at[broken, "close"] > level_a and data.at[broken - 1, "close"] <= level_a
        if not first_break or a in used_a:
            continue
        bearish = data.loc[a + 1: broken - 1]
        bearish = bearish[bearish["close"] < bearish["open"]]
        if bearish.empty:
            continue
        c = int(bearish.index[-1])

        # B must form after the break and must subsequently be confirmed.
        for b in range(broken, len(data) - 2):
            if bool(swings.iloc[b]) and data.at[b, "high"] > level_a:
                confirmed_at = b + 2
                structures.append({
                    "a_date": data.at[a, "timestamp"], "a_price": level_a,
                    "break_date": data.at[broken, "timestamp"],
                    "c_date": data.at[c, "timestamp"],
                    "zone_low": float(data.at[c, "low"]),
                    "zone_high": float(data.at[c, "high"]),
                    "b_date": data.at[b, "timestamp"], "b_price": float(data.at[b, "high"]),
                    "tradable_from": data.at[confirmed_at, "timestamp"],
                })
                used_a.add(a)
                break
    return pd.DataFrame(structures)


def lower_timeframe_retest_trades(higher: pd.DataFrame, lower: pd.DataFrame,
                                  symbol: str, higher_timeframe: str,
                                  lower_timeframe: str) -> pd.DataFrame:
    """Buy first lower-TF return to C after B confirmation; target B, stop C low."""
    structures = higher_timeframe_bullish_order_blocks(higher)
    if structures.empty:
        return pd.DataFrame()
    data = lower.copy().reset_index(drop=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    records: list[dict] = []
    for structure in structures.to_dict("records"):
        entry = float(structure["zone_high"])
        stop = float(structure["zone_low"])
        target = float(structure["b_price"])
        if not stop < entry < target:
            continue
        eligible = data.index[data["timestamp"] > structure["tradable_from"]]
        for signal in eligible:
            touched = data.at[signal, "low"] <= entry and data.at[signal, "high"] >= stop
            if not touched:
                continue
            outcome, exit_date, exit_price, holding = "OPEN", None, None, None
            for exited in range(signal, len(data)):
                target_hit = data.at[exited, "high"] >= target
                stop_hit = data.at[exited, "low"] <= stop
                if target_hit and stop_hit:
                    outcome, exit_date, holding = "AMBIGUOUS", data.at[exited, "timestamp"], exited - signal
                    break
                if target_hit or stop_hit:
                    outcome = "TARGET" if target_hit else "STOP"
                    exit_date = data.at[exited, "timestamp"]
                    exit_price = target if target_hit else stop
                    holding = exited - signal
                    break
            records.append({
                "symbol": symbol.upper(), "higher_timeframe": higher_timeframe,
                "lower_timeframe": lower_timeframe, **structure,
                "signal_date": data.at[signal, "timestamp"], "entry_price": entry,
                "target_price": target, "stop_loss": stop,
                "reward_risk": (target - entry) / (entry - stop),
                "outcome": outcome, "exit_date": exit_date, "exit_price": exit_price,
                "holding_candles": holding,
                "pnl_pct": ((exit_price / entry - 1) * 100) if exit_price is not None else None,
            })
            break
    return pd.DataFrame(records)
