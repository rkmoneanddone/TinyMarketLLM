from __future__ import annotations

import pandas as pd


VARIANTS = {
    "A_ORDER_BLOCK_RETEST": (),
    "B_PLUS_LIQUIDITY_SWEEP": ("liquidity_sweep",),
    "C_PLUS_STRONG_DISPLACEMENT": ("strong_displacement",),
    "D_PLUS_FVG": ("fvg_created",),
    "E_SWEEP_AND_FVG": ("liquidity_sweep", "fvg_created"),
    "F_COMPLETE_REJECTION": ("liquidity_sweep", "strong_displacement", "fvg_created", "rejection"),
    "G_COMPLETE_PRICE_VOLUME": ("liquidity_sweep", "strong_displacement", "fvg_created", "rejection", "price_volume"),
    "H_COMPLETE_RSI_VOLUME": ("liquidity_sweep", "strong_displacement", "fvg_created", "rejection", "price_volume", "rsi_confirmation"),
}


def _raw_swings(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    high = ((data.high > data.high.shift(1)) & (data.high > data.high.shift(2))
            & (data.high >= data.high.shift(-1)) & (data.high >= data.high.shift(-2)))
    low = ((data.low < data.low.shift(1)) & (data.low < data.low.shift(2))
           & (data.low <= data.low.shift(-1)) & (data.low <= data.low.shift(-2)))
    return high.fillna(False), low.fillna(False)


def _outcome(data: pd.DataFrame, signal: int, direction: str, entry: float,
             target: float, stop: float) -> dict:
    favourable, adverse = 0.0, 0.0
    result = {"outcome": "OPEN", "exit_date": None, "exit_price": None,
              "holding_candles": None, "mfe_pct": None, "mae_pct": None}
    for exited in range(signal + 1, len(data)):
        if direction == "LONG":
            target_hit, stop_hit = data.at[exited, "high"] >= target, data.at[exited, "low"] <= stop
            favourable = max(favourable, float(data.at[exited, "high"] / entry - 1) * 100)
            adverse = min(adverse, float(data.at[exited, "low"] / entry - 1) * 100)
        else:
            target_hit, stop_hit = data.at[exited, "low"] <= target, data.at[exited, "high"] >= stop
            favourable = max(favourable, float(1 - data.at[exited, "low"] / entry) * 100)
            adverse = min(adverse, float(1 - data.at[exited, "high"] / entry) * 100)
        if target_hit and stop_hit:
            result.update(outcome="AMBIGUOUS", exit_date=data.at[exited, "timestamp"], holding_candles=exited-signal)
            break
        if target_hit or stop_hit:
            result.update(outcome="TARGET" if target_hit else "STOP",
                          exit_date=data.at[exited, "timestamp"],
                          exit_price=target if target_hit else stop,
                          holding_candles=exited-signal)
            break
    result.update(mfe_pct=favourable, mae_pct=adverse)
    return result


def order_block_trade_records(prepared: pd.DataFrame, symbol: str, *, timeframe: str,
                              maximum_retest_candles: int = 12,
                              structure_lookback: int = 20) -> pd.DataFrame:
    """Build a causal long/short OB ledger; variants are filters over one ledger.

    A swing at i is only usable from i+2. Same-bar target+stop is AMBIGUOUS.
    """
    data = prepared.copy().reset_index(drop=True)
    required = {"timestamp", "open", "high", "low", "close", "volume", "atr_14_pct", "rsi_14"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Missing Order Block columns: {sorted(missing)}")
    atr = data.atr_14_pct / 100 * data.close
    swing_high, swing_low = _raw_swings(data)
    records: list[dict] = []

    for created in range(max(structure_lookback, 3), len(data)):
        body = abs(float(data.at[created, "close"] - data.at[created, "open"]))
        for direction in ("LONG", "SHORT"):
            bullish = direction == "LONG"
            displacement = data.at[created, "close"] > data.at[created, "open"] if bullish else data.at[created, "close"] < data.at[created, "open"]
            if not displacement:
                continue
            prior = data.iloc[max(0, created - 5):created]
            bases = prior[prior.close < prior.open] if bullish else prior[prior.close > prior.open]
            if bases.empty:
                continue
            ob = int(bases.index[-1])
            zone_low = float(data.at[ob, "low"] if bullish else data.at[ob, "close"])
            zone_high = float(data.at[ob, "open"] if bullish else data.at[ob, "high"])
            known_end = created - 2
            known = data.iloc[:known_end + 1]
            if bullish:
                swings = known.index[swing_low.iloc[:known_end + 1]]
                swept_level = float(data.at[int(swings[-1]), "low"]) if len(swings) else None
                swept = swept_level is not None and data.at[ob, "low"] < swept_level and data.at[created, "close"] > swept_level
                fvg = data.at[created, "low"] > data.at[created - 2, "high"]
                broke_structure = data.at[created, "close"] > known.high.tail(structure_lookback).max()
            else:
                swings = known.index[swing_high.iloc[:known_end + 1]]
                swept_level = float(data.at[int(swings[-1]), "high"]) if len(swings) else None
                swept = swept_level is not None and data.at[ob, "high"] > swept_level and data.at[created, "close"] < swept_level
                fvg = data.at[created, "high"] < data.at[created - 2, "low"]
                broke_structure = data.at[created, "close"] < known.low.tail(structure_lookback).min()
            strong = body >= 0.8 * atr.iloc[created] and bool(broke_structure)

            for signal in range(created + 1, min(created + maximum_retest_candles + 1, len(data))):
                touched = data.at[signal, "low"] <= zone_high and data.at[signal, "high"] >= zone_low
                if not touched:
                    continue
                if bullish:
                    rejection = data.at[signal, "close"] > data.at[signal, "open"] and data.at[signal, "close"] >= zone_high
                    price_volume = data.at[signal, "close"] > data.at[signal-1, "close"] and data.at[signal, "volume"] > data.at[signal-1, "volume"]
                    rsi_ok = data.at[signal, "rsi_14"] > data.at[signal-1, "rsi_14"]
                    candidates = known.index[swing_high.iloc[:known_end + 1] & (known.high > data.at[signal, "close"])]
                    if not len(candidates): break
                    target_index, entry = int(candidates[-1]), float(data.at[signal, "close"])
                    target, stop = float(data.at[target_index, "high"]), float(min(zone_low, data.at[signal, "low"]) - .25 * atr.iloc[signal])
                    risk, reward = entry-stop, target-entry
                else:
                    rejection = data.at[signal, "close"] < data.at[signal, "open"] and data.at[signal, "close"] <= zone_low
                    price_volume = data.at[signal, "close"] < data.at[signal-1, "close"] and data.at[signal, "volume"] > data.at[signal-1, "volume"]
                    rsi_ok = data.at[signal, "rsi_14"] < data.at[signal-1, "rsi_14"]
                    candidates = known.index[swing_low.iloc[:known_end + 1] & (known.low < data.at[signal, "close"])]
                    if not len(candidates): break
                    target_index, entry = int(candidates[-1]), float(data.at[signal, "close"])
                    target, stop = float(data.at[target_index, "low"]), float(max(zone_high, data.at[signal, "high"]) + .25 * atr.iloc[signal])
                    risk, reward = stop-entry, entry-target
                if risk <= 0 or reward <= 0:
                    break
                outcome = _outcome(data, signal, direction, entry, target, stop)
                exit_price = outcome["exit_price"]
                pnl = None if exit_price is None else ((exit_price/entry-1)*100 if bullish else (entry-exit_price)/entry*100)
                records.append({
                    "symbol": symbol.upper(), "timeframe": timeframe.upper(), "direction": direction,
                    "order_block_date": data.at[ob, "timestamp"], "displacement_date": data.at[created, "timestamp"],
                    "signal_date": data.at[signal, "timestamp"], "zone_low": zone_low, "zone_high": zone_high,
                    "entry_price": entry, "target_price": target, "stop_loss": stop, "reward_risk": reward/risk,
                    "liquidity_sweep": bool(swept), "strong_displacement": bool(strong), "fvg_created": bool(fvg),
                    "rejection": bool(rejection), "price_volume": bool(price_volume), "rsi_confirmation": bool(rsi_ok),
                    **outcome, "pnl_pct": pnl,
                })
                break
    if not records:
        return pd.DataFrame()
    return (pd.DataFrame(records).sort_values("displacement_date", ascending=False)
            .drop_duplicates(["symbol", "direction", "signal_date"])
            .sort_values("signal_date").reset_index(drop=True))


def select_variant(ledger: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    selected = ledger
    for condition in VARIANTS[variant]:
        selected = selected[selected[condition]]
    return selected
