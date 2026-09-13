from __future__ import annotations

import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def high_breakout_history(
    frame: pd.DataFrame,
    symbol: str,
    lookback_years: int = 10,
    minimum_history_candles: int = TRADING_DAYS_PER_YEAR,
    volume_threshold: float = 1.2,
) -> pd.DataFrame:
    """Return a causal, bounded high-breakout history for one symbol.

    The current candle is excluded from both the prior-high and average-volume
    references. With fewer than ten years available, the label remains a
    stored-period high and never claims a lifetime all-time high.
    """
    required = {"timestamp", "high", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing high-breakout columns: {sorted(missing)}")
    if lookback_years < 1 or minimum_history_candles < 1 or volume_threshold <= 0:
        raise ValueError("High-breakout parameters must be positive.")

    data = frame.copy().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    window = lookback_years * TRADING_DAYS_PER_YEAR
    prior_high = data["high"].shift(1).rolling(window, min_periods=minimum_history_candles).max()
    prior_volume = data["volume"].shift(1).rolling(20, min_periods=20).mean()
    volume_ratio = data["volume"] / prior_volume.replace(0, pd.NA)
    price_breakout = prior_high.notna() & (data["close"] > prior_high)
    confirmed = price_breakout & (volume_ratio >= volume_threshold)
    distance = (data["close"] - prior_high) / prior_high.replace(0, pd.NA) * 100

    result = pd.DataFrame({
        "timestamp": data["timestamp"],
        "symbol": symbol.upper(),
        "close": data["close"],
        "prior_stored_high": prior_high,
        "distance_from_high_pct": distance,
        "volume_ratio_20": volume_ratio,
        "price_breakout": price_breakout,
        "volume_confirmed": confirmed,
    })
    result["status"] = "NO_BREAKOUT"
    result.loc[distance.between(-2.0, 0.0, inclusive="both"), "status"] = "NEAR_10Y_HIGH"
    result.loc[price_breakout, "status"] = "PRICE_BREAKOUT"
    result.loc[confirmed, "status"] = "CONFIRMED_10Y_HIGH_BREAKOUT"
    if "buy_trade_outcome" in data:
        result["trade_outcome"] = data["buy_trade_outcome"]
    return result


def independent_breakouts(events: pd.DataFrame, cooldown_candles: int) -> pd.DataFrame:
    kept: list[dict] = []
    for _, group in events.sort_values("timestamp").groupby("symbol"):
        last_index = -10**9
        for row in group.itertuples():
            if row.Index - last_index >= cooldown_candles:
                kept.append(row._asdict())
                last_index = row.Index
    if not kept:
        return events.iloc[0:0].copy()
    return pd.DataFrame(kept).drop(columns=["Index"], errors="ignore")


def next_day_setup_events(
    prepared: pd.DataFrame,
    symbol: str,
    setup_names: tuple[str, ...],
    cooldown_candles: int = 5,
    minimum_net_move_pct: float = 0.1,
) -> pd.DataFrame:
    """Build independent next-session outcomes for explicitly named setups."""
    records = []
    for row in prepared.reset_index(drop=True).itertuples():
        present = set(str(row.setup).split("|"))
        for setup_name in setup_names:
            if setup_name in present and pd.notna(row.future_move_1_pct):
                records.append({
                    "timestamp": row.timestamp,
                    "symbol": symbol.upper(),
                    "bar_index": row.Index,
                    "setup_name": setup_name,
                    "future_move_1_pct": row.future_move_1_pct,
                    "success_after_cost_buffer": bool(row.future_move_1_pct > minimum_net_move_pct),
                })
    if not records:
        return pd.DataFrame(columns=[
            "timestamp", "symbol", "bar_index", "setup_name",
            "future_move_1_pct", "success_after_cost_buffer",
        ])
    events = pd.DataFrame(records)
    kept = []
    for (_, _), group in events.groupby(["symbol", "setup_name"]):
        last_bar = -10**9
        for row in group.sort_values("bar_index").itertuples(index=False):
            if row.bar_index - last_bar >= cooldown_candles:
                kept.append(row._asdict())
                last_bar = row.bar_index
    return pd.DataFrame(kept, columns=events.columns)


def swing_setup_events(
    prepared: pd.DataFrame,
    symbol: str,
    setup_names: tuple[str, ...],
    horizons: dict[str, int],
    minimum_net_move_pct: float = 0.2,
) -> pd.DataFrame:
    """Create non-overlapping forward-return events for daily swing horizons."""
    data = prepared.reset_index(drop=True).copy()
    records = []
    for setup_name in setup_names:
        setup_mask = data["setup"].str.split("|").apply(lambda names: setup_name in names)
        for horizon_name, candles in horizons.items():
            future_move = (data["close"].shift(-candles) / data["close"] - 1) * 100
            candidates = data.index[setup_mask & future_move.notna()]
            last_bar = -10**9
            for bar_index in candidates:
                if bar_index - last_bar < candles:
                    continue
                move = float(future_move.loc[bar_index])
                records.append({
                    "timestamp": data.loc[bar_index, "timestamp"],
                    "symbol": symbol.upper(),
                    "bar_index": int(bar_index),
                    "setup_name": setup_name,
                    "horizon": horizon_name,
                    "horizon_candles": candles,
                    "future_move_pct": move,
                    "success_after_cost_buffer": move > minimum_net_move_pct,
                })
                last_bar = bar_index
    return pd.DataFrame(records)


def ema_alignment_history(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Detect causal price transitions through a 9/21/50/200 EMA stack."""
    required = {"timestamp", "open", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing EMA-alignment columns: {sorted(missing)}")
    data = frame.copy().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    emas = {span: data["close"].ewm(span=span, adjust=False, min_periods=span).mean() for span in (9, 21, 50, 200)}
    maximum = pd.concat(emas.values(), axis=1).max(axis=1)
    minimum = pd.concat(emas.values(), axis=1).min(axis=1)
    above_all = data["close"] > maximum
    below_all = data["close"] < minimum
    bull_stack = (emas[9] > emas[21]) & (emas[21] > emas[50]) & (emas[50] > emas[200])
    bear_stack = (emas[9] < emas[21]) & (emas[21] < emas[50]) & (emas[50] < emas[200])
    long_entry = above_all & ~above_all.shift(1, fill_value=False) & bull_stack & (data["close"] > data["open"])
    short_entry = below_all & ~below_all.shift(1, fill_value=False) & bear_stack & (data["close"] < data["open"])
    result = pd.DataFrame({
        "timestamp": data["timestamp"], "symbol": symbol.upper(), "close": data["close"],
        "ema_9": emas[9], "ema_21": emas[21], "ema_50": emas[50], "ema_200": emas[200],
        "above_all_emas": above_all, "below_all_emas": below_all,
        "bull_stack": bull_stack, "bear_stack": bear_stack,
        "long_entry": long_entry, "short_entry": short_entry,
    })
    result["setup"] = "NONE"
    result.loc[long_entry, "setup"] = "EMA_9_21_50_200_LONG"
    result.loc[short_entry, "setup"] = "EMA_9_21_50_200_SHORT"
    if "buy_trade_outcome" in data:
        result["trade_outcome"] = "NO_SETUP"
        result.loc[long_entry, "trade_outcome"] = data.loc[long_entry, "buy_trade_outcome"]
        result.loc[short_entry, "trade_outcome"] = data.loc[short_entry, "sell_trade_outcome"]
    return result


def resample_ohlc(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Causally aggregate daily OHLCV, excluding the potentially open bucket."""
    data = frame.copy().sort_values("timestamp").set_index("timestamp")
    result = data.resample(frequency, label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna().reset_index()
    return result.iloc[:-1].reset_index(drop=True)


def rsi_reversal_history(prepared: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """Detect confirmed RSI recovery from <25 or rejection from >80."""
    required = {"timestamp", "open", "close", "rsi_14"}
    missing = required - set(prepared.columns)
    if missing:
        raise ValueError(f"Missing RSI-reversal columns: {sorted(missing)}")
    data = prepared.copy().reset_index(drop=True)
    long_entry = (data["rsi_14"].shift(1) < 25) & (data["rsi_14"] >= 25) & (data["close"] > data["open"])
    short_entry = (data["rsi_14"].shift(1) > 80) & (data["rsi_14"] <= 80) & (data["close"] < data["open"])
    result = data[["timestamp", "close", "rsi_14"]].copy()
    result["symbol"] = symbol.upper()
    result["timeframe"] = timeframe
    result["below_25"] = data["rsi_14"] < 25
    result["above_80"] = data["rsi_14"] > 80
    result["setup"] = "NONE"
    result.loc[long_entry, "setup"] = "RSI_RECLAIM_25_LONG"
    result.loc[short_entry, "setup"] = "RSI_REJECT_80_SHORT"
    if "buy_trade_outcome" in data:
        result["trade_outcome"] = "NO_SETUP"
        result.loc[long_entry, "trade_outcome"] = data.loc[long_entry, "buy_trade_outcome"]
        result.loc[short_entry, "trade_outcome"] = data.loc[short_entry, "sell_trade_outcome"]
    return result


def breakout_pullback_history(
    prepared: pd.DataFrame,
    symbol: str,
    zone_lifetime_candles: int = 10,
) -> pd.DataFrame:
    """Detect objective bullish breakout pullbacks into OB and FVG zones."""
    data = prepared.copy().reset_index(drop=True)
    prior_high = data["high"].shift(1).rolling(20, min_periods=20).max()
    breakout = data["close"] > prior_high
    events: list[dict] = []
    active: list[dict] = []
    for index, row in data.iterrows():
        remaining = []
        for zone in active:
            if index > zone["expires"]:
                continue
            touched = row["low"] <= zone["high"] and row["high"] >= zone["low"]
            confirmed = touched and row["close"] > zone["high"] and row["close"] > row["open"]
            if confirmed:
                events.append({
                    "timestamp": row["timestamp"], "symbol": symbol.upper(), "bar_index": index,
                    "setup": zone["setup"], "zone_low": zone["low"], "zone_high": zone["high"],
                    "breakout_bar_index": zone["created"],
                    "trade_outcome": row.get("buy_trade_outcome", "UNKNOWN"),
                })
            else:
                remaining.append(zone)
        active = remaining
        if not bool(breakout.iloc[index]):
            continue
        prior = data.iloc[max(0, index - 5):index]
        bearish = prior[prior["close"] < prior["open"]]
        if not bearish.empty:
            candle = bearish.iloc[-1]
            active.append({
                "setup": "BREAKOUT_ORDER_BLOCK_PULLBACK", "low": float(candle["low"]),
                "high": float(candle["open"]), "created": index, "expires": index + zone_lifetime_candles,
            })
        if index >= 2 and row["low"] > data.loc[index - 2, "high"]:
            active.append({
                "setup": "BREAKOUT_FVG_PULLBACK", "low": float(data.loc[index - 2, "high"]),
                "high": float(row["low"]), "created": index, "expires": index + zone_lifetime_candles,
            })
    return pd.DataFrame(events, columns=[
        "timestamp", "symbol", "bar_index", "setup", "zone_low", "zone_high",
        "breakout_bar_index", "trade_outcome",
    ])


def morning_star_history(prepared: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Detect an ATR-normalized Morning Star at prior 20-candle support."""
    data = prepared.copy().reset_index(drop=True)
    atr = data["atr_14_pct"] / 100 * data["close"]
    first_body = data["open"].shift(2) - data["close"].shift(2)
    middle_body = (data["close"].shift(1) - data["open"].shift(1)).abs()
    third_body = data["close"] - data["open"]
    first_midpoint = (data["open"].shift(2) + data["close"].shift(2)) / 2
    pattern_low = pd.concat([data["low"].shift(2), data["low"].shift(1), data["low"]], axis=1).min(axis=1)
    prior_support = data["low"].shift(3).rolling(20, min_periods=20).min()
    pattern = (
        (first_body >= 0.5 * atr.shift(2))
        & (middle_body <= 0.35 * first_body)
        & (third_body >= 0.4 * atr)
        & (data["close"] > first_midpoint)
        & (pattern_low <= prior_support + 0.25 * atr)
    )
    result = data[["timestamp", "close"]].copy()
    result["symbol"] = symbol.upper()
    result["setup"] = "NONE"
    result.loc[pattern, "setup"] = "MORNING_STAR_AT_SUPPORT"
    result["prior_support"] = prior_support
    result["pattern_low"] = pattern_low
    if "buy_trade_outcome" in data:
        result["trade_outcome"] = "NO_SETUP"
        result.loc[pattern, "trade_outcome"] = data.loc[pattern, "buy_trade_outcome"]
    return result


def weekly_fvg_hold_sequences(
    prepared: pd.DataFrame,
    symbol: str,
    maximum_wait_weeks: int = 12,
    expansion_weeks: int = 8,
) -> pd.DataFrame:
    """Find bullish FVG creation, later hold with rising RSI, then prior-high test."""
    data = prepared.copy().reset_index(drop=True)
    atr = data["atr_14_pct"] / 100 * data["close"]
    bullish_fvg = (
        (data["low"] > data["high"].shift(2))
        & (data["close"] > data["open"])
        & ((data["close"] - data["open"]) >= 0.4 * atr)
    )
    prior_highs = data["high"].shift(1).rolling(20, min_periods=20).max()
    records = []
    for created in data.index[bullish_fvg.fillna(False)]:
        zone_low = float(data.loc[created - 2, "high"])
        zone_high = float(data.loc[created, "low"])
        prior_high = float(prior_highs.loc[created])
        for held in range(created + 1, min(created + maximum_wait_weeks + 1, len(data))):
            tolerance = 0.25 * atr.loc[held]
            touches = data.loc[held, "low"] <= zone_high + tolerance
            holds = touches and data.loc[held, "close"] >= zone_high and data.loc[held, "close"] >= data.loc[held, "open"]
            rsi_rising = data.loc[held, "rsi_14"] > data.loc[held - 1, "rsi_14"] and data.loc[held, "rsi_change_3"] > 0
            if not (holds and rsi_rising):
                continue
            future = data.iloc[held + 1:min(held + expansion_weeks + 1, len(data))]
            reached = future[future["high"] >= prior_high * 0.99]
            records.append({
                "symbol": symbol.upper(), "timeframe": "1W",
                "fvg_date": data.loc[created, "timestamp"], "hold_date": data.loc[held, "timestamp"],
                "fvg_zone_low": zone_low, "fvg_zone_high": zone_high,
                "hold_close": float(data.loc[held, "close"]), "hold_rsi_14": float(data.loc[held, "rsi_14"]),
                "prior_20w_high": prior_high, "reached_prior_high": not reached.empty,
                "weeks_to_prior_high": int(reached.index[0] - held) if not reached.empty else None,
                "maximum_8w_move_pct": float((future["high"].max() / data.loc[held, "close"] - 1) * 100) if len(future) else None,
            })
            break
    if not records:
        return pd.DataFrame()
    result = pd.DataFrame(records)
    # One hold candle is one market event even when several stacked FVGs overlap.
    return result.sort_values("fvg_date", ascending=False).drop_duplicates(["symbol", "hold_date"]).sort_values("hold_date").reset_index(drop=True)
