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
