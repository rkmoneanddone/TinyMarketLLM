from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tiny_market_llm.scanner import TinyMarketScanner
from src.tiny_market_llm.scanner.market_sections import (
    intraday_bearish_fvg_previous_low_records,
    intraday_fvg_previous_high_records,
)

SYMBOLS = ("GRASIM", "TCS", "RELIANCE", "TBZ")
START = pd.Timestamp("2026-01-01", tz="UTC")
UNSEEN_START = pd.Timestamp("2026-03-01", tz="UTC")


def daily_regimes(hourly: pd.DataFrame) -> pd.DataFrame:
    data = hourly.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["day"] = data["timestamp"].dt.floor("D")
    daily = data.groupby("day", as_index=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    )
    ema21 = daily["close"].ewm(span=21, adjust=False, min_periods=21).mean()
    ema50 = daily["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    recent_high = daily["high"].rolling(5).max()
    recent_low = daily["low"].rolling(5).min()
    prior_high = daily["high"].shift(5).rolling(5).max()
    prior_low = daily["low"].shift(5).rolling(5).min()
    bullish = (
        (daily["close"] > ema21) & (ema21 > ema50) & (ema21 > ema21.shift(3))
        & (recent_high > prior_high) & (recent_low > prior_low)
    )
    bearish = (
        (daily["close"] < ema21) & (ema21 < ema50) & (ema21 < ema21.shift(3))
        & (recent_high < prior_high) & (recent_low < prior_low)
    )
    daily["daily_regime"] = np.select([bullish, bearish], ["BULLISH", "BEARISH"], default="SIDEWAYS")
    # Today's 1H signals may only use yesterday's completed Daily regime.
    daily["effective_from"] = daily["day"] + pd.Timedelta(days=1)
    return daily[["effective_from", "daily_regime"]]


def attach_regime(records: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records
    return pd.merge_asof(
        records.sort_values("signal_date"), regimes.sort_values("effective_from"),
        left_on="signal_date", right_on="effective_from", direction="backward",
    ).drop(columns="effective_from")


def trade_ledger(symbol: str, hourly: pd.DataFrame) -> pd.DataFrame:
    prepared = TinyMarketScanner().prepare(hourly)
    longs = intraday_fvg_previous_high_records(prepared, symbol, START)
    if not longs.empty:
        longs = longs.rename(columns={"buy_price": "entry_price", "sell_target": "target_price"})
        longs["direction"] = "LONG"
    shorts = intraday_bearish_fvg_previous_low_records(prepared, symbol, START)
    combined = pd.concat([longs, shorts], ignore_index=True, sort=False)
    return attach_regime(combined, daily_regimes(hourly))


def select_variant(rows: pd.DataFrame, variant: str) -> pd.DataFrame:
    direction_match = (
        ((rows["direction"] == "LONG") & (rows["daily_regime"] == "BULLISH"))
        | ((rows["direction"] == "SHORT") & (rows["daily_regime"] == "BEARISH"))
    )
    selected = rows[direction_match]
    if variant == "DIRECTION_PRICE_VOLUME":
        confirmation = (
            ((selected["direction"] == "LONG") & (selected["price_change_3_pct"] > 0))
            | ((selected["direction"] == "SHORT") & (selected["price_change_3_pct"] < 0))
        ) & (selected["volume_change_3_pct"] > 0)
        selected = selected[confirmation]
    return selected


def summary(rows: pd.DataFrame, symbol: str, period: str, variant: str, direction: str) -> dict:
    selected = rows if direction == "ALL" else rows[rows["direction"] == direction]
    resolved = selected[selected["outcome"].isin(["TARGET", "STOP"])]
    wins = int((resolved["outcome"] == "TARGET").sum())
    return {
        "symbol": symbol, "period": period, "variant": variant, "direction": direction,
        "signals": len(selected), "resolved": len(resolved), "targets": wins,
        "stops": int((resolved["outcome"] == "STOP").sum()),
        "success_rate_pct": round(wins / len(resolved) * 100, 2) if len(resolved) else None,
        "average_reward_risk": round(float(resolved["reward_risk"].mean()), 3) if len(resolved) else None,
        "average_pnl_pct": round(float(resolved["pnl_pct"].mean()), 4) if len(resolved) else None,
    }


def main() -> None:
    ledgers = []
    for symbol in SYMBOLS:
        path = ROOT / "data" / "market" / symbol / "1H" / "ohlc.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        ledgers.append(trade_ledger(symbol, pd.read_parquet(path)))
    trades = pd.concat(ledgers, ignore_index=True, sort=False)

    results = []
    for symbol in SYMBOLS:
        stock = trades[trades["symbol"] == symbol]
        periods = {
            "DEVELOPMENT_JAN_FEB": stock[stock["signal_date"] < UNSEEN_START],
            "UNSEEN_MAR_ONWARD": stock[stock["signal_date"] >= UNSEEN_START],
        }
        for period, period_rows in periods.items():
            for variant in ("DIRECTION_ONLY", "DIRECTION_PRICE_VOLUME"):
                chosen = select_variant(period_rows, variant)
                for direction in ("ALL", "LONG", "SHORT"):
                    results.append(summary(chosen, symbol, period, variant, direction))
    result_frame = pd.DataFrame(results)

    output = ROOT / "data" / "research"
    output.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output / "direction_aware_fvg_1H_trades.csv", index=False)
    result_frame.to_csv(output / "direction_aware_fvg_1H_validation.csv", index=False)
    record = output / "direction_aware_fvg_1H_validation.json"
    record.write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "daily_regime": "completed Daily structure plus EMA21/EMA50",
        "entry_timeframe": "1H", "unseen_start": "2026-03-01",
        "results": result_frame.to_dict("records"), "trades": trades.to_dict("records"),
    }, indent=2, default=str), encoding="utf-8")

    for symbol in SYMBOLS:
        print(f"\n{'=' * 22} {symbol} {'=' * 22}")
        print(result_frame[result_frame["symbol"] == symbol].to_string(index=False))
    print(f"\n[RECORD] {record}")


if __name__ == "__main__":
    main()
