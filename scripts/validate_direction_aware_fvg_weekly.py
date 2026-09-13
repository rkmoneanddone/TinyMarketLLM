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
START = pd.Timestamp("2016-09-01", tz="UTC")
UNSEEN_START = pd.Timestamp("2024-01-01", tz="UTC")


def resample_ohlc(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return (
        frame.set_index("timestamp").resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"]).reset_index()
    )


def monthly_regimes(daily: pd.DataFrame) -> pd.DataFrame:
    monthly = resample_ohlc(daily, "ME")
    ema21 = monthly["close"].ewm(span=21, adjust=False, min_periods=21).mean()
    ema50 = monthly["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    recent_high, recent_low = monthly["high"].rolling(3).max(), monthly["low"].rolling(3).min()
    prior_high = monthly["high"].shift(3).rolling(3).max()
    prior_low = monthly["low"].shift(3).rolling(3).min()
    bullish = ((monthly["close"] > ema21) & (ema21 > ema50) & (ema21 > ema21.shift(2))
               & (recent_high > prior_high) & (recent_low > prior_low))
    bearish = ((monthly["close"] < ema21) & (ema21 < ema50) & (ema21 < ema21.shift(2))
               & (recent_high < prior_high) & (recent_low < prior_low))
    monthly["higher_regime"] = np.select([bullish, bearish], ["BULLISH", "BEARISH"], default="SIDEWAYS")
    monthly["effective_from"] = monthly["timestamp"] + pd.Timedelta(days=1)
    return monthly[["effective_from", "higher_regime"]]


def ledger(symbol: str, daily: pd.DataFrame) -> pd.DataFrame:
    weekly = resample_ohlc(daily, "W-FRI")
    prepared = TinyMarketScanner().prepare(weekly)
    longs = intraday_fvg_previous_high_records(prepared, symbol, START, timeframe="1W")
    if not longs.empty:
        longs = longs.rename(columns={"buy_price": "entry_price", "sell_target": "target_price"})
        longs["direction"] = "LONG"
    shorts = intraday_bearish_fvg_previous_low_records(prepared, symbol, START, timeframe="1W")
    combined = pd.concat([longs, shorts], ignore_index=True, sort=False).sort_values("signal_date")
    return pd.merge_asof(
        combined, monthly_regimes(daily).sort_values("effective_from"),
        left_on="signal_date", right_on="effective_from", direction="backward",
    ).drop(columns="effective_from")


def choose(rows: pd.DataFrame, variant: str) -> pd.DataFrame:
    match = (((rows["direction"] == "LONG") & (rows["higher_regime"] == "BULLISH"))
             | ((rows["direction"] == "SHORT") & (rows["higher_regime"] == "BEARISH")))
    selected = rows[match]
    if variant == "DIRECTION_PRICE_VOLUME":
        aligned_price = (((selected["direction"] == "LONG") & (selected["price_change_3_pct"] > 0))
                         | ((selected["direction"] == "SHORT") & (selected["price_change_3_pct"] < 0)))
        selected = selected[aligned_price & (selected["volume_change_3_pct"] > 0)]
    return selected


def metrics(rows: pd.DataFrame, symbol: str, period: str, variant: str, direction: str) -> dict:
    selected = rows if direction == "ALL" else rows[rows["direction"] == direction]
    resolved = selected[selected["outcome"].isin(["TARGET", "STOP"])]
    wins = int((resolved["outcome"] == "TARGET").sum())
    return {
        "symbol": symbol, "period": period, "variant": variant, "direction": direction,
        "signals": len(selected), "targets": wins, "stops": int((resolved["outcome"] == "STOP").sum()),
        "success_rate_pct": round(wins / len(resolved) * 100, 2) if len(resolved) else None,
        "average_reward_risk": round(float(resolved["reward_risk"].mean()), 3) if len(resolved) else None,
        "average_pnl_pct": round(float(resolved["pnl_pct"].mean()), 4) if len(resolved) else None,
    }


def main() -> None:
    ledgers = []
    for symbol in SYMBOLS:
        path = ROOT / "data" / "market" / symbol / "1D" / "ohlc.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; download Daily history first")
        ledgers.append(ledger(symbol, pd.read_parquet(path)))
    trades = pd.concat(ledgers, ignore_index=True, sort=False)

    rows = []
    for symbol in SYMBOLS:
        stock = trades[trades["symbol"] == symbol]
        periods = {"DEVELOPMENT_PRE_2024": stock[stock["signal_date"] < UNSEEN_START],
                   "UNSEEN_2024_ONWARD": stock[stock["signal_date"] >= UNSEEN_START]}
        for period, period_rows in periods.items():
            for variant in ("DIRECTION_ONLY", "DIRECTION_PRICE_VOLUME"):
                selected = choose(period_rows, variant)
                for direction in ("ALL", "LONG", "SHORT"):
                    rows.append(metrics(selected, symbol, period, variant, direction))
    results = pd.DataFrame(rows)
    output = ROOT / "data" / "research"
    output.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output / "direction_aware_fvg_1W_trades.csv", index=False)
    results.to_csv(output / "direction_aware_fvg_1W_validation.csv", index=False)
    record = output / "direction_aware_fvg_1W_validation.json"
    record.write_text(json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "direction_timeframe": "completed 1M", "entry_timeframe": "1W",
        "unseen_start": "2024-01-01", "results": results.to_dict("records"),
        "trades": trades.to_dict("records")}, indent=2, default=str), encoding="utf-8")
    for symbol in SYMBOLS:
        print(f"\n{'=' * 22} {symbol} {'=' * 22}")
        print(results[results["symbol"] == symbol].to_string(index=False))
    print(f"\n[RECORD] {record}")


if __name__ == "__main__":
    main()
