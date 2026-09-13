from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tiny_market_llm.scanner.high_value_order_block import lower_timeframe_retest_trades
from src.tiny_market_llm.scanner.market_sections import resample_ohlc

SYMBOLS = ("GRASIM", "TCS", "RELIANCE", "TBZ")
TYPES = ("DESCENDING_CONTINUATION", "CONSOLIDATION_LIQUIDITY_FAKEOUT",
         "SIGNIFICANT_LOW_LIQUIDITY_FAKEOUT")
UNSEEN_START = pd.Timestamp("2024-01-01", tz="UTC")


def summary(rows: pd.DataFrame, symbol: str, mapping: str, period: str, kind: str) -> dict:
    chosen = rows if kind == "ALL_3_TYPES" else rows[rows["retest_type"] == kind]
    resolved = chosen[chosen["outcome"].isin(["TARGET", "STOP"])]
    wins = int((resolved["outcome"] == "TARGET").sum())
    returns_r = resolved.apply(
        lambda row: row["reward_risk"] if row["outcome"] == "TARGET" else -1.0, axis=1)
    return {"symbol": symbol, "mapping": mapping, "period": period, "retest_type": kind,
            "signals": len(chosen), "resolved": len(resolved), "targets": wins,
            "stops": int((resolved["outcome"] == "STOP").sum()),
            "success_rate_pct": round(wins / len(resolved) * 100, 2) if len(resolved) else None,
            "expectancy_r": round(float(returns_r.mean()), 3) if len(returns_r) else None,
            "average_pnl_pct": round(float(resolved["pnl_pct"].mean()), 4) if len(resolved) else None}


def main() -> None:
    ledgers = []
    for symbol in SYMBOLS:
        daily_path = ROOT / "data" / "market" / symbol / "1D" / "ohlc.parquet"
        if not daily_path.exists():
            raise FileNotFoundError(f"Missing {daily_path}")
        daily = pd.read_parquet(daily_path); daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)
        weekly = resample_ohlc(daily, "W-FRI")
        result = lower_timeframe_retest_trades(weekly, daily, symbol, "1W", "1D", TYPES)
        if len(result): ledgers.append(result)
        hourly_path = ROOT / "data" / "market" / symbol / "1H" / "ohlc.parquet"
        if hourly_path.exists():
            hourly = pd.read_parquet(hourly_path); hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True)
            higher_daily = resample_ohlc(hourly, "1D")
            result = lower_timeframe_retest_trades(higher_daily, hourly, symbol, "1D", "1H", TYPES)
            if len(result): ledgers.append(result)
    ledger = pd.concat(ledgers, ignore_index=True, sort=False) if ledgers else pd.DataFrame()
    rows = []
    for symbol in SYMBOLS:
        for mapping in ("1W->1D", "1D->1H"):
            high, low = mapping.split("->")
            stock = ledger[(ledger["symbol"] == symbol) & (ledger["higher_timeframe"] == high)
                           & (ledger["lower_timeframe"] == low)] if len(ledger) else ledger
            periods = {"DEVELOPMENT_PRE_2024": stock[stock["signal_date"] < UNSEEN_START],
                       "UNSEEN_2024_ONWARD": stock[stock["signal_date"] >= UNSEEN_START]}
            for period, period_rows in periods.items():
                for kind in ("ALL_3_TYPES", *TYPES):
                    rows.append(summary(period_rows, symbol, mapping, period, kind))
    results = pd.DataFrame(rows)
    output = ROOT / "data" / "research"; output.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output / "order_block_retest_type_trades.csv", index=False)
    results.to_csv(output / "order_block_retest_type_validation.csv", index=False)
    record = output / "order_block_retest_type_validation.json"
    record.write_text(json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "retest_types": TYPES, "results": results.to_dict("records"),
        "trades": ledger.to_dict("records")}, indent=2, default=str), encoding="utf-8")
    print(results.to_string(index=False)); print(f"\n[RECORD] {record}")


if __name__ == "__main__": main()
