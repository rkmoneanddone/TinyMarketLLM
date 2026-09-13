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
UNSEEN_START = pd.Timestamp("2024-01-01", tz="UTC")


def summarize(rows: pd.DataFrame, symbol: str, mapping: str, period: str) -> dict:
    resolved = rows[rows["outcome"].isin(["TARGET", "STOP"])]
    wins = int((resolved["outcome"] == "TARGET").sum())
    r = resolved.apply(lambda x: x["reward_risk"] if x["outcome"] == "TARGET" else -1.0, axis=1)
    return {
        "symbol": symbol, "mapping": mapping, "period": period, "signals": len(rows),
        "resolved": len(resolved), "targets": wins, "stops": int((resolved["outcome"] == "STOP").sum()),
        "success_rate_pct": round(wins / len(resolved) * 100, 2) if len(resolved) else None,
        "average_reward_risk": round(float(resolved["reward_risk"].mean()), 3) if len(resolved) else None,
        "expectancy_r": round(float(r.mean()), 3) if len(r) else None,
        "average_pnl_pct": round(float(resolved["pnl_pct"].mean()), 4) if len(resolved) else None,
    }


def main() -> None:
    trades: list[pd.DataFrame] = []
    for symbol in SYMBOLS:
        daily_path = ROOT / "data" / "market" / symbol / "1D" / "ohlc.parquet"
        if not daily_path.exists():
            raise FileNotFoundError(f"Missing {daily_path}")
        daily = pd.read_parquet(daily_path)
        daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)
        weekly = resample_ohlc(daily, "W-FRI")
        weekly_daily = lower_timeframe_retest_trades(weekly, daily, symbol, "1W", "1D")
        if len(weekly_daily):
            trades.append(weekly_daily)

        hourly_path = ROOT / "data" / "market" / symbol / "1H" / "ohlc.parquet"
        if hourly_path.exists():
            hourly = pd.read_parquet(hourly_path)
            hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True)
            hourly_daily = resample_ohlc(hourly, "1D")
            daily_hourly = lower_timeframe_retest_trades(hourly_daily, hourly, symbol, "1D", "1H")
            if len(daily_hourly):
                trades.append(daily_hourly)

    ledger = pd.concat(trades, ignore_index=True, sort=False) if trades else pd.DataFrame()
    results = []
    for symbol in SYMBOLS:
        for mapping in ("1W->1D", "1D->1H"):
            high, low = mapping.split("->")
            selected = ledger[(ledger["symbol"] == symbol) & (ledger["higher_timeframe"] == high)
                              & (ledger["lower_timeframe"] == low)] if len(ledger) else ledger
            periods = {"DEVELOPMENT_PRE_2024": selected[selected["signal_date"] < UNSEEN_START],
                       "UNSEEN_2024_ONWARD": selected[selected["signal_date"] >= UNSEEN_START]}
            for period, rows in periods.items():
                results.append(summarize(rows, symbol, mapping, period))
    result_frame = pd.DataFrame(results)
    output = ROOT / "data" / "research"; output.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output / "high_value_order_block_trades.csv", index=False)
    result_frame.to_csv(output / "high_value_order_block_validation.csv", index=False)
    record = output / "high_value_order_block_validation.json"
    record.write_text(json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "HTF A break -> last bearish candle C -> confirmed new high B -> LTF return to C -> target B",
        "results": result_frame.to_dict("records"), "trades": ledger.to_dict("records")},
        indent=2, default=str), encoding="utf-8")
    print(result_frame.to_string(index=False))
    print(f"\n[RECORD] {record}")


if __name__ == "__main__":
    main()
