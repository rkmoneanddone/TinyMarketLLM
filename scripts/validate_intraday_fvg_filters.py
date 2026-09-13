from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tiny_market_llm.scanner import TinyMarketScanner
from src.tiny_market_llm.scanner.market_sections import intraday_fvg_previous_high_records

SYMBOLS = ("GRASIM", "TCS", "RELIANCE", "TBZ")
START = pd.Timestamp("2026-01-01", tz="UTC")
VALIDATION_START = pd.Timestamp("2026-03-01", tz="UTC")


def filtered(records: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "BASE":
        return records
    mask = (
        (records["ema_21_distance"] > 0)
        & (records["ema_50_distance"] > 0)
        & (records["ema_21_slope"] > 0)
    )
    return records[mask]


def metrics(records: pd.DataFrame, symbol: str, period: str, variant: str) -> dict:
    resolved = records[records["outcome"].isin(["TARGET", "STOP"])]
    wins = int((resolved["outcome"] == "TARGET").sum())
    return {
        "symbol": symbol, "period": period, "variant": variant,
        "signals": len(records), "resolved": len(resolved), "targets": wins,
        "stops": int((resolved["outcome"] == "STOP").sum()),
        "open": int((records["outcome"] == "OPEN").sum()),
        "ambiguous": int((records["outcome"] == "AMBIGUOUS").sum()),
        "success_rate_pct": round(wins / len(resolved) * 100, 2) if len(resolved) else None,
        "average_reward_risk": round(float(resolved["reward_risk"].mean()), 3) if len(resolved) else None,
        "average_pnl_pct": round(float(resolved["pnl_pct"].mean()), 4) if len(resolved) else None,
        "average_holding_hours": round(float(resolved["holding_candles"].mean()), 1) if len(resolved) else None,
    }


def main() -> None:
    scanner = TinyMarketScanner()
    ledgers = []
    for symbol in SYMBOLS:
        path = ROOT / "data" / "market" / symbol / "1H" / "ohlc.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        ledger = intraday_fvg_previous_high_records(scanner.prepare(pd.read_parquet(path)), symbol, START)
        if not ledger.empty:
            ledgers.append(ledger)

    trades = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    rows = []
    for symbol in SYMBOLS:
        stock = trades[trades["symbol"] == symbol]
        periods = {
            "DEVELOPMENT_JAN_FEB": stock[stock["signal_date"] < VALIDATION_START],
            "UNSEEN_MAR_ONWARD": stock[stock["signal_date"] >= VALIDATION_START],
        }
        for period, period_rows in periods.items():
            for variant in ("BASE", "EMA_TREND"):
                rows.append(metrics(filtered(period_rows, variant), symbol, period, variant))

    results = pd.DataFrame(rows)
    eligible = results[
        (results["period"] == "UNSEEN_MAR_ONWARD")
        & (results["resolved"] >= 10)
        & (results["success_rate_pct"] >= 70)
        & (results["average_pnl_pct"] > 0)
    ][["symbol", "variant"]].to_dict("records")

    output = ROOT / "data" / "research"
    output.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output / "fvg_previous_high_1H_trades.csv", index=False)
    results.to_csv(output / "fvg_previous_high_1H_validation.csv", index=False)
    record = output / "fvg_previous_high_1H_validation.json"
    record.write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "logic": "bullish FVG -> hold -> rising RSI -> nearest confirmed previous swing high",
        "development": "2026-01-01 through 2026-02-28",
        "unseen": "2026-03-01 onward", "maximum_holding": None,
        "eligible": eligible, "results": results.to_dict("records"),
        "trades": trades.to_dict("records"),
    }, indent=2, default=str), encoding="utf-8")

    for symbol in SYMBOLS:
        print(f"\n{'=' * 22} {symbol} {'=' * 22}")
        print(results[results["symbol"] == symbol].to_string(index=False))
    print(f"\n[ELIGIBLE] {eligible if eligible else 'NONE - keep 1H setup disabled'}")
    print(f"[RECORD] {record}")


if __name__ == "__main__":
    main()
