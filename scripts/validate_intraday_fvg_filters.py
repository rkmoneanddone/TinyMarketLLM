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
from src.tiny_market_llm.scanner.market_sections import fvg_trade_records


SYMBOLS = ("GRASIM", "TCS", "RELIANCE", "TBZ")
START = pd.Timestamp("2026-01-01", tz="UTC")
VALIDATION_START = pd.Timestamp("2026-06-01", tz="UTC")


def candidate_mask(records: pd.DataFrame, variant: str) -> pd.Series:
    trend = (
        (records["ema_21_distance"] > 0)
        & (records["ema_50_distance"] > 0)
        & (records["ema_21_slope"] > 0)
    )
    volume = records["volume_ratio_20"] >= 1.0
    masks = {
        "BASE": pd.Series(True, index=records.index),
        "EMA_TREND": trend,
        "VOLUME": volume,
        "EMA_TREND_VOLUME": trend & volume,
    }
    return masks[variant]


def metrics(records: pd.DataFrame, period: str, variant: str) -> dict:
    resolved = records[records["outcome"].isin(["TARGET", "STOP", "TIME_EXIT"])]
    profitable = int((resolved["pnl_pct"] > 0).sum())
    return {
        "period": period,
        "variant": variant,
        "signals": len(records),
        "resolved": len(resolved),
        "targets": int((records["outcome"] == "TARGET").sum()),
        "stops": int((records["outcome"] == "STOP").sum()),
        "time_exits": int((records["outcome"] == "TIME_EXIT").sum()),
        "profitable": profitable,
        "success_rate_pct": round(profitable / len(resolved) * 100, 2) if len(resolved) else None,
        "average_pnl_pct": round(float(resolved["pnl_pct"].mean()), 4) if len(resolved) else None,
    }


def main() -> None:
    scanner = TinyMarketScanner()
    ledgers = []
    for symbol in SYMBOLS:
        path = ROOT / "data" / "market" / symbol / "1H" / "ohlc.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        prepared = scanner.prepare(pd.read_parquet(path))
        ledgers.append(fvg_trade_records(prepared, symbol, START, timeframe="1H"))

    trades = pd.concat(ledgers, ignore_index=True)
    development = trades[trades["signal_date"] < VALIDATION_START]
    validation = trades[trades["signal_date"] >= VALIDATION_START]
    variants = ("BASE", "EMA_TREND", "VOLUME", "EMA_TREND_VOLUME")
    rows = []
    for variant in variants:
        rows.append(metrics(development[candidate_mask(development, variant)], "DEVELOPMENT_JAN_MAY", variant))
        rows.append(metrics(validation[candidate_mask(validation, variant)], "UNSEEN_JUN_ONWARD", variant))

    results = pd.DataFrame(rows)
    eligible = results[
        (results["period"] == "UNSEEN_JUN_ONWARD")
        & (results["resolved"] >= 10)
        & (results["success_rate_pct"] >= 70)
        & (results["average_pnl_pct"] > 0)
    ]["variant"].tolist()

    output = ROOT / "data" / "research"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "fvg_hold_rsi_1H_filter_validation.csv"
    json_path = output / "fvg_hold_rsi_1H_filter_validation.json"
    results.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_period": "2026-01-01 through 2026-05-31",
        "unseen_period": "2026-06-01 onward",
        "activation_rule": "unseen resolved >= 10, success >= 70%, average P&L > 0",
        "eligible_variants": eligible,
        "results": results.to_dict("records"),
    }, indent=2, default=str), encoding="utf-8")

    print(results.to_string(index=False))
    print(f"\n[ELIGIBLE] {eligible if eligible else 'NONE - keep 1H setup disabled'}")
    print(f"[RECORD] {json_path}")


if __name__ == "__main__":
    main()
