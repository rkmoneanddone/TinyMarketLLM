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
UNSEEN_START = pd.Timestamp("2026-03-01", tz="UTC")
REGIMES = (
    "PRICE_UP_VOLUME_UP",
    "PRICE_UP_VOLUME_DOWN",
    "PRICE_DOWN_VOLUME_DOWN",
    "PRICE_DOWN_VOLUME_UP",
)


def add_regime(records: pd.DataFrame) -> pd.DataFrame:
    result = records.copy()
    price_up = result["price_change_3_pct"] > 0
    volume_up = result["volume_change_3_pct"] > 0
    result["price_volume_regime"] = "PRICE_DOWN_VOLUME_DOWN"
    result.loc[price_up & volume_up, "price_volume_regime"] = "PRICE_UP_VOLUME_UP"
    result.loc[price_up & ~volume_up, "price_volume_regime"] = "PRICE_UP_VOLUME_DOWN"
    result.loc[~price_up & volume_up, "price_volume_regime"] = "PRICE_DOWN_VOLUME_UP"
    result["interpretation"] = result["price_volume_regime"].map({
        "PRICE_UP_VOLUME_UP": "LONG_CONFIRMATION",
        "PRICE_UP_VOLUME_DOWN": "RISE_WEAKENING",
        "PRICE_DOWN_VOLUME_DOWN": "WAIT_POSSIBLE_REVERSAL",
        "PRICE_DOWN_VOLUME_UP": "BEARISH_CONTINUATION",
    })
    return result


def summarize(rows: pd.DataFrame, symbol: str, period: str, regime: str) -> dict:
    selected = rows[rows["price_volume_regime"] == regime]
    resolved = selected[selected["outcome"].isin(["TARGET", "STOP"])]
    wins = int((resolved["outcome"] == "TARGET").sum())
    return {
        "symbol": symbol, "period": period, "regime": regime,
        "signals": len(selected), "resolved": len(resolved), "targets": wins,
        "stops": int((resolved["outcome"] == "STOP").sum()),
        "success_rate_pct": round(wins / len(resolved) * 100, 2) if len(resolved) else None,
        "average_reward_risk": round(float(resolved["reward_risk"].mean()), 3) if len(resolved) else None,
        "average_pnl_pct": round(float(resolved["pnl_pct"].mean()), 4) if len(resolved) else None,
    }


def main() -> None:
    scanner = TinyMarketScanner()
    ledgers = []
    for symbol in SYMBOLS:
        path = ROOT / "data" / "market" / symbol / "1H" / "ohlc.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        records = intraday_fvg_previous_high_records(
            scanner.prepare(pd.read_parquet(path)), symbol, START
        )
        if not records.empty:
            ledgers.append(add_regime(records))

    trades = pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    summaries = []
    for symbol in SYMBOLS:
        stock = trades[trades["symbol"] == symbol]
        periods = {
            "DEVELOPMENT_JAN_FEB": stock[stock["signal_date"] < UNSEEN_START],
            "UNSEEN_MAR_ONWARD": stock[stock["signal_date"] >= UNSEEN_START],
        }
        for period, rows in periods.items():
            for regime in REGIMES:
                summaries.append(summarize(rows, symbol, period, regime))

    results = pd.DataFrame(summaries)
    bullish = results[
        (results["period"] == "UNSEEN_MAR_ONWARD")
        & (results["regime"] == "PRICE_UP_VOLUME_UP")
        & (results["resolved"] >= 10)
        & (results["success_rate_pct"] >= 70)
        & (results["average_pnl_pct"] > 0)
    ][["symbol", "regime"]].to_dict("records")

    output = ROOT / "data" / "research"
    output.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output / "fvg_1H_price_volume_trades.csv", index=False)
    results.to_csv(output / "fvg_1H_price_volume_validation.csv", index=False)
    record = output / "fvg_1H_price_volume_validation.json"
    record.write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trend_window_candles": 3,
        "unseen_start": "2026-03-01",
        "eligible_long": bullish,
        "results": results.to_dict("records"),
        "trades": trades.to_dict("records"),
    }, indent=2, default=str), encoding="utf-8")

    for symbol in SYMBOLS:
        print(f"\n{'=' * 22} {symbol} {'=' * 22}")
        print(results[results["symbol"] == symbol].to_string(index=False))
    print(f"\n[LONG ELIGIBLE] {bullish if bullish else 'NONE'}")
    print(f"[RECORD] {record}")


if __name__ == "__main__":
    main()
