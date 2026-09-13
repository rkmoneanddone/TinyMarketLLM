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
from src.tiny_market_llm.scanner.market_sections import resample_ohlc
from src.tiny_market_llm.scanner.order_block_experiment import VARIANTS, order_block_trade_records, select_variant

SYMBOLS = ("GRASIM", "TCS", "RELIANCE", "TBZ")
UNSEEN_START = pd.Timestamp("2024-01-01", tz="UTC")


def metrics(rows: pd.DataFrame, symbol: str, timeframe: str, period: str, variant: str, direction: str) -> dict:
    chosen = rows if direction == "ALL" else rows[rows.direction == direction]
    resolved = chosen[chosen.outcome.isin(["TARGET", "STOP"])]
    wins = int((resolved.outcome == "TARGET").sum())
    expectancy = None
    maximum_drawdown = None
    if len(resolved):
        risk_units = resolved.apply(lambda x: x.reward_risk if x.outcome == "TARGET" else -1.0, axis=1)
        expectancy = round(float(risk_units.mean()), 3)
        equity = risk_units.cumsum()
        maximum_drawdown = round(float((equity.cummax() - equity).max()), 3)
    return {"symbol": symbol, "timeframe": timeframe, "period": period, "variant": variant,
            "direction": direction, "signals": len(chosen), "resolved": len(resolved), "targets": wins,
            "stops": int((resolved.outcome == "STOP").sum()),
            "success_rate_pct": round(wins/len(resolved)*100, 2) if len(resolved) else None,
            "average_reward_risk": round(float(resolved.reward_risk.mean()), 3) if len(resolved) else None,
            "expectancy_r": expectancy, "maximum_drawdown_r": maximum_drawdown,
            "average_pnl_pct": round(float(resolved.pnl_pct.mean()), 4) if len(resolved) else None,
            "average_mfe_pct": round(float(resolved.mfe_pct.mean()), 4) if len(resolved) else None,
            "average_mae_pct": round(float(resolved.mae_pct.mean()), 4) if len(resolved) else None}


def main() -> None:
    ledgers = []
    scanner = TinyMarketScanner()
    for symbol in SYMBOLS:
        path = ROOT / "data" / "market" / symbol / "1D" / "ohlc.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run scripts/download_daily.py first")
        daily = pd.read_parquet(path)
        daily["timestamp"] = pd.to_datetime(daily.timestamp, utc=True)
        for timeframe, frame in (("1D", daily), ("1W", resample_ohlc(daily, "W-FRI"))):
            ledger = order_block_trade_records(scanner.prepare(frame), symbol, timeframe=timeframe)
            if len(ledger): ledgers.append(ledger)
    trades = pd.concat(ledgers, ignore_index=True, sort=False) if ledgers else pd.DataFrame()
    rows = []
    for symbol in SYMBOLS:
        for timeframe in ("1D", "1W"):
            base = trades[(trades.symbol == symbol) & (trades.timeframe == timeframe)]
            periods = {"DEVELOPMENT_PRE_2024": base[base.signal_date < UNSEEN_START],
                       "UNSEEN_2024_ONWARD": base[base.signal_date >= UNSEEN_START]}
            for period, period_rows in periods.items():
                for variant in VARIANTS:
                    selected = select_variant(period_rows, variant)
                    for direction in ("ALL", "LONG", "SHORT"):
                        rows.append(metrics(selected, symbol, timeframe, period, variant, direction))
    results = pd.DataFrame(rows)
    output = ROOT / "data" / "research"; output.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output / "order_block_experiment_trades.csv", index=False)
    results.to_csv(output / "order_block_experiment_validation.csv", index=False)
    record = output / "order_block_experiment_validation.json"
    record.write_text(json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unseen_start": str(UNSEEN_START), "variants": VARIANTS, "results": results.to_dict("records"),
        "trades": trades.to_dict("records")}, indent=2, default=str), encoding="utf-8")
    for timeframe in ("1D", "1W"):
        print(f"\n{'='*24} {timeframe} {'='*24}")
        view = results[(results.timeframe == timeframe) & (results.direction == "ALL")]
        print(view.to_string(index=False))
    print(f"\n[RECORD] {record}")


if __name__ == "__main__": main()
