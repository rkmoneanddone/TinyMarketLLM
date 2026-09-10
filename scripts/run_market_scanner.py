from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tiny_market_llm.scanner import ScannerConfig, TinyMarketScanner


CONFIG_PATH = ROOT / "config" / "three_stock_scanner.json"


def load_configuration() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_frames(config: dict) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in config["symbols"]:
        path = ROOT / config["storage"]["root"] / symbol / config["timeframe"] / "ohlc.parquet"
        if path.exists():
            frames[symbol] = pd.read_parquet(path)
        else:
            print(f"[SKIP] {symbol}: missing {path}")
    if not frames:
        raise FileNotFoundError("No configured stock datasets were found. Run update_three_stocks.py first.")
    return frames


def scanner_from(config: dict) -> TinyMarketScanner:
    return TinyMarketScanner(ScannerConfig(
        horizon=config["prediction_horizon_candles"],
        horizons=tuple(config["prediction_horizons"]),
        flat_threshold_pct=config["flat_move_threshold_pct"],
        minimum_confidence=config["minimum_confidence"],
        minimum_training_rows=config["minimum_training_rows"],
        minimum_horizon_agreement=config["minimum_horizon_agreement"],
        target_atr_multiple=config["target_atr_multiple"],
        stop_atr_multiple=config["stop_atr_multiple"],
    ))


def write_report(rows: pd.DataFrame, summary: dict | None, config: dict, name: str) -> Path:
    report_dir = ROOT / config["reports"]["root"]
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"{name}.csv"
    json_path = report_dir / f"{name}.json"
    html_path = report_dir / f"{name}.html"
    rows.to_csv(csv_path, index=False)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "rows": rows.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    columns = list(rows.columns)
    table_rows = []
    for record in rows.to_dict(orient="records"):
        cells = "".join(f"<td>{html.escape(str(record.get(column, '')))}</td>" for column in columns)
        table_rows.append(f"<tr>{cells}</tr>")
    summary_html = html.escape(json.dumps(summary, indent=2, default=str)) if summary else "Latest scan"
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TinyMarketLLM</title>
<style>body{{font-family:Segoe UI,Arial;margin:32px;background:#f4f7fb;color:#152033}}
.card{{background:white;padding:20px;border-radius:12px;box-shadow:0 3px 14px #0001;margin-bottom:20px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #dde3ec;text-align:left}}
th{{background:#17243b;color:white;position:sticky;top:0}}pre{{white-space:pre-wrap}}</style></head>
<body><h1>TinyMarketLLM</h1><div class="card"><h2>{html.escape(name)}</h2><pre>{summary_html}</pre></div>
<div class="card" style="overflow:auto"><table><thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in columns)}</tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></div><p>Research/paper-trading output; not a guaranteed trade.</p></body></html>"""
    html_path.write_text(document, encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyMarketLLM three-stock scanner")
    parser.add_argument("--symbol", choices=("GRASIM", "RELIANCE", "TCS"))
    parser.add_argument("--train-start", help="Optional first training date, e.g. 2026-01-01")
    parser.add_argument("--train-end", help="Run fixed-cutoff unseen historical test, e.g. 2025-03-31")
    parser.add_argument("--test-start", help="Optional first unseen-test date, e.g. 2026-04-01")
    parser.add_argument("--test-end", help="Optional final unseen-test date")
    args = parser.parse_args()

    config = load_configuration()
    frames = load_frames(config)
    if args.symbol:
        frames = {args.symbol: frames[args.symbol]}
    scanner = scanner_from(config)

    if args.train_end:
        rows, summary = scanner.historical_test(
            frames,
            train_end=args.train_end,
            train_start=args.train_start,
            test_start=args.test_start,
            test_end=args.test_end,
        )
        train_label = f"{args.train_start or 'earliest'}_to_{args.train_end}"
        test_label = f"{args.test_start or 'after-train'}_to_{args.test_end or 'latest'}"
        name = f"historical_train_{train_label}_test_{test_label}"
    else:
        rows = scanner.scan_latest(frames).head(config["maximum_candidates"])
        summary = {"mode": "latest", "stocks_loaded": sorted(frames)}
        name = f"latest_{args.symbol}" if args.symbol else "latest_all"

    report = write_report(rows, summary, config, name)
    print(rows.to_string(index=False))
    print(f"\n[REPORT] {report}")


if __name__ == "__main__":
    main()
