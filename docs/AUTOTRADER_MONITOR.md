# Paper market monitor

The monitor accepts a stream of quotes for up to three configured symbols, captures LTP/volume/OI, rejects stale timestamped quotes, evaluates protective paper SL/TP, and produces a machine-readable dashboard snapshot with daily/weekly summaries.

Default mode is **observe only** (no automatic entries). Use `--demo` to enable the existing unvalidated 3/5 SMA demonstration entry logic. There is no broker connection and no real trading.

```powershell
python scripts/autotrader_monitor.py
python scripts/autotrader_monitor.py --demo --interval 0.1
python scripts/autotrader_monitor.py --csv prices.csv --ledger data/autotrader/paper_ledger.json --status data/autotrader/status.json
python -m unittest discover -s tests -p "test_autotrader*.py"
```

CSV header: `symbol,price,volume,oi`; price required, volume/OI optional. Replay ignores historical timestamps and is not a live feed. The JSON snapshot is not a browser dashboard. Restart with the same ledger to restore paper positions and counters. No real orders, no live Dhan API, no autonomous process scheduling, and no trained learning model.

**Limitations:** SL/TP execution is simulated at the observed quote, not guaranteed in live markets. No broker reconciliation, atomic order management, commissions, exchange holidays, margin, partial fills, or production risk controls. Do not use with live funds.
