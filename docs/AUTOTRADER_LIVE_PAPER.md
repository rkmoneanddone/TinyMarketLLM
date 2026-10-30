# Live-data paper testing

This feature is **read-only market data**. The adapter cannot place orders.

## Offline checks
```powershell
git pull
python -m py_compile src/autotrader/learning.py src/autotrader/dhan_live_data.py scripts/autotrader_dashboard.py
python -m unittest discover -s tests -p "test_autotrader*.py"
```

## Optional live Dhan quotes

Requires an active Dhan data entitlement, valid access token, and a locally downloaded instrument master with the expected columns. Do not commit secrets or instrument files.

```powershell
$env:DHAN_CLIENT_ID="YOUR_CLIENT_ID"
$env:DHAN_ACCESS_TOKEN="YOUR_ACCESS_TOKEN"
python scripts/autotrader_dashboard.py --live-dhan --dhan-master "C:\path\to\dhan-master.csv" --interval 2
```

The watchlist must contain exact mapped NSE equity symbols. Previously saved unknown symbols such as APPOLI must be removed from Configuration before live polling.

Live mode is **observation-only**: protective paper exits still apply to existing paper positions, but no automatic entry strategy is enabled. Do not interpret synthetic/replay results as live evidence. Live LTP alone does not include volume or futures OI.

On any feed failure, the system persists a safety halt and records an error. Restore the data feed and resolve the halt through the proper safety workflow; START does not override a data-error halt.

**Limitations before deployment:** verify Dhan's current marketfeed schema and API rate limits against your entitlement, test the CSV instrument-master format, add session-aware scheduling, heartbeat/staleness monitoring, durable learning setup capture, historical parquet replay, fee/settlement accounting, and comprehensive broker reconciliation. No live orders are supported.
