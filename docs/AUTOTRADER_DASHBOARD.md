# AutoTrader paper dashboard

Run from repository root:

```powershell
git pull
python scripts/autotrader_dashboard.py
```

Open **http://127.0.0.1:8765**. Synthetic prices continuously cycle for three stocks. No API token required.

Optional:

```powershell
python scripts/autotrader_dashboard.py --demo --interval 1 --ledger data/autotrader/paper_ledger.json
python scripts/autotrader_dashboard.py --csv my_quotes.csv --interval 0.5
python -m unittest discover -s tests -p "test_autotrader*.py"
```

Default mode is **observation only**. `--demo` enables the **unvalidated** paper SMA demonstration strategy. The ledger is optional for demonstration but required to retain positions and kill switch across restarts. A CSV replay ends after its final row, while the dashboard stays available.

The dashboard is **read-only**, binds only to localhost, and does not expose BUY/SELL, deposit, withdrawal or Dhan API endpoints. The quotes and OI in synthetic mode are artificial. Real market streaming, broker reconciliation, margin and intraday trading remain disabled. Do not expose the dashboard to the internet.
