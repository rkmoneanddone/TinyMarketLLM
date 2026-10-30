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

## Watchlist and paper controls

The compact desktop header contains Monitor and Configuration. Configuration contains START, STOP (new entries only), and stock search. Open Positions contains individual EXIT actions and the only EXIT ALL / KILL control.

Stock search is fail-closed: only the built-in demonstration symbols SBIN, BEL, ITC are recognized without an instrument master. For additional NSE equity instruments, provide a **local downloaded Dhan instrument CSV** with fields `SEM_TRADING_SYMBOL`, `SEM_SMST_SECURITY_ID`, `SEM_EXM_EXCH_ID`, and `SEM_SEGMENT`:

```powershell
python scripts/autotrader_dashboard.py --dhan-master path/to/dhan_instruments.csv
```

This resolves exact trading symbols/security IDs for display and validation; it does **not** fetch prices or connect to Dhan. No guessing from company names. Use instrument search and click a match, then Add selected stock. Existing symbols are preserved, up to three. Remove individually. A stock without a configured quote feed displays **NO DATA FEED**. The synthetic stream only covers SBIN/BEL/ITC.

START only lifts an ordinary operator STOP. The existing demonstration entry strategy runs **only** when launched with `--demo`. STOP leaves positions open; emergency exit uses fresh observed paper quotes; missing/stale quotes leave the position open. Emergency lockouts cannot be reset by START. Individual EXIT closes only the selected paper position. No real broker orders are sent.

**Known limitation:** a broker instrument master may change its CSV schema. Verify the downloaded Dhan CSV column names and exchange/segment values before relying on catalog imports.
