# AutoTrader paper chain

Implemented offline chain: quote -> demonstration 3/5 SMA crossing or explicit paper BUY -> hard checks -> position -> quote-triggered SL/TP -> SELL -> journal -> daily/weekly report -> learning record.

The strategy is a **demonstration only** and is NOT validated or suitable for live capital. Breakout/retest remains unimplemented. No live Dhan feed, broker orders, UI, true trade-learning model, scheduler, commission model, partial fills or broker-side stop order exists. The learning record is a labeled outcome, not an adaptive model.

Run:
```bash
python scripts/autotrader_chain_demo.py
python -m unittest discover -s tests -p "test_autotrader*.py"
```

Hard checks live in `src/autotrader/hard_checks.py`. This paper ledger assumes immediate simulated fills and sequential calls. Real execution requires atomic reservations, broker reconciliation, persistent stop orders, rate-limit handling, exchange/session compliance, fees/slippage, and a restart-safe kill switch. No credentials are required; real orders remain disabled.
