# AutoTrader (foundation)

Separate module in the TinyMarketLLM repository. No live execution, trading signal, or running service is implemented yet.

## Rules
- Initial watchlist: 12 configurable candidates (see `src/autotrader/core.py`).
- Cash equity long-only; paper mode only.
- Reward:risk fixed at 2:1, computed from entry and stop.
- Breakout/retest column is reserved and always `NOT_IMPLEMENTED`.
- Optional volume and derivatives OI fields may be absent until Dhan API access is renewed.
- Position size limited by cash capital and planned ₹100 per-trade price risk. Brokerage, taxes, gaps and slippage can cause larger losses.
- No Dhan credentials are needed for the offline smoke test.

## Run
```bash
python scripts/autotrader_demo.py
python -m unittest discover -s tests -p "test_autotrader.py"
```

## Not yet implemented
Live Dhan feed, instrument-ID mapping, OI normalization, signal logic, persistent paper portfolio, exit orders, market-hours scheduler, duplicate-order protection, kill switch, restart reconciliation, dashboard integration and broker order handling. **Never enable live trading before these are implemented and tested.**
