# Chart-derived hypotheses

The supplied chart examples are used as hypothesis teachers, not as training
labels. Their annotations can contain hindsight and selection bias, while the
underlying OHLC and volume data provide reproducible evidence.

Candidate features to test, one group at a time:

- resistance/support age, touch count, and distance;
- range compression before a breakout;
- breakout close location and strength relative to ATR;
- volume contraction followed by expansion;
- breakout retest and hold/failure;
- EMA pullback, reclaim, and distance/overextension;
- EMA200 overhead or support;
- higher-low/lower-high slope and failed breakout structure.

Each feature group must be computed using only candles available at prediction
time. It is retained only if daily walk-forward evaluation improves results
across more than one stock and clears the existing statistical credibility
gates. Timeframes are evaluated separately: 1D first, then intraday models when
intraday data are available.
