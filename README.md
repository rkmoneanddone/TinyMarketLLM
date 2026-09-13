# TinyMarketLLM

## Current milestone

Build a visual market-memory dataset from ANY chart image.

The chart does NOT need to show:

- price
- symbol
- timeframe
- exchange

If those are unavailable, the analyzer records them as unknown and still
extracts visual price behavior.

### Pipeline

```text
ANY CHART
   ↓
scanner
   ↓
SHA-256 duplicate detection
   ↓
vision teacher
   ↓
visual market memory
   ↓
processed / review / failed
```

### Important

The current vision analyzer uses OpenAI as a **teacher**. It is NOT the final
TinyMarketLLM.

The long-term goal is:

```text
teacher-generated market memory
        ↓
high-quality training dataset
        ↓
small specialized model
```

## Setup

From PowerShell:

```powershell
cd F:\projects\TinyMarketLLM
python -m pip install -r requirements.txt
```

Set the API key for the current PowerShell session:

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
```

Optional model override:

```powershell
$env:TINYMARKETLLM_VISION_MODEL="gpt-5-mini"
```

Put a chart into:

```text
charts\incoming
```

Run:

```powershell
python scripts\scan_charts.py
```

The result is stored in:

```text
data\raw\chart_records.jsonl
```

The image is moved to:

```text
charts\processed
```

or:

```text
charts\review
```

API/model errors leave the image in `charts\incoming` so it can be retried.

## Dataset principle

We store two complementary forms of knowledge:

1. Visual price behavior
2. Numerical price movement

A chart without price values is still useful for learning the visual behavior.

When OHLC data is later available, it can be paired with the visual record to create
a normalized movement signature such as:

```text
maximum favorable move: +6.4%
maximum adverse move: -1.3%
final move: +5.1%
```

The model should ultimately learn:

> When price behaves like this, what tends to happen next and how much does it move?


## Schema 2.1 — evidence separation

Every record now keeps:

```text
teacher_analysis
    ↓
raw response from the vision teacher

trusted_analysis
    ↓
validated/normalized version for future training
```

The trusted analysis explicitly separates:

- observed
- derived
- inferred
- unknown_or_unreliable

Numerical movement is kept separate from visual movement.

If actual numerical OHLC/outcome data is unavailable:

```text
numerical_movement.available = false
```

and percentage movement fields remain null.

This prevents the future model from learning fabricated price outcomes.

Future date values are also flagged/cleared by the validator when they appear
inconsistent with the current date.


## v5.1 schema fix

Removed the obsolete `volume_visible` field from the response schema. Volume remains explicitly outside the analysis scope.


## Environment configuration

The scanner automatically loads:

```text
F:\projects\TinyMarketLLM\.env
```

No PowerShell `$env:OPENAI_API_KEY=...` command is required.

Example:

```text
OPENAI_API_KEY=your_real_key
TINYMARKETLLM_VISION_MODEL=gpt-5.6-luna
```

The `.env` file is excluded from Git via `.gitignore`.


## Schema 2.3 — structured market sequence

Price swings are now represented as machine-readable `H`, `L`, `HH`, `HL`, `LH`, and `LL`.
RSI gets the same structured swing representation. A separate price-vs-RSI relationship
record captures potential divergence without forcing the future model to parse English prose.

Visible levels distinguish exact printed values from approximate visual levels.

## Three-stock market scanner

The first numerical scanner is intentionally limited to `GRASIM`, `RELIANCE`,
and `TCS`. It has no Firebase, Supabase, or OpenAI runtime dependency. Dhan is
used only to refresh portable local OHLC files; model training and reports use
the saved data.

Install the numerical dependencies:

```powershell
cd F:\projects\TinyMarketLLM
python -m pip install -r requirements.txt
```

Put a current Dhan client ID and token in the local `.env`, then download or
incrementally refresh the three datasets:

```powershell
python scripts\update_three_stocks.py
```

Run a leakage-safe fixed-cutoff experiment. This trains only on candles through
March and evaluates later candles as unseen data:

```powershell
python scripts\run_market_scanner.py --train-end 2025-03-31 --test-end 2025-09-30
```

To train on only January-March and evaluate only April-September, provide all
four boundaries explicitly:

```powershell
python scripts\run_market_scanner.py --train-start 2026-01-01 --train-end 2026-03-31 --test-start 2026-04-01 --test-end 2026-09-30
```

For the more realistic daily walk-forward test, retrain before every trading
day using only the preceding rolling history:

```powershell
python scripts\run_market_scanner.py --walk-forward --training-years 3 --test-start 2026-04-01 --test-end 2026-09-30
```

This is the preferred validation mode for daily trade ideas. Intraday/hourly
signals require separate intraday OHLC data and a separately validated model;
daily and hourly candles are not mixed in one learner.

Analyse one configured stock:

```powershell
python scripts\run_market_scanner.py --symbol RELIANCE
```

Scan all three and create the latest zero-to-three candidate report:

```powershell
python scripts\run_market_scanner.py
```

Reports are written as CSV, JSON, and a locally openable HTML dashboard under
`reports\daily`. Full scans use `latest_all`; named scans use a separate
`latest_SYMBOL` filename, so neither overwrites the other. `WAIT` is a valid
result; the scanner never forces a BUY or
SELL merely to fill a quota. Outputs are research/paper-trading candidates and
not guaranteed trades.

Historical summaries include majority-baseline accuracy and mark an
underperforming model as `REJECTED`. A rejected model must not be promoted to
paper-trade or live-trade use.

The scanner reports separate accuracy and confidence for its 1-, 3-, and
5-candle classifiers. The configured 1-candle classifier is the primary
BUY/SELL/WAIT direction; the other horizons confirm it rather than having their
probabilities averaged into a mismatched target. BUY or SELL requires sufficient
primary-model confidence, agreement from at least two horizons, and matching
EMA/RSI/volume/price-structure evidence. Historical
training purges rows whose future label crosses the training cutoff. Reports
also measure whether a 1 ATR target was reached before a 0.75 ATR stop within
five candles; a same-candle target/stop touch is recorded as `AMBIGUOUS`, never
as a win. A model is a `CANDIDATE` only when it beats the majority baseline,
beats random balanced accuracy, and demonstrates target-before-stop quality on
enough resolved trades across multiple stocks. Both classification accuracy and
target-before-stop rate use 95% Wilson uncertainty bounds; their lower bounds
must clear the relevant baseline. The default requires at least 20 resolved
trades covering at least two stocks. Otherwise it remains
`REJECTED`/research-only.

The primary one-candle direction matches the “tomorrow” scanner goal. Trade
path quality is measured separately over the next five candles, so reports make
a clear distinction between direction horizon and target/stop evaluation window.

Chart-derived ideas are isolated from the production feature set until they
prove useful. Run the automatic leakage-safe comparison with:

```powershell
python scripts\compare_scanner_features.py --training-years 3 --test-start 2026-04-01 --test-end 2026-09-30
```

This creates one core-versus-chart report without changing the configured core
scanner. The candidate group includes longer support/resistance, breakout
strength, range compression, volume expansion, candle close location, EMA
overextension, and higher-low structure. It is never promoted automatically.

## Selective trade setups

The scanner does not trade on an indicator alone. It detects discrete EMA21/50
crossovers, EMA pullbacks, volume-confirmed breakouts/breakdowns, breakout
retests, and support/resistance rejections. A BUY or SELL requires a detected
setup plus matching model direction, multi-horizon agreement, technical
evidence, and confidence. Every setup is also evaluated independently by
target-before-stop outcome in historical reports; otherwise the result is WAIT.

Before allowing the probability model to veto or approve setups, evaluate the
setups themselves over all locally stored history:

```powershell
python scripts\evaluate_trade_setups.py
```

This fast evaluator does not retrain the model. It removes overlapping repeated
signals using the five-candle trade window, reports expectancy before costs,
and requires at least 20 resolved events across at least two stocks whose 95%
target-rate lower bound exceeds the strategy breakeven rate.

Trade eligibility is configuration-driven through `eligible_setups`. Candidate
setups remain visible in research reports, but only approved setup names can
reach BUY or SELL. The optional `compact` feature profile uses candle behaviour,
EMA trend, RSI momentum, and relative volume; it is research-only unless unseen
walk-forward tests outperform the configured `core` profile.

## 10-year high-breakout section

`python scripts\run_market_sections.py --section high-breakouts` creates an
offline section and report from locally stored data. It identifies price-only
and volume-confirmed stored-history breakouts, collapses clustered candles into
independent events, and reports target-before-stop results. The output is
deliberately labelled 10Y/stored-history high rather than lifetime ATH.

## Next-day trade section

`python scripts\run_market_sections.py --section next-day` evaluates approved
setups specifically against the following daily candle with a 0.1% cost/slippage
buffer. A setup must have at least 20 independent events and a 95% success-rate
lower bound above 50% in both the pre-2024 discovery period and the unseen
2024+ period. Otherwise every stock is explicitly reported as WAIT.

## Swing trade section

`python scripts\run_market_sections.py --section swing` evaluates 1/2/3-week
and 1/2/3-month horizons independently from daily candles. Events cannot overlap
within their own horizon. Promotion requires at least 20 events, positive mean
return after a 0.2% buffer, and a 95% success-rate lower bound above 50% in both
the discovery and unseen periods; otherwise the horizon reports WAIT.

## EMA 9/21/50/200 alignment section

`python scripts\run_market_sections.py --section ema-alignment` reports whether
price is above or below all four EMAs and whether the averages form a proper
bullish or bearish stack. A setup occurs only on a new candle transition through
the complete stack and is trade-eligible only after discovery and unseen
target-before-stop validation.

## Daily/weekly/monthly RSI reversal section

`python scripts\run_market_sections.py --section rsi-reversal` resamples stored
daily OHLCV into completed weekly and monthly candles and evaluates RSI reversal
separately for every timeframe. RSI below 25 or above 80 is informational; an
entry requires a bullish reclaim of 25 or bearish rejection of 80 plus candle
confirmation and successful discovery/unseen validation.
