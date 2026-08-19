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
