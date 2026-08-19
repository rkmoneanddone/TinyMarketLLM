from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from typing import Any

from openai import OpenAI


ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "identity": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "symbol": {"type": ["string", "null"]},
                "company_name": {"type": ["string", "null"]},
                "exchange": {"type": ["string", "null"]},
                "chart_source": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": [
                "symbol", "company_name", "exchange",
                "chart_source", "confidence"
            ],
        },
        "time": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "timeframe": {"type": ["string", "null"]},
                "visible_start": {"type": ["string", "null"]},
                "visible_end": {"type": ["string", "null"]},
                "timezone": {"type": ["string", "null"]},
                "confidence": {"type": "number"},
            },
            "required": [
                "timeframe", "visible_start", "visible_end",
                "timezone", "confidence"
            ],
        },
        "chart": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"type": "string"},
                "candlestick_present": {"type": ["boolean", "null"]},
                "ohlc_present": {"type": ["boolean", "null"]},
                "line_chart": {"type": ["boolean", "null"]},
                "multiple_panels": {"type": ["boolean", "null"]},
                "price_scale_available": {"type": ["boolean", "null"]},
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "type", "candlestick_present", "ohlc_present",
                "line_chart", "multiple_panels",
                "price_scale_available", "indicators"
            ],
        },
        "price": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "open": {"type": ["number", "null"]},
                "high": {"type": ["number", "null"]},
                "low": {"type": ["number", "null"]},
                "close": {"type": ["number", "null"]},
                "price_scale": {"type": ["string", "null"]},
                "values_reliably_read": {"type": "boolean"},
            },
            "required": [
                "open", "high", "low", "close",
                "price_scale", "values_reliably_read"
            ],
        },
        "price_structure": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "swing_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "enum": ["H", "L"]},
                            "classification": {"type": "string", "enum": ["INITIAL", "HH", "HL", "LH", "LL", "UNKNOWN"]},
                            "level": {"type": ["number", "null"]},
                            "level_is_exact": {"type": "boolean"},
                            "confidence": {"type": "number"}
                        },
                        "required": ["type", "classification", "level", "level_is_exact", "confidence"]
                    }
                },
                "sequence": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["H", "L", "HH", "HL", "LH", "LL", "UNKNOWN"]}
                },
                "confirmed_sequence": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["H", "L", "HH", "HL", "LH", "LL", "UNKNOWN"]}
                },
                "current_unconfirmed": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "present": {"type": "boolean"},
                        "description": {"type": ["string", "null"]},
                        "type": {"type": ["string", "null"], "enum": ["H", "L", "recovery", "decline", "breakout_attempt", "breakdown_attempt", "UNKNOWN", None]}
                    },
                    "required": ["present", "description", "type"]
                },
                "trend": {"type": ["string", "null"]},
                "structure": {"type": ["string", "null"]}
            },
            "required": ["swing_points", "sequence", "confirmed_sequence", "current_unconfirmed", "trend", "structure"]
        },
        "price_action": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "visible_price_levels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "kind": {"type": "string", "enum": ["HIGH", "LOW", "CLOSE", "SUPPORT", "RESISTANCE"]},
                            "value": {"type": ["number", "null"]},
                            "exact": {"type": "boolean"},
                            "confidence": {"type": "number"}
                        },
                        "required": ["kind", "value", "exact", "confidence"]
                    }
                },
                "high_low_visible": {"type": "boolean"},
                "swing_highs_visible": {"type": "boolean"},
                "swing_lows_visible": {"type": "boolean"},
                "sequence": {"type": "array", "items": {"type": "string"}},
                "trend": {"type": ["string", "null"]},
                "structure": {"type": ["string", "null"]},
                "support_visible": {"type": "boolean"},
                "resistance_visible": {"type": "boolean"},
                "breakout": {"type": "boolean"},
                "breakdown": {"type": "boolean"},
                "pullback": {"type": "boolean"},
                "reversal": {"type": "boolean"}
            },
            "required": ["visible_price_levels","high_low_visible","swing_highs_visible","swing_lows_visible","sequence","trend","structure","support_visible","resistance_visible","breakout","breakdown","pullback","reversal"]
        },
        "rsi": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "visible": {"type": "boolean"},
                "current_value": {"type": ["number", "null"]},
                "direction": {"type": ["string", "null"]},
                "trend": {"type": ["string", "null"]},
                "sequence": {"type": "array", "items": {"type": "string", "enum": ["H", "L", "HH", "HL", "LH", "LL", "UNKNOWN"]}},
                "swing_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "enum": ["H", "L"]},
                            "classification": {"type": "string", "enum": ["INITIAL", "HH", "HL", "LH", "LL", "UNKNOWN"]},
                            "value": {"type": ["number", "null"]},
                            "value_is_exact": {"type": "boolean"},
                            "confidence": {"type": "number"}
                        },
                        "required": ["type", "classification", "value", "value_is_exact", "confidence"]
                    }
                },
                "confirmed_sequence": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["H", "L", "HH", "HL", "LH", "LL", "UNKNOWN"]}
                },
                "current_unconfirmed": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "present": {"type": "boolean"},
                        "description": {"type": ["string", "null"]},
                        "type": {"type": ["string", "null"]}
                    },
                    "required": ["present", "description", "type"]
                },
                "overbought": {"type": "boolean"},
                "oversold": {"type": "boolean"},
                "divergence": {"type": ["string", "null"]},
                "confidence": {"type": "number"}
            },
            "required": ["visible","current_value","direction","trend","sequence","swing_points","confirmed_sequence","current_unconfirmed","overbought","oversold","divergence","confidence"]
        },
        "moving_average": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "visible": {"type": "boolean"},
                "type": {"type": ["string", "null"]},
                "period": {"type": ["integer", "null"]},
                "direction": {"type": ["string", "null"]},
                "price_position": {"type": ["string", "null"]},
                "cross": {"type": ["string", "null"]},
                "distance_character": {"type": ["string", "null"]},
                "confidence": {"type": "number"}
            },
            "required": ["visible","type","period","direction","price_position","cross","distance_character","confidence"]
        },
        "visual_behavior": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "trend": {"type": ["string", "null"]},
                "structure": {"type": ["string", "null"]},
                "higher_highs": {"type": ["boolean", "null"]},
                "higher_lows": {"type": ["boolean", "null"]},
                "lower_highs": {"type": ["boolean", "null"]},
                "lower_lows": {"type": ["boolean", "null"]},
                "channel": {"type": ["string", "null"]},
                "range_or_consolidation": {"type": ["boolean", "null"]},
                "breakout": {"type": ["boolean", "null"]},
                "breakdown": {"type": ["boolean", "null"]},
                "pullback_present": {"type": ["boolean", "null"]},
                "reversal_present": {"type": ["boolean", "null"]},
                "momentum": {"type": ["string", "null"]},
                "latest_movement": {"type": ["string", "null"]},
                "volume_behavior": {"type": ["string", "null"]},
            },
            "required": [
                "trend", "structure", "higher_highs", "higher_lows",
                "lower_highs", "lower_lows", "channel",
                "range_or_consolidation", "breakout", "breakdown",
                "pullback_present", "reversal_present", "momentum",
                "latest_movement", "volume_behavior"
            ],
        },
        "levels": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "support_visible": {"type": "boolean"},
                "resistance_visible": {"type": "boolean"},
                "support_count": {"type": "integer"},
                "resistance_count": {"type": "integer"},
            },
            "required": [
                "support_visible", "resistance_visible",
                "support_count", "resistance_count"
            ],
        },
        "patterns": {
            "type": "array",
            "items": {"type": "string"},
        },
        "price_rsi_relationship": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "relationship": {"type": ["string", "null"]},
                "price_sequence": {"type": "array", "items": {"type": "string"}},
                "rsi_sequence": {"type": "array", "items": {"type": "string"}},
                "divergence_confirmed": {"type": "boolean"},
                "confidence": {"type": "number"}
            },
            "required": ["relationship", "price_sequence", "rsi_sequence", "divergence_confirmed", "confidence"]
        },
        "movement_signature_visual": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "direction": {"type": ["string", "null"]},
                "relative_strength": {"type": ["string", "null"]},
                "movement_shape": {"type": ["string", "null"]},
                "pullback_depth_relative": {"type": ["string", "null"]},
                "acceleration": {"type": ["string", "null"]},
                "duration_character": {"type": ["string", "null"]},
            },
            "required": [
                "direction", "relative_strength", "movement_shape",
                "pullback_depth_relative", "acceleration",
                "duration_character"
            ],
        },
        "evidence": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "observed": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "derived": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "inferred": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "unknown_or_unreliable": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": [
                "observed", "derived", "inferred", "unknown_or_unreliable"
            ]
        },
        "numerical_movement": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "available": {"type": "boolean"},
                "source": {"type": ["string", "null"]},
                "entry_price": {"type": ["number", "null"]},
                "future_high": {"type": ["number", "null"]},
                "future_low": {"type": ["number", "null"]},
                "future_close": {"type": ["number", "null"]},
                "max_favorable_move_pct": {"type": ["number", "null"]},
                "max_adverse_move_pct": {"type": ["number", "null"]},
                "final_move_pct": {"type": ["number", "null"]},
                "time_to_max_move": {"type": ["string", "null"]},
                "time_to_adverse_move": {"type": ["string", "null"]}
            },
            "required": [
                "available", "source", "entry_price", "future_high",
                "future_low", "future_close", "max_favorable_move_pct",
                "max_adverse_move_pct", "final_move_pct",
                "time_to_max_move", "time_to_adverse_move"
            ]
        },
        "quality": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "overall_confidence": {"type": "number"},
                "needs_review": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": [
                "overall_confidence", "needs_review", "reason"
            ],
        },
    },
    "required": [
        "identity", "time", "chart", "price", "price_structure", "price_action", "rsi", "moving_average", "visual_behavior", "levels", "patterns",
        "movement_signature_visual", "evidence", "price_rsi_relationship", "numerical_movement", "quality"
    ],
}


VISION_INSTRUCTIONS = r"""
You are the visual chart teacher for TinyMarketLLM.

Analyze ONLY:
1. PRICE ACTION / MARKET STRUCTURE
2. RSI
3. MOVING AVERAGES
4. VISUAL PRICE MOVEMENT

Ignore completely:
- MACD
- VWAP
- Bollinger Bands
- Volume
- Any other indicator

PRICE STRUCTURE:
- Identify visible swing highs (H) and swing lows (L).
- Classify each swing relative to the previous comparable swing:
  HH = higher high
  HL = higher low
  LH = lower high
  LL = lower low
- Return the ordered swing sequence in structured form.
- Separate confirmed swing classifications from the current, still-forming movement.
- Do not label the latest high/low as HH/HL/LH/LL unless the swing is visually confirmed.
- Put the latest unfinished movement in current_unconfirmed.
- If a level is read from a printed price label/scale with confidence, mark exact=true;
  otherwise use an approximate level and exact=false.
- Never invent exact OHLC.

RSI:
- Only analyze RSI if an RSI panel/line is visibly present.
- Capture current value only if readable.
- Identify RSI H/L and HH/HL/LH/LL structure.
- Separate confirmed RSI swings from the current, still-forming RSI movement.
- Capture overbought/oversold.
- Compare price swing structure against RSI swing structure.
- Only call divergence confirmed when the relationship is visually supported.

MOVING AVERAGE:
- Only analyze MA if visible.
- Capture type/period only when identifiable.
- Capture direction, price above/below, cross, and relative distance.
- Never invent a period.

MOVEMENT:
- Describe direction, relative strength, shape, acceleration/deceleration,
  duration, pullback depth, and breakout extension.
- Do not convert image geometry into fake percentages.

IMPORTANT:
- Charts without symbol, price, timeframe, RSI, or MA are still valid.
- Missing information must remain null/unknown.
- Keep observed, derived, inferred, and unknown information separate.
- Do not provide BUY, SELL, targets, stop-loss, or investment advice.
- The purpose is to create structured training memory so a future tiny model can learn
  recurring relationships between price structure, RSI, MA and subsequent price movement.
"""

def _image_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def analyze_chart(image_path: Path) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    env_file = project_root / ".env"
    load_dotenv(dotenv_path=env_file, override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in PowerShell before scanning."
        )

    model = os.getenv("TINYMARKETLLM_VISION_MODEL", "gpt-5-mini")
    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        instructions=VISION_INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Analyze this chart image for TinyMarketLLM. "
                            "Return only the requested structured analysis."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(image_path),
                        "detail": "high",
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "chart_analysis",
                "description": "Structured visual market-behavior analysis.",
                "strict": True,
                "schema": ANALYSIS_SCHEMA,
            }
        },
    )

    if not response.output_text:
        raise RuntimeError("Vision model returned no structured output.")

    try:
        result = json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Vision model returned invalid JSON: {exc}"
        ) from exc

    result["_teacher"] = {
        "provider": "openai",
        "model": model,
        "analysis_type": "visual_chart_behavior",
    }

    return result


