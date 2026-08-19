from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any


def _is_reasonable_year(value: str | None) -> bool:
    if not value:
        return False
    try:
        year = int(value[:4])
    except (ValueError, TypeError):
        return False
    return 1900 <= year <= datetime.now().year


def validate_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Create a trusted/normalized view without deleting the raw teacher output.

    The teacher output is retained verbatim by the caller. This function only
    marks suspicious or unsupported information as unknown.
    """
    trusted = deepcopy(analysis)

    evidence = trusted.setdefault("evidence", {})
    evidence.setdefault("observed", [])
    evidence.setdefault("derived", [])
    evidence.setdefault("inferred", [])
    evidence.setdefault("unknown_or_unreliable", [])

    numerical = trusted.setdefault("numerical_movement", {})
    numerical.setdefault("available", False)

    # Numerical movement is never inferred from image geometry.
    if not numerical.get("available"):
        for key in (
            "entry_price",
            "future_high",
            "future_low",
            "future_close",
            "max_favorable_move_pct",
            "max_adverse_move_pct",
            "final_move_pct",
            "time_to_max_move",
            "time_to_adverse_move",
        ):
            numerical[key] = None
        numerical["source"] = None

    # Guard against future-looking visible dates. The chart may be a projection,
    # but we do not want the training record to silently treat them as historical.
    time = trusted.setdefault("time", {})
    for key in ("visible_start", "visible_end"):
        value = time.get(key)
        if value and not _is_reasonable_year(str(value)):
            time[key] = None
            evidence["unknown_or_unreliable"].append(
                f"{key} was not accepted as a reliable calendar year."
            )

    current_year = datetime.now().year
    for key in ("visible_start", "visible_end"):
        value = time.get(key)
        if value:
            try:
                year = int(str(value)[:4])
                if year > current_year:
                    time[key] = None
                    evidence["unknown_or_unreliable"].append(
                        f"{key} appears to be a future year and was cleared."
                    )
            except ValueError:
                pass

    # If exact price values were not reliably read, do not retain them as facts.
    price = trusted.setdefault("price", {})
    if not price.get("values_reliably_read", False):
        for key in ("open", "high", "low", "close"):
            price[key] = None

    quality = trusted.setdefault("quality", {})
    issues = evidence["unknown_or_unreliable"]

    if issues:
        quality["needs_review"] = True
        reason = quality.get("reason", "")
        extra = " Validation flagged: " + " ".join(dict.fromkeys(issues))
        quality["reason"] = (reason + extra).strip()

    return trusted
