"""
sustainability.py — A relative "Green Score" (0-100) for one recipe.

Explicitly NOT a certified life-cycle assessment (LCA): a real LCA needs
energy-per-kg, transport, water use, and end-of-life data this app has no
way to know. What IS available, and honest to score from, are three signals
already computed elsewhere in the app, all traceable to either real
user-entered data or the recipe itself — nothing here is invented:

  * reagent hazard class / corrosive flag — from cost_model.py's
    CostDatabase, which is itself only ever populated by the user (see
    cost_model.py's own docstring). A reagent with no hazard entered is
    listed separately as "not scored", never assumed safe.
  * process complexity — pyrolysis-stage and optional-step counts, the
    same objectively-countable structure planner.feasibility_score already
    uses (more stages/steps = more reheating energy and reagent use).
  * process temperature — expressed as a PERCENTILE within this dataset's
    own observed range (not an absolute "high" threshold this app has no
    basis to assert), so the deduction is always relative to what this
    dataset actually contains.

The score and its per-item deductions are a transparent heuristic for
COMPARING recipes against each other, not an absolute certification.
"""

from __future__ import annotations

import re

import numpy as np

import constraint_engine
import cost_model

HAZARD_POINTS = {"None": 0, "Low": 5, "Moderate": 15, "High": 30, "Severe": 50}
CORROSIVE_POINTS = 10
STAGE_POINTS_PER_EXTRA = 8
STEP_POINTS_EACH = 4
TEMPERATURE_POINTS = {90: 15, 75: 8}   # percentile threshold -> points lost

_TEMPERATURE_PATTERN = re.compile(r"temp|pyro", re.I)


def grade_for(score: float) -> str:
    """Letter grade purely for display — a coarse banding of the same
    heuristic score, not a separate judgement."""
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 50:
        return "C"
    if score >= 30:
        return "D"
    return "F"


def estimate_temperature_percentile(recipe_dict, X_raw, numeric_cols) -> float | None:
    """Mean percentile rank (0-100) of this recipe's temperature-like
    knob value(s) within the dataset's own observed distribution for that
    knob — None if no temperature-like numeric knob is present. Relative
    to THIS dataset only; never an absolute degree-C threshold."""
    percentiles = []
    for col in numeric_cols:
        if not _TEMPERATURE_PATTERN.search(str(col)) or col not in recipe_dict:
            continue
        try:
            v = float(recipe_dict[col])
        except (TypeError, ValueError):
            continue
        series = X_raw[col].dropna().values
        if len(series) == 0:
            continue
        percentiles.append(100.0 * float((series <= v).mean()))
    if not percentiles:
        return None
    return float(np.mean(percentiles))


def green_score(recipe_dict, chemical_names, *, temperature_percentile=None,
                cost_engine=None) -> dict:
    """0-100 relative sustainability score for one recipe. ``chemical_names``
    is the list of reagent names used (e.g. from a candidate's
    ``chemical_recommendations``); ``temperature_percentile`` is typically
    ``estimate_temperature_percentile(...)``'s result. ``cost_engine``
    defaults to ``cost_model.ENGINE`` (the user's saved cost/hazard data).

    Returns ``{"score", "grade", "deductions": [{"reason","points"}, ...],
    "unscored_reagents", "hazard"}``.
    """
    cost_engine = cost_engine or cost_model.ENGINE
    score = 100.0
    deductions = []

    hazard = cost_engine.recipe_hazard(chemical_names)
    pts = HAZARD_POINTS.get(hazard["max_hazard"], 0)
    if pts:
        score -= pts
        deductions.append({
            "reason": f"Most hazardous reagent used is rated '{hazard['max_hazard']}'",
            "points": pts})
    if hazard["corrosive"]:
        score -= CORROSIVE_POINTS
        deductions.append({"reason": "Uses a reagent flagged corrosive",
                           "points": CORROSIVE_POINTS})

    n_stages = constraint_engine.count_stages(recipe_dict)
    n_steps = constraint_engine.count_steps(recipe_dict)
    if n_stages > 1:
        pts = (n_stages - 1) * STAGE_POINTS_PER_EXTRA
        score -= pts
        deductions.append({
            "reason": f"{n_stages} pyrolysis stages (extra reheating energy vs. a single stage)",
            "points": pts})
    if n_steps > 0:
        pts = n_steps * STEP_POINTS_EACH
        score -= pts
        deductions.append({
            "reason": f"{n_steps} optional processing step(s) (extra reagent/energy use)",
            "points": pts})

    if temperature_percentile is not None:
        pts = 0
        for threshold, p in sorted(TEMPERATURE_POINTS.items(), reverse=True):
            if temperature_percentile >= threshold:
                pts = p
                break
        if pts:
            score -= pts
            deductions.append({
                "reason": f"Processing temperature is in the top "
                         f"{100 - temperature_percentile:.0f}% of conditions "
                         "seen in this dataset (energy-intensive)",
                "points": pts})

    score = max(0.0, min(100.0, score))
    return {
        "score": round(score, 1), "grade": grade_for(score),
        "deductions": deductions, "unscored_reagents": hazard["unknown"],
        "hazard": hazard,
    }
