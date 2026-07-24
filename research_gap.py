"""
research_gap.py — Identify under-studied regions of the training data.

Pure analysis over the training data already collected (no new experiments,
no model): rare category values, sparse numeric windows, and untested
category combinations — each reported as a concrete, actionable gap a
researcher can read and know exactly what experiment would fill, rather than
a vague "explore more."
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def rare_categories(df: pd.DataFrame, cat_choices: dict,
                    min_frac: float = 0.05, min_count: int = 3) -> list:
    """Categorical knob values seen fewer than ``min_frac`` (or
    ``min_count``, whichever is looser) of the time — under-studied
    reagent/biomass/atmosphere choices. Sorted rarest first.
    """
    gaps = []
    n = len(df)
    if n == 0:
        return gaps
    for col, choices in cat_choices.items():
        if col not in df.columns:
            continue
        counts = df[col].astype(str).value_counts()
        threshold = max(min_count, min_frac * n)
        for choice in choices:
            c = int(counts.get(str(choice), 0))
            if 0 < c < threshold:
                gaps.append({
                    "type": "rare_category", "column": col, "value": choice,
                    "count": c, "total": n,
                    "description": f"'{choice}' for {col} appears in only {c}/{n} "
                                   "experiment(s) — under-studied.",
                })
    return sorted(gaps, key=lambda g: g["count"])


def numeric_windows(df: pd.DataFrame, numeric_cols, n_bins: int = 8,
                    min_frac: float = 0.03) -> list:
    """Sub-ranges of a numeric knob's observed span with few/no
    observations — a "missing temperature window"-style gap. Sorted
    emptiest first.
    """
    gaps = []
    n = len(df)
    if n == 0:
        return gaps
    for col in numeric_cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) < 5:
            continue
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            continue
        edges = np.linspace(lo, hi, n_bins + 1)
        counts, _ = np.histogram(vals, bins=edges)
        threshold = max(1, min_frac * len(vals))
        for i, c in enumerate(counts):
            if c < threshold:
                gaps.append({
                    "type": "numeric_gap", "column": col,
                    "range": (round(float(edges[i]), 3), round(float(edges[i + 1]), 3)),
                    "count": int(c), "total": int(len(vals)),
                    "description": f"{col} in [{edges[i]:.3g}, {edges[i + 1]:.3g}] has "
                                   f"only {int(c)}/{len(vals)} experiment(s) — sparse window.",
                })
    return sorted(gaps, key=lambda g: g["count"])


def untested_combinations(df: pd.DataFrame, cat_choices: dict, max_columns: int = 4,
                          min_choices: int = 2, max_choices: int = 8) -> list:
    """Category PAIRS across two categorical knobs never observed together.
    Limited to a handful of columns with a manageable number of choices
    each — the pair count grows fast, and this is meant to surface a few
    concrete "X with Y has never been tried" ideas, not an exhaustive matrix.
    """
    gaps = []
    cols = [c for c, choices in cat_choices.items()
           if min_choices <= len(choices) <= max_choices][:max_columns]
    for col_a, col_b in itertools.combinations(cols, 2):
        if col_a not in df.columns or col_b not in df.columns:
            continue
        observed = set(zip(df[col_a].astype(str), df[col_b].astype(str)))
        for a, b in itertools.product(cat_choices[col_a], cat_choices[col_b]):
            if (str(a), str(b)) not in observed:
                gaps.append({
                    "type": "untested_combination", "column_a": col_a, "value_a": a,
                    "column_b": col_b, "value_b": b,
                    "description": f"{col_a}='{a}' with {col_b}='{b}' has never been tried.",
                })
    return gaps


def detect_gaps(df: pd.DataFrame, numeric_cols, cat_choices: dict, max_gaps: int = 15) -> list:
    """All gap types, priority-ordered (rare categories and sparse numeric
    windows — concrete, single-variable gaps — before untested combinations,
    which can be numerous and less specific) and capped to ``max_gaps``.
    Returns a flat list of gap dicts, each with a ready-to-display
    ``"description"``. Best-effort: never raises — an analysis failure on
    one gap type just yields fewer gaps, not an error.
    """
    gaps = []
    for fn in (lambda: rare_categories(df, cat_choices),
              lambda: numeric_windows(df, numeric_cols),
              lambda: untested_combinations(df, cat_choices)):
        try:
            gaps.extend(fn())
        except Exception:  # noqa: BLE001
            continue
    order = {"rare_category": 0, "numeric_gap": 1, "untested_combination": 2}
    gaps.sort(key=lambda g: (order.get(g["type"], 9), g.get("count", 0)))
    return gaps[:max_gaps]
