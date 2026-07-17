"""
units.py  —  Auto-standardize units of measurement, one spreadsheet column at a time.
=====================================================================================

Lab spreadsheets record the same physical quantity in inconsistent units: a
"Time" column mixing ``"30 min"`` and ``"2 h"``, or a header that names the unit
(``"Py.3 temp. (oC)"``) while the cells are bare numbers.  :func:`standardize_units`
detects each column's physical dimension and rewrites every numeric cell into a
single canonical unit, returning clean float values the model can consume.

It is deliberately CONSERVATIVE — a column is only touched when its unit signal
is unambiguous:

  * cells carry explicit units (``"30 min"``, ``"900 oC"``, ``"5 mg"``), or
  * the header names a recognized unit in brackets (``"Time (h)"``, ``"(K)"``).

Columns with no unit information, columns that are mostly free text, and columns
whose cells span more than one physical dimension are left EXACTLY as they were.
Empty / "not-done" tokens (``""``, ``"--"``, ``"none"`` …) are preserved verbatim
so the downstream missing-data logic still works.  Nothing here ever raises.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd


# =============================================================================
# Unit registry
# =============================================================================
# Each physical DIMENSION declares a default canonical unit plus the aliases
# that map onto it.  A linear unit's value is the multiplier that converts it
# INTO the canonical unit (canonical itself is 1.0).  Temperature is affine, so
# its non-canonical aliases store a callable ``source_value -> °C`` instead.
def _c_from_k(v: float) -> float:
    return v - 273.15


def _c_from_f(v: float) -> float:
    return (v - 32.0) * 5.0 / 9.0


DIMENSIONS: dict[str, dict] = {
    "temperature": {
        "canonical": "°C",
        "units": {
            "°c": 1.0, "c": 1.0, "degc": 1.0, "celsius": 1.0,
            "k": _c_from_k, "kelvin": _c_from_k,
            "°f": _c_from_f, "f": _c_from_f, "degf": _c_from_f, "fahrenheit": _c_from_f,
        },
    },
    "time": {
        "canonical": "min",
        "units": {
            "s": 1 / 60, "sec": 1 / 60, "secs": 1 / 60, "second": 1 / 60, "seconds": 1 / 60,
            "min": 1.0, "mins": 1.0, "minute": 1.0, "minutes": 1.0,
            "h": 60.0, "hr": 60.0, "hrs": 60.0, "hour": 60.0, "hours": 60.0,
            "d": 1440.0, "day": 1440.0, "days": 1440.0,
        },
    },
    "heating_rate": {
        "canonical": "°C/min",
        "units": {
            "°c/min": 1.0, "c/min": 1.0, "k/min": 1.0, "degc/min": 1.0,
            "°c/h": 1 / 60, "c/h": 1 / 60, "k/h": 1 / 60, "c/hr": 1 / 60,
            "°c/s": 60.0, "c/s": 60.0, "k/s": 60.0,
        },
    },
    "mass": {
        "canonical": "g",
        "units": {
            "ng": 1e-9, "µg": 1e-6, "mcg": 1e-6,
            "mg": 1e-3, "g": 1.0, "gm": 1.0, "gram": 1.0, "grams": 1.0,
            "kg": 1e3,
        },
    },
    "length": {
        "canonical": "nm",
        "units": {
            "å": 0.1, "ang": 0.1, "angstrom": 0.1,
            "nm": 1.0, "µm": 1e3, "micron": 1e3, "microns": 1e3,
            "mm": 1e6, "cm": 1e7, "m": 1e9,
        },
    },
    "pressure": {
        "canonical": "bar",
        "units": {
            "pa": 1e-5, "kpa": 1e-2, "hpa": 1e-3, "mpa": 10.0,
            "bar": 1.0, "mbar": 1e-3, "atm": 1.01325, "psi": 0.0689476, "torr": 0.00133322,
        },
    },
    "surface_area": {
        "canonical": "m²/g",
        "units": {"m²/g": 1.0, "cm²/g": 1e-4},
    },
    "flow": {
        "canonical": "mL/min",
        "units": {
            "ml/min": 1.0, "sccm": 1.0, "cm³/min": 1.0, "cc/min": 1.0,
            "l/min": 1000.0, "l/h": 1000 / 60, "ml/h": 1 / 60,
        },
    },
}

CANONICAL_UNITS = {dim: spec["canonical"] for dim, spec in DIMENSIONS.items()}

# Empty-ish tokens that must survive standardization untouched (mirrors the
# app's missing-data vocabulary; kept local so this module stays GUI-free).
_BLANK_TOKENS = {
    "", "-", "--", "---", "----", "—", "–", "none", "nil", "nan", "na", "n/a",
    "n.a.", "missing", "unknown", "unspecified", "tbd", "?", "..", ".", "x", "input",
}


def _norm_unit(u: str) -> str:
    """Canonicalize a raw unit token: lowercase, unify micro/degree/superscripts."""
    s = str(u).strip().lower()
    s = s.replace("µ", "u").replace("μ", "u")   # micro sign & Greek mu -> 'u'
    s = s.replace("º", "°")                     # masculine ordinal -> degree
    s = s.replace("²", "2").replace("³", "3")   # superscript 2/3 -> 2/3
    s = s.replace(" ", "")
    if s in ("oc", "of"):        # 'o' typed as the degree symbol (oC, oF)
        s = s[1:]
    return s


# Reverse lookup: normalized alias -> (dimension, conversion factor|callable).
_LOOKUP: dict[str, tuple[str, object]] = {}
for _dim, _spec in DIMENSIONS.items():
    for _alias, _conv in _spec["units"].items():
        _LOOKUP[_norm_unit(_alias)] = (_dim, _conv)

# Normalized canonical token per dimension (always a factor of 1.0 in _LOOKUP).
_CANON_NORM = {dim: _norm_unit(spec["canonical"]) for dim, spec in DIMENSIONS.items()}

# A leading number, then everything after it is the unit candidate (validated by
# an exact registry lookup, so unrecognized trailers fall back to "bare number").
_NUM_RE = re.compile(r"^\s*([-+]?[\d,]*\.?\d+)\s*(.*)$")

# Single-letter tokens too easily a variable symbol (not a unit) to trust from a
# header alone — both are 'length' with huge scale factors, so a false positive
# would be badly wrong. Explicit per-cell units (e.g. "5 m") still convert.
_AMBIGUOUS_HEADER_UNITS = {"m", "a"}


def _is_blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return str(v).strip().lower() in _BLANK_TOKENS


def _split_num_unit(v):
    """('30 min') -> (30.0, 'min'); ('1,200') -> (1200.0, None); ('Ar') -> (None, None)."""
    m = _NUM_RE.match(str(v).strip())
    if not m:
        return None, None
    try:
        num = float(m.group(1).replace(",", ""))
    except ValueError:
        return None, None
    rest = m.group(2).strip()
    return num, (rest or None)


def _header_unit(name: str):
    """Recognized unit named in a column header, e.g. 'Time (h)' -> ('time', 'h').

    Only accepts a bracketed token or a clear trailing token, and only when it
    maps to a known unit.  Bare single-letter tokens (e.g. '(m)') are rejected as
    too ambiguous for a header — explicit per-cell units still handle those.
    """
    m = re.search(r"[\(\[]\s*([^)\]]+?)\s*[\)\]]", name)
    token = m.group(1) if m else None
    if token is None:
        m2 = re.search(r"[,\s]\s*([^\s,]+)\s*$", name)
        token = m2.group(1) if m2 else None
    if not token:
        return None, None
    norm = _norm_unit(token)
    hit = _LOOKUP.get(norm)
    if hit is None:
        return None, None
    if norm in _AMBIGUOUS_HEADER_UNITS:     # header-safety: skip ambiguous symbols
        return None, None
    return hit[0], norm


def _convert(value: float, dim: str, src_norm: str, target_norm: str) -> float:
    """Convert ``value`` from ``src_norm`` to ``target_norm`` within one dimension."""
    _, conv = _LOOKUP[src_norm]
    base = conv(value) if callable(conv) else value * conv     # -> default canonical
    tconv = DIMENSIONS[dim]["units"][target_norm]
    if callable(tconv):        # affine target unsupported -> leave at default canonical
        return base
    return base / tconv


def _resolve_target(dim: str, canonical: dict) -> tuple[str, str]:
    """Return (target_norm, human_label) honoring a caller override when valid."""
    override = canonical.get(dim)
    if override is not None:
        ov = _norm_unit(override)
        conv = DIMENSIONS[dim]["units"].get(ov)
        if conv is not None and not callable(conv):
            return ov, override
    return _CANON_NORM[dim], DIMENSIONS[dim]["canonical"]


def _standardize_column(series: pd.Series, colname: str, canonical: dict):
    """Return (new_series, note) if the column was standardized, else None."""
    vals = list(series)
    cells = []                      # (kind, number, dim, unit_norm) per cell
    n_nonblank = n_numeric = 0
    dim_votes: Counter = Counter()
    unit_votes: dict[str, Counter] = {}

    for v in vals:
        if _is_blank(v):
            cells.append(("blank", None, None, None))
            continue
        n_nonblank += 1
        num, unit = _split_num_unit(v)
        if num is None:
            cells.append(("text", None, None, None))
            continue
        n_numeric += 1
        if unit:
            un = _norm_unit(unit)
            hit = _LOOKUP.get(un)
            if hit:
                dim_votes[hit[0]] += 1
                unit_votes.setdefault(hit[0], Counter())[un] += 1
                cells.append(("num_unit", num, hit[0], un))
                continue
        cells.append(("num_bare", num, None, None))   # plain or unrecognized-unit number

    # Guardrails: need enough numeric signal, and not a mostly-text column.
    if n_numeric < 2 or n_nonblank == 0 or n_numeric / n_nonblank < 0.7:
        return None

    # Decide the column's single physical dimension.
    dim = None
    if dim_votes:
        top, top_n = dim_votes.most_common(1)[0]
        if len(dim_votes) == 1 or top_n / sum(dim_votes.values()) >= 0.8:
            dim = top
        else:
            return None        # cells span multiple dimensions -> too risky
    hdim, hunit = _header_unit(colname)
    if dim is None:
        if hdim is None:
            return None        # no unit information anywhere -> leave untouched
        dim = hdim

    # Source unit for bare numbers: the header's (if it agrees) else the most
    # common explicit cell unit; None means "assume already canonical".
    if hdim == dim and hunit is not None:
        bare_src = hunit
    elif dim in unit_votes:
        bare_src = unit_votes[dim].most_common(1)[0][0]
    else:
        bare_src = None

    target_norm, target_label = _resolve_target(dim, canonical)

    out_vals, used_units = [], set()
    n_conv = 0
    value_changed = stripped_unit = False
    for (kind, num, cdim, un), orig in zip(cells, vals):
        if kind == "num_unit" and cdim == dim:
            conv = _convert(num, dim, un, target_norm)
            out_vals.append(conv)
            used_units.add(un); n_conv += 1; stripped_unit = True
            value_changed |= abs(conv - num) > 1e-9 * (1 + abs(num))
        elif kind == "num_bare" and bare_src is not None:
            conv = _convert(num, dim, bare_src, target_norm)
            out_vals.append(conv)
            used_units.add(bare_src); n_conv += 1
            value_changed |= abs(conv - num) > 1e-9 * (1 + abs(num))
        elif kind == "num_bare":
            out_vals.append(num)        # no known source -> keep number as-is
        else:
            out_vals.append(orig)       # blank / text / other-dimension -> untouched

    # Skip pure no-ops (e.g. a bare-number column already in the canonical unit).
    if not (value_changed or stripped_unit):
        return None

    src_desc = ", ".join(sorted(used_units)) if used_units else "?"
    note = (f"Standardized units in '{colname}' [{dim}]: "
            f"{src_desc} -> {target_label} ({n_conv} value(s)).")
    return pd.Series(out_vals, index=series.index), note


def standardize_units(df: pd.DataFrame, canonical: dict | None = None,
                      protected=(), notes: list | None = None) -> pd.DataFrame:
    """Auto-standardize measurement units column by column.

    Parameters
    ----------
    df : DataFrame
        Raw (already messy-column-parsed) data.
    canonical : {dimension: unit}, optional
        Override the target unit for a dimension, e.g. ``{"time": "h"}``.  Only
        linear targets are honored; an unknown or affine (temperature) override
        falls back to the dimension's default (see :data:`CANONICAL_UNITS`).
    protected : iterable of column names
        Columns never touched (targets, ids, user-typed overrides).
    notes : list, optional
        Human-readable messages describing each conversion are appended here.

    Returns
    -------
    DataFrame
        A copy with recognized unit columns rewritten to a single canonical
        unit (as floats).  Untouched columns are returned unchanged.
    """
    out = df.copy()
    canonical = dict(canonical or {})
    protected = {str(c) for c in protected}
    for col in list(out.columns):
        if str(col) in protected:
            continue
        try:
            result = _standardize_column(out[col], str(col), canonical)
        except Exception:      # noqa: BLE001 - standardization must never break a load
            result = None
        if result is None:
            continue
        new_series, note = result
        out[col] = new_series
        if notes is not None:
            notes.append(note)
    return out
