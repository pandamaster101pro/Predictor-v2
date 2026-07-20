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
# Chemistry registry for percent concentration -> molarity
# =============================================================================
# Values are intentionally small and explicit.  A percent can only become mol/L
# when we know the solute's molar mass, and density is also required for v/v or
# w/w conversions.  Unknown solutes/bases are left untouched.
CHEMICALS: dict[str, dict] = {
    "h2so4": {
        "label": "H2SO4",
        "molar_mass_g_mol": 98.079,
        "pure_density_g_ml": 1.8305,
        "solution_density_g_ml": 1.84,
        "aliases": ["sulfuricacid", "sulphuricacid"],
    },
    "h2o2": {
        "label": "H2O2",
        "molar_mass_g_mol": 34.0147,
        "pure_density_g_ml": 1.45,
        "solution_density_g_ml": 1.13,
        "aliases": ["hydrogenperoxide"],
    },
    "hcl": {
        "label": "HCl",
        "molar_mass_g_mol": 36.4609,
        "solution_density_g_ml": 1.19,
        "aliases": ["hydrochloricacid"],
    },
    "hno3": {
        "label": "HNO3",
        "molar_mass_g_mol": 63.012,
        "solution_density_g_ml": 1.41,
        "aliases": ["nitricacid"],
    },
    "koh": {
        "label": "KOH",
        "molar_mass_g_mol": 56.1056,
        "aliases": ["potassiumhydroxide"],
    },
    "naoh": {
        "label": "NaOH",
        "molar_mass_g_mol": 39.997,
        "aliases": ["sodiumhydroxide"],
    },
    "nahco3": {
        "label": "NaHCO3",
        "molar_mass_g_mol": 84.0066,
        "aliases": ["sodiumbicarbonate", "sodiumhydrogencarbonate"],
    },
    "na2co3": {
        "label": "Na2CO3",
        "molar_mass_g_mol": 105.9888,
        "aliases": ["sodiumcarbonate"],
    },
    "li2co3": {
        "label": "Li2CO3",
        "molar_mass_g_mol": 73.891,
        # Common typo/abbreviation in the existing spreadsheet.
        "aliases": ["lico3", "lithiumcarbonate"],
    },
}


def _norm_chemical_text(text: str) -> str:
    """Normalize a formula/name enough for conservative substring matching."""
    s = str(text).lower()
    s = s.replace("⋅", "").replace("·", "").replace(".", "")
    s = s.replace(" ", "").replace("-", "").replace("_", "")
    return re.sub(r"[^a-z0-9]", "", s)


_CHEM_LOOKUP: dict[str, dict] = {}
for _formula, _info in CHEMICALS.items():
    _CHEM_LOOKUP[_norm_chemical_text(_formula)] = _info
    for _alias in _info.get("aliases", []):
        _CHEM_LOOKUP[_norm_chemical_text(_alias)] = _info


def _find_chemical(*texts):
    haystack = " ".join(str(t) for t in texts if t is not None)
    norm = _norm_chemical_text(haystack)
    hits = []
    for key, info in _CHEM_LOOKUP.items():
        if key and key in norm:
            hits.append((len(key), key, info))
    if not hits:
        return None
    # Longest match wins so Na2CO3 beats shorter accidental pieces.
    return sorted(hits, reverse=True)[0][2]


_MOLAR_RE = re.compile(
    r"^\s*([-+]?[\d,]*\.?\d+)\s*"
    r"(?:M(?=\s|[A-Z0-9(]|$)|[Mm]olar\b|[Mm]ol\s*/\s*[Ll]\b)"
    r"\s*([A-Za-z0-9()⋅·.\-\s]*)",
)
_PERCENT_RE = re.compile(
    r"^\s*([-+]?[\d,]*\.?\d+)\s*"
    r"(?P<prefix>w\s*/\s*v|v\s*/\s*v|w\s*/\s*w|vol|wt|v|w)?\s*"
    r"%\s*"
    r"(?P<suffix>w\s*/\s*v|v\s*/\s*v|w\s*/\s*w|vol|wt|v|w)?\s*"
    r"(?P<rest>.*)$",
    re.I,
)


def _percent_basis(prefix, suffix, column_name):
    raw = (prefix or suffix or "").strip().lower().replace(" ", "")
    if not raw:
        # Let an explicit header such as "H2SO4 concentration (%V)" provide the
        # basis for bare cell values like "3.2".
        lower = str(column_name).lower().replace(" ", "")
        if "%v" in lower or "v%" in lower or "v/v" in lower or "vol%" in lower:
            raw = "v"
        elif "%w/v" in lower or "w/v" in lower:
            raw = "w/v"
        elif "wt%" in lower or "%wt" in lower or "w/w" in lower:
            raw = "wt"
    if raw in {"v", "vol", "v/v"}:
        return "v/v"
    if raw in {"w/v"}:
        return "w/v"
    if raw in {"w", "wt", "w/w"}:
        return "w/w"
    return None


def _molarity_from_percent(pct: float, basis: str, chem: dict):
    mm = chem.get("molar_mass_g_mol")
    if not mm:
        return None
    if basis == "w/v":
        return 10.0 * pct / mm
    if basis == "v/v":
        density = chem.get("pure_density_g_ml")
        return None if density is None else 10.0 * pct * density / mm
    if basis == "w/w":
        density = chem.get("solution_density_g_ml")
        return None if density is None else 10.0 * pct * density / mm
    return None


def parse_concentration_to_molarity(value, column_name: str = ""):
    """
    Convert concentration text to molarity when chemistry is unambiguous.

    Supports already-molar values (``1M KOH``), explicit percent v/v
    (``1.25v% H2SO4``), percent w/v, and percent w/w when the registry has a
    solution density.  Returns ``(molarity, solute_label, basis)`` or ``None``.
    """
    if _is_blank(value):
        return None
    text = str(value).strip()

    m = _MOLAR_RE.match(text)
    if m:
        try:
            molarity = float(m.group(1).replace(",", ""))
        except ValueError:
            return None
        chem = _find_chemical(m.group(2), column_name)
        label = chem["label"] if chem else "M"
        return molarity, label, "M"

    m = _PERCENT_RE.match(text)
    if not m:
        return None
    try:
        pct = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    # Values such as "30mole%" are ratios/doping, not solution concentration.
    if "mole" in text.lower() or "mol%" in text.lower():
        return None

    basis = _percent_basis(m.group("prefix"), m.group("suffix"), column_name)
    if basis is None:
        return None
    chem = _find_chemical(m.group("rest"), column_name)
    if chem is None:
        return None

    molarity = _molarity_from_percent(pct, basis, chem)
    if molarity is None:
        return None
    return molarity, chem["label"], basis


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


def _header_percent_context(colname: str):
    basis = _percent_basis(None, None, colname)
    chem = _find_chemical(colname)
    if basis is None or chem is None:
        return None, None
    return basis, chem


def _standardize_concentration_column(series: pd.Series, colname: str):
    """Return (new_series, note) for concentration -> M, else None."""
    vals = list(series)
    header_basis, header_chem = _header_percent_context(colname)
    out_vals, bases, solutes = [], set(), set()
    n_nonblank = n_conv = 0

    for v in vals:
        if _is_blank(v):
            out_vals.append(v)
            continue
        n_nonblank += 1

        parsed = parse_concentration_to_molarity(v, column_name=colname)
        if parsed is not None:
            molarity, solute, basis = parsed
            out_vals.append(molarity)
            bases.add(basis); solutes.add(solute); n_conv += 1
            continue

        # Header-driven case: column says "H2SO4 concentration (%V)" and cells
        # are bare numbers like 3.2.
        if header_basis is not None and header_chem is not None:
            num, unit = _split_num_unit(v)
            if num is not None and unit is None:
                molarity = _molarity_from_percent(num, header_basis, header_chem)
                if molarity is not None:
                    out_vals.append(molarity)
                    bases.add(header_basis); solutes.add(header_chem["label"]); n_conv += 1
                    continue

        out_vals.append(v)

    if n_conv == 0 or n_nonblank == 0 or n_conv / n_nonblank < 0.7:
        return None
    basis_desc = ", ".join(sorted(bases))
    solute_desc = ", ".join(sorted(solutes))
    note = (f"Standardized concentration in '{colname}': "
            f"{solute_desc} {basis_desc} -> M ({n_conv} value(s)).")
    return pd.Series(out_vals, index=series.index), note


# =============================================================================
# Parsed messy-column triples  (numeric_feature_A* + their label columns)
# =============================================================================
# The app's messy-column parser splits a cell like '3.2%V H2SO4' into a number
# plus two LABEL columns:  numeric_feature_A*=3.2, group_label_B*='V',
# text_modifier_C*='h2so4'.  When those labels identify a percent basis and a
# registry chemical, the number can be rewritten as molarity and the label
# columns removed — their information then lives entirely in the number.
_TRIPLE_A_RE = re.compile(r"^numeric_feature_A(\d*)$")

# Percent-basis codes as they appear in the parser's "code" label (the token
# that followed the number in the raw cell, upper-cased by the parser).
_BASIS_CODES = {"v": "v/v", "vv": "v/v", "vol": "v/v",
                "wv": "w/v",
                "w": "w/w", "wt": "w/w", "ww": "w/w"}


def molarity_from_parts(amount, code, text):
    """Molarity from one parsed messy-cell triple, or None if ambiguous.

    ``code``/``text`` are the label parts the messy-column parser emits:
    '3.2%V H2SO4' -> (3.2, 'V', 'h2so4') and '1M KOH' -> (1.0, 'M', 'koh').
    A conversion needs a registry chemical AND either an explicit molar 'M'
    code or a recognized percent basis, so doping codes like '15% Mn' and
    plain amounts never convert.
    """
    code_n = str(code or "").strip().lower()
    text_n = str(text or "").strip().lower()
    # '0.2 w/v% NaHCO3' parses as code 'W' + text '/v nahco3' — reunite them.
    m = re.match(r"^/\s*([a-z])\b\s*(.*)$", text_n)
    if m:
        code_n, text_n = code_n + m.group(1), m.group(2)
    chem = _find_chemical(text_n)
    if chem is None:
        return None
    if code_n == "m":                     # '1M KOH' — already molar
        try:
            return float(amount)
        except (TypeError, ValueError):
            return None
    basis = _BASIS_CODES.get(code_n)
    if basis is None:
        return None
    try:
        return _molarity_from_percent(float(amount), basis, chem)
    except (TypeError, ValueError):
        return None


def standardize_parsed_mixed(df: pd.DataFrame, notes: list | None = None,
                             labels: dict | None = None, protected=()) -> pd.DataFrame:
    """Convert parsed messy-column triples to molarity and DROP their labels.

    Runs after :func:`standardize_units`.  For every triple whose label columns
    give an unambiguous basis + chemical for at least 70% of the informative
    rows (and at least 2 rows), the numeric column is rewritten as mol/L, the
    solute identity is kept as one categorical ``solute_label_D*`` column
    (NaOH/KOH/... ; 'None' for not-done rows, 'Missing' when unknown), and the
    raw label columns are removed — so '1M NaOH' round-trips as 1.0 + 'NaOH'.
    'Not-done' rows (labels blank/'None') keep their 0.0.  Triples that don't
    look like concentrations — dopants, plain amounts, unknown chemicals — are
    left untouched, labels included.

    ``labels`` optionally maps internal column names to display names for the
    ``notes`` messages.
    """
    out = df.copy()
    labels = dict(labels or {})
    protected = {str(c) for c in protected}
    for a_col in list(out.columns):
        m = _TRIPLE_A_RE.match(str(a_col))
        if m is None or str(a_col) in protected:
            continue
        b_col = f"group_label_B{m.group(1)}"
        c_col = f"text_modifier_C{m.group(1)}"
        d_col = f"solute_label_D{m.group(1)}"
        if b_col not in out.columns or c_col not in out.columns:
            continue

        new_vals, new_solutes, n_informative, n_conv = [], [], 0, 0
        solutes = set()
        for amt, code, txt in zip(out[a_col], out[b_col], out[c_col]):
            code_blank = _is_blank(code) or str(code).strip().lower() == "none"
            txt_blank = _is_blank(txt) or str(txt).strip().lower() == "none"
            if code_blank and txt_blank:
                new_vals.append(amt)          # 'not done' -> stays 0.0
                new_solutes.append("None")
                continue
            n_informative += 1
            mol = molarity_from_parts(amt, code, txt)
            if mol is None:
                new_vals.append(amt)
                new_solutes.append("Missing")
            else:
                new_vals.append(mol)
                n_conv += 1
                chem = _find_chemical(txt)
                label = chem["label"] if chem else "Missing"
                new_solutes.append(label)
                if chem is not None:
                    solutes.add(label)

        if n_conv < 2 or n_informative == 0 or n_conv / n_informative < 0.7:
            continue
        out[a_col] = pd.Series(new_vals, index=out.index, dtype=float)
        out[d_col] = pd.Series(new_solutes, index=out.index, dtype=object)
        out = out.drop(columns=[b_col, c_col])
        if notes is not None:
            solute_desc = ", ".join(sorted(solutes)) or "?"
            notes.append(f"Standardized '{labels.get(a_col, a_col)}' to molarity "
                         f"(M; {solute_desc}), kept the chemical as "
                         f"'{labels.get(d_col, d_col)}', and removed its label columns "
                         f"'{labels.get(b_col, b_col)}' and '{labels.get(c_col, c_col)}'.")
    return out


# =============================================================================
# Display helpers  —  show '1M NaOH' instead of the internal split columns
# =============================================================================
_PART_PREFIXES = (("A", "numeric_feature_A"), ("B", "group_label_B"),
                  ("C", "text_modifier_C"), ("D", "solute_label_D"))


def _meaningful(v) -> bool:
    return not (_is_blank(v) or str(v).strip().lower() in ("none", "missing"))


def recipe_groups(columns) -> dict:
    """Detect parsed messy-column groups among ``columns``.

    Returns ``{suffix: {"A": col, "B": col?, "C": col?, "D": col?}}`` for every
    numeric_feature_A* that has at least one companion label/solute column.
    """
    cols = {str(c) for c in columns}
    groups = {}
    for c in cols:
        m = _TRIPLE_A_RE.match(c)
        if m is None:
            continue
        s = m.group(1)
        g = {"A": c}
        for key, pre in _PART_PREFIXES[1:]:
            if f"{pre}{s}" in cols:
                g[key] = f"{pre}{s}"
        if len(g) > 1:
            groups[s] = g
    return groups


def compose_value(amount, solute=None, code=None, detail=None) -> str:
    """One human-readable value from the parsed parts.

    (1.0, 'NaOH')          -> '1M NaOH'         (converted concentration)
    (15.0, None, 'MN')     -> '15% MN'          (unconverted triple, dopant)
    (0.0, 'None')          -> '--'              (step not performed)
    (5.0, None, 'ZN', 'znso4-7h2o') -> '5% ZN (znso4-7h2o)'
    """
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        amt = 0.0
    if solute is not None:                       # molarity + chemical identity
        s = str(solute).strip()
        if s.lower() == "none" and amt == 0:
            return "--"
        if _meaningful(s):
            return f"{amt:.4g}M {s}"
        return f"{amt:.4g} M"
    has_code, has_detail = _meaningful(code), _meaningful(detail)
    if has_code:
        base = f"{amt:.4g}% {str(code).strip()}"
        return f"{base} ({str(detail).strip()})" if has_detail else base
    if has_detail:
        return str(detail).strip() if amt == 0 else f"{amt:.4g} {str(detail).strip()}"
    return "--" if amt == 0 else f"{amt:.4g}"


def compose_group(group: dict, getval) -> str:
    """Compose the display value for one recipe group via ``getval(col)``."""
    return compose_value(
        getval(group["A"]),
        solute=(getval(group["D"]) if "D" in group else None),
        code=(getval(group["B"]) if "B" in group else None),
        detail=(getval(group["C"]) if "C" in group else None))


def compose_grouped(raw: dict, labels: dict | None = None) -> list:
    """Collapse the parsed-part entries of a raw {col: value} dict for display.

    Returns ``[(display_name, display_value), ...]`` where each recipe group
    becomes one row (named after its source column, e.g. 'Pretreat 1') showing
    e.g. '1M NaOH'; every other key passes through (display-labelled when a
    label exists).
    """
    labels = dict(labels or {})
    groups = recipe_groups(raw.keys())
    member = {col: s for s, g in groups.items() for col in g.values()}
    done, rows = set(), []
    for key, val in raw.items():
        s = member.get(str(key))
        if s is None:
            rows.append((labels.get(str(key), str(key)), val))
            continue
        if s in done:
            continue
        done.add(s)
        g = groups[s]
        base = labels.get(g["A"], g["A"])
        base = base.rsplit(":", 1)[0].strip() if ":" in base else base
        rows.append((base, compose_group(g, lambda c: raw.get(c))))
    return rows


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
            result = _standardize_concentration_column(out[col], str(col))
            if result is None:
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
