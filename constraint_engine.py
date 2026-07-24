"""
constraint_engine.py — Parse and apply laboratory constraints.

Same per-line syntax and behaviour app_imgui.py has used since Phase 2
(``<column> <op> <value>`` / ``IN`` / ``NOT IN``), extracted into its own
testable module, plus two convenience layers that expand into it:

  * chemical-class shortcuts   "NO STRONG ACID" -> a numeric <= 0 bound on
                                the real Is_Strong_Acid descriptor column
                                for every chemistry role present in this
                                run's schema (there's no single "acid"
                                column to name directly — the descriptor
                                lives per reagent role, e.g.
                                Pretreat1_Is_Strong_Acid, Activator_Is_Strong_Acid).
  * process shortcuts          "stages <= 2" / "steps <= 1" -> a limit on
                                the DECODED recipe's pyrolysis-stage or
                                optional-step count. Not expressible as a
                                bound on any single search dimension (stage
                                count is a property of which knobs are
                                non-blank, not a value any one knob takes),
                                so these are applied as a post-hoc filter
                                on each candidate recipe instead of tightening
                                the search bounds up front.

Nothing here fabricates chemistry or hazard data — the chemical-class
shortcuts key off descriptor columns the app already computes from known
reagent structure (chemistry_features.py), the same descriptors already
shown in "Chemically feasible reagent mapping".
"""

from __future__ import annotations

import re

import numpy as np

# Shared with planner.py's feasibility_score (imported from there, not
# duplicated) so "how many stages/steps does this recipe use" has exactly
# one definition across the app.
_STEP_PATTERN = re.compile(r"pretreat|post.?treat|activat|additive|dop", re.I)
_STAGE_PATTERN = re.compile(r"(?:py|pyrolysis|stage)\D*(\d+)", re.I)
_BLANK_VALUES = {"none", "missing", "--", "0", "0.0", ""}


def _is_blank(value) -> bool:
    return str(value).strip().lower() in _BLANK_VALUES


def count_steps(recipe_dict) -> int:
    """Number of optional processing steps (pretreat/post-treat/activation/
    additive/doping) with a non-blank value in a decoded recipe dict."""
    return sum(1 for name, val in recipe_dict.items()
              if _STEP_PATTERN.search(str(name)) and not _is_blank(val))


def count_stages(recipe_dict) -> int:
    """Number of distinct pyrolysis stages (Py.1/Py.2/... temperature knobs)
    with a non-blank value in a decoded recipe dict. Always >= 1."""
    stages = set()
    for name, val in recipe_dict.items():
        m = _STAGE_PATTERN.search(str(name))
        if m and not _is_blank(val):
            stages.add(m.group(1))
    return len(stages) or 1


# ---- chemical-class shortcuts -------------------------------------------------
CHEMICAL_CLASS_KEYWORDS = {
    "strong acid": "Is_Strong_Acid", "strong base": "Is_Strong_Base",
    "oxidizer": "Is_Oxidizer", "oxidiser": "Is_Oxidizer",
    "reducing agent": "Is_Reducing_Agent",
    "chloride": "Contains_Chloride", "fluoride": "Contains_Fluoride",
    "sulfate": "Contains_Sulfate",
}
# Longest keyword first, so "strong acid" matches before a hypothetical
# shorter "acid" alias would.
_CHEMICAL_CLASS_RE = re.compile(
    r"^\s*NO\s+(" + "|".join(re.escape(k) for k in
                             sorted(CHEMICAL_CLASS_KEYWORDS, key=len, reverse=True))
    + r")\s*$", re.I)

_PROCESS_KEYWORDS = {"stages": "max_stages", "stage": "max_stages",
                    "steps": "max_steps", "step": "max_steps"}


def parse(text) -> list:
    """Parse one constraint per line into a list of raw constraint dicts,
    each tagged with a ``"kind"``:

      * ``{"kind": "chemical_class", "class": "strong acid", "raw": line}``
      * ``{"kind": "process", "column": "max_stages"|"max_steps",
          "op": "<=", "value": N, "raw": line}``
      * ``{"kind": "column", "column": ..., "op": ..., "value"/"values": ...}``
        — identical to the original per-column constraint dict.

    Blank lines and lines starting with '#' are ignored. Never raises — an
    unparsable line is silently skipped (surfaced instead as a status note
    by the caller; a typo shouldn't abort the whole run).
    """
    constraints = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m = _CHEMICAL_CLASS_RE.match(line)
        if m:
            cls = m.group(1).lower()
            constraints.append({"kind": "chemical_class", "class": cls, "raw": line})
            continue

        upper = f" {line.upper()} "
        op = None
        if " NOT IN " in upper:
            op = "NOT IN"
        elif " IN " in upper:
            op = "IN"
        else:
            for cand in ("<=", ">=", "==", "!=", "<", ">"):
                if cand in line:
                    op = cand
                    break
        if op is None:
            continue
        idx = upper.find(f" {op} ") if op in ("IN", "NOT IN") else line.find(op)
        col = line[:idx].strip()
        val = (line[idx + len(op) + 2:].strip() if op in ("IN", "NOT IN")
              else line[idx + len(op):].strip())
        if not col:
            continue

        process_key = _PROCESS_KEYWORDS.get(col.lower())
        if process_key is not None and op in ("<=", "<", ">=", ">", "=="):
            try:
                n = int(round(float(val.strip().strip("'\""))))
            except ValueError:
                continue
            constraints.append({"kind": "process", "column": process_key,
                                "op": op, "value": n, "raw": line})
            continue

        if op in ("IN", "NOT IN"):
            val = val.strip("[]() ")
            values = [v.strip().strip("'\"") for v in val.split(",") if v.strip()]
            if not values:
                continue
            constraints.append({"kind": "column", "column": col, "op": op, "values": values})
        else:
            val = val.strip().strip("'\"")
            if not val:
                continue
            constraints.append({"kind": "column", "column": col, "op": op, "value": val})
    return constraints


def expand_chemical_constraints(constraints, chemistry_schema) -> tuple:
    """Expand every ``chemical_class`` entry into one ``column`` entry per
    chemistry role that actually has the matching descriptor (a run with
    no chemistry columns configured, or none of the relevant class, simply
    has no effect — reported in a note rather than silently ignored).

    Returns ``(expanded, notes)`` — ``expanded`` has every chemical_class
    entry replaced by its per-role column entries (other entries pass
    through unchanged); ``notes`` explains what each shortcut expanded to.
    """
    expanded, notes = [], []
    columns = (chemistry_schema or {}).get("columns", {})
    for c in constraints:
        if c.get("kind") != "chemical_class":
            expanded.append(c)
            continue
        descriptor = CHEMICAL_CLASS_KEYWORDS.get(c["class"])
        hits = []
        for source, info in columns.items():
            feature = f"{info['prefix']}_{descriptor}"
            if feature in info.get("descriptor_columns", []):
                expanded.append({"kind": "column", "column": feature,
                                 "op": "<=", "value": "0"})
                hits.append(source)
        if hits:
            notes.append(f"'NO {c['class'].upper()}' -> excluded from: {', '.join(hits)}")
        else:
            notes.append(f"'NO {c['class'].upper()}' has no effect — no configured "
                         "chemistry role tracks this descriptor.")
    return expanded, notes


def process_limits(constraints) -> dict:
    """Pull every ``process`` entry into a simple ``{"max_stages": N,
    "max_steps": N}`` dict for ``satisfies_process`` — the tightest (lowest)
    limit wins if a limit is specified more than once."""
    limits = {}
    for c in constraints:
        if c.get("kind") != "process" or c.get("op") not in ("<=", "<"):
            continue
        n = c["value"] if c["op"] == "<=" else c["value"] - 1
        key = c["column"]
        limits[key] = min(n, limits[key]) if key in limits else n
    return limits


def satisfies_process(recipe_dict, limits) -> bool:
    """True if a decoded recipe respects every entry in ``limits`` (from
    ``process_limits``). An empty/None ``limits`` always satisfies."""
    if not limits:
        return True
    if "max_stages" in limits and count_stages(recipe_dict) > limits["max_stages"]:
        return False
    if "max_steps" in limits and count_steps(recipe_dict) > limits["max_steps"]:
        return False
    return True


def apply(constraints, numeric_cols, cat_choices, X_raw, by_label):
    """Tighten numeric ranges / filter categorical choice lists per
    ``column``-kind constraint (``chemical_class`` entries must already be
    expanded via ``expand_chemical_constraints``; ``process`` entries are
    consulted separately via ``process_limits``/``satisfies_process``).

    Returns ``(numeric_overrides, notes)``: ``numeric_overrides`` is
    consulted when the search bounds are built (``{col: (lo, hi)}``);
    ``cat_choices`` is filtered in place. A categorical constraint that
    would eliminate every option (e.g. a typo'd value) is IGNORED rather
    than left unsatisfiable — noted as such rather than silently dropped.
    """
    numeric_overrides = {}
    notes = []
    for c in constraints:
        if c.get("kind") not in (None, "column"):
            continue
        col = by_label.get(c["column"], c["column"])
        op = c["op"]
        if col in numeric_cols:
            try:
                v = float(c["value"])
            except (KeyError, ValueError):
                notes.append(f"'{c['column']} {op} {c.get('value', '')}' is not numeric — ignored.")
                continue
            lo, hi = numeric_overrides.get(
                col, (float(np.percentile(X_raw[col], 1)), float(np.percentile(X_raw[col], 99))))
            if op == "<=":
                hi = min(hi, v)
            elif op == "<":
                hi = min(hi, v - 1e-9)
            elif op == ">=":
                lo = max(lo, v)
            elif op == ">":
                lo = max(lo, v + 1e-9)
            elif op == "==":
                lo = hi = v
            elif op == "!=":
                notes.append(f"'{c['column']} != {v:g}' skipped — "
                             "not-equal isn't supported for numeric knobs "
                             "(the value is continuous, not a fixed point).")
                continue
            else:
                continue
            numeric_overrides[col] = (lo, hi)
            notes.append(f"{c['column']} {op} {v:g}")
        elif col in cat_choices:
            choices = cat_choices[col]
            if op in ("<=", ">=", "<", ">"):
                # A knob classified as categorical (e.g. temperature values
                # left as strings when the sheet mixed numbers with "Missing")
                # can still take a numeric comparison — filter by parsing each
                # choice, dropping non-numeric ones ("Missing" cannot satisfy
                # a numeric bound).
                try:
                    v = float(c["value"])
                except (KeyError, ValueError):
                    notes.append(f"'{c['column']} {op} {c.get('value', '')}' "
                                 "is not numeric — ignored.")
                    continue
                cmp = {"<=": lambda x: x <= v, ">=": lambda x: x >= v,
                      "<": lambda x: x < v, ">": lambda x: x > v}[op]
                new_choices = []
                for ch in choices:
                    try:
                        if cmp(float(ch)):
                            new_choices.append(ch)
                    except (TypeError, ValueError):
                        continue
            elif op == "IN":
                allowed = set(c["values"])
                new_choices = [ch for ch in choices if ch in allowed]
            elif op == "NOT IN":
                forbidden = set(c["values"])
                new_choices = [ch for ch in choices if ch not in forbidden]
            elif op == "!=":
                new_choices = [ch for ch in choices if ch != c["value"]]
            elif op == "==":
                new_choices = [ch for ch in choices if ch == c["value"]]
            else:
                continue
            if new_choices:
                cat_choices[col] = new_choices
                shown = ",".join(c["values"]) if op in ("IN", "NOT IN") else c.get("value", "")
                notes.append(f"{c['column']} {op} {shown}")
            else:
                notes.append(f"'{c['column']} {op} …' matches none of the observed "
                             f"categories ({', '.join(choices[:6])}…) — ignored.")
        else:
            notes.append(f"'{c['column']}' is not a controllable knob — constraint ignored.")
    return numeric_overrides, notes
