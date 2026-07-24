"""
Dear ImGui desktop app  —  train + predict, all in one window.
==============================================================

A native GUI (via `imgui-bundle`) that does the WHOLE workflow with no command
line:

  * TRAIN tab   -> pick a spreadsheet, set the messy/target/id columns, and train
                   a MultiOutputRegressor(RandomForest). Shows per-target R²/RMSE
                   and the most influential features. Training runs on a
                   background thread so the window stays responsive.
  * PREDICT tab -> widgets are built automatically from your data (number inputs
                   for numeric features, dropdowns for text features). Predicts
                   all 6 targets at once.
  * BATCH tab   -> run predictions for an entire spreadsheet and save the results.

HOW TO RUN
----------
    pip install imgui-bundle pandas numpy scikit-learn openpyxl joblib
    python app_imgui.py

This app runs fully offline: it never installs packages or reaches the
network. If a required package is missing it prints the pip command and
exits, leaving it to you to install (from a mirror/wheelhouse if offline).

The trained model can be saved to / loaded from 'model.joblib'.
==============================================================
"""

import sys
import threading
import importlib.util
from dataclasses import asdict


# -----------------------------------------------------------------------------
# Dependency check (offline — never installs anything or touches the network).
# Required packages must be present to launch; optional ones only unlock extra
# tabs (Compare models, Charts, PDF reports) and just produce a note.
# -----------------------------------------------------------------------------
def check_dependencies():
    required = {
        "imgui_bundle": "imgui-bundle",
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "scipy": "scipy",          # differential evolution for the "Optimize" tab
        "joblib": "joblib",
        "openpyxl": "openpyxl",    # lets pandas read .xlsx files
    }
    optional = {
        "matplotlib": "matplotlib",  # chart rendering for the "Charts" tab
        "xgboost": "xgboost",        # extra algorithms for "Compare models"
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "shap": "shap",              # SHAP summary / dependence plots
        "reportlab": "reportlab",    # PDF screening reports
    }

    def absent(mods):
        return [pip for mod, pip in mods.items()
                if importlib.util.find_spec(mod) is None]

    missing_opt = absent(optional)
    if missing_opt:
        print("[i] Optional packages not installed (their tabs stay disabled): "
              + " ".join(missing_opt))
    missing_req = absent(required)
    if missing_req:
        print("[X] Missing required packages: " + " ".join(missing_req))
        print("    This app runs fully offline and will not install anything.")
        print("    Install them yourself, then re-run:")
        print("        pip install " + " ".join(missing_req + missing_opt))
        sys.exit(1)


check_dependencies()

import os
import re
import json
import math
import numpy as np
import time
import pandas as pd
import joblib
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
from imgui_bundle import imgui, immapp, hello_imgui, portable_file_dialogs as pfd
import latent
import intelligence
import screening
import report
import units
import slideshow
import validation as V
import chemistry_features as chemistry
import cost_model
import planner
import research_gap
import pareto
import constraint_engine
import sustainability
import tradeoffs
import portfolio
import bayesopt

# Pre-warm heavy, complex optional imports ONCE, here, on the main thread —
# before immapp.run() / main() ever starts, so before any background worker
# thread can exist. Charts, Latent, Dataset Intelligence, and Optimize each
# independently do `import charts` inside their OWN threading.Thread body;
# charts.py in turn lazily imports shap (which pulls in IPython/tqdm) the
# first time a SHAP chart runs. If two of those threads raced to import the
# same complex package for the first time simultaneously, Python's shared
# sys.modules state could produce "partially initialized module ... circular
# import" errors. Doing it once, up front, single-threaded, makes every later
# `import charts`/`import shap` a cheap no-op instead of a race.
import charts  # noqa: F401 - matplotlib(Agg) side effect; also used directly below
try:
    import shap  # noqa: F401
except ImportError:
    pass

MODEL_OUT = "model.joblib"
RANDOM_STATE = 42

# --- Missing-data semantics -------------------------------------------------
# Two DISTINCT kinds of "empty" cell, handled differently:
#   * NOT-DONE ("--", "none", …) -> the step was DELIBERATELY skipped. Real
#     information: numeric columns take 0 (no amount used), text columns take the
#     explicit "None" category. Rows made of these are still informative.
#   * UNKNOWN  ("", "n/a", "unknown", NaN) -> simply not recorded. Numeric columns
#     are median-imputed; text columns become "Missing"; these carry no info.
NOT_DONE_TOKENS = {"--", "---", "----", "—", "–", "none", "nil", "no additive",
                   "no pretreatment", "no post-treatment", "not applied"}
UNKNOWN_TOKENS = {"", "-", "unknown", "unspecified", "nan", "na", "n/a", "n.a.",
                  "missing", "missing or unspecified", "tbd", "?", "input",
                  "..", ".", "x"}
NOT_DONE_LABEL = "None"       # explicit "step not performed" category label
# Any empty-ish token (either kind) — used only for numeric auto-detection.
BLANK_TOKENS = UNKNOWN_TOKENS | NOT_DONE_TOKENS
MIN_INFORMATIVE_FRAC = 0.10   # drop a row if <10% of its feature cells carry real info
MAX_MISSING_FRAC = 0.60       # drop a feature if missing in >60% of the kept rows


def classify_cell(v):
    """Classify a raw cell as 'not_done', 'unknown', or None (a real value)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "unknown"
    s = str(v).strip().lower()
    if s in NOT_DONE_TOKENS:
        return "not_done"
    if s in UNKNOWN_TOKENS:
        return "unknown"
    return None


def coerce_numeric_series(s):
    """Numeric view of a messy column: 'not-done' -> 0, unknown/unparseable -> NaN."""
    def one(v):
        k = classify_cell(v)
        if k == "not_done":
            return "0"          # a skipped step contributes no amount
        if k == "unknown":
            return None         # -> NaN -> imputed downstream
        return v
    return pd.to_numeric(s.map(one), errors="coerce")


def normalize_categorical_series(s):
    """Text view: 'not-done' -> 'None' category, unknown -> 'Missing', else str."""
    def one(v):
        k = classify_cell(v)
        if k == "not_done":
            return NOT_DONE_LABEL
        if k == "unknown":
            return "Missing"
        return str(v)
    return s.map(one)
MAX_CATEGORIES = 25           # drop a text feature with more unique values than this


# =============================================================================
# PREPROCESSING  (identical logic to train_model.py, kept self-contained)
# =============================================================================
def parse_mixed_value(value):
    """
    Split one messy cell into three ANONYMIZED parts.
        "15% mn"            -> (15.0, "MN",   "None")
        "bio oil"           -> (0.0,  "None", "bio oil")
        "5%zn(znso4·7h2o)"  -> (5.0,  "ZN",   "znso4·7h2o")
        NaN / "" / "None"   -> (0.0,  "None", "None")
    Returns (float, str, str); never raises.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0, "None", "None"
    s = str(value).strip()
    if s.lower() in {"", "none", "nan", "missing", "na"}:
        return 0.0, "None", "None"
    s_low = s.lower()

    num_match = re.search(r"\d+(?:\.\d+)?", s_low)
    numeric_feature_A = float(num_match.group()) if num_match else 0.0

    paren_match = re.search(r"\(([^)]*)\)", s_low)
    code_match = re.search(r"\d+\s*%?\s*([a-z]{1,4})", s_low)
    group_label_B = code_match.group(1).upper() if code_match else "None"

    if paren_match:
        text_modifier_C = paren_match.group(1).strip() or "None"
    else:
        leftover = s_low
        if num_match:
            leftover = leftover.replace(num_match.group(), " ")
        leftover = leftover.replace("%", " ")
        if code_match:
            leftover = re.sub(
                r"\b" + re.escape(code_match.group(1)) + r"\b", " ", leftover, count=1
            )
        leftover = leftover.strip()
        text_modifier_C = leftover if leftover else "None"

    return numeric_feature_A, group_label_B, text_modifier_C


def read_any(path, nrows=None, sheet=None):
    """
    Read a CSV or Excel file into a DataFrame based on its extension.
    `sheet` (name or index) picks the worksheet for multi-tab .xlsx files;
    None reads the first sheet. Ignored for CSVs.
    """
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, nrows=nrows)
    return pd.read_excel(path, nrows=nrows, sheet_name=(0 if sheet is None else sheet))


def list_sheets(path):
    """Return the worksheet/tab names for an Excel file ([] for CSV or on error)."""
    if not path or path.lower().endswith(".csv"):
        return []
    try:
        return list(pd.ExcelFile(path).sheet_names)
    except Exception:  # noqa: BLE001
        return []


def auto_detect_columns(path, sheet=None):
    """
    Peek at a spreadsheet and guess the three config fields for step 2:
      * targets -> the 6 known current columns if present, else numeric columns
                   that look like '<num>_<num><A|mA>'.
      * ids     -> 'serial_orig'/'serial_clean' if present, else near-unique
                   text columns whose name hints at an identifier.
      * mixed   -> every text column whose cells look "messy"
                   (contain %, parentheses, or a number stuck to letters).
    Returns (mixed_list, id_list, target_list). Reads only a 200-row sample.
    """
    df = read_any(path, nrows=200, sheet=sheet)
    cols = [str(c) for c in df.columns]
    n = max(len(df), 1)

    # --- targets ---
    known = ["100_1A", "100_0p1A", "100_10mA", "500_1A", "500_0p1A", "500_10mA"]
    targets = [c for c in known if c in cols]
    if not targets:
        # Rate-style performance columns: '100_1A', 'LIB_1A', 'SIB_0_1A',
        # 'LIB_30mA', ... — any name ending in _<num>[p<num>]A / mA.
        tpat = re.compile(r"^\S+_\d+p?\d*\s*(a|ma)$", re.I)
        targets = [c for c in cols
                   if tpat.match(c) and pd.api.types.is_numeric_dtype(df[c])]

    # --- id columns ---
    ids = [c for c in ["serial_orig", "serial_clean"] if c in cols]
    if not ids:
        for c in cols:
            if df[c].dtype == object:
                uniq_ratio = df[c].nunique(dropna=True) / n
                name_hint = "serial" in c.lower() or c.lower() in {"id", "index"} \
                    or c.lower().endswith("_id") \
                    or "批號" in c or "編號" in c or "batch" in c.lower()
                # A hinted name is an id even with replicates (rows sharing a
                # serial); without a hint require near-total uniqueness.
                if (name_hint and uniq_ratio > 0.2) or uniq_ratio >= 0.98:
                    ids.append(c)

    # --- messy/mixed columns (there may be several) ---
    messy_re = re.compile(r"[%()·]|\d\s*[a-z]{1,4}", re.I)
    scored = []
    for c in cols:
        if c in ids or c in targets or df[c].dtype != object:
            continue
        vals = df[c].dropna().astype(str)
        if len(vals) == 0:
            continue
        score = float(vals.str.contains(messy_re).mean())
        if score >= 0.25:  # enough messy cells to be worth parsing
            scored.append((score, c))
    # Most-messy first, but keep it stable/readable.
    mixed = [c for _, c in sorted(scored, key=lambda t: -t[0])]

    return mixed, ids, targets


def _anon_names(index, total):
    """
    Anonymized output names for the messy column at position `index`.
    One messy column -> A/B/C (no suffix). Several -> A1/B1/C1, A2/B2/C2, …
    so multiple messy columns never collide.
    """
    if total <= 1:
        return "numeric_feature_A", "group_label_B", "text_modifier_C"
    s = index + 1
    return f"numeric_feature_A{s}", f"group_label_B{s}", f"text_modifier_C{s}"


def _mixed_list(mixed):
    """Normalize the messy-column setting to a clean ordered list."""
    if isinstance(mixed, str):
        return [c.strip() for c in mixed.split(",") if c.strip()]
    return [str(c).strip() for c in (mixed or []) if str(c).strip()]


def mixed_feature_labels(mixed, columns=None):
    """
    Human-readable labels for parsed messy-column features.

    The model still uses stable internal names such as numeric_feature_A1, but
    the UI should show the original column and the parsed part instead.
    """
    names = _mixed_list(mixed)
    if columns is not None:
        available = {str(c) for c in columns}
        names = [c for c in names if c in available]
    labels = {}
    total = len(names)
    for i, col in enumerate(names):
        amount, code, modifier = _anon_names(i, total)
        labels[amount] = f"{col}: number/percent"
        labels[code] = f"{col}: code"
        labels[modifier] = f"{col}: detail"
        # Chemical-identity column emitted when the triple is converted to
        # molarity (units.standardize_parsed_mixed). Same suffix scheme.
        suffix = amount[len("numeric_feature_A"):]
        labels[f"solute_label_D{suffix}"] = f"{col}: chemical"
    return labels


def feature_label(col):
    """Display name for an internal feature column.

    Parsed messy-column parts (numeric_feature_A1, group_label_B5, ...) map to
    '<original column>: number/percent | code | detail'; anything else is
    returned unchanged. The map is set at train/load time (STATE.feature_labels).
    """
    return STATE.feature_labels.get(str(col), str(col))


def pretty(col):
    """Human-friendly label for any feature/target name shown in the UI."""
    return screening._pretty(feature_label(col))


def group_base_label(group):
    """Original column name a recipe group came from ('Pretreat 1')."""
    lab = feature_label(group["A"])
    return lab.rsplit(":", 1)[0].strip() if ":" in lab else lab


def _set_widget_categorical(col, value):
    """Point a categorical widget at ``value`` (custom text when unseen)."""
    choices = STATE.categorical_schema.get(col, [])
    value = str(value)
    if value in choices:
        STATE.category_index[col] = choices.index(value)
        STATE.screen_custom_category[col] = ""
    else:
        STATE.screen_custom_category[col] = value


def apply_recipe_text(group, text):
    """Parse one typed recipe value ('1M NaOH', '3.2%V H2SO4', '15% Mn', '--')
    back into the group's internal widget values."""
    text = str(text).strip()
    if not text or units._is_blank(text):        # step not performed
        STATE.numeric_values[group["A"]] = 0.0
        for k in ("B", "C", "D"):
            if k in group:
                _set_widget_categorical(group[k], "None")
        return
    conc = units.parse_concentration_to_molarity(text)
    if conc is not None and "D" in group:
        molarity, solute, _basis = conc
        STATE.numeric_values[group["A"]] = float(molarity)
        _set_widget_categorical(group["D"], solute)
        return
    amount, code, detail = parse_mixed_value(text)
    STATE.numeric_values[group["A"]] = float(amount)
    if "B" in group:
        _set_widget_categorical(group["B"], code)
    if "C" in group:
        _set_widget_categorical(group["C"], detail)
    if "D" in group:
        chem = units._find_chemical(detail, code)
        _set_widget_categorical(group["D"], chem["label"] if chem else "Missing")

def _widget_cat_value(col):
    custom = STATE.screen_custom_category.get(col, "").strip()
    if custom:
        return custom
    choices = STATE.categorical_schema.get(col, [])
    idx = min(STATE.category_index.get(col, 0), max(len(choices) - 1, 0))
    return choices[idx] if choices else "None"


def draw_recipe_inputs(id_prefix, mark_dirty=False):
    """One combined text input per parsed messy column ('1M NaOH' style).

    Returns the set of internal columns consumed, so callers skip their
    individual widgets.
    """
    groups = units.recipe_groups(set(STATE.numeric_schema) | set(STATE.categorical_schema))
    consumed = set()
    for s in sorted(groups):
        g = groups[s]
        consumed.update(g.values())
        key = f"{id_prefix}{s}"
        cur = STATE.mixed_text.get(key)
        if cur is None:
            cur = units.compose_group(
                g, lambda c: STATE.numeric_values.get(c, STATE.numeric_schema.get(c))
                if c in STATE.numeric_schema else _widget_cat_value(c))
            STATE.mixed_text[key] = cur
        imgui.set_next_item_width(360)
        changed, txt = imgui.input_text(f"{group_base_label(g)}##recipe_{key}", cur)
        if changed:
            STATE.mixed_text[key] = txt
            apply_recipe_text(g, txt)
            if mark_dirty:
                STATE.screen_dirty = True
    return consumed


def _chemistry_model_columns():
    columns = set(STATE.chemistry_schema.get("interactions", []))
    for source, info in STATE.chemistry_schema.get("columns", {}).items():
        columns.add(source)
        columns.update(info.get("descriptor_columns", []))
    return columns


def draw_chemical_inputs(id_prefix, mark_dirty=False):
    """Render original chemical selectors while descriptors stay model-internal."""
    for source, info in STATE.chemistry_schema.get("columns", {}).items():
        choices = list(info.get("observed_chemicals", []))
        custom_label = "Custom / inferred..."
        display_choices = choices + [custom_label]
        custom = STATE.chemistry_custom_values.get(source, "")
        selected = STATE.chemistry_values.get(source, choices[0] if choices else "Unknown")
        idx = len(choices) if custom else (choices.index(selected) if selected in choices else len(choices))
        imgui.set_next_item_width(300)
        changed, idx = imgui.combo(f"{source}##chem_{id_prefix}_{source}", idx, display_choices)
        if changed:
            if idx < len(choices):
                STATE.chemistry_values[source] = choices[idx]
                STATE.chemistry_custom_values[source] = ""
            else:
                STATE.chemistry_custom_values[source] = custom or "Unknown"
            if mark_dirty:
                STATE.screen_dirty = True
        if idx == len(choices) or STATE.chemistry_custom_values.get(source):
            imgui.set_next_item_width(300)
            changed, value = imgui.input_text(
                f"New chemical formula/name##chem_custom_{id_prefix}_{source}",
                STATE.chemistry_custom_values.get(source, ""))
            if changed:
                STATE.chemistry_custom_values[source] = value
                if mark_dirty:
                    STATE.screen_dirty = True
            if value.strip():
                descriptor = chemistry.ENGINE.generator.describe(value.strip())
                sims = chemistry.ENGINE.generator.similarities(value.strip(), top=2)
                sim_text = ", ".join(f"{s.name} {s.score:.2f}" for s in sims)
                imgui.text_colored(
                    DIM, f"{descriptor.categorical['ChemicalClass']} · inferred confidence "
                         f"{descriptor.confidence:.2f} · nearest: {sim_text}")


def prepare_raw(df, ids, mixed):
    """
    Drop id columns and parse+drop every messy column into its own anonymized
    A/B/C triple. `mixed` may be a single name or a list of names.
    """
    df = df.drop(columns=ids, errors="ignore")

    mixed = _mixed_list(mixed)
    present = [c for c in mixed if c and c in df.columns]
    if not present:
        return df

    df = df.copy()
    total = len(present)
    for i, col in enumerate(present):
        na, nb, nc = _anon_names(i, total)
        parsed = df[col].apply(parse_mixed_value).apply(pd.Series)
        parsed.columns = [na, nb, nc]
        df[na] = parsed[na].astype(float)
        df[nb] = parsed[nb].astype(str)
        df[nc] = parsed[nc].astype(str)
    df = df.drop(columns=present)
    return df


def auto_coerce_numeric(df, protected=(), threshold=0.85, min_unique=3):
    """
    Coerce object columns that are *mostly* numeric to real numbers.

    Real spreadsheets often carry one stray annotation (e.g. an 'INPUT' marker
    row) that flips an otherwise-numeric column to text, which would then be
    mis-handled as a categorical.  A column whose non-blank cells parse as
    numbers at least ``threshold`` of the time (and has >= ``min_unique``
    distinct numeric values) is treated as numeric; unparseable cells become
    NaN and are imputed downstream.  Columns in ``protected`` (manual overrides)
    are never touched.
    """
    df = df.copy()
    protected = set(protected)
    for c in df.columns:
        if c in protected or pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = df[c]
        # Empty-ish tokens (unknown OR not-done) don't count against the parse
        # rate — a column of numbers sprinkled with '--' is still numeric.
        nonblank = s[~s.astype(str).str.strip().str.lower().isin(BLANK_TOKENS)]
        if len(nonblank) == 0:
            continue
        parsed = pd.to_numeric(nonblank, errors="coerce")
        if parsed.notna().mean() >= threshold and parsed.dropna().nunique() >= min_unique:
            # 'not-done' cells become 0; unknown/unparseable become NaN (imputed).
            df[c] = coerce_numeric_series(s)
    return df


def apply_col_type_overrides(df, overrides):
    """
    Force chosen columns to a user-selected type before feature building.
      * "numeric"     -> pd.to_numeric (unparseable cells become NaN, then imputed)
      * "categorical" -> cast to string so it is one-hot encoded
    `overrides` is {col: "numeric" | "categorical"}; unknown columns are ignored.
    """
    if not overrides:
        return df
    df = df.copy()
    for col, kind in overrides.items():
        if col not in df.columns:
            continue
        if kind == "numeric":
            df[col] = coerce_numeric_series(df[col])   # 'not-done' -> 0
        elif kind == "categorical":
            df[col] = df[col].astype(str)
    return df


def _looks_like_battery_target(column):
    """Conservative detector used to withhold other outcomes in publication mode."""
    name = str(column)
    return bool(
        re.search(r"(?:^|_)(?:LIB|SIB)(?:_|$)", name, re.I)
        or re.search(r"(?:capacity|retention|coulombic|ICE)(?:_|$)", name, re.I)
    )


def build_training_data(cfg, notes=None):
    """Build aligned raw/encoded predictors, targets, groups, and schemas.

    The raw frame is retained for leakage-safe fold-local preprocessing.  The
    encoded frame is only used to fit the final deployable model on all rows and
    to preserve the existing prediction/screening workflow.
    """
    def log(msg):
        if notes is not None:
            notes.append(msg)

    raw_df = read_any(cfg["data_path"], sheet=cfg.get("sheet")).reset_index(drop=True)
    validation_cfg = V.normalize_validation_config(cfg.get("validation"))
    group_column = validation_cfg.get("group_column", "")
    grouped = validation_cfg["method"] != "random_kfold"
    if grouped and not group_column:
        raise ValueError("Choose a grouping column for grouped validation.")
    if grouped and group_column not in raw_df.columns:
        raise ValueError(f"Grouping column '{group_column}' was not found in the dataset.")
    groups = (raw_df[group_column].copy() if group_column in raw_df.columns
              else pd.Series(np.arange(len(raw_df)), name="row"))
    chemistry_groups = groups.copy()
    if not grouped:
        # An experiment ID still defines independent chemistry support even when
        # random K-fold validation was selected.
        chemistry_group_column = next(
            (c for c in cfg.get("ids", []) if c in raw_df.columns and
             1 < raw_df[c].nunique(dropna=True) < len(raw_df)), None)
        if chemistry_group_column:
            chemistry_groups = raw_df[chemistry_group_column].copy()
            log(f"Chemistry feature budgeting uses {chemistry_groups.nunique()} independent "
                f"groups from ID column '{chemistry_group_column}'.")

    # IDs are never eligible predictors.  The grouping column is held separately
    # even when the user did not also classify it as an ID.
    drop_ids = list(dict.fromkeys(list(cfg.get("ids", [])) + ([group_column] if group_column else [])))
    df = prepare_raw(raw_df, drop_ids, cfg["mixed"])
    # Standardize measurement units per column BEFORE numeric coercion, so cells
    # like "2 h" become 120 (minutes) and a "(K)" temperature column becomes °C.
    # Targets (labels) and manually-typed columns are protected from conversion.
    # Optional: governed by the "Auto-standardize units" checkbox on the Train
    # tab — when off, values are used exactly as they appear in the sheet.
    if cfg.get("standardize_units", True):
        df = units.standardize_units(
            df,
            protected=set(cfg["targets"]) | set(cfg.get("col_types", {}).keys()),
            notes=notes,
        )
        # The messy-column parser splits concentrations like '3.2%V H2SO4' into
        # a number plus label columns ('<col>: code' / '<col>: detail').  Rebuild
        # molarity from those parts and drop the label columns — their
        # information now lives in the standardized number.
        df = units.standardize_parsed_mixed(
            df, notes=notes, labels=mixed_feature_labels(cfg["mixed"]))
    else:
        log("Unit standardization is OFF — values used exactly as in the sheet.")
    # Auto-fix columns that are numeric except for a stray text marker, so a real
    # temperature/time column isn't mistaken for a categorical. Manual choices win.
    df = auto_coerce_numeric(df, protected=set(cfg.get("col_types", {}).keys()))
    # Honor manual Numeric/Categorical choices from the "Column types" tab.
    df = apply_col_type_overrides(df, cfg.get("col_types"))

    # Drop columns the user marked "Exclude" in the Column types tab.
    excluded = [c for c in cfg.get("exclude", []) if c in df.columns]
    if excluded:
        df = df.drop(columns=excluded)
        log(f"Excluded {len(excluded)} column(s): {', '.join(excluded)}.")

    # Drop entirely-empty columns.
    empty = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty:
        df = df.drop(columns=empty)

    targets = list(cfg["targets"])
    if not targets:
        raise ValueError(
            "No target column selected. Type the target column name(s) in the "
            "Train tab (step 2), or set a Target role in the Column types tab. "
            f"Available columns: {', '.join(str(c) for c in df.columns[:20])}…")
    missing = [c for c in targets if c not in df.columns]
    if missing:
        raise ValueError(f"Target columns not found: {missing}")

    # In single-target publication mode, measured battery outcomes other than the
    # chosen target are explicitly withheld so one capacity cannot predict another.
    if cfg.get("single_target_mode"):
        if len(targets) != 1:
            raise ValueError("Single-target publication mode requires exactly one active target.")
        other_targets = [c for c in df.columns
                         if c not in targets and (_looks_like_battery_target(c)
                                                  or c in cfg.get("all_target_columns", []))]
        if other_targets:
            df = df.drop(columns=other_targets, errors="ignore")
            log(f"Publication leakage guard excluded {len(other_targets)} other outcome column(s).")

    # Targets: coerce to numeric, then DROP rows with no label (never impute y).
    for col in targets:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n0 = len(df)
    target_mask = df[targets].notna().all(axis=1)
    df = df.loc[target_mask].reset_index(drop=True)
    groups = groups.loc[target_mask].reset_index(drop=True)
    chemistry_groups = chemistry_groups.loc[target_mask].reset_index(drop=True)
    log(f"Kept {len(df)}/{n0} rows with an observed target.")

    # Drop rows whose features are essentially all blank / 'Unknown'.
    feat_only = [c for c in df.columns if c not in targets]

    def informative_frac(row):
        # 'Not-done' cells ARE information (a deliberate no-op); only truly
        # unknown / unrecorded cells count as uninformative.
        hits = sum(0 if ((v != v) or str(v).strip().lower() in UNKNOWN_TOKENS) else 1
                   for v in row)
        return hits / len(feat_only) if feat_only else 0.0

    frac = df[feat_only].apply(informative_frac, axis=1)
    n1 = len(df)
    informative_mask = (frac >= MIN_INFORMATIVE_FRAC).to_numpy()
    df = df.loc[informative_mask].reset_index(drop=True)
    groups = groups.loc[informative_mask].reset_index(drop=True)
    chemistry_groups = chemistry_groups.loc[informative_mask].reset_index(drop=True)
    log(f"Dropped {n1 - len(df)} blank/'Unknown'-only rows -> {len(df)} rows.")

    X = df[feat_only].copy()
    y = df[targets]
    original_predictor_count = int(X.shape[1])

    chemistry_schema = {"columns": {}, "interactions": [],
                        "descriptor_feature_count": 0,
                        "rdkit_available": chemistry.ENGINE.generator.rdkit.available}
    chemistry_originals = pd.DataFrame(index=X.index)
    requested_mode = cfg.get("chemistry_mode", "automatic")
    if not cfg.get("chemistry_enabled", True):
        requested_mode = "off"
    chemistry_target = (y[targets[0]] if cfg.get("single_target_mode") and targets else None)
    chemistry_detection = chemistry.ENGINE.detect_column_details(X)
    detection_overrides = dict(cfg.get("chemistry_column_overrides", {}))
    chemistry_columns = [
        item.column for item in chemistry_detection
        if detection_overrides.get(item.column, item.confidence >= .70)
    ]
    chemistry_config = chemistry.ENGINE.auto_configure(
        X, chemical_columns=chemistry_columns, groups=chemistry_groups,
        target=chemistry_target, requested_mode=requested_mode)
    if requested_mode == "custom":
        chemistry.apply_custom_families(
            chemistry_config, cfg.get("chemistry_custom_families", []))
    expansion = chemistry.ENGINE.transform(X, config=chemistry_config)
    X = expansion.frame
    chemistry_schema = expansion.metadata
    chemistry_originals = expansion.original_values
    chemistry_diagnostics = dict(
        chemistry_schema.get("chemistry_feature_diagnostics", {}))
    if chemistry_schema.get("columns"):
        log(
            f"Chemistry {chemistry_config.mode} mode selected "
            f"{chemistry_schema['descriptor_feature_count']} descriptors and "
            f"{len(chemistry_schema['interactions'])} supported interaction(s) from "
            f"{chemistry_schema.get('candidate_descriptor_count', 0)} candidates."
        )
        for reason in chemistry_config.rationale:
            log(reason)

    # Feature selection: drop too-sparse and too-high-cardinality columns.
    dropped = 0
    for col in list(X.columns):
        if X[col].isna().mean() > MAX_MISSING_FRAC:
            X = X.drop(columns=col); dropped += 1
        elif not pd.api.types.is_numeric_dtype(X[col]) and \
                X[col].dropna().astype(str).nunique() > MAX_CATEGORIES:
            X = X.drop(columns=col); dropped += 1
    log(f"Feature selection: kept {X.shape[1]} columns (dropped {dropped} noisy).")

    # Record a raw frame for fold-local preprocessing. Categorical semantic
    # normalization is deterministic; numeric imputation remains unfitted here.
    numeric_schema, categorical_schema = {}, {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            med = X[col].median()
            med = 0.0 if pd.isna(med) else float(med)
            numeric_schema[col] = med
        else:
            X[col] = normalize_categorical_series(X[col])
            categorical_schema[col] = sorted(X[col].unique().tolist())

    X_raw = X.reset_index(drop=True)
    X_for_final = X_raw.copy()
    for col, med in numeric_schema.items():
        X_for_final[col] = X_for_final[col].fillna(med)
    X_enc = pd.get_dummies(X_for_final, drop_first=True).astype(float)
    y = y.reset_index(drop=True)
    groups.name = group_column or groups.name
    if len(X_raw) != len(y) or len(groups) != len(y):
        raise ValueError("Internal alignment error after cleaning X, y, and groups.")
    return {
        "X_raw": X_raw,
        "X_encoded": X_enc,
        "y": y,
        "groups": groups,
        "numeric_schema": numeric_schema,
        "categorical_schema": categorical_schema,
        "original_predictors": original_predictor_count,
        "chemistry_config": chemistry.chemistry_config_as_dict(chemistry_config),
        "chemistry_schema": chemistry_schema,
        "chemistry_feature_diagnostics": chemistry_diagnostics,
        "chemistry_originals": chemistry_originals.reset_index(drop=True),
    }


def build_Xy(cfg, notes=None):
    """Backward-friendly five-value data builder requested by the Train workflow."""
    data = build_training_data(cfg, notes=notes)
    return (data["X_encoded"], data["y"], data["groups"],
            data["numeric_schema"], data["categorical_schema"])


# =============================================================================
# MODEL COMPARISON  —  7 architectures benchmarked with 5-fold CV
# =============================================================================
def build_models():
    """{name: estimator} for every architecture. Boosting libs import lazily so
    a missing one is skipped rather than crashing the benchmark."""
    models = {}

    try:
        from xgboost import XGBRegressor
        models["MultiOutput(XGBRegressor)"] = MultiOutputRegressor(
            XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        models["MultiOutput(LGBMRegressor)"] = MultiOutputRegressor(
            LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        )
    except ImportError:
        pass

    models["MultiOutput(ExtraTrees)"] = MultiOutputRegressor(
        ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    )

    try:
        from catboost import CatBoostRegressor
        models["RegressorChain(CatBoost)"] = RegressorChain(
            CatBoostRegressor(iterations=300, depth=5, learning_rate=0.05,
                              random_state=RANDOM_STATE, verbose=0),
            order=list(range(6)),
        )
    except ImportError:
        pass

    models["MultiOutput(SVR)"] = MultiOutputRegressor(
        make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale"))
    )

    stack_estimators = [
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("et", ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
    ]
    try:
        from catboost import CatBoostRegressor
        stack_estimators.insert(
            1, ("cat", CatBoostRegressor(iterations=200, depth=5, learning_rate=0.05,
                                         random_state=RANDOM_STATE, verbose=0))
        )
    except ImportError:
        pass
    models["Stacking(Ridge+CatBoost+ExtraTrees)"] = MultiOutputRegressor(
        StackingRegressor(estimators=stack_estimators,
                          final_estimator=Ridge(alpha=1.0), n_jobs=-1)
    )

    models["MLPRegressor(64,32)"] = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800,
                     early_stopping=True, random_state=RANDOM_STATE),
    )

    return models


# =============================================================================
# CAPACITY OPTIMIZER  —  train on controllable knobs, search for the best recipe
# =============================================================================
_RISK_FROM_APPLICABILITY_LABEL = {
    "well inside the training domain": "Low",
    "inside the training domain": "Low",
    "at the edge of the training domain": "Moderate",
    "OUTSIDE the training domain": "High",
}


def _parse_objectives_text(text, primary_column, primary_direction):
    """Parse 'column:maximize|minimize:weight' entries (comma- or newline-
    separated) into a normalized objective list.

    The primary target is always objectives[0] — added automatically with
    weight 1.0 if not explicitly listed. Weights auto-normalize to sum to 1.
    An entry with an unparsable weight defaults to 1.0 (equal say). Returns a
    list with at least one entry (the primary) even given empty/unparsable
    text, so single-objective callers can treat "no extra objectives" and
    "objectives=[primary]" identically.
    """
    entries = []
    for part in re.split(r"[,\n]", str(text or "")):
        part = part.strip()
        if not part:
            continue
        bits = [b.strip() for b in part.split(":")]
        if len(bits) < 2 or not bits[0]:
            continue
        col, raw_dir = bits[0], bits[1].lower()
        if raw_dir.startswith("max"):
            d = "maximise"
        elif raw_dir.startswith("min"):
            d = "minimise"
        else:
            continue
        try:
            weight = float(bits[2]) if len(bits) > 2 else 1.0
        except ValueError:
            weight = 1.0
        entries.append({"column": col, "direction": d, "weight": max(weight, 0.0)})

    if not any(e["column"] == primary_column for e in entries):
        entries.insert(0, {"column": primary_column, "direction": primary_direction,
                           "weight": 1.0})
    total = sum(e["weight"] for e in entries) or 1.0
    for e in entries:
        e["weight"] = e["weight"] / total
    return entries


def _pareto_front_candidates(archive_vecs, archive_obj, *, objectives, target, model, y,
                             extra_models, vectorize, decode_vec, annotate_recipe,
                             feat_cols, X_enc, numeric_cols, cat_choices, X_raw,
                             chemistry_schema, chemistry_originals_df, label_map,
                             cv_rmse, cv_r2, min_applicability, fixed,
                             n_priority_candidates=0, random_state=RANDOM_STATE,
                             pool_cap=600, max_front=15, process_limits=None):
    """Non-dominated (Pareto-optimal) recipes across every objective.

    Reuses the same DE-search + training-row-seeded archive the (separate,
    single-objective) Top-N pass uses, but selects candidates by Pareto
    dominance across ALL objectives (via ``pareto.non_dominated_sort`` —
    every rank, not just the optimal front) rather than one weighted
    utility, then diversity-filters the OPTIMAL front down to ``max_front``
    points ranked by weighted utility so the result is a genuinely
    different set of trade-offs, not dozens of near-duplicates.

    Best-effort: any failure returns ``([], [])`` rather than propagating,
    matching ``_rank_top_recipes``'s contract — the single-best recipe and
    Top-N list must keep working regardless of whether this succeeds.
    Returns ``(pareto_points, front_sizes)`` — ``front_sizes[0]`` is the
    optimal front's size before diversity-filtering down to ``max_front``,
    ``front_sizes[1:]`` are the next-best ranks' sizes (rank-2 front is
    optimal only once every rank-1 point is set aside, and so on).
    """
    if not archive_vecs or len(objectives) < 2:
        return [], []
    all_vecs = np.vstack(archive_vecs)
    n = len(all_vecs)
    if n == 0:
        return [], []

    # Subsample exactly like _rank_top_recipes: priority (training-row-
    # derived) candidates kept in full, the rest randomly thinned.
    n_priority_candidates = min(n_priority_candidates, n)
    split = n - n_priority_candidates
    idx = np.arange(n)
    priority_idx, search_idx = idx[split:], idx[:split]
    search_budget = pool_cap - n_priority_candidates
    if search_budget < len(search_idx):
        rng_np = np.random.RandomState(random_state)
        search_idx = rng_np.choice(search_idx, max(search_budget, 0), replace=False)
    all_vecs = all_vecs[np.concatenate([priority_idx, search_idx]).astype(int)]

    try:
        numeric_schema = {c: float(X_raw[c].median()) for c in numeric_cols}
        screener = screening.Screener(
            model, feat_cols, numeric_schema, cat_choices, [target],
            X_enc.astype(float), pd.DataFrame({target: y}, index=X_enc.index),
            cv_rmse={target: cv_rmse}, cv_r2={target: cv_r2},
            display_labels=label_map, chemistry_schema=chemistry_schema,
            chemistry_originals=chemistry_originals_df,
        )
    except Exception:  # noqa: BLE001
        return [], []

    rows = []
    for vec in all_vecs:
        try:
            X_row = pd.DataFrame([vectorize(vec)], columns=feat_cols).astype(float)
            ad = screener.applicability(X_row)
        except Exception:  # noqa: BLE001
            continue
        applicability_pct = 100.0 - ad["percentile"]
        if applicability_pct < min_applicability:
            continue
        values = {}
        ok = True
        recipe_dict = None
        for obj in objectives:
            col = obj["column"]
            try:
                if col in extra_models:
                    values[col] = float(extra_models[col].predict(
                        vectorize(vec).reshape(1, -1))[0])
                elif col == target:
                    values[col] = float(model.predict(vectorize(vec).reshape(1, -1))[0])
                else:               # a controllable-knob objective (e.g. temperature)
                    if recipe_dict is None:
                        recipe_dict, _ = decode_vec(vec)
                    values[col] = float(recipe_dict[col])
            except (KeyError, TypeError, ValueError):
                ok = False
                break
        if not ok:
            continue
        if process_limits:
            if recipe_dict is None:
                recipe_dict, _ = decode_vec(vec)
            if not constraint_engine.satisfies_process(recipe_dict, process_limits):
                continue
        rows.append({"vec": vec, "X_row": X_row, "values": values, "ad": ad})
    if not rows:
        return [], []

    def score(row, obj):    # "higher is better" for every objective, uniformly
        v = row["values"][obj["column"]]
        return v if obj["direction"] == "maximise" else -v

    directions = [o["direction"] for o in objectives]
    raw_values = [[row["values"][o["column"]] for o in objectives] for row in rows]
    fronts = pareto.non_dominated_sort(raw_values, directions)
    if not fronts:
        return [], []
    front_idx = fronts[0]
    front_sizes = [len(f) for f in fronts]

    def weighted_utility(row):
        return sum(o["weight"] * score(row, o) for o in objectives)

    front = sorted((rows[i] for i in front_idx), key=weighted_utility, reverse=True)
    diversity_gap = 0.75 * screener._d_ref
    selected, selected_std = [], []
    for row in front:
        if len(selected) >= max(max_front, 1):
            break
        std_vec = screener.scaler.transform(row["X_row"].values)[0]
        if any(np.linalg.norm(std_vec - other) < diversity_gap for other in selected_std):
            continue
        selected.append(row)
        selected_std.append(std_vec)

    # Crowding distance (spacing in objective space) over the FINAL
    # selected set — informational (surfaced per point as "crowding", e.g.
    # "most unique trade-off"), not a second selection filter: which
    # candidates survive is still decided by the applicability-gated,
    # spatially-diverse pass above (verified against real data), so this
    # is purely additive.
    crowd_values = [[row["values"][o["column"]] for o in objectives] for row in selected]
    crowding = pareto.crowding_distance(crowd_values) if selected else []

    pareto_points = []
    for rank, (row, crowd) in enumerate(zip(selected, crowding), start=1):
        recipe_dict, cand_edges = decode_vec(row["vec"])
        ordered, chem_recs = annotate_recipe(recipe_dict)
        risk = _RISK_FROM_APPLICABILITY_LABEL.get(row["ad"]["label"], "Moderate")
        pareto_points.append({
            "rank": rank, "recipe": ordered, "fixed": set(fixed),
            "objectives": dict(row["values"]),
            "applicability_pct": 100.0 - row["ad"]["percentile"],
            "risk": risk, "utility": weighted_utility(row),
            "chemical_recommendations": chem_recs, "edges": cand_edges,
            "crowding": (round(crowd, 3) if np.isfinite(crowd) else None),
        })
    return pareto_points, front_sizes


def _numeric_categorical_choices(choices):
    """Sorted (float, original_string) pairs for a categorical knob's
    numeric-parseable choices, e.g. a temperature column left as text
    ("700", "750", ..., "Missing") when a sheet mixed numbers with blanks."""
    out = []
    for ch in choices:
        try:
            out.append((float(ch), ch))
        except (TypeError, ValueError):
            continue
    return sorted(out)


def _pick_sensitivity_knobs(spec, cat_choices=None, max_knobs=3):
    """Choose a few search dimensions to sweep for robustness analysis —
    preferring named temperature/time/concentration knobs (the variables a
    researcher most often asks "how sensitive is this to small changes"
    about), falling back to other numeric dims otherwise. A categorical knob
    whose choices are mostly numeric-looking text (a temperature column left
    as strings, say) still qualifies — only genuinely non-orderable
    categories (reagent names, atmosphere) are excluded.
    """
    cat_choices = cat_choices or {}

    def sweepable(s):
        if s[0] == "num":
            return True
        choices = cat_choices.get(s[1], [])
        numeric = _numeric_categorical_choices(choices)
        return bool(choices) and len(numeric) / len(choices) >= 0.7

    candidates = [s for s in spec if sweepable(s)]
    patterns = [re.compile(r"temp|pyro", re.I),
               re.compile(r"time|holding|hold", re.I),
               re.compile(r"molar|concentration|conc\b", re.I)]
    picked = []
    for pat in patterns:
        for s in candidates:
            if s not in picked and pat.search(str(s[1])):
                picked.append(s)
                break
    for s in candidates:
        if len(picked) >= max_knobs:
            break
        if s not in picked:
            picked.append(s)
    return picked[:max_knobs]


def _sensitivity_analysis(vec, spec, vectorize, model, X_raw, cat_choices=None,
                          max_knobs=3, n_points=5, frac=0.15):
    """Sweep a few key knobs around one candidate's value, predicting the
    target at each point, to show whether the recommendation is robust to
    small process variations or requires unrealistic precision.

    Numeric knobs sweep continuously (±``frac`` of the observed range);
    numeric-looking categorical knobs (see ``_pick_sensitivity_knobs``) sweep
    through the nearest ``n_points`` discrete observed values instead.

    Returns a list of ``{"knob", "center", "sweep": [(value, predicted), ...],
    "pct_range", "robust"}``. Best-effort: a knob that fails to sweep (e.g. an
    unusual dtype) is skipped rather than aborting the whole analysis.
    """
    cat_choices = cat_choices or {}
    results = []
    try:
        base_pred = float(model.predict(vectorize(vec).reshape(1, -1))[0])
    except Exception:  # noqa: BLE001
        return results
    chosen = _pick_sensitivity_knobs(spec, cat_choices=cat_choices, max_knobs=max_knobs)
    for i, s in enumerate(spec):
        if s not in chosen:
            continue
        col = s[1]
        try:
            if s[0] == "num":
                lo, hi = float(np.percentile(X_raw[col], 1)), float(np.percentile(X_raw[col], 99))
                span = (hi - lo) or 1.0
                center = float(vec[i])
                delta = frac * span
                sweep_vals = np.clip(np.linspace(center - delta, center + delta, n_points), lo, hi)
            else:
                numeric = _numeric_categorical_choices(cat_choices[col])
                nvals = [v for v, _ in numeric]
                cur_idx = max(0, min(int(round(vec[i])), len(s[2]) - 1))
                center = float(nvals[min(cur_idx, len(nvals) - 1)]) if nvals else 0.0
                near = sorted(nvals, key=lambda v: abs(v - center))[:n_points]
                sweep_vals = sorted(near)
            points = []
            for sv in sweep_vals:
                v2 = np.array(vec, dtype=float)
                if s[0] == "num":
                    v2[i] = sv
                else:
                    numeric = _numeric_categorical_choices(cat_choices[col])
                    label = next((lbl for v, lbl in numeric if v == sv), None)
                    if label is None:      # shouldn't happen: sv came from `numeric` itself
                        continue
                    v2[i] = cat_choices[col].index(label)
                pred = float(model.predict(vectorize(v2).reshape(1, -1))[0])
                points.append((round(float(sv), 4), round(pred, 4)))
            preds = np.array([p[1] for p in points])
            pred_range = float(preds.max() - preds.min())
            pct_range = 100.0 * pred_range / (abs(base_pred) + 1e-9)
            results.append({"knob": col, "center": round(center, 4), "sweep": points,
                            "pct_range": round(pct_range, 1), "robust": pct_range < 15.0})
        except Exception:  # noqa: BLE001
            continue
    return results


def _reagent_info_for_candidate(candidate):
    """Per-reagent cost/hazard info (whatever the user has entered) for one
    candidate's recommended chemicals — informational only, never a
    fabricated recipe-level total: computing a true total needs the actual
    mass/volume used, which this optimizer's descriptor-space search doesn't
    track precisely enough to claim as a real number.
    """
    rows = []
    for rec in candidate.get("chemical_recommendations") or []:
        name = rec.get("recommended")
        if not name:
            continue
        entry = cost_model.ENGINE.get(name)
        rows.append({
            "reagent": name,
            "cost_per_kg": entry.cost_per_kg if entry else None,
            "cost_per_liter": entry.cost_per_liter if entry else None,
            "hazard_class": entry.hazard_class if entry else "Unknown",
            "corrosive": entry.corrosive if entry else False,
            "priced": bool(entry and entry.has_cost),
        })
    return rows


def _sustainability_for_candidate(candidate, X_raw, numeric_cols):
    """Green Score (see sustainability.py's docstring for what is and
    isn't scored — never a fabricated hazard/energy fact, only the user's
    own entered reagent data plus the recipe's own countable structure)
    for one candidate's decoded recipe."""
    recipe_dict = dict(candidate.get("recipe", []))
    chem_names = [rec.get("recommended")
                 for rec in candidate.get("chemical_recommendations") or []
                 if rec.get("recommended")]
    temp_pct = sustainability.estimate_temperature_percentile(recipe_dict, X_raw, numeric_cols)
    return sustainability.green_score(recipe_dict, chem_names, temperature_percentile=temp_pct)


def _rank_top_recipes(archive_vecs, archive_obj, *, target, sign, y, model,
                      vectorize, decode_vec, annotate_recipe, feat_cols, X_enc,
                      numeric_cols, cat_choices, X_raw, chemistry_schema,
                      chemistry_originals_df, label_map, cv_rmse, cv_r2,
                      top_n, risk_lambda, min_applicability, fixed,
                      n_priority_candidates=0, random_state=RANDOM_STATE,
                      pool_cap=800, process_limits=None):
    """Score every archived DE candidate; return up to `top_n` diverse,
    in-domain recommendations (see run_capacity_optimization's docstring).

    ``n_priority_candidates`` marks how many entries at the END of the
    concatenated archive are real-training-row-derived (see the caller) —
    those are always kept in full rather than being subject to subsampling,
    since there are at most a few hundred of them and they anchor the pool
    with genuinely in-domain options.

    Best-effort throughout: any failure here returns an empty list rather
    than propagating, since the single-best recipe must keep working
    regardless of whether this richer pass succeeds. Returns
    (top_recipes, n_candidates_considered).
    """
    if not archive_vecs:
        return [], 0
    all_vecs = np.vstack(archive_vecs)
    all_obj = np.concatenate(archive_obj)
    n_considered = len(all_vecs)
    if n_considered == 0:
        return [], 0

    # Cap the expensive per-candidate screening pass to a manageable size —
    # the DE-search portion of the archive can run into the tens of
    # thousands of points for a search with many knobs / generations.
    # Subsample RANDOMLY, not by best-raw-objective: a model will often
    # predict its most extreme values for combinations that push several
    # knobs to their bounds simultaneously, so objective-based truncation
    # would systematically fill the pool with the least trustworthy
    # (most extrapolated) candidates — exactly what applicability scoring
    # below is supposed to catch, not what should decide who gets IN.
    n_priority_candidates = min(n_priority_candidates, len(all_vecs))
    search_budget = pool_cap - n_priority_candidates
    if n_priority_candidates > 0:
        split = len(all_vecs) - n_priority_candidates
        priority_vecs, priority_obj = all_vecs[split:], all_obj[split:]
        search_vecs, search_obj = all_vecs[:split], all_obj[:split]
    else:
        priority_vecs = priority_obj = np.empty((0,))
        search_vecs, search_obj = all_vecs, all_obj
    if search_budget < len(search_vecs):
        rng_np = np.random.RandomState(random_state)
        keep = rng_np.choice(len(search_vecs), max(search_budget, 0), replace=False)
        search_vecs, search_obj = search_vecs[keep], search_obj[keep]
    if n_priority_candidates > 0:
        all_vecs = np.vstack([priority_vecs, search_vecs]) if len(search_vecs) else priority_vecs
        all_obj = np.concatenate([priority_obj, search_obj]) if len(search_obj) else priority_obj
    else:
        all_vecs, all_obj = search_vecs, search_obj

    try:
        numeric_schema = {c: float(X_raw[c].median()) for c in numeric_cols}
        screener = screening.Screener(
            model, feat_cols, numeric_schema, cat_choices, [target],
            X_enc.astype(float), pd.DataFrame({target: y}, index=X_enc.index),
            cv_rmse={target: cv_rmse}, cv_r2={target: cv_r2},
            display_labels=label_map, chemistry_schema=chemistry_schema,
            chemistry_originals=chemistry_originals_df,
        )
    except Exception:  # noqa: BLE001 - Top-N is best-effort, never fatal
        return [], n_considered

    y_range = float(y.max() - y.min()) or 1.0
    # A candidate must sit at least this far (in the same standardised space
    # the screener's own nearest-neighbour model uses) from every already-
    # selected one to count as genuinely different, not a near-duplicate.
    diversity_gap = 0.75 * screener._d_ref

    scored = []
    for vec, raw_obj in zip(all_vecs, all_obj):
        try:
            X_row = pd.DataFrame([vectorize(vec)], columns=feat_cols).astype(float)
            unc = screener.uncertainty(X_row)[target]
            ad = screener.applicability(X_row)
        except Exception:  # noqa: BLE001
            continue
        applicability_pct = 100.0 - ad["percentile"]
        if applicability_pct < min_applicability:
            continue
        if process_limits:
            cand_recipe, _ = decode_vec(vec)
            if not constraint_engine.satisfies_process(cand_recipe, process_limits):
                continue
        # raw_obj = sign * predicted, so -raw_obj is "higher is better" in
        # both maximise and minimise directions alike.
        utility = (-raw_obj
                  - risk_lambda * unc["sigma"]
                  - risk_lambda * (ad["percentile"] / 100.0) * y_range)
        scored.append({"vec": vec, "X_row": X_row, "predicted": sign * raw_obj,
                       "unc": unc, "ad": ad, "applicability_pct": applicability_pct,
                       "utility": utility})
    scored.sort(key=lambda s: s["utility"], reverse=True)

    selected, selected_std = [], []
    for cand in scored:
        if len(selected) >= max(int(top_n), 1):
            break
        std_vec = screener.scaler.transform(cand["X_row"].values)[0]
        if any(np.linalg.norm(std_vec - other) < diversity_gap for other in selected_std):
            continue
        selected.append(cand)
        selected_std.append(std_vec)

    top_recipes = []
    for rank, cand in enumerate(selected, start=1):
        recipe_dict, cand_edges = decode_vec(cand["vec"])
        ordered, chem_recs = annotate_recipe(recipe_dict)
        unc, ad = cand["unc"], cand["ad"]
        risk = _RISK_FROM_APPLICABILITY_LABEL.get(ad["label"], "Moderate")
        sim_rows = screener.similar(cand["X_row"], k=1)
        sim = sim_rows[0] if sim_rows else None

        reason = [ad["label"][:1].upper() + ad["label"][1:] + "."]
        if sim:
            measured = sim["measured"].get(target)
            reason.append(
                f"Closest known experiment is {sim['similarity']:.0f}% similar"
                + (f" (measured {measured:.3g})." if measured is not None else "."))
        if chem_recs:
            best_chem = max(chem_recs, key=lambda c: c["similarity"])
            reason.append(f"Uses {best_chem['recommended']} "
                          f"(descriptor similarity {best_chem['similarity']:.2f}).")
        reason.append(f"Confidence: {unc['conf_raw']}.")
        if cand_edges:
            reason.append("Sits at the edge of the observed range for "
                          + ", ".join(cand_edges) + ".")

        top_recipes.append({
            "rank": rank, "recipe": ordered, "fixed": set(fixed),
            "predicted": cand["predicted"], "lo": unc["lo"], "hi": unc["hi"],
            "sigma": unc["sigma"], "applicability_pct": cand["applicability_pct"],
            "risk": risk, "confidence": unc["conf_raw"],
            "similarity_pct": (sim["similarity"] if sim else None),
            "nearest_measured": (sim["measured"].get(target) if sim else None),
            "utility": cand["utility"], "chemical_recommendations": chem_recs,
            "reason": reason, "edges": cand_edges,
        })
    return top_recipes, n_considered


def _search_with_archive(objective_fn, bounds, random_state, de_kwargs,
                         on_generation=None):
    """Run one differential-evolution search, returning (best_vec, archive_vecs,
    archive_obj) — the winning point plus every candidate visited across all
    generations (not just the final population), for the diversity/Pareto
    passes downstream.

    Iterates the solver generation-by-generation via scipy's internal
    ``DifferentialEvolutionSolver`` class instead of the one-shot functional
    API, so the archive can be captured — this reproduces the functional
    API's result bit-for-bit for the same seed (verified against
    ``differential_evolution()`` directly). Falls back to the plain
    single-best functional search (empty archive) if the internal solver is
    unavailable for any reason — callers must treat an empty archive as
    "no diversity/Pareto pass possible, single best still valid".
    """
    archive_vecs, archive_obj = [], []
    try:
        from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
        solver = DifferentialEvolutionSolver(
            objective_fn, bounds, rng=random_state, **de_kwargs)
        for _nit in range(1, solver.maxiter + 1):
            try:
                next(solver)
            except StopIteration:
                break
            archive_vecs.append(np.array(
                [solver._scale_parameters(p) for p in solver.population]))
            archive_obj.append(solver.population_energies.copy())
            if on_generation is not None:
                on_generation(_nit, solver.maxiter)
            if solver.converged():
                break
        best_i = int(np.argmin(solver.population_energies))
        best_vec = solver._scale_parameters(solver.population[best_i])
    except Exception:  # noqa: BLE001 - fall back to the original, simpler search
        archive_vecs, archive_obj = [], []
        result = differential_evolution(objective_fn, bounds, seed=random_state, **de_kwargs)
        best_vec = result.x
    return best_vec, archive_vecs, archive_obj


def _run_bo_proposals(*, X_enc, y, spec, bounds, integrality, vectorize,
                      decode_vec, annotate_recipe, direction, target, cv_r2,
                      cv_rmse, label_map, fixed, n_rows, n_numeric, n_categorical,
                      column_type_mode, q, xi, random_state, progress=None):
    """Fit a Gaussian-process surrogate on the already-built feature space and
    propose the next ``q`` experiments to run by Expected-Improvement Bayesian
    optimization. Reuses the exact ``vectorize``/``decode_vec``/``annotate_recipe``
    machinery the differential-evolution optimizer builds, so the search space,
    chemistry descriptors and constraints are identical — only the surrogate and
    the selection criterion differ (probabilistic GP + acquisition, rather than
    a point-model optimum)."""
    def _p(msg, frac=None):
        if progress is not None:
            progress(msg, frac)

    notes = []
    bo_direction = "maximize" if direction == "maximise" else "minimize"
    y_best = float(y.max()) if direction == "maximise" else float(y.min())
    proposals = []
    if not bounds:
        notes.append("No free knobs to vary — every controllable column is "
                     "fixed or constant, so there is nothing to propose.")
    else:
        _p("Fitting Gaussian-process surrogate over the search space…", 0.45)
        surrogate = bayesopt.fit_surrogate(
            X_enc.astype(float).values, y, direction=bo_direction,
            random_state=random_state)
        _bo_cb = progress and (lambda i, qq: _p(
            f"Proposing experiment {i + 1}/{qq} (maximizing Expected "
            f"Improvement)…", 0.5 + 0.45 * i / max(qq, 1)))
        raw = bayesopt.propose_batch(
            surrogate, bounds, lambda v: vectorize(v), q=q,
            integrality=integrality, xi=xi, random_state=random_state,
            de_kwargs=dict(popsize=12, maxiter=40), callback=_bo_cb or None)
        _p("Decoding proposed recipes and mapping reagents…", 0.97)
        for rank, p in enumerate(raw, start=1):
            recipe_, edges_ = decode_vec(p.x)
            ordered_, chem_recs_ = annotate_recipe(recipe_)
            # Expected improvement over the best result observed so far, in the
            # target's own units (EI itself is in internal maximization space).
            proposals.append({
                "rank": rank, "recipe": ordered_, "fixed": set(fixed),
                "predicted": p.mean, "sigma": p.sigma, "ei": p.ei,
                "lo": p.mean - 1.96 * p.sigma, "hi": p.mean + 1.96 * p.sigma,
                "edges": edges_, "chemical_recommendations": chem_recs_,
            })
    return dict(
        bo=True, target=target, direction=direction,
        r2=cv_r2, rmse=cv_rmse, n_rows=n_rows, n_knobs=len(spec),
        n_numeric=n_numeric, n_categorical=n_categorical,
        obs_min=float(y.min()), obs_max=float(y.max()), y_best=y_best,
        labels=label_map, proposals=proposals, notes=notes,
        mode=("column-type" if column_type_mode else "manual"))


def run_capacity_optimization(data_path, target, excluded, fixed, direction,
                              min_support=3, random_state=RANDOM_STATE, sheet=None,
                              features=None, col_types=None, ids=None, mixed=None,
                              chemistry_enabled=True, chemistry_schema=None,
                              chemistry_mode="automatic",
                              top_n=10, risk_lambda=1.0, min_applicability=60.0,
                              constraints_text="", objectives_text="",
                              bayesopt_mode=False, bo_batch=5, bo_xi=0.01,
                              run_de=True, progress=None):
    """
    Train XGBoost to predict `target` from CONTROLLABLE inputs only, then use
    differential evolution to find the input recipe with the best predicted
    target. `fixed` pins chosen knobs (e.g. a test current density).

    Beyond the single best recipe, also returns ``top_recipes``: up to
    ``top_n`` diverse, in-domain candidates pulled from every generation the
    search visited (not just the final, converged population), each scored
    with a screening.Screener built from this optimizer's own model so the
    same uncertainty/applicability/similarity machinery used elsewhere in the
    app backs the recommendations. ``risk_lambda`` penalises both prediction
    uncertainty and how far a candidate sits from the training domain (an
    XGBoost point-model has no per-sample epistemic variance the way a forest
    does, so the domain-distance term is what actually keeps the ranking from
    just picking whatever extrapolates furthest). ``min_applicability`` (0-100)
    rejects candidates below that in-domain score outright.

    ``constraints_text``: one laboratory constraint per line (see
    ``constraint_engine.parse``), tightening the search space itself before
    any candidate is generated — e.g. ``"Temperature <= 1000\\nBiomass IN
    [Rice husk, Bamboo]\\nNO STRONG ACID\\nstages <= 2"``.

    ``objectives_text``: 'column:maximize|minimize:weight' entries (see
    ``_parse_objectives_text``). When it names more than just the primary
    target, a SECOND, separate multi-objective search runs (the single-best
    recipe and Top-N list above are always driven by the primary target
    alone, unaffected) and its result is returned as ``pareto_front`` —
    non-dominated recipes across every named objective, each showing every
    objective's value so you can see the actual trade-offs rather than one
    compromise point. Outcome-type objectives (e.g. a second measured
    property) get their own trained model; objectives that name a
    controllable knob itself (e.g. "minimise Temperature") need no model —
    read directly from the candidate.

    Which columns are controllable is driven by the Column-types configuration
    when available: pass ``features`` (the columns tagged role='Feature') and
    ``col_types`` ({col: 'numeric'|'categorical'}). Everything else — targets,
    ids, excluded and measured-after-synthesis outcomes — is held out
    automatically. Without ``features`` it falls back to "everything except the
    target and the `excluded` list".

    Returns a result dict for the UI.
    """
    from xgboost import XGBRegressor

    def _p(msg, frac=None):
        if progress is not None:
            progress(msg, frac)

    _p("Loading and cleaning the dataset…", 0.04)
    raw_df = read_any(data_path, sheet=sheet)
    label_map = mixed_feature_labels(mixed or [], columns=raw_df.columns)
    generated_cols = set(label_map)
    # Drop id columns and split any messy columns into anonymized A/B/C features,
    # so the optimizer's knobs line up with the training pipeline.
    df = prepare_raw(raw_df, ids or [], mixed or [])
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in the file.")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).reset_index(drop=True)

    col_types = col_types or {}
    # Let fixed knobs use either the internal id or the readable UI label.
    by_label = {v: k for k, v in label_map.items()}
    fixed = {by_label.get(k, k): v for k, v in (fixed or {}).items()}
    column_type_mode = features is not None
    if column_type_mode:
        # COLUMN-TYPE MODE: optimise only over role='Feature' columns (plus any
        # features generated from parsed messy columns). Targets / ids / excluded
        # / measured-outcome columns are automatically held out.
        feat_set = set(features)
        controllable = [c for c in df.columns
                        if c != target and (c in feat_set or c in generated_cols)]
    else:
        controllable = [c for c in df.columns if c not in (set(excluded) | {target})]
    if not controllable:
        raise ValueError("No controllable columns to optimize. In the Column types "
                         "tab, mark the synthesis variables you can set as role "
                         "'Feature' (and the outcome as 'Target').")
    fixed = {k: v for k, v in fixed.items() if k in controllable}
    y = df[target].values

    # Split knobs into numeric ranges and categorical choices, honoring the
    # Column-types Numeric/Categorical assignment (falling back to dtype).
    X_raw = df[controllable].copy()
    if chemistry_enabled:
        if chemistry_schema and chemistry_schema.get("columns"):
            saved_schema = dict(chemistry_schema)
            X_raw = chemistry.ENGINE.transform_with_schema(X_raw, saved_schema)
            chemistry_schema = saved_schema
        else:
            chemistry_config = chemistry.ENGINE.auto_configure(
                X_raw, groups=pd.Series(np.arange(len(X_raw))),
                requested_mode=chemistry_mode)
            chemistry_expansion = chemistry.ENGINE.transform(
                X_raw, config=chemistry_config)
            chemistry_schema = chemistry_expansion.metadata
            X_raw = chemistry_expansion.frame
        # Optimization occurs in descriptor space; any retained identity label
        # is mapped back to a feasible known reagent after the search. Snapshot
        # the original chemical names first so the screening engine (used for
        # the Top-N recommendation pass below) can still show real reagent
        # names in "nearest known experiment" lookups.
        retained_sources = [c for c in chemistry_schema.get("columns", {}) if c in X_raw]
        chemistry_originals_df = X_raw[retained_sources].copy() if retained_sources else None
        X_raw = X_raw.drop(columns=retained_sources, errors="ignore")
    else:
        chemistry_schema = {"columns": {}, "interactions": []}
        chemistry_originals_df = None
    controllable = list(X_raw.columns)

    # Fixed chemical names become fixed descriptor profiles; the optimizer never
    # treats a reagent label as a one-hot category.
    expanded_fixed = dict(fixed)
    for source, info in chemistry_schema.get("columns", {}).items():
        if source not in fixed:
            continue
        descriptor = chemistry.ENGINE.generator.describe(str(fixed[source]))
        for key, value in descriptor.feature_values().items():
            expanded_fixed[f"{info['prefix']}_{key}"] = value
        expanded_fixed.pop(source, None)
    fixed = {k: v for k, v in expanded_fixed.items() if k in controllable}

    numeric_cols, cat_choices = [], {}
    for c in controllable:
        kind = col_types.get(c)
        if c.startswith("numeric_feature_"):
            kind = "numeric"
        elif c.startswith("group_label_") or c.startswith("text_modifier_"):
            kind = "categorical"
        is_num = (kind == "numeric") or (kind is None
                                         and pd.api.types.is_numeric_dtype(X_raw[c]))
        if is_num:
            s = coerce_numeric_series(X_raw[c])   # '--'/'none' -> 0, unknown -> NaN
            med = s.median()
            X_raw[c] = s.fillna(0.0 if pd.isna(med) else med)
            numeric_cols.append(c)
        else:
            X_raw[c] = normalize_categorical_series(X_raw[c])
            counts = X_raw[c].value_counts()
            cat_choices[c] = sorted(counts[counts >= min_support].index.tolist()) \
                or sorted(counts.index.tolist())

    # Laboratory constraints (temperature <= 1000, forbidden reagents, only
    # certain biomass, chemical-class shortcuts like "NO STRONG ACID",
    # process shortcuts like "stages <= 2", ...) tighten the search space
    # itself — an eliminated candidate is never generated in the first
    # place, rather than generated and then filtered out after the fact.
    # Process shortcuts are the one exception: stage/step count is a
    # property of the DECODED recipe, not a single search dimension, so
    # they're checked per-candidate later (see process_limits below).
    constraint_notes = []
    numeric_overrides = {}
    parsed_constraints = constraint_engine.parse(constraints_text)
    process_limits = constraint_engine.process_limits(parsed_constraints)
    if parsed_constraints:
        expanded_constraints, chem_notes = constraint_engine.expand_chemical_constraints(
            parsed_constraints, chemistry_schema)
        constraint_notes.extend(chem_notes)
        numeric_overrides, column_notes = constraint_engine.apply(
            expanded_constraints, numeric_cols, cat_choices, X_raw, by_label)
        constraint_notes.extend(column_notes)
        for c in parsed_constraints:
            if c.get("kind") == "process":
                constraint_notes.append(f"{c['column']} {c['op']} {c['value']}")
                continue
            if c.get("kind") != "column":
                continue
            col = by_label.get(c["column"], c["column"])
            if col in fixed:
                constraint_notes.append(
                    f"'{c['column']}' is a fixed knob — its constraint isn't "
                    "checked against the pinned value.")

    # Interaction columns stay in the fitted model but are derived inside each
    # optimizer evaluation; they are not independent knobs a user could set.
    interaction_columns = set(chemistry_schema.get("interactions", []))
    search_numeric_cols = [c for c in numeric_cols if c not in interaction_columns]

    X_enc = pd.get_dummies(X_raw, drop_first=False)
    feat_cols = X_enc.columns.tolist()
    col_pos = {c: i for i, c in enumerate(feat_cols)}
    n_feat = len(feat_cols)

    _p(f"Training surrogate model on {len(search_numeric_cols) + len(cat_choices)} "
       "knobs (5-fold cross-validation)…", 0.22)
    model = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=random_state, n_jobs=-1, verbosity=0)
    oof_pred = cross_val_predict(
        model, X_enc.astype(float).values, y,
        cv=KFold(5, shuffle=True, random_state=random_state))
    cv_r2 = float(r2_score(y, oof_pred))
    cv_rmse = float(np.sqrt(mean_squared_error(y, oof_pred)))
    model.fit(X_enc.astype(float).values, y)
    _p(f"Surrogate ready (cross-validated R² = {cv_r2:.2f}).", 0.38)

    # Multi-objective setup. objectives[0] is always the primary target above;
    # additional entries either name a controllable knob (no model needed —
    # read directly from a candidate) or a second measured/outcome column
    # (gets its own trained model, reusing the same encoded feature space).
    objectives = _parse_objectives_text(objectives_text, target, direction)
    extra_models, objective_notes = {}, []
    if len(objectives) > 1:
        keep_objectives = [objectives[0]]
        for obj in objectives[1:]:
            col = obj["column"]
            if col in numeric_cols:
                keep_objectives.append(obj)
                continue
            if col in cat_choices:
                # A knob classified categorical (e.g. temperature left as text
                # strings when the sheet mixed numbers with "Missing") can
                # still be a numeric objective if its choices parse as numbers
                # — same reasoning as the numeric-comparison constraint fix.
                choices = cat_choices[col]
                numeric_choices = 0
                for ch in choices:
                    try:
                        float(ch)
                        numeric_choices += 1
                    except (TypeError, ValueError):
                        pass
                if choices and numeric_choices / len(choices) >= 0.7:
                    keep_objectives.append(obj)
                else:
                    objective_notes.append(
                        f"Objective '{col}' names a non-numeric categorical knob "
                        "(e.g. a reagent or atmosphere choice) — only numeric "
                        "objectives are supported. Skipped.")
                continue
            if col not in df.columns:
                objective_notes.append(f"Objective '{col}' was not found in the data — skipped.")
                continue
            try:
                y_obj_full = pd.to_numeric(df[col], errors="coerce")
                mask = y_obj_full.notna().values
                if mask.sum() < 10:
                    objective_notes.append(
                        f"Objective '{col}' has too few observed values ({int(mask.sum())}) "
                        "— skipped.")
                    continue
                obj_model = XGBRegressor(
                    n_estimators=400, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=random_state, n_jobs=-1, verbosity=0)
                obj_model.fit(X_enc.astype(float).values[mask], y_obj_full.values[mask])
                extra_models[col] = obj_model
                keep_objectives.append(obj)
            except Exception as e:  # noqa: BLE001 - one bad objective shouldn't sink the run
                objective_notes.append(f"Objective '{col}' failed to train ({e}) — skipped.")
        objectives = keep_objectives
        if len(objectives) < 2:
            objective_notes.append("Fewer than 2 usable objectives — running single-objective.")

    # Base vector holds the fixed-knob contributions (never varied by the search).
    base = np.zeros(n_feat)
    for c, v in fixed.items():
        if c in numeric_cols:
            try:
                base[col_pos[c]] = float(v)
            except ValueError:
                pass
        else:
            pos = col_pos.get(f"{c}_{v}")
            if pos is not None:
                base[pos] = 1.0

    # Search space = controllable knobs that are NOT fixed.
    bounds, integrality, spec = [], [], []
    for c in search_numeric_cols:
        if c in fixed:
            continue
        if c in numeric_overrides:
            lo, hi = numeric_overrides[c]
        else:
            lo, hi = np.percentile(X_raw[c], 1), np.percentile(X_raw[c], 99)
        if lo == hi:
            hi = lo + 1e-6
        if lo > hi:      # contradictory constraints (e.g. >=1000 and <=700) -> widest safe fallback
            lo, hi = float(np.percentile(X_raw[c], 1)), float(np.percentile(X_raw[c], 99))
            constraint_notes.append(f"Contradictory constraints on {c} — ignored, using observed range.")
        bounds.append((lo, hi)); integrality.append(
            chemistry.descriptor_is_discrete(c))
        spec.append(("num", c, col_pos.get(c)))
    for c, choices in cat_choices.items():
        if c in fixed:
            continue
        bounds.append((0, len(choices) - 1)); integrality.append(True)
        spec.append(("cat", c, [col_pos.get(f"{c}_{ch}") for ch in choices]))

    sign = -1.0 if direction == "maximise" else 1.0

    def vectorize(vec):
        x = base.copy()
        for v, s in zip(vec, spec):
            if s[0] == "num":
                if s[2] is not None:
                    x[s[2]] = v
            else:
                i = max(0, min(int(round(v)), len(s[2]) - 1))
                if s[2][i] is not None:
                    x[s[2][i]] = 1.0
        for interaction in chemistry_schema.get("interaction_specs", []):
            out = col_pos.get(interaction["feature"])
            left = col_pos.get(interaction["left"])
            right = col_pos.get(interaction["right"])
            if out is not None and left is not None and right is not None:
                x[out] = x[left] * x[right]
        return x

    def decode_vec(vec):
        """One search vector -> (recipe dict incl. fixed knobs, edge-of-data flags)."""
        recipe_ = dict(fixed)
        edges_ = []
        for v, s in zip(vec, spec):
            if s[0] == "num":
                recipe_[s[1]] = round(float(v), 4)
                lo, hi = np.percentile(X_raw[s[1]], 1), np.percentile(X_raw[s[1]], 99)
                span = (hi - lo) or 1.0
                if abs(v - lo) < 0.02 * span or abs(v - hi) < 0.02 * span:
                    edges_.append(s[1])
            else:
                i = max(0, min(int(round(v)), len(cat_choices[s[1]]) - 1))
                recipe_[s[1]] = cat_choices[s[1]][i]
        return recipe_, edges_

    chemistry_model_columns = set(chemistry_schema.get("interactions", []))
    for info in chemistry_schema.get("columns", {}).values():
        chemistry_model_columns.update(info.get("descriptor_columns", []))

    def annotate_recipe(recipe_dict):
        """Ordered display list + nearest-known-reagent recommendations for one recipe."""
        ordered_ = [(c, recipe_dict[c]) for c in controllable
                   if c not in chemistry_model_columns]
        chem_recs_ = []
        central = ("Is_Strong_Acid", "Is_Strong_Base", "pKa", "pKb",
                  "EstimatedOxidationTendency", "MolecularWeight", "Contains_Hydroxide")
        for source, info in chemistry_schema.get("columns", {}).items():
            nearest = chemistry.ENGINE.nearest_known_profile(recipe_dict, info["prefix"], top=3)
            if nearest:
                ordered_.append((f"Recommended {source}", nearest[0].name))
                chem_recs_.append({
                    "column": source, "recommended": nearest[0].name,
                    "similarity": nearest[0].score,
                    "alternatives": [{"name": item.name, "score": item.score}
                                     for item in nearest[1:]],
                })
            for key in central:
                feature = f"{info['prefix']}_{key}"
                if feature in recipe_dict:
                    ordered_.append((chemistry.descriptor_display_name(feature),
                                     recipe_dict[feature]))
        return ordered_, chem_recs_

    # BAYESIAN OPTIMIZATION path: the whole feature space (search bounds,
    # chemistry descriptors, constraints, encode/decode) is now built exactly as
    # the differential-evolution optimizer uses it. Fit a GP surrogate over it
    # and propose the next experiments. run_de=False returns here without the DE
    # search / Top-N / Pareto passes, so the "Suggest Experiments" tab is fast.
    if bayesopt_mode:
        bo_result = _run_bo_proposals(
            X_enc=X_enc, y=y, spec=spec, bounds=bounds, integrality=integrality,
            vectorize=vectorize, decode_vec=decode_vec,
            annotate_recipe=annotate_recipe, direction=direction, target=target,
            cv_r2=cv_r2, cv_rmse=cv_rmse, label_map=label_map, fixed=fixed,
            n_rows=len(df), n_numeric=len(search_numeric_cols),
            n_categorical=len(cat_choices), column_type_mode=column_type_mode,
            q=bo_batch, xi=bo_xi, random_state=random_state, progress=progress)
        if not run_de:
            return bo_result

    def objective(vec):
        return sign * float(model.predict(vectorize(vec).reshape(1, -1))[0])

    de_kwargs = dict(
        integrality=integrality, popsize=15, maxiter=80, tol=1e-4,
        mutation=(0.5, 1.0), recombination=0.9, polish=False, updating="immediate")

    # Search generation-by-generation (instead of the one-shot functional API)
    # so every candidate DE visits along the way can be archived, not just the
    # final, tightly-converged population. This is what makes the Top-N
    # diversity pass below meaningful: by convergence DE's population has
    # narrowed to near-duplicates of the single best point.
    _p("Searching the recipe space (differential evolution)…", 0.42)
    _de_cb = progress and (lambda g, m: _p(
        f"Searching recipe space — generation {g}/{m}…", 0.42 + 0.38 * g / max(m, 1)))
    best_vec, archive_vecs, archive_obj = _search_with_archive(
        objective, bounds, random_state, de_kwargs, on_generation=_de_cb or None)

    # Also seed the Top-N candidate pool with real training recipes (what
    # does the fitted model predict for experiments actually run?). A free
    # search that varies every knob independently will almost always land on
    # combinations that read as "novel" from curse-of-dimensionality alone —
    # ~800 rows cannot densely cover a 20-30 dimensional space — even when
    # every individual knob value stays well inside the observed range. Real
    # recipes score high applicability by construction, so this is what
    # actually gives the applicability-gated ranking below trustworthy,
    # non-extrapolated options to choose from, not just DE's exploration.
    n_priority_candidates = 0
    if archive_vecs and spec:
        try:
            sample_idx = X_raw.index
            if len(sample_idx) > 500:
                rng_np = np.random.RandomState(random_state)
                sample_idx = pd.Index(rng_np.choice(sample_idx, 500, replace=False))
            train_vecs = []
            for ridx in sample_idx:
                row = X_raw.loc[ridx]
                v = []
                for s in spec:
                    if s[0] == "num":
                        v.append(float(row[s[1]]))
                    else:
                        choices = cat_choices[s[1]]
                        val = row[s[1]]
                        v.append(float(choices.index(val)) if val in choices else 0.0)
                train_vecs.append(v)
            if train_vecs:
                train_vecs = np.array(train_vecs, dtype=float)
                train_obj = np.array([objective(v) for v in train_vecs])
                archive_vecs.append(train_vecs)
                archive_obj.append(train_obj)
                n_priority_candidates = len(train_vecs)
        except Exception:  # noqa: BLE001 - purely additive, never fatal
            pass

    recipe, edges = decode_vec(best_vec)
    predicted = float(model.predict(vectorize(best_vec).reshape(1, -1))[0])
    ordered, chemical_recommendations = annotate_recipe(recipe)
    sensitivity = _sensitivity_analysis(best_vec, spec, vectorize, model, X_raw,
                                        cat_choices=cat_choices)

    _p("Ranking recommended experiments (uncertainty + applicability)…", 0.82)
    top_recipes, n_candidates_considered = _rank_top_recipes(
        archive_vecs, archive_obj, target=target, sign=sign, y=y, model=model,
        vectorize=vectorize, decode_vec=decode_vec, annotate_recipe=annotate_recipe,
        feat_cols=feat_cols, X_enc=X_enc, numeric_cols=numeric_cols,
        cat_choices=cat_choices, X_raw=X_raw, chemistry_schema=chemistry_schema,
        chemistry_originals_df=chemistry_originals_df, label_map=label_map,
        cv_rmse=cv_rmse, cv_r2=cv_r2, top_n=top_n, risk_lambda=risk_lambda,
        min_applicability=min_applicability, fixed=fixed,
        n_priority_candidates=n_priority_candidates, random_state=random_state,
        process_limits=process_limits)

    # ---- Multi-objective Pareto front (separate search; single-best/Top-N
    # above are always driven by the primary target alone, unaffected). ----
    pareto_front = []
    pareto_fronts_summary = []
    if len(objectives) > 1:
        _p("Computing the multi-objective Pareto front…", 0.9)
        try:
            # Normalise every objective's raw value to roughly [0, 1] (min-max
            # over its OWN observed range) before weighting, so a target
            # spanning hundreds of mAh/g doesn't drown out one spanning a few
            # percent. Direction is folded in here too (minimise -> flipped).
            obj_ranges = {}
            for obj in objectives:
                col = obj["column"]
                if col == target:
                    vals = y
                elif col in extra_models:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna().values
                else:
                    # Knob objective — may be a numeric column OR a numeric-
                    # looking categorical one (values stored as strings); coerce
                    # either way rather than taking a lexicographic min/max.
                    vals = pd.to_numeric(X_raw[col], errors="coerce").dropna().values
                if len(vals) == 0:
                    raise ValueError(f"objective '{col}' has no numeric values to range over")
                lo_r, hi_r = float(np.min(vals)), float(np.max(vals))
                obj_ranges[col] = (lo_r, (hi_r - lo_r) or 1.0)

            def objective_raw(vec, col):
                """Raw value of one objective for a candidate, or None if this
                candidate doesn't have a meaningful value for it — e.g. a
                numeric-looking categorical knob objective landed on a
                non-numeric choice like "Missing" (unrecorded data)."""
                if col in extra_models:
                    return float(extra_models[col].predict(vectorize(vec).reshape(1, -1))[0])
                if col == target:
                    return float(model.predict(vectorize(vec).reshape(1, -1))[0])
                recipe_v, _ = decode_vec(vec)
                try:
                    return float(recipe_v[col])
                except (TypeError, ValueError):
                    return None

            def multi_objective(vec):    # DE minimises -> this is "lower is better"
                total = 0.0
                for obj in objectives:
                    lo_r, span_r = obj_ranges[obj["column"]]
                    raw = objective_raw(vec, obj["column"])
                    # No usable value for this objective on this candidate ->
                    # worst case (1.0, post-normalisation), steering the
                    # search away rather than crashing on it.
                    norm = 1.0 if raw is None else (raw - lo_r) / span_r
                    total += obj["weight"] * (norm if obj["direction"] == "minimise" else -norm)
                return total

            _, pareto_archive_vecs, pareto_archive_obj = _search_with_archive(
                multi_objective, bounds, random_state, de_kwargs)

            pareto_n_priority = 0
            if pareto_archive_vecs and n_priority_candidates:
                try:
                    train_pareto_obj = np.array([multi_objective(v) for v in train_vecs])
                    pareto_archive_vecs.append(train_vecs)
                    pareto_archive_obj.append(train_pareto_obj)
                    pareto_n_priority = len(train_vecs)
                except Exception:  # noqa: BLE001
                    pass

            pareto_front, pareto_fronts_summary = _pareto_front_candidates(
                pareto_archive_vecs, pareto_archive_obj, objectives=objectives,
                target=target, model=model, y=y, extra_models=extra_models,
                vectorize=vectorize, decode_vec=decode_vec, annotate_recipe=annotate_recipe,
                feat_cols=feat_cols, X_enc=X_enc, numeric_cols=numeric_cols,
                cat_choices=cat_choices, X_raw=X_raw, chemistry_schema=chemistry_schema,
                chemistry_originals_df=chemistry_originals_df, label_map=label_map,
                cv_rmse=cv_rmse, cv_r2=cv_r2, min_applicability=min_applicability,
                fixed=fixed, n_priority_candidates=pareto_n_priority,
                random_state=random_state, process_limits=process_limits)
        except Exception as e:  # noqa: BLE001 - Pareto is additive, never fatal
            objective_notes.append(f"Pareto front computation failed: {e}")

    # ---- Decision-support enrichment: strategy labels, active-learning
    # score, feasibility, reagent cost/hazard (whatever's entered), a
    # Green Score sustainability heuristic, a balanced portfolio, a
    # Pareto-front tradeoff explanation, and training-data coverage gaps.
    # All best-effort over data already computed above — never able to
    # affect the search or the single-best/Top-N/Pareto results themselves.
    _p("Scoring cost, sustainability, feasibility, and coverage gaps…", 0.95)
    try:
        top_recipes = planner.enrich_candidates(top_recipes) if top_recipes else top_recipes
        for cand in top_recipes:
            cand["reagents"] = _reagent_info_for_candidate(cand)
            cand["sustainability"] = _sustainability_for_candidate(cand, X_raw, numeric_cols)
    except Exception:  # noqa: BLE001
        pass
    try:
        if pareto_front:
            pareto_front = planner.enrich_candidates(pareto_front)
            for cand in pareto_front:
                cand["reagents"] = _reagent_info_for_candidate(cand)
                cand["sustainability"] = _sustainability_for_candidate(cand, X_raw, numeric_cols)
            pareto_front = tradeoffs.explain(pareto_front, objectives)
    except Exception:  # noqa: BLE001
        pass
    batch_plan, batch_plan_summary = [], {}
    try:
        if top_recipes:
            batch_plan = portfolio.build_portfolio(top_recipes, size=min(10, top_n))
            batch_plan_summary = portfolio.summarize(batch_plan)
    except Exception:  # noqa: BLE001
        pass
    research_gaps = []
    try:
        research_gaps = research_gap.detect_gaps(X_raw, numeric_cols, cat_choices)
        # research_gap.py is generic and knows nothing about this app's
        # parsed-messy-column display labels — relabel here so a gap on
        # e.g. 'group_label_B4' reads as "Pretreat 2: code" instead.
        def relabel(col):
            return screening._pretty(label_map.get(col, col))
        for g in research_gaps:
            for key in ("column", "column_a", "column_b"):
                if key in g:
                    pretty_col = relabel(g[key])
                    g["description"] = g["description"].replace(g[key], pretty_col)
                    g[key] = pretty_col
    except Exception:  # noqa: BLE001
        pass

    return dict(target=target, r2=cv_r2, rmse=cv_rmse, predicted=predicted,
                obs_min=float(y.min()), obs_max=float(y.max()),
                recipe=ordered, fixed=set(fixed), edges=edges,
                labels=label_map,
                n_rows=len(df), n_knobs=len(spec),
                mode=("column-type" if column_type_mode else "manual"),
                n_numeric=len(search_numeric_cols), n_categorical=len(cat_choices),
                chemistry_schema=chemistry_schema,
                chemical_recommendations=chemical_recommendations,
                optimized_over="chemistry descriptors, mapped back to known reagents",
                top_recipes=top_recipes, n_candidates_considered=n_candidates_considered,
                risk_lambda=risk_lambda, min_applicability=min_applicability,
                constraint_notes=constraint_notes,
                pareto_front=pareto_front, objectives_used=objectives,
                pareto_fronts_summary=pareto_fronts_summary,
                objective_notes=objective_notes, sensitivity=sensitivity,
                batch_plan=batch_plan, batch_plan_summary=batch_plan_summary,
                research_gaps=research_gaps)


# =============================================================================
# PROGRESS  (shared, visible feedback for every long-running background task)
# =============================================================================
class Progress:
    """Progress state for one long operation, driven from a worker thread and
    rendered every frame by ``draw_progress_panel``. Workers call
    ``begin()`` / ``step()`` / ``finish()`` / ``fail()``; the UI shows a real
    progress bar, an elapsed-time readout, and a live, timestamped log of every
    step so the user can watch the processing happen rather than staring at a
    frozen window."""

    def __init__(self, idle_message=""):
        self.idle_message = idle_message
        self.active = False
        self.finished_ok = False
        self.fraction = 0.0
        self.message = idle_message
        self.error = ""
        self.log = []            # list of (elapsed_seconds, text)
        self.elapsed = 0.0
        self._t0 = 0.0

    def begin(self, message="Starting…"):
        self.active = True
        self.finished_ok = False
        self.fraction = 0.0
        self.error = ""
        self.log = []
        self._t0 = time.time()
        self.step(message, 0.0)

    def step(self, message, fraction=None):
        if fraction is not None:
            self.fraction = max(0.0, min(1.0, float(fraction)))
        self.message = message
        self.elapsed = (time.time() - self._t0) if self._t0 else 0.0
        self.log.append((self.elapsed, message))

    def finish(self, message="Done."):
        self.step(message, 1.0)
        self.active = False
        self.finished_ok = True

    def fail(self, message):
        self.error = message
        self.step("Failed.")
        self.active = False

    def sub(self, lo, hi):
        """Return a ``(message, frac)`` callback mapping frac in [0,1] onto the
        band [lo, hi] — lets a nested loop (DE generations, BO proposals) report
        its own 0→1 progress into a slice of the overall bar."""
        def cb(message, frac=None):
            if frac is None:
                self.step(message)
            else:
                self.step(message, lo + (hi - lo) * max(0.0, min(1.0, float(frac))))
        return cb


def draw_progress_panel(prog, *, show_when_idle=False, log_height=150):
    """Render a Progress object: coloured bar + status + elapsed + live log."""
    if not (prog.active or prog.finished_ok or prog.error or show_when_idle):
        return
    frac = 1.0 if prog.finished_ok else prog.fraction
    if prog.error:
        fill = _rgba("#c62828")
    elif prog.finished_ok:
        fill = _rgba("#2e7d32")
    else:
        fill = _rgba(_PALETTE["accent"])
    imgui.push_style_color(imgui.Col_.plot_histogram, fill)
    imgui.progress_bar(frac, imgui.ImVec2(-1.0, 0.0), f"{int(round(frac * 100))}%")
    imgui.pop_style_color()

    if prog.active:
        spin = " " + "|/-\\"[int(time.time() * 8) % 4]
    else:
        spin = ""
    status_col = _rgba("#c62828") if prog.error else DIM
    imgui.text_colored(status_col, f"{prog.message}{spin}    ({prog.elapsed:.1f}s)")

    if prog.log:
        node = imgui.tree_node(
            f"Processing details — {len(prog.log)} step(s)###plog{id(prog)}")
        if node:
            # begin_child must always be matched by end_child, even when it
            # returns false (content clipped) — hence the unconditional pair.
            imgui.begin_child(f"##log{id(prog)}", imgui.ImVec2(0.0, log_height),
                              imgui.ChildFlags_.border)
            for t, msg in prog.log[-300:]:
                imgui.text_colored(DIM, f"[{t:6.1f}s]  {msg}")
            if prog.active:                     # follow the tail while running
                imgui.set_scroll_here_y(1.0)
            imgui.end_child()
            imgui.tree_pop()


# =============================================================================
# APPLICATION STATE  (immediate-mode UI redraws every frame; state lives here)
# =============================================================================
class AppState:
    def __init__(self):
        # --- Config (editable in the Train tab) ---
        self.data_path = ""
        self.mixed_column = "your_mixed_column"
        self.id_columns = "serial_orig, serial_clean"
        self.target_columns = "100_1A, 100_0p1A, 100_10mA, 500_1A, 500_0p1A, 500_10mA"

        # Publication validation controls shared by Train, Compare, and Charts.
        self.validation_method_idx = 0     # Random KFold / GroupKFold / repeated grouped
        self.group_column = ""
        self.n_splits = 5
        self.n_repeats = 10
        self.confidence_level = 95.0
        self.random_state = RANDOM_STATE
        self.single_target_mode = False
        self.single_target = ""
        self.top_feature_count = 20
        self.show_feature_ratio_warnings = True
        self.chemistry_enabled = True
        self.chemistry_mode_idx = 1        # Off / Automatic / Compact / Standard / Full / Custom
        self.chemistry_config = {}
        self.chemistry_feature_diagnostics = {}
        self.chemistry_detection = []
        self.chemistry_estimate = {}
        self.chemistry_full_risk_confirmed = False
        self.chemistry_column_overrides = {}
        self.chemistry_custom_families = {
            family: family in {
                "acid_base", "functional_groups", "redox",
                "physical_properties", "confidence"
            }
            for family in chemistry.DESCRIPTOR_FAMILIES
        }
        self.chemistry_pubchem_enabled = False
        self.chemistry_schema = {}
        self.chemistry_originals = None
        self.chemical_knowledge = []
        self.chemistry_status = "Load a dataset, then scan chemical knowledge."
        self.chemistry_selected_idx = 0
        self.chemistry_values = {}         # {original chemical column: selected value}
        self.chemistry_custom_values = {}  # free-form unseen chemicals such as HBr

        # --- Worksheet/tab selection for multi-sheet .xlsx files ---
        self.sheet_names = []              # tabs in the chosen data file ([] = single/CSV)
        self.sheet_idx = 0                 # which tab is selected
        self.batch_sheet_names = []        # tabs in the batch-predict file
        self.batch_sheet_idx = 0

        # --- File dialogs (async) ---
        self.data_dialog = None
        self.batch_dialog = None
        self.json_dialog = None
        self.json_save_dialog = None

        # --- Progress trackers (one per long-running background task; each
        # drives a visible bar + live processing log via draw_progress_panel) ---
        self.prog_train = Progress()
        self.prog_compare = Progress()
        self.prog_opt = Progress()
        self.prog_bo = Progress()
        self.prog_charts = Progress()
        self.prog_intel = Progress()

        # --- Training status (written by the background thread) ---
        self.is_training = False
        self.progress = 0.0
        self.status = "Load a spreadsheet, then press Train."
        self.train_error = ""

        # --- Trained-model bundle + schema for building predict widgets ---
        self.trained = False
        self.model = None
        self.feature_columns = []          # post-one-hot column order
        self.numeric_schema = {}           # {col: median}
        self.categorical_schema = {}       # {col: [choices]}
        self.targets = []
        self.cfg = {}                      # {ids, mixed} used at train time

        # --- Evaluation results for display ---
        self.metrics = []                  # rich per-target metric dictionaries
        self.metric_distributions = {}
        self.validation_config = V.normalize_validation_config()
        self.feature_diagnostics = {}
        self.oof_predictions = None
        self.oof_prediction_std = None
        self.oof_prediction_samples = None
        self.oof_truth = None
        self.preprocessing_statement = ""
        self.importances = []             # [(feature, importance), ...]
        self.summary = ""

        # --- Predict tab widget values ---
        self.numeric_values = {}           # {col: float}
        self.category_index = {}           # {col: chosen index}
        self.feature_labels = {}           # {internal col: display label}
        self.mixed_text = {}               # {widget key: '1M NaOH' style text}
        self.single_pred = None            # [(target, value), ...]
        self.predict_error = ""

        # --- Batch tab ---
        self.batch_path = ""
        self.batch_status = ""
        self.batch_error = ""
        self.batch_results = None

        # Auto-standardize measurement units during preprocessing (Train tab
        # checkbox): '2 h' -> 120 min, K -> C, '3.2%V H2SO4' -> mol/L, and
        # parsed messy-column label removal. Off = data used exactly as-is.
        self.standardize_units = True

        # --- Column-types tab (manual numeric/categorical overrides + roles) ---
        self.coltype_columns = []          # ordered list of column names scanned
        self.coltype_map = {}              # {col: "numeric" | "categorical"}
        self.coltype_role = {}             # {col: feature|target|id|messy|exclude}
        self.exclude_columns = ""          # comma-sep cols dropped from training
        self.coltype_filter = ""           # type-to-filter the column list
        self.coltype_status = "Choose a spreadsheet in the Train tab, then Scan columns."

        # --- Compare-models tab ---
        self.is_comparing = False
        self.compare_status = "Set the columns in the Train tab, then run the benchmark."
        self.compare_error = ""
        self.compare_results = []          # [(model_name, avg_oof_r2), ...] completed
        self.compare_validation_summary = {}
        self.compare_current = ""          # model currently being evaluated
        self.compare_total = 0
        self.compare_done = 0
        self.compare_chart_path = ""
        self.compare_run = 0

        # --- Optimize tab ---
        self.opt_target = "Reversible_Capacity_mAh_per_g"
        # Measured-after-synthesis + other-outcome columns to EXCLUDE (can't be set).
        self.opt_excluded = (
            "Sample_ID, Lignin_Purity_wt%, Ash_Content_wt%, Sulfur_Content_wt%, "
            "d002_Angstrom, La_nm, Lc_nm, ID_IG_Ratio, BET_Surface_Area_m2_per_g, "
            "Total_Pore_Volume_cm3_per_g, Micropore_Fraction, Closed_Pore_Fraction, "
            "True_Density_g_per_cm3, Carbon_Yield_wt%, Plateau_Capacity_mAh_per_g, "
            "Slope_Capacity_mAh_per_g, ICE_%, Rate_Cap_Retention_%, "
            "Cycle_Retention_100cyc_%, Avg_Sodiation_Voltage_V"
        )
        self.opt_fixed = ""                # e.g. "Current_Density_mA_per_g=100"
        self.opt_direction_idx = 0         # 0 = maximise, 1 = minimise
        self.is_optimizing = False
        self.opt_status = "Pick a file (Train tab), set the target/exclusions, then run."
        self.opt_error = ""
        self.opt_result = None

        # Top-N recommended-experiments pass (uncertainty- and applicability-
        # aware ranking alongside the single best recipe above).
        self.opt_top_n = 10                # how many diverse candidates to return
        self.opt_risk_lambda = 1.0         # penalises both prediction uncertainty
                                           # and distance from the training domain
        self.opt_min_applicability = 60.0  # reject candidates below this in-domain score (0-100)

        # Laboratory constraints and multi-objective Pareto front (Phase 2).
        self.opt_constraints = ""          # one 'Column <= value' etc. per line
        self.opt_objectives = ""           # 'Column:maximize|minimize:weight, ...'

        # Pareto trade-off chart: which two objectives to plot.
        self.opt_pareto_chart_a = 0
        self.opt_pareto_chart_b = 1

        # --- Suggest Experiments tab (Bayesian optimization) ---
        # Reuses the Optimize tab's target / direction / constraints / column
        # roles; only the batch size and exploration weight are BO-specific.
        self.bo_batch = 5                  # how many experiments to propose
        self.bo_xi = 0.01                  # EI exploration margin (higher = bolder)
        self.is_bayesopt = False
        self.bo_status = ("Set the target + knobs (shared with the Optimize tab), "
                          "then suggest experiments.")
        self.bo_error = ""
        self.bo_result = None
        self.opt_pareto_chart_path = None
        self.opt_pareto_chart_error = ""

        # --- Charts tab ---
        self.charts_target_idx = 0         # target for SHAP + optimization heatmap
        self.charts_pareto_a = 0           # Pareto x-axis target
        self.charts_pareto_b = 1           # Pareto y-axis target
        self.charts_featx_idx = 0          # optimization-heatmap x feature
        self.charts_featy_idx = 1          # optimization-heatmap y feature
        self.charts_imp_top_idx = 2        # Top 20 by default
        self.is_charting = False
        self.charts_status = "Train (or load) a model, then Generate charts."
        self.charts_error = ""
        self.chart_items = []              # [(title, rel_path), ...] rendered PNGs
        self.charts_run = 0                # bumped each run -> unique names (texture cache)
        self.slideshow_dialog = None       # async save-file dialog for the slideshow
        self.slideshow_status = ""         # result message for the slideshow export
        # Charts render into CHART_DIR only as a transient display cache (wiped
        # at startup); "Save charts…" exports the current ones to a chosen folder.
        self.save_charts_dialog = None     # async select-folder dialog
        self.save_charts_status = ""

        # --- Latent Variables tab ---
        self.lat_columns = []              # available columns (from file/sheet)
        self.lat_target_idx = 0
        self.lat_categorical = ", ".join(latent.DEFAULT_CATEGORICAL)
        self.lat_numerical = ", ".join(latent.DEFAULT_NUMERICAL)
        self.lat_excluded = ""
        self.lat_chem_w = [1.0, 1.0, 1.0]  # editable chemical-index weights
        self.lat_biomass_pca = False       # False = equal-weight mean, True = PCA-1
        self.lat_pca_components = 3         # 2-10
        self.lat_pls_components = 2         # 1-10
        self.is_lat_running = False
        self.lat_status = "Choose a file (Train tab), Scan columns, then Run analysis."
        self.lat_error = ""
        self.lat_run = 0                   # unique chart filenames
        self.lat_pls_result = None
        self.lat_compare_result = None
        self.lat_chart_items = []          # [(title, rel_path), ...]
        self.lat_last = {}                 # cached df/config/pca for exports

        # --- Dataset Intelligence tab ---
        self.intel_target_idx = 0          # which configured target to analyse
        self.intel_pca_components = 4       # interactive PCA component selector (2-10)
        self.is_intel_running = False
        self.intel_status = "Configure columns in the Train / Column types tab, then Run."
        self.intel_error = ""
        self.intel_run = 0                 # unique chart filenames
        self.intel_results = None          # dict of all section results
        self.intel_chart_items = []        # [(title, rel_path), ...]
        self.intel_insights = []           # AI Research Assistant lines
        self.intel_conclusion = ""

        # --- Screening engine (BioCarbon Screen decision support) ---
        self.X_train = None                # encoded training matrix (for AD/similarity)
        self.y_train = None                # observed targets (DataFrame)
        self.cv_rmse = {}                  # {target: cross-validated RMSE}
        self.cv_r2 = {}                    # {target: cross-validated R2}
        self.screener = None               # screening.Screener built after training

        # --- Screen tab (primary workflow) ---
        self.screen_target_idx = 0         # which target to screen for
        self.screen_result = None          # last full screening result dict
        self.is_screening = False
        self.screen_error = ""
        self.screen_live = False           # live What-if: rescreen on any change
        self.screen_dirty = False          # inputs changed since last screen
        self.screen_pdf_dialog = None
        self.screen_xlsx_dialog = None
        self.screen_import_dialog = None
        self.screen_import_status = ""
        self.screen_export_msg = ""
        self.screen_custom_category = {}   # {categorical col: user-entered unknown value}

        # --- Experiment Prioritization tab ---
        self.prio_target_idx = 0
        self.prio_source_path = ""         # candidate spreadsheet (optional)
        self.prio_source_sheet_names = []
        self.prio_source_sheet_idx = 0
        self.prio_source_dialog = None
        self.prio_save_dialog = None
        self.is_prioritizing = False
        self.prio_status = ("Train/load a model, then rank candidates from the "
                            "optimizer or an imported spreadsheet.")
        self.prio_error = ""
        self.prio_df = None                # ranked DataFrame
        self.prio_w_perf = 0.40
        self.prio_w_conf = 0.20
        self.prio_w_novel = 0.15
        self.prio_w_sim = 0.15
        self.prio_w_feas = 0.10


STATE = AppState()


# =============================================================================
# TRAINING  (runs on a worker thread; updates STATE as it goes)
# =============================================================================
def _split_cols(text):
    """Turn 'a, b ,c' into ['a','b','c'] (dropping blanks)."""
    return [c.strip() for c in text.split(",") if c.strip()]


def _current_sheet():
    """Name of the selected worksheet/tab, or None for single-sheet / CSV files."""
    if STATE.sheet_names and 0 <= STATE.sheet_idx < len(STATE.sheet_names):
        return STATE.sheet_names[STATE.sheet_idx]
    return None


def _validation_config_from_state():
    methods = ["random_kfold", "group_kfold", "repeated_grouped_cv"]
    idx = min(max(int(STATE.validation_method_idx), 0), len(methods) - 1)
    return V.normalize_validation_config({
        "method": methods[idx],
        "group_column": STATE.group_column,
        "n_splits": STATE.n_splits,
        "n_repeats": STATE.n_repeats,
        "confidence_level": STATE.confidence_level / 100.0,
        "random_state": STATE.random_state,
        "interval_method": "percentile",
    })


def _active_targets_from_state():
    targets = _split_cols(STATE.target_columns)
    if not STATE.single_target_mode:
        return targets
    return [STATE.single_target] if STATE.single_target else []


def _training_config_from_state():
    chemistry_modes = ["off", "automatic", "compact", "standard", "full", "custom"]
    mode = chemistry_modes[min(STATE.chemistry_mode_idx, len(chemistry_modes) - 1)]
    return dict(
        data_path=STATE.data_path,
        ids=_split_cols(STATE.id_columns),
        mixed=_split_cols(STATE.mixed_column),
        targets=_active_targets_from_state(),
        all_target_columns=[c for c in STATE.coltype_columns
                            if STATE.coltype_role.get(c) == "target"],
        single_target_mode=STATE.single_target_mode,
        col_types=dict(STATE.coltype_map),
        exclude=_split_cols(STATE.exclude_columns),
        sheet=_current_sheet(),
        standardize_units=STATE.standardize_units,
        validation=_validation_config_from_state(),
        chemistry_enabled=STATE.chemistry_enabled,
        chemistry_mode=mode,
        chemistry_custom_families=[
            family for family, enabled in STATE.chemistry_custom_families.items()
            if enabled
        ],
        chemistry_column_overrides=dict(STATE.chemistry_column_overrides),
    )


def _sync_validation_column_defaults(columns):
    columns = list(map(str, columns))
    if STATE.group_column not in columns:
        STATE.group_column = next(
            (c for c in ("Condition_ID", "Group_ID", "Batch_ID") if c in columns), ""
        )
    if STATE.single_target not in columns:
        configured = [c for c in _split_cols(STATE.target_columns) if c in columns]
        STATE.single_target = ("SIB_0_1A" if "SIB_0_1A" in columns
                               else (configured[0] if configured else ""))


def _set_chemistry_state(schema, originals=None):
    """Publish chemistry metadata and sensible prediction-widget defaults."""
    STATE.chemistry_schema = dict(schema or {})
    STATE.chemistry_config = dict(STATE.chemistry_schema.get("chemistry_config") or
                                  STATE.chemistry_config or {})
    STATE.chemistry_feature_diagnostics = dict(
        STATE.chemistry_schema.get("chemistry_feature_diagnostics") or
        STATE.chemistry_feature_diagnostics or {})
    STATE.chemistry_originals = originals
    all_values = []
    for source, info in STATE.chemistry_schema.get("columns", {}).items():
        choices = list(info.get("observed_chemicals", []))
        current = STATE.chemistry_values.get(source)
        if current not in choices:
            current = choices[0] if choices else "Unknown"
        STATE.chemistry_values[source] = current
        STATE.chemistry_custom_values.setdefault(source, "")
        all_values.extend(choices)
    STATE.chemical_knowledge = chemistry.ENGINE.knowledge_for_values(all_values)
    if STATE.chemistry_schema.get("columns"):
        STATE.chemistry_status = (
            f"Detected {len(STATE.chemistry_schema['columns'])} chemical column(s); "
            f"generated {STATE.chemistry_schema.get('descriptor_feature_count', 0)} descriptors."
        )


def _chemistry_feature_labels(schema):
    labels = {}
    prefix_labels = {}
    for source, info in (schema or {}).get("columns", {}).items():
        source_label = STATE.feature_labels.get(source, source)
        source_label = source_label.split(":", 1)[0].strip()
        if source_label == source:
            source_label = chemistry.ChemistryFeatureEngineer.prefix(source).replace("_", " ")
        prefix_labels[info.get("prefix", "")] = source_label
        for feature in info.get("descriptor_columns", []):
            prefix = info.get("prefix", "")
            token = feature[len(prefix) + 1:] if feature.startswith(prefix + "_") else feature
            display = f"{source_label}: {chemistry.descriptor_display_name(token)}"
            labels[feature] = display
            for level in STATE.categorical_schema.get(feature, []):
                labels[f"{feature}_{level}"] = f"{display} = {level}"
    for feature in (schema or {}).get("interactions", []):
        prefix = next((p for p in prefix_labels if feature.startswith(p + "_")), "")
        token = feature[len(prefix) + 1:] if prefix else feature
        label = chemistry.descriptor_display_name(token)
        labels[feature] = f"{prefix_labels[prefix]}: {label}" if prefix else label
    return labels


def _autofill_columns(path):
    """Detect and fill the step-2 fields from the chosen spreadsheet."""
    try:
        mixed, ids, targets = auto_detect_columns(path, sheet=_current_sheet())  # mixed is a list
        if mixed:
            STATE.mixed_column = ", ".join(mixed)
        if ids:
            STATE.id_columns = ", ".join(ids)
        if targets:
            STATE.target_columns = ", ".join(targets)
        scan_column_types()  # seed the Column types tab from this file
        shown = ", ".join(mixed) if mixed else "?"
        STATE.status = (
            f"Auto-detected  ·  {len(mixed)} messy col(s): {shown}  ·  "
            f"{len(ids)} id col(s)  ·  {len(targets)} target(s). Adjust if needed, then Train."
        )
    except Exception as e:  # noqa: BLE001
        STATE.status = f"Loaded file, but auto-detect failed: {e}"


# Per-column role choices for the Column types tab. Internal keys <-> UI labels.
ROLE_KEYS = ["feature", "target", "id", "messy", "exclude"]
ROLE_LABELS = ["Feature", "Target", "ID", "Messy", "Exclude"]


def scan_column_types():
    """
    Read a sample of the chosen spreadsheet and seed each column's default type
    (numeric if pandas reads it as a number, else categorical) and role (from the
    current Train-tab config). Any choice the user already made is preserved.
    """
    if not STATE.data_path:
        STATE.coltype_status = "Choose a spreadsheet in the Train tab first."
        return
    try:
        df = read_any(STATE.data_path, nrows=200, sheet=_current_sheet())
        cols = [str(c) for c in df.columns]

        # Seed default roles from whatever the Train-tab fields currently say.
        targets = set(_split_cols(STATE.target_columns))
        ids = set(_split_cols(STATE.id_columns))
        messy = set(_split_cols(STATE.mixed_column))

        new_map, new_role = {}, {}
        for c in cols:
            if c in STATE.coltype_map:          # keep prior manual type choice
                new_map[c] = STATE.coltype_map[c]
            else:
                new_map[c] = ("numeric" if pd.api.types.is_numeric_dtype(df[c])
                              else "categorical")
            if c in STATE.coltype_role:         # keep prior manual role choice
                new_role[c] = STATE.coltype_role[c]
            elif c in targets:
                new_role[c] = "target"
            elif c in ids:
                new_role[c] = "id"
            elif c in messy:
                new_role[c] = "messy"
            else:
                new_role[c] = "feature"

        STATE.coltype_columns = cols
        STATE.coltype_map = new_map
        STATE.coltype_role = new_role
        sync_roles_to_cfg()                     # push roles into the Train-tab fields
        _sync_validation_column_defaults(cols)
        n_num = sum(1 for v in new_map.values() if v == "numeric")
        n_tgt = sum(1 for v in new_role.values() if v == "target")
        STATE.coltype_status = (
            f"Scanned {len(cols)} column(s) · {n_num} numeric / "
            f"{len(cols) - n_num} categorical · {n_tgt} target(s). Adjust below, then Train."
        )
        scan_chemical_knowledge(silent=True)
    except Exception as e:  # noqa: BLE001
        STATE.coltype_status = f"Scan failed: {e}"


def scan_chemical_knowledge(silent=False):
    """Auto-configure chemistry from the same rows and roles used for training."""
    if not STATE.data_path:
        if not silent:
            STATE.chemistry_status = "Choose a spreadsheet in the Train tab first."
        return
    try:
        raw = read_any(STATE.data_path, sheet=_current_sheet()).reset_index(drop=True)
        validation_cfg = _validation_config_from_state()
        group_column = validation_cfg.get("group_column", "")
        id_columns = _split_cols(STATE.id_columns)
        chemistry_group_column = (
            group_column if group_column in raw.columns else next(
                (c for c in id_columns if c in raw.columns and
                 1 < raw[c].nunique(dropna=True) < len(raw)), None)
        )
        groups = (raw[chemistry_group_column].copy() if chemistry_group_column
                  else pd.Series(np.arange(len(raw)), name="row"))
        drop_ids = list(dict.fromkeys(id_columns + ([group_column] if group_column else [])))
        prepared = prepare_raw(raw, drop_ids, _split_cols(STATE.mixed_column))
        if STATE.standardize_units:
            prepared = units.standardize_units(
                prepared, protected=set(_active_targets_from_state()) |
                set(STATE.coltype_map))
            prepared = units.standardize_parsed_mixed(
                prepared, labels=mixed_feature_labels(_split_cols(STATE.mixed_column)))
        prepared = auto_coerce_numeric(prepared, protected=set(STATE.coltype_map))
        prepared = apply_col_type_overrides(prepared, STATE.coltype_map)
        prepared = prepared.drop(columns=_split_cols(STATE.exclude_columns), errors="ignore")
        targets = [c for c in _active_targets_from_state() if c in prepared.columns]
        if targets:
            target_frame = prepared[targets].apply(pd.to_numeric, errors="coerce")
            mask = target_frame.notna().all(axis=1)
            prepared = prepared.loc[mask].reset_index(drop=True)
            groups = groups.loc[mask].reset_index(drop=True)
        X = prepared.drop(columns=targets, errors="ignore")
        details = chemistry.ENGINE.detect_column_details(X)
        STATE.chemistry_detection = [asdict(item) for item in details]
        columns = [
            item.column for item in details
            if STATE.chemistry_column_overrides.get(item.column,
                                                    item.confidence >= .70)
        ]
        modes = ["off", "automatic", "compact", "standard", "full", "custom"]
        mode = modes[min(STATE.chemistry_mode_idx, len(modes) - 1)]
        config = chemistry.ENGINE.auto_configure(
            X, chemical_columns=columns, groups=groups,
            requested_mode=mode)
        if mode == "custom":
            chemistry.apply_custom_families(
                config, [name for name, enabled in STATE.chemistry_custom_families.items()
                         if enabled])
        estimate = chemistry.ENGINE.estimate_feature_counts(X, config, groups=groups)
        expansion = estimate.pop("expansion")
        STATE.chemistry_config = chemistry.chemistry_config_as_dict(config)
        STATE.chemistry_feature_diagnostics = dict(
            expansion.metadata.get("chemistry_feature_diagnostics", {}))
        STATE.chemistry_estimate = estimate
        _set_chemistry_state(expansion.metadata, expansion.original_values)
        if not expansion.metadata.get("columns"):
            STATE.chemistry_status = "No chemical columns were detected automatically."
        else:
            STATE.chemistry_status = (
                f"Auto configuration ready: {estimate['chemistry_features']} chemistry "
                f"feature(s), {estimate['estimated_total_encoded_predictors']} estimated "
                f"encoded predictors from {estimate['independent_groups']} groups."
            )
    except Exception as e:  # noqa: BLE001
        STATE.chemistry_status = f"Chemical scan failed: {e}"


def sync_roles_to_cfg():
    """Rebuild the Train-tab column fields from the per-column role choices so the
    rest of the pipeline (which reads those fields) uses the same configuration."""
    def by_role(role):
        return [c for c in STATE.coltype_columns if STATE.coltype_role.get(c) == role]
    STATE.target_columns = ", ".join(by_role("target"))
    STATE.id_columns = ", ".join(by_role("id"))
    STATE.mixed_column = ", ".join(by_role("messy"))
    STATE.exclude_columns = ", ".join(by_role("exclude"))


def apply_json_config(path):
    """
    Load a JSON config and apply it to the Column types tab. Recognised keys:
      dataset  -> selects the worksheet/tab with that name (if present)
      target / targets      -> role "target"  (typed numeric)
      features.categorical  -> role "feature", type categorical
      features.numerical    -> role "feature", type numeric
      exclude               -> role "exclude"
    Columns named in the JSON are added to the table even if a scan hasn't run.
    """
    with open(path, "r", encoding="utf-8") as f:
        spec = V.load_settings_compat(json.load(f))

    feats = spec.get("features", {}) or {}
    categorical = [str(c) for c in feats.get("categorical", [])]
    numerical = [str(c) for c in feats.get("numerical", [])]
    targets = [str(c) for c in spec.get("targets", [])]
    if not targets and spec.get("target"):
        targets = [str(spec["target"])]
    excluded = [str(c) for c in spec.get("exclude", [])]
    ids = [str(c) for c in spec.get("ids", [])]
    messy = [str(c) for c in spec.get("messy", [])]

    validation_cfg = spec["validation"]
    methods = ["random_kfold", "group_kfold", "repeated_grouped_cv"]
    STATE.validation_method_idx = methods.index(validation_cfg["method"])
    STATE.group_column = validation_cfg["group_column"]
    STATE.n_splits = validation_cfg["n_splits"]
    STATE.n_repeats = validation_cfg["n_repeats"]
    STATE.confidence_level = validation_cfg["confidence_level"] * 100.0
    STATE.random_state = validation_cfg["random_state"]
    STATE.single_target_mode = bool(spec["training"].get("single_target_mode", False))
    STATE.single_target = str(spec["training"].get("target") or "")
    STATE.top_feature_count = max(1, int(spec["reporting"].get("top_feature_count", 20)))
    STATE.show_feature_ratio_warnings = bool(
        spec["reporting"].get("show_feature_ratio_warnings", True))
    chemistry_spec = dict(spec.get("chemistry") or {})
    feature_spec = dict(spec.get("chemistry_features") or {})
    STATE.chemistry_enabled = bool(feature_spec.get("enabled", True))
    chemistry_modes = ["off", "automatic", "compact", "standard", "full", "custom"]
    requested_mode = str(feature_spec.get(
        "mode", "automatic" if STATE.chemistry_enabled else "off")).lower()
    STATE.chemistry_mode_idx = (chemistry_modes.index(requested_mode)
                                if requested_mode in chemistry_modes else 1)
    STATE.chemistry_column_overrides = dict(
        feature_spec.get("column_overrides") or {})
    STATE.chemistry_pubchem_enabled = bool(
        chemistry_spec.get("pubchem_enabled", False))
    chemistry.ENGINE.generator.set_pubchem_enabled(STATE.chemistry_pubchem_enabled)
    chemistry.ENGINE.generator.clear_overrides()
    for chemical_name, values in chemistry_spec.get("overrides", {}).items():
        chemistry.ENGINE.generator.set_override(chemical_name, values)

    # Restore the source file when it is still available, then switch worksheet.
    source_file = spec.get("source_file")
    if source_file and os.path.exists(str(source_file)):
        STATE.data_path = str(source_file)
        STATE.sheet_names = list_sheets(STATE.data_path)
        STATE.sheet_idx = 0

    # dataset -> worksheet/tab: switch to it and (re)scan so the table reflects it.
    dataset = spec.get("dataset")
    if dataset and dataset in STATE.sheet_names:
        STATE.sheet_idx = STATE.sheet_names.index(dataset)
    if STATE.data_path:
        scan_column_types()  # populate/refresh columns from the (possibly new) tab

    # Assign types + roles. Targets are numeric outcomes for a regression task.
    for c in numerical:
        STATE.coltype_map[c] = "numeric"
        STATE.coltype_role[c] = "feature"
    for c in categorical:
        STATE.coltype_map[c] = "categorical"
        STATE.coltype_role[c] = "feature"
    for c in targets:
        STATE.coltype_map[c] = "numeric"
        STATE.coltype_role[c] = "target"
    for c in excluded:
        STATE.coltype_role[c] = "exclude"
    for c in ids:
        STATE.coltype_role[c] = "id"
    for c in messy:
        STATE.coltype_role[c] = "messy"

    # Make sure every JSON-named column is present in the table + config, keeping
    # any already-scanned order first, then appending anything new.
    referenced = numerical + categorical + targets + excluded + ids + messy
    seen = set(STATE.coltype_columns)
    for c in referenced:
        if c not in seen:
            STATE.coltype_columns.append(c)
            seen.add(c)
            STATE.coltype_map.setdefault(c, "categorical")

    sync_roles_to_cfg()
    if STATE.data_path:
        scan_chemical_knowledge(silent=True)
    tab = f" · tab '{dataset}'" if dataset and dataset in STATE.sheet_names else ""
    STATE.coltype_status = (
        f"Loaded JSON: {len(targets)} target(s), {len(numerical)} numeric + "
        f"{len(categorical)} categorical feature(s), {len(excluded)} excluded{tab}."
    )


def export_json_config(path):
    """Save column roles plus the publication validation/reporting controls."""
    def by_role(role):
        return [c for c in STATE.coltype_columns if STATE.coltype_role.get(c) == role]

    features = by_role("feature")
    spec = {
        "source_file": STATE.data_path,
        "dataset": _current_sheet(),
        "targets": _split_cols(STATE.target_columns),
        "ids": by_role("id") or _split_cols(STATE.id_columns),
        "messy": by_role("messy") or _split_cols(STATE.mixed_column),
        "features": {
            "numerical": [c for c in features if STATE.coltype_map.get(c) == "numeric"],
            "categorical": [c for c in features if STATE.coltype_map.get(c) == "categorical"],
        },
        "exclude": by_role("exclude") or _split_cols(STATE.exclude_columns),
        "validation": _validation_config_from_state(),
        "training": {
            "single_target_mode": STATE.single_target_mode,
            "target": STATE.single_target,
        },
        "reporting": {
            "top_feature_count": STATE.top_feature_count,
            "show_feature_ratio_warnings": STATE.show_feature_ratio_warnings,
        },
        "chemistry": {
            "enabled": STATE.chemistry_enabled,
            "pubchem_enabled": STATE.chemistry_pubchem_enabled,
            "overrides": chemistry.ENGINE.generator.export_overrides(),
        },
        "chemistry_features": {
            "enabled": STATE.chemistry_enabled,
            "mode": ["off", "automatic", "compact", "standard", "full", "custom"][
                min(STATE.chemistry_mode_idx, 5)],
            "auto_configure": True,
            "max_chemistry_features": None,
            "retain_original_labels": "auto",
            "rare_category_min_groups": 5,
            "near_constant_threshold": 0.99,
            "correlation_threshold": 0.95,
            "enable_interactions": "auto",
            "enable_rdkit_descriptors": "auto",
            "enable_morgan_fingerprints": False,
            "column_overrides": dict(STATE.chemistry_column_overrides),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    STATE.coltype_status = f"Saved settings to {path}."


def start_training():
    """Kick off training on a background thread so the UI stays responsive."""
    if STATE.is_training or not STATE.data_path:
        return
    ratio = float(STATE.chemistry_estimate.get(
        "groups_per_encoded_predictor", math.inf))
    if (STATE.chemistry_mode_idx == 4 and ratio < 1 and
            not STATE.chemistry_full_risk_confirmed):
        STATE.train_error = (
            "Full chemistry mode has fewer than one independent group per encoded "
            "predictor. Confirm the high-risk Full-mode warning before training.")
        STATE.status = "Training confirmation required."
        return
    STATE.is_training = True
    STATE.trained = False
    STATE.train_error = ""
    STATE.progress = 0.0
    STATE.status = "Starting…"

    # Snapshot config so the thread doesn't read fields being edited mid-run.
    cfg = _training_config_from_state()
    threading.Thread(target=_train_worker, args=(cfg,), daemon=True).start()


def _train_worker(cfg):
    """Train in the background using the shared leakage-safe validation engine."""
    STATE.prog_train.begin("Loading and cleaning data…")
    try:
        STATE.status = "Loading and cleaning data..."
        STATE.progress = 0.2
        targets = list(cfg["targets"])
        clean_notes = []
        STATE.prog_train.step("Encoding features and chemistry descriptors…", 0.35)
        data = build_training_data(cfg, notes=clean_notes)
        X_raw = data["X_raw"]
        X_enc = data["X_encoded"]
        y = data["y"]
        groups = data["groups"]
        numeric_schema = data["numeric_schema"]
        categorical_schema = data["categorical_schema"]
        feature_columns = X_enc.columns.tolist()
        if len(X_enc) < cfg["validation"]["n_splits"]:
            raise ValueError(
                f"Only {len(X_enc)} usable rows remain after cleaning; "
                f"{cfg['validation']['n_splits']} folds were requested. "
                + "  ".join(clean_notes)
            )

        base_model = ExtraTreesRegressor(
            n_estimators=300, max_depth=12, min_samples_split=4,
            max_features="sqrt", random_state=cfg["validation"]["random_state"],
            n_jobs=-1,
        )
        model = base_model if len(targets) == 1 else MultiOutputRegressor(base_model)
        validation_label = V.validation_method_label(cfg["validation"])
        STATE.status = f"Evaluating with {validation_label}..."
        STATE.progress = 0.5
        STATE.prog_train.step(f"Cross-validating ({validation_label})…", 0.55)
        evaluation = V.evaluate_model_cv(
            model, X_raw, y, cfg["validation"], groups=groups,
            numeric_columns=list(numeric_schema),
            categorical_columns=list(categorical_schema),
        )

        STATE.status = "Fitting final model on all usable rows..."
        STATE.progress = 0.9
        STATE.prog_train.step("Fitting final model on all usable rows…", 0.9)
        fit_y = y.iloc[:, 0] if len(targets) == 1 else y
        model.fit(X_enc, fit_y)
        pred_tr = np.asarray(model.predict(X_enc), dtype=float)
        if pred_tr.ndim == 1:
            pred_tr = pred_tr.reshape(-1, 1)
        if hasattr(model, "estimators_") and len(targets) > 1:
            importances = np.mean(
                [est.feature_importances_ for est in model.estimators_], axis=0)
        else:
            importances = np.asarray(model.feature_importances_)
        imp_series = pd.Series(importances, index=X_enc.columns).sort_values(ascending=False)

        method = cfg["validation"]["method"]
        independent_groups = int(groups.nunique()) if method != "random_kfold" else len(X_enc)
        effective_repeats = (cfg["validation"]["n_repeats"]
                             if method == "repeated_grouped_cv" else 1)
        metrics = []
        for i, target in enumerate(targets):
            metrics.append({
                "target": target,
                "train_r2": float(r2_score(y.iloc[:, i], pred_tr[:, i])),
                "train_rmse": float(np.sqrt(mean_squared_error(y.iloc[:, i], pred_tr[:, i]))),
                "cv": evaluation["metric_summaries"][target],
                "pooled_oof": evaluation["pooled_metrics"][target],
                "n_rows": len(X_enc),
                "n_groups": independent_groups,
                "n_splits": cfg["validation"]["n_splits"],
                "n_repeats": effective_repeats,
                "validation_method": validation_label,
            })

        diagnostics = V.feature_diagnostics(
            data["original_predictors"], len(feature_columns), len(X_enc),
            independent_groups,
        )
        STATE.model = model
        STATE.feature_columns = feature_columns
        STATE.numeric_schema = numeric_schema
        STATE.categorical_schema = categorical_schema
        STATE.targets = targets
        STATE.cfg = dict(cfg)
        STATE.feature_labels = mixed_feature_labels(cfg["mixed"])
        STATE.feature_labels.update(_chemistry_feature_labels(data["chemistry_schema"]))
        STATE.chemistry_config = dict(data.get("chemistry_config") or {})
        STATE.chemistry_feature_diagnostics = dict(
            data.get("chemistry_feature_diagnostics") or {})
        _set_chemistry_state(data["chemistry_schema"], data["chemistry_originals"])
        STATE.metrics = metrics
        STATE.metric_distributions = evaluation["metric_distributions"]
        STATE.validation_config = dict(cfg["validation"])
        STATE.feature_diagnostics = diagnostics
        STATE.oof_predictions = evaluation["oof_predictions"]
        STATE.oof_prediction_std = evaluation["oof_prediction_std"]
        STATE.oof_prediction_samples = evaluation["oof_prediction_samples"]
        STATE.oof_truth = y.reset_index(drop=True)
        STATE.preprocessing_statement = evaluation["preprocessing"]
        STATE.importances = list(imp_series.items())
        STATE.X_train = X_enc
        STATE.y_train = y.reset_index(drop=True)
        STATE.cv_rmse = {m["target"]: m["pooled_oof"]["rmse"] for m in metrics}
        STATE.cv_r2 = {m["target"]: m["pooled_oof"]["r2"] for m in metrics}
        rebuild_screener()

        mean_tr = float(np.mean([m["train_r2"] for m in metrics]))
        mean_cv = float(np.mean([m["pooled_oof"]["r2"] for m in metrics]))
        STATE.summary = (
            f"ExtraTrees - {validation_label} on {len(X_enc)} rows.  "
            f"Train R2 = {mean_tr:.3f} - pooled OOF R2 = {mean_cv:.3f} - "
            f"{len(feature_columns)} encoded features.  " + "  ".join(clean_notes)
        )
        STATE.numeric_values = dict(numeric_schema)
        STATE.category_index = {c: 0 for c in categorical_schema}
        STATE.screen_custom_category = {c: "" for c in categorical_schema}
        STATE.mixed_text = {}
        STATE.single_pred = None
        STATE.predict_error = ""
        STATE.trained = True
        STATE.status = "Done. Review the grouped OOF metrics below, then use Predict."
        STATE.progress = 1.0
        STATE.prog_train.finish(
            f"Trained — pooled OOF R² = {mean_cv:.3f} on {len(X_enc)} rows.")
    except Exception as e:  # noqa: BLE001
        STATE.train_error = f"{type(e).__name__}: {e}"
        STATE.status = "Training failed."
        STATE.progress = 0.0
        STATE.prog_train.fail(f"{type(e).__name__}: {e}")
    finally:
        STATE.is_training = False


def start_comparison():
    """Kick off the 5-fold model benchmark on a background thread."""
    if STATE.is_comparing or not STATE.data_path:
        return
    STATE.is_comparing = True
    STATE.compare_error = ""
    STATE.compare_results = []
    STATE.compare_current = ""
    STATE.compare_done = 0
    STATE.compare_total = 0
    STATE.compare_status = "Preparing data…"

    cfg = _training_config_from_state()
    threading.Thread(target=_compare_worker, args=(cfg,), daemon=True).start()


def _comparison_estimator(model, n_targets):
    """Use scalar estimators in single-target mode and retain multi-output models."""
    if n_targets == 1:
        if isinstance(model, MultiOutputRegressor):
            return model.estimator
        if isinstance(model, RegressorChain):
            return getattr(model, "estimator", getattr(model, "base_estimator", model))
        return model
    if isinstance(model, RegressorChain):
        model.set_params(order=list(range(n_targets)))
    return model


def _compare_worker(cfg):
    STATE.prog_compare.begin("Loading and cleaning data…")
    try:
        data = build_training_data(cfg)
        X, y, groups = data["X_raw"], data["y"], data["groups"]
        numeric = list(data["numeric_schema"])
        categorical = list(data["categorical_schema"])
        STATE.prog_compare.step("Assembling model architectures…", 0.05)
        models = build_models()
        STATE.compare_total = len(models)
        validation_label = V.validation_method_label(cfg["validation"])
        STATE.compare_validation_summary = {
            "label": validation_label,
            "group_column": cfg["validation"].get("group_column", ""),
            "n_splits": cfg["validation"]["n_splits"],
            "n_repeats": (cfg["validation"]["n_repeats"]
                          if cfg["validation"]["method"] == "repeated_grouped_cv" else 1),
            "n_groups": (int(groups.nunique())
                         if cfg["validation"]["method"] != "random_kfold" else len(X)),
        }

        for name, candidate in models.items():
            STATE.compare_current = name
            STATE.compare_status = (
                f"Evaluating {name} with {validation_label} "
                f"({STATE.compare_done + 1}/{STATE.compare_total})..."
            )
            STATE.prog_compare.step(
                f"Evaluating {name} ({STATE.compare_done + 1}/{STATE.compare_total})…",
                0.05 + 0.9 * STATE.compare_done / max(STATE.compare_total, 1))
            t0 = time.time()
            row = {"name": name, "runtime": float("nan"),
                   "train_r2": float("nan"), "predict_ms": float("nan")}
            try:
                model = _comparison_estimator(candidate, y.shape[1])
                result = V.evaluate_model_cv(
                    model, X, y, cfg["validation"], groups=groups,
                    numeric_columns=numeric, categorical_columns=categorical,
                )
                aggregate = {}
                for metric in ("r2", "rmse", "mae"):
                    arrays = [result["metric_distributions"][t][metric] for t in y.columns]
                    aggregate_scores = np.nanmean(np.asarray(arrays, dtype=float), axis=0)
                    aggregate[metric] = V.calculate_metric_summary(
                        aggregate_scores, cfg["validation"]["confidence_level"], "percentile")

                final_pipe = make_pipeline(V.build_preprocessor(numeric, categorical), model)
                fit_y = y.iloc[:, 0] if y.shape[1] == 1 else y
                final_pipe.fit(X, fit_y)
                pred_train = np.asarray(final_pipe.predict(X))
                train_r2 = float(r2_score(fit_y, pred_train,
                                          multioutput="uniform_average"))
                tp = time.time()
                final_pipe.predict(X)
                predict_ms = 1000.0 * (time.time() - tp) / max(len(X), 1)
                row.update({
                    "r2": aggregate["r2"], "rmse": aggregate["rmse"],
                    "mae": aggregate["mae"], "train_r2": train_r2,
                    "predict_ms": predict_ms,
                    "validation_method": validation_label,
                })
            except Exception as e:  # noqa: BLE001
                print(f"[compare] {name} failed: {e}")
                row["error"] = str(e)
            row["runtime"] = time.time() - t0
            STATE.compare_results.append(row)
            STATE.compare_done += 1

        STATE.compare_current = ""
        try:
            import charts as C
            os.makedirs(CHART_DIR, exist_ok=True)
            STATE.compare_run += 1
            STATE.compare_chart_path = C.model_comparison_plot(
                STATE.compare_results,
                f"{CHART_DIR}/compare_models_{STATE.compare_run}.png",
                title=f"Model comparison - {validation_label}")
        except Exception as e:  # noqa: BLE001
            print(f"[compare chart] failed: {e}")
        STATE.compare_status = f"Done. Benchmarked {STATE.compare_total} architectures."
        STATE.prog_compare.finish(
            f"Done — {STATE.compare_total} architectures benchmarked.")
    except Exception as e:  # noqa: BLE001
        STATE.compare_error = f"{type(e).__name__}: {e}"
        STATE.compare_status = "Comparison failed."
        STATE.prog_compare.fail(f"{type(e).__name__}: {e}")
    finally:
        STATE.is_comparing = False


def _parse_fixed(text):
    """Parse 'col=val, col2=val2' into a dict."""
    out = {}
    for part in text.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip():
                out[k.strip()] = v.strip()
    return out


def start_optimize():
    """Kick off the capacity optimizer on a background thread."""
    if STATE.is_optimizing or not STATE.data_path:
        return
    STATE.is_optimizing = True
    STATE.opt_error = ""
    STATE.opt_result = None
    STATE.opt_status = "Training model on controllable knobs…"

    # COLUMN-TYPE MODE: when the Column types tab has been configured, drive the
    # optimizer from the per-column roles/types instead of a hand-typed exclusion
    # list — role='Feature' columns become the knobs, the chosen role='Target'
    # is the objective, and everything else is held out automatically.
    features = col_types = ids = mixed = None
    excluded = _split_cols(STATE.opt_excluded)
    if STATE.coltype_columns:
        sync_roles_to_cfg()
        features = [c for c in STATE.coltype_columns
                    if STATE.coltype_role.get(c) == "feature"]
        ids = [c for c in STATE.coltype_columns
               if STATE.coltype_role.get(c) == "id"]
        mixed = [c for c in STATE.coltype_columns
                 if STATE.coltype_role.get(c) == "messy"]
        excluded = [c for c in STATE.coltype_columns
                    if STATE.coltype_role.get(c) == "exclude"]
        col_types = dict(STATE.coltype_map)
    else:
        ids = _split_cols(STATE.id_columns)
        mixed = _split_cols(STATE.mixed_column)

    cfg = dict(
        data_path=STATE.data_path,
        target=STATE.opt_target.strip(),
        excluded=excluded,
        fixed=_parse_fixed(STATE.opt_fixed),
        direction="maximise" if STATE.opt_direction_idx == 0 else "minimise",
        sheet=_current_sheet(),                  # selected worksheet/tab
        features=features,
        col_types=col_types,
        ids=ids,
        mixed=mixed,
        chemistry_enabled=STATE.chemistry_enabled,
        chemistry_schema=STATE.chemistry_schema,
        chemistry_mode=["off", "automatic", "compact", "standard", "full", "custom"][
            min(STATE.chemistry_mode_idx, 5)],
        top_n=STATE.opt_top_n,
        risk_lambda=STATE.opt_risk_lambda,
        min_applicability=STATE.opt_min_applicability,
        constraints_text=STATE.opt_constraints,
        objectives_text=STATE.opt_objectives,
    )
    threading.Thread(target=_optimize_worker, args=(cfg,), daemon=True).start()


def _optimize_worker(cfg):
    STATE.prog_opt.begin("Starting optimization…")
    try:
        STATE.opt_status = "Training + searching (this can take a minute)…"
        res = run_capacity_optimization(
            cfg["data_path"], cfg["target"], cfg["excluded"],
            cfg["fixed"], cfg["direction"], sheet=cfg.get("sheet"),
            features=cfg.get("features"), col_types=cfg.get("col_types"),
            ids=cfg.get("ids"), mixed=cfg.get("mixed"),
            chemistry_enabled=cfg.get("chemistry_enabled", True),
            chemistry_schema=cfg.get("chemistry_schema"),
            chemistry_mode=cfg.get("chemistry_mode", "automatic"),
            top_n=cfg.get("top_n", 10), risk_lambda=cfg.get("risk_lambda", 1.0),
            min_applicability=cfg.get("min_applicability", 60.0),
            constraints_text=cfg.get("constraints_text", ""),
            objectives_text=cfg.get("objectives_text", ""),
            progress=STATE.prog_opt.step)
        STATE.opt_result = res
        src = ("column-type roles" if res.get("mode") == "column-type"
               else "manual exclusions")
        n_top = len(res.get("top_recipes", []))
        n_pareto = len(res.get("pareto_front", []))
        pareto_bit = f", {n_pareto} Pareto-optimal trade-off(s)" if len(
            res.get("objectives_used", [])) > 1 else ""
        STATE.opt_status = (
            f"Done ({src}). Model R²={res['r2']:.2f} on {res['n_rows']} rows, "
            f"{res['n_knobs']} knobs "
            f"({res.get('n_numeric', 0)} numeric / {res.get('n_categorical', 0)} categorical). "
            f"{n_top} recommended experiment(s) from "
            f"{res.get('n_candidates_considered', 0)} candidates considered{pareto_bit}.")
        STATE.prog_opt.finish(f"Done — {n_top} experiment(s) recommended.")
    except Exception as e:  # noqa: BLE001
        STATE.opt_error = f"{type(e).__name__}: {e}"
        STATE.opt_status = "Optimization failed."
        STATE.prog_opt.fail(f"{type(e).__name__}: {e}")
    finally:
        STATE.is_optimizing = False


def start_bayesopt():
    """Kick off Bayesian-optimization experiment suggestion on a background
    thread. Shares the Optimize tab's target / direction / constraints / column
    roles — only the surrogate (GP) and selection (Expected Improvement) differ."""
    if STATE.is_bayesopt or not STATE.data_path:
        return
    STATE.is_bayesopt = True
    STATE.bo_error = ""
    STATE.bo_result = None
    STATE.bo_status = "Fitting Gaussian-process surrogate…"

    features = col_types = ids = mixed = None
    excluded = _split_cols(STATE.opt_excluded)
    if STATE.coltype_columns:
        sync_roles_to_cfg()
        features = [c for c in STATE.coltype_columns
                    if STATE.coltype_role.get(c) == "feature"]
        ids = [c for c in STATE.coltype_columns
               if STATE.coltype_role.get(c) == "id"]
        mixed = [c for c in STATE.coltype_columns
                 if STATE.coltype_role.get(c) == "messy"]
        excluded = [c for c in STATE.coltype_columns
                    if STATE.coltype_role.get(c) == "exclude"]
        col_types = dict(STATE.coltype_map)
    else:
        ids = _split_cols(STATE.id_columns)
        mixed = _split_cols(STATE.mixed_column)

    cfg = dict(
        data_path=STATE.data_path,
        target=STATE.opt_target.strip(),
        excluded=excluded,
        fixed=_parse_fixed(STATE.opt_fixed),
        direction="maximise" if STATE.opt_direction_idx == 0 else "minimise",
        sheet=_current_sheet(),
        features=features,
        col_types=col_types,
        ids=ids,
        mixed=mixed,
        chemistry_enabled=STATE.chemistry_enabled,
        chemistry_schema=STATE.chemistry_schema,
        chemistry_mode=["off", "automatic", "compact", "standard", "full", "custom"][
            min(STATE.chemistry_mode_idx, 5)],
        constraints_text=STATE.opt_constraints,
        bo_batch=int(STATE.bo_batch),
        bo_xi=float(STATE.bo_xi),
    )
    threading.Thread(target=_bayesopt_worker, args=(cfg,), daemon=True).start()


def _bayesopt_worker(cfg):
    STATE.prog_bo.begin("Starting Bayesian optimization…")
    try:
        STATE.bo_status = "Fitting GP surrogate + maximizing acquisition…"
        res = run_capacity_optimization(
            cfg["data_path"], cfg["target"], cfg["excluded"],
            cfg["fixed"], cfg["direction"], sheet=cfg.get("sheet"),
            features=cfg.get("features"), col_types=cfg.get("col_types"),
            ids=cfg.get("ids"), mixed=cfg.get("mixed"),
            chemistry_enabled=cfg.get("chemistry_enabled", True),
            chemistry_schema=cfg.get("chemistry_schema"),
            chemistry_mode=cfg.get("chemistry_mode", "automatic"),
            constraints_text=cfg.get("constraints_text", ""),
            bayesopt_mode=True, run_de=False,
            bo_batch=cfg.get("bo_batch", 5), bo_xi=cfg.get("bo_xi", 0.01),
            progress=STATE.prog_bo.step)
        STATE.bo_result = res
        src = ("column-type roles" if res.get("mode") == "column-type"
               else "manual exclusions")
        STATE.bo_status = (
            f"Done ({src}). GP surrogate on {res['n_rows']} rows, "
            f"{res['n_knobs']} knobs "
            f"({res.get('n_numeric', 0)} numeric / {res.get('n_categorical', 0)} "
            f"categorical). {len(res.get('proposals', []))} experiment(s) proposed.")
        STATE.prog_bo.finish(
            f"Done — {len(res.get('proposals', []))} experiment(s) proposed.")
    except Exception as e:  # noqa: BLE001
        STATE.bo_error = f"{type(e).__name__}: {e}"
        STATE.bo_status = "Suggestion failed."
        STATE.prog_bo.fail(f"{type(e).__name__}: {e}")
    finally:
        STATE.is_bayesopt = False


# =============================================================================
# CHARTS  (renders 8 diagnostic plots as PNGs on a worker thread)
# =============================================================================
CHART_DIR = "charts"


def _aggregate_importance(importances, numeric_schema, categorical_schema):
    """Fold one-hot dummy importances back onto their source feature name."""
    num_names = set(numeric_schema.keys())
    cat_names = list(categorical_schema.keys())
    agg = {}
    for col, imp in importances:
        src = col
        if col not in num_names:
            for cn in cat_names:
                if col == cn or col.startswith(cn + "_"):
                    src = cn
                    break
        agg[src] = agg.get(src, 0.0) + float(imp)
    return agg


def start_charts():
    """Kick off chart generation on a background thread (needs a trained model)."""
    if STATE.is_charting or not STATE.trained:
        return
    STATE.is_charting = True
    STATE.charts_error = ""
    STATE.chart_items = []
    STATE.charts_status = "Preparing data…"

    cfg = _training_config_from_state()
    cfg["targets"] = list(STATE.targets)
    cfg["validation"] = dict(STATE.validation_config)
    numeric_feats = list(STATE.numeric_schema.keys())

    def pick(idx):
        return numeric_feats[idx] if 0 <= idx < len(numeric_feats) else None

    opts = dict(
        target_idx=STATE.charts_target_idx,
        pareto_a=STATE.charts_pareto_a,
        pareto_b=STATE.charts_pareto_b,
        featx=pick(STATE.charts_featx_idx),
        featy=pick(STATE.charts_featy_idx),
        imp_top=STATE.top_feature_count,
    )
    threading.Thread(target=_charts_worker, args=(cfg, opts), daemon=True).start()


def _charts_worker(cfg, opts):
    import charts as C
    try:
        os.makedirs(CHART_DIR, exist_ok=True)
        STATE.charts_run += 1
        rid = STATE.charts_run

        def path(name):
            return f"{CHART_DIR}/{name}_{rid}.png"

        items = []
        targets = cfg["targets"]
        ti = min(max(opts["target_idx"], 0), len(targets) - 1)

        STATE.prog_charts.begin("Loading & cleaning data…")

        def cs(msg, i):
            STATE.charts_status = msg
            STATE.prog_charts.step(msg, i / 13.0)

        cs("Loading & cleaning data…", 0)
        data = build_training_data(cfg)
        X_enc, y = data["X_encoded"], data["y"]
        numeric_schema = data["numeric_schema"]
        categorical_schema = data["categorical_schema"]
        numeric_feats = list(numeric_schema.keys())
        validation_label = V.validation_method_label(cfg["validation"])
        grouping = cfg["validation"].get("group_column", "")
        validation_title = validation_label + (f" by {grouping}" if grouping else "")

        # Correlation and target-signal analysis.
        cs("1/13 · correlation analysis…", 1)
        corr_df = pd.concat(
            [X_enc[numeric_feats].reset_index(drop=True), y.reset_index(drop=True)],
            axis=1,
        )
        items.append(("Pearson correlation heatmap",
                      C.correlation_heatmap(corr_df, path("corr_pearson"),
                                            method="pearson",
                                            title="Feature and target correlation")))
        items.append(("Spearman correlation heatmap",
                      C.correlation_heatmap(corr_df, path("corr_spearman"),
                                            method="spearman",
                                            title="Monotonic feature and target correlation")))
        items.append(("Mutual information ranking",
                      C.mutual_information_bar(X_enc, y.iloc[:, ti], path("mi"),
                                               target_name=targets[ti], top=20)))
        items.append(("Target distribution",
                      C.target_distribution(y.iloc[:, ti].values, path("target_dist"),
                                            target_name=targets[ti])))
        items.append(("Model performance summary",
                      C.model_performance_summary(STATE.metrics, path("model_perf"))))

        # Out-of-fold predictions (shared by charts 2 & 3), same model as training.
        cs("6/13 · out-of-fold predictions…", 6)
        if STATE.oof_predictions is None or len(STATE.oof_predictions) != len(y):
            base = ExtraTreesRegressor(
                n_estimators=300, max_depth=12, min_samples_split=4,
                max_features="sqrt", random_state=cfg["validation"]["random_state"],
                n_jobs=-1)
            oof_model = base if len(targets) == 1 else MultiOutputRegressor(base)
            evaluation = V.evaluate_model_cv(
                oof_model, data["X_raw"], y, cfg["validation"], groups=data["groups"],
                numeric_columns=list(numeric_schema),
                categorical_columns=list(categorical_schema))
            oof = evaluation["oof_predictions"].to_numpy()
            distributions = evaluation["metric_distributions"]
        else:
            oof = STATE.oof_predictions.to_numpy()
            distributions = STATE.metric_distributions
        r2_by = {t: float(r2_score(y.iloc[:, i], oof[:, i])) for i, t in enumerate(targets)}

        items.append(("Predicted vs actual",
                      C.predicted_vs_actual(y.values, oof, targets, path("pva"), r2_by,
                                            validation_method=validation_title)))
        cs("7/13 · residual diagnostics…", 7)
        items.append(("Residual plot",
                      C.residual_plot(y.values, oof, targets, path("resid"),
                                      validation_method=validation_title)))
        items.append(("Residual distribution",
                      C.residual_distribution(y.values, oof, targets, path("resid_dist"),
                                              validation_method=validation_title)))
        items.append(("Fold/repeat metric distributions",
                      C.cv_metric_distribution(distributions, path("cv_dist"),
                                               validation_method=validation_title)))

        # 4. Feature importance (folded back to source columns).
        cs("9/13 · feature importance…", 9)
        imp_src = _aggregate_importance(STATE.importances, numeric_schema, categorical_schema)
        top = opts.get("imp_top", 20)
        items.append(("Feature importance", C.feature_importance(
            {feature_label(k): v for k, v in imp_src.items()}, path("imp"), top=top,
            title="Grouped feature importance"
            + (" (all features)" if not top else f" (top {top})"))))

        # 5 & 6. SHAP on the trained single-target estimator.
        Xs_shap = X_enc.reindex(columns=STATE.feature_columns, fill_value=0)
        est = (STATE.model.estimators_[ti]
               if len(targets) > 1 and hasattr(STATE.model, "estimators_")
               else STATE.model)
        cs("10/13 · SHAP summary (can be slow)…", 10)
        items.append(("SHAP summary",
                      C.shap_summary(est, Xs_shap, path("shap_sum"), targets[ti],
                                     display_labels=STATE.feature_labels)))
        cs("11/13 · SHAP dependence…", 11)
        items.append(("SHAP dependence",
                      C.shap_dependence(est, Xs_shap, path("shap_dep"), targets[ti],
                                        display_labels=STATE.feature_labels)))

        # 7. Optimization heatmap — sweep two numeric knobs through the model.
        cs("12/13 · optimization heatmap…", 12)
        fx, fy = opts.get("featx"), opts.get("featy")
        if fx not in numeric_feats or fy not in numeric_feats or fx == fy:
            ranked = [f for f in sorted(imp_src, key=lambda k: -imp_src[k])
                      if f in numeric_feats]
            picks = (ranked or numeric_feats)[:2]
            fx = fx if fx in numeric_feats else (picks[0] if picks else None)
            fy = fy if fy in numeric_feats and fy != fx else (
                picks[1] if len(picks) > 1 else None)

        def predict_fn(raw_df):
            return _predict_2d(STATE.model, build_matrix(raw_df))

        if fx and fy and fx != fy:
            xr = (float(np.percentile(X_enc[fx], 2)), float(np.percentile(X_enc[fx], 98)))
            yr = (float(np.percentile(X_enc[fy], 2)), float(np.percentile(X_enc[fy], 98)))
            items.append(("Optimization heatmap", C.optimization_heatmap(
                predict_fn, numeric_schema, categorical_schema, fx, fy, path("opt"),
                target_index=ti, target_name=targets[ti], x_range=xr, y_range=yr)))
        else:
            items.append(("Optimization heatmap", C._placeholder(
                path("opt"), "Optimization heatmap",
                "Need at least two numeric features to sweep.")))

        # Pareto front — trade-off between two targets (both maximised).
        cs("13/13 · Pareto front…", 13)
        if len(targets) >= 2:
            a = targets[min(max(opts["pareto_a"], 0), len(targets) - 1)]
            b = targets[min(max(opts["pareto_b"], 0), len(targets) - 1)]
            items.append(("Pareto front", C.pareto_front(y, a, b, path("pareto"))))
        else:
            items.append(("Pareto front", C._placeholder(
                path("pareto"), "Pareto front", "Need at least two targets.")))

        STATE.chart_items = items
        STATE.charts_status = f"Done. Generated {len(items)} charts."
        STATE.prog_charts.finish(f"Done — {len(items)} charts generated.")
    except Exception as e:  # noqa: BLE001
        STATE.charts_error = f"{type(e).__name__}: {e}"
        STATE.charts_status = "Chart generation failed."
        STATE.prog_charts.fail(f"{type(e).__name__}: {e}")
    finally:
        STATE.is_charting = False


# =============================================================================
# LATENT VARIABLES  (engineered indices + PCA/PLS + leakage-safe A/B/C compare)
# =============================================================================
def lat_scan_columns():
    """Populate the target dropdown + default exclusions from the chosen file/tab."""
    if not STATE.data_path:
        STATE.lat_status = "Choose a spreadsheet in the Train tab first."
        return
    try:
        df = read_any(STATE.data_path, nrows=50, sheet=_current_sheet())
        STATE.lat_columns = [str(c) for c in df.columns]
        # Default the target to the first capacity column present, else first column.
        caps = [c for c in latent.CAPACITY_TARGETS if c in STATE.lat_columns]
        target = caps[0] if caps else (STATE.lat_columns[0] if STATE.lat_columns else "")
        if target in STATE.lat_columns:
            STATE.lat_target_idx = STATE.lat_columns.index(target)
        STATE.lat_excluded = ", ".join(latent.default_excluded(target, STATE.lat_columns))
        STATE.lat_status = f"Scanned {len(STATE.lat_columns)} columns. Adjust, then Run analysis."
    except Exception as e:  # noqa: BLE001
        STATE.lat_status = f"Scan failed: {e}"


def _lat_current_config():
    "grade in current config"
    """Assemble a latent.LatentConfig from the current UI fields."""
    target = (STATE.lat_columns[STATE.lat_target_idx]
              if 0 <= STATE.lat_target_idx < len(STATE.lat_columns) else "")
    return latent.LatentConfig(
        target=target,
        categorical=_split_cols(STATE.lat_categorical),
        numerical=_split_cols(STATE.lat_numerical),
        excluded=_split_cols(STATE.lat_excluded),
        chem_weights=tuple(float(w) for w in STATE.lat_chem_w),
        biomass_method="pca" if STATE.lat_biomass_pca else "mean",
    )


def start_latent():
    """Kick off the full latent-variable analysis on a background thread."""
    if STATE.is_lat_running or not STATE.data_path:
        return
    if not STATE.lat_columns:
        lat_scan_columns()
    STATE.is_lat_running = True
    STATE.lat_error = ""
    STATE.lat_chart_items = []
    STATE.lat_status = "Starting…"
    cfg = _lat_current_config()
    opts = dict(sheet=_current_sheet(), n_pca=int(STATE.lat_pca_components),
                n_pls=int(STATE.lat_pls_components))
    threading.Thread(target=_latent_worker, args=(cfg, opts), daemon=True).start()


def _latent_worker(cfg, opts):
    import charts as C
    try:
        os.makedirs(CHART_DIR, exist_ok=True)
        STATE.lat_run += 1
        rid = STATE.lat_run

        def path(name):
            return f"{CHART_DIR}/lat_{name}_{rid}.png"

        STATE.lat_status = "Loading data…"
        df = read_any(STATE.data_path, sheet=opts["sheet"])
        if cfg.target not in df.columns:
            raise ValueError(f"Target '{cfg.target}' not found. Scan columns first.")

        items = []

        # A. Engineered indices (full-data, for display/export + correlation chart).
        STATE.lat_status = "Computing engineered indices…"
        eng = latent.compute_engineered_latents(df, cfg.chem_weights, cfg.biomass_method)
        corr_df = pd.concat(
            [eng.reset_index(drop=True),
             pd.to_numeric(df[cfg.target], errors="coerce").reset_index(drop=True)],
            axis=1)
        items.append(("Engineered latents vs target (correlation)",
                      C.correlation_heatmap(corr_df, path("eng_corr"),
                                            title="Engineered latents vs target")))

        # B. PCA (full-data fit for display) + its three charts.
        STATE.lat_status = "Fitting PCA…"
        pca = latent.fit_pca(df, cfg, opts["n_pca"])
        items.append(("PCA explained variance",
                      C.explained_variance(pca["explained_variance_ratio"], path("evr"))))
        material = df["Material"] if "Material" in df.columns else None
        items.append(("PCA scores",
                      C.pca_score_scatter(pca["scores"], material, path("scores"),
                                          color_name="Material")))
        items.append(("PCA loadings (PC1)",
                      C.pca_loading_bar(pca["loadings"], path("load"), pc="PC1")))
        items.append(("PCA variable contribution",
                      C.pca_variable_contribution(pca["loadings"], path("contrib"),
                                                  pcs=("PC1", "PC2"))))
        items.append(("Target distribution",
                      C.target_distribution(pd.to_numeric(df[cfg.target], errors="coerce"),
                                            path("target"), target_name=cfg.target)))

        # PLS 5-fold CV.
        STATE.lat_status = "Cross-validating PLS…"
        pls = latent.evaluate_pls(df, cfg, opts["n_pls"])

        # C. Pipeline comparison A/B/C (leakage-safe).
        def prog(msg):
            STATE.lat_status = msg
        comp = latent.compare_pipelines(df, cfg, n_pca=opts["n_pca"], progress=prog)
        for metric in ("r2", "rmse", "mae"):
            items.append((f"Pipeline A/B/C · {metric.upper()}",
                          C.pipeline_comparison_bar(comp, path(f"cmp_{metric}"), metric=metric)))

        # Actual-vs-predicted + residuals for the strongest pipeline (C).
        STATE.lat_status = "Out-of-fold predictions (pipeline C)…"
        y_true, y_pred, used = latent.oof_predict(df, cfg, variant="C",
                                                  n_pca=opts["n_pca"])
        r2 = {cfg.target: float(r2_score(y_true, y_pred))}
        items.append((f"Actual vs predicted (C · {used})",
                      C.predicted_vs_actual(y_true, y_pred, [cfg.target],
                                            path("ava"), r2)))
        items.append((f"Residuals (C · {used})",
                      C.residual_plot(y_true, y_pred, [cfg.target], path("resid"))))
        items.append((f"Residual distribution (C · {used})",
                      C.residual_distribution(y_true, y_pred, [cfg.target],
                                              path("resid_dist"))))

        # Publish.
        STATE.lat_pls_result = pls
        STATE.lat_compare_result = comp
        STATE.lat_chart_items = items
        STATE.lat_last = {"cfg": cfg, "sheet": opts["sheet"], "pca": pca,
                          "n_pca": opts["n_pca"]}
        STATE.lat_status = (f"Done. PLS(k={pls['n_components']}) CV R²="
                            f"{pls['r2_mean']:.3f}±{pls['r2_std']:.3f}. See charts below.")
    except Exception as e:  # noqa: BLE001
        STATE.lat_error = f"{type(e).__name__}: {e}"
        STATE.lat_status = "Latent analysis failed."
    finally:
        STATE.is_lat_running = False


def lat_export(kind):
    """Export latent artifacts to files in the working directory."""
    last = STATE.lat_last
    if not last:
        STATE.lat_status = "Run the analysis first."
        return
    cfg = last["cfg"]
    try:
        df = read_any(STATE.data_path, sheet=last["sheet"])
        if kind == "xlsx":
            frame = latent.build_export_frame(df, cfg, last["pca"]["scores"])
            frame.to_excel("latent_export.xlsx", index=False)
            STATE.lat_status = f"Saved latent_export.xlsx ({len(frame)} rows)."
        elif kind == "loadings":
            last["pca"]["loadings"].to_csv("pca_loadings.csv")
            STATE.lat_status = "Saved pca_loadings.csv."
        elif kind == "comparison":
            latent.comparison_to_frame(STATE.lat_compare_result).to_csv(
                "latent_comparison.csv", index=False)
            STATE.lat_status = "Saved latent_comparison.csv."
        elif kind == "pipeline":
            pipe = latent.fit_full_pipeline(df, cfg, variant="C", n_pca=last["n_pca"])
            joblib.dump({"pipeline": pipe, "target": cfg.target,
                         "categorical": cfg.categorical, "numerical": cfg.numerical},
                        "latent_pipeline.joblib")
            STATE.lat_status = "Saved latent_pipeline.joblib."
    except Exception as e:  # noqa: BLE001
        STATE.lat_error = f"Export failed: {e}"


# =============================================================================
# DATASET INTELLIGENCE  (why prediction succeeds / fails — reuses Train config)
# =============================================================================
def _configured_targets():
    """Targets currently configured in the Train / Column types tabs."""
    roles = [c for c in STATE.coltype_columns if STATE.coltype_role.get(c) == "target"]
    return roles or _split_cols(STATE.target_columns)


def _intel_config(target):
    """
    Build a leakage-safe latent.LatentConfig from the EXISTING Train / Column
    types selections — no duplicate configuration in the Intelligence tab.
    Categorical/numerical come from the Column types roles when available, else
    from the latent defaults filtered to the file's columns.
    """
    all_targets = _configured_targets()
    if STATE.coltype_columns:
        cat = [c for c in STATE.coltype_columns
               if STATE.coltype_role.get(c) == "feature"
               and STATE.coltype_map.get(c) == "categorical"]
        num = [c for c in STATE.coltype_columns
               if STATE.coltype_role.get(c) == "feature"
               and STATE.coltype_map.get(c) == "numeric"]
        excluded = [c for c in STATE.coltype_columns
                    if STATE.coltype_role.get(c) in ("exclude", "id")]
    else:  # fall back to the latent recommended defaults
        cat = list(latent.DEFAULT_CATEGORICAL)
        num = list(latent.DEFAULT_NUMERICAL)
        excluded = list(latent.DEFAULT_ALWAYS_EXCLUDE)
    excluded += [t for t in all_targets if t != target]  # never leak other targets
    return latent.LatentConfig(
        target=target, categorical=cat, numerical=num, excluded=excluded,
        chem_weights=tuple(float(w) for w in STATE.lat_chem_w),
        biomass_method="pca" if STATE.lat_biomass_pca else "mean")


def start_intelligence():
    """Kick off the full Dataset Intelligence analysis on a background thread."""
    if STATE.is_intel_running or not STATE.data_path:
        return
    targets = _configured_targets()
    if not targets:
        STATE.intel_status = "Set at least one Target in the Train / Column types tab first."
        return
    ti = min(STATE.intel_target_idx, len(targets) - 1)
    STATE.is_intel_running = True
    STATE.intel_error = ""
    STATE.intel_chart_items = []
    STATE.intel_results = None
    STATE.intel_status = "Starting…"
    cfg = _intel_config(targets[ti])
    opts = dict(sheet=_current_sheet(), n_pca=int(STATE.intel_pca_components))
    threading.Thread(target=_intel_worker, args=(cfg, opts), daemon=True).start()


def _intel_worker(cfg, opts):
    """Run all 10 sections + AI insights, generate charts, cache for the report."""
    import charts as C
    try:
        os.makedirs(CHART_DIR, exist_ok=True)
        STATE.intel_run += 1
        rid = STATE.intel_run

        def path(name):
            return f"{CHART_DIR}/intel_{name}_{rid}.png"

        STATE.prog_intel.begin("Loading data…")
        _intel_step = [0]
        _intel_total = 11

        def prog(msg):
            STATE.intel_status = msg
            _intel_step[0] += 1
            STATE.prog_intel.step(msg, min(_intel_step[0] / _intel_total, 0.98))

        prog("Loading data…")
        df = read_any(STATE.data_path, sheet=opts["sheet"])
        if cfg.target not in df.columns:
            raise ValueError(f"Target '{cfg.target}' not found in the sheet.")
        if STATE.chemistry_enabled:
            original_cat, original_num = cfg.feature_columns(list(df.columns))
            feature_frame = df[original_cat + original_num].copy()
            if STATE.chemistry_schema.get("columns"):
                chemistry_schema = STATE.chemistry_schema
                exp_frame = chemistry.ENGINE.transform_with_schema(
                    feature_frame, chemistry_schema).reset_index(drop=True)
            else:
                modes = ["off", "automatic", "compact", "standard", "full", "custom"]
                config = chemistry.ENGINE.auto_configure(
                    feature_frame, groups=pd.Series(np.arange(len(feature_frame))),
                    requested_mode=modes[min(STATE.chemistry_mode_idx, 5)])
                expansion = chemistry.ENGINE.transform(feature_frame, config=config)
                chemistry_schema = expansion.metadata
                exp_frame = expansion.frame.reset_index(drop=True)
            if chemistry_schema.get("columns"):
                chemical_sources = set(chemistry_schema["columns"])
                # Merge the expanded columns in one concat (adding them one by one
                # fragments the DataFrame); drop any pre-existing duplicates first.
                drop = [c for c in df.columns
                        if c in chemical_sources or c in exp_frame.columns]
                df = pd.concat(
                    [df.drop(columns=drop, errors="ignore").reset_index(drop=True),
                     exp_frame], axis=1)
                new_cat = [c for c in exp_frame
                           if not pd.api.types.is_numeric_dtype(exp_frame[c])]
                new_num = [c for c in exp_frame if c not in new_cat]
                cfg = latent.LatentConfig(
                    target=cfg.target, categorical=new_cat, numerical=new_num,
                    excluded=list(cfg.excluded), chem_weights=cfg.chem_weights,
                    biomass_method=cfg.biomass_method)
        # Fix numeric columns polluted by a stray text marker (e.g. an 'INPUT'
        # annotation row) so median imputation / PCA don't choke; never touch
        # columns the config treats as categorical.
        df = auto_coerce_numeric(df, protected=set(cfg.categorical) | {cfg.target})
        # The config EXPLICITLY declares these columns numeric, so force them.
        # 'not-done' cells ('--', 'none') -> 0; unknown/stray text -> NaN (imputed
        # or dropped) instead of crashing SimpleImputer(strategy='median') / PCA.
        _cat_cols, _num_cols = cfg.feature_columns(list(df.columns))
        for _c in _num_cols:
            df[_c] = coerce_numeric_series(df[_c])
        # Text features: '--'/'none' -> explicit "None" category (distinct from an
        # unrecorded "Missing"), so the model can learn "step skipped" as a signal.
        for _c in _cat_cols:
            df[_c] = normalize_categorical_series(df[_c])

        prog("1/9 · dataset summary…")
        summary = intelligence.dataset_summary(df, cfg)
        prog("2/9 · predictability…")
        pred = intelligence.predictability(df, cfg)
        prog("3/9 · redundancy / VIF…")
        redund = intelligence.redundancy(df, cfg)
        prog("4/9 · learnability (5-fold CV)…")
        learn = intelligence.learnability(df, cfg, progress=prog)
        diff = intelligence.difficulty_score(summary, pred, learn)
        prog("6/9 · causal structure…")
        causal = intelligence.causal_structure(df)
        prog("7/9 · latent analysis…")
        lat = intelligence.latent_analysis(df, cfg)
        prog("8/9 · target analysis…")
        tgt = intelligence.target_analysis(df, cfg)
        prog("9/9 · PCA…")
        pca = latent.fit_pca(df, cfg, opts["n_pca"])

        # ---- Charts (reuse existing plotting code) ----
        items = []
        cat, num = cfg.feature_columns(list(df.columns))
        items.append(("Missing values",
                      C.missingness_chart(df, path("missing"))))
        # Predictability
        strength = pred["table"].set_index("feature")["mutual_info"]
        items.append(("Predictability — mutual information",
                      C.ranked_bar(strength, path("predict"),
                                   title="Feature → target mutual information",
                                   xlabel="mutual information")))
        corr_cols = num + [cfg.target]
        items.append(("Pearson correlation heatmap",
                      C.correlation_heatmap(df[corr_cols].apply(pd.to_numeric, errors="coerce"),
                                            path("corr"), title="Feature correlation",
                                            method="pearson")))
        items.append(("Spearman correlation heatmap",
                      C.correlation_heatmap(df[corr_cols].apply(pd.to_numeric, errors="coerce"),
                                            path("corr_spearman"),
                                            title="Monotonic feature correlation",
                                            method="spearman")))
        # Redundancy — VIF
        vif_ser = pd.Series({k: v for k, v in redund["vif"].items() if np.isfinite(v)})
        items.append(("Feature redundancy — VIF",
                      C.ranked_bar(vif_ser, path("vif"),
                                   title="Variance Inflation Factor (>10 = redundant)",
                                   xlabel="VIF")))
        items.append(("Dataset difficulty",
                      C.dataset_difficulty_chart(summary, diff, learn, path("difficulty"))))
        # Learnability comparison
        learn_r2 = pd.Series({n: s["r2_mean"] for n, s in learn["results"].items()
                              if "r2_mean" in s})
        items.append(("Learnability — CV R² by model",
                      C.ranked_bar(learn_r2, path("learn"), diverging=True,
                                   title="5-fold CV R² by model", xlabel="CV R²")))
        # Causal pipeline
        items.append(("Causal pipeline",
                      C.causal_diagram(causal["structure_present"], path("causal"),
                                       message=causal["message"])))
        # Latent radar + heatmap
        contrib = lat["contribution"]
        items.append(("Latent contribution (radar)",
                      C.radar_chart(list(contrib.keys()), list(contrib.values()),
                                    path("radar"), title="Latent contribution (MI share)")))
        lat_corr = pd.concat(
            [lat["latents"].reset_index(drop=True),
             pd.to_numeric(df[cfg.target], errors="coerce").reset_index(drop=True)], axis=1)
        items.append(("Latent variables vs target",
                      C.correlation_heatmap(lat_corr, path("lat_corr"),
                                            title="Latent vs target")))
        # Target analysis
        items.append(("Target distribution",
                      C.target_distribution(tgt.get("values", []), path("target"),
                                            target_name=cfg.target, stats=tgt)))
        # PCA scree + biplot
        items.append(("PCA explained variance (scree)",
                      C.explained_variance(pca["explained_variance_ratio"], path("scree"))))
        material = df["Material"] if "Material" in df.columns else None
        items.append(("PCA biplot",
                      C.biplot(pca["scores"], pca["loadings"], path("biplot"),
                               color_series=material, color_name="Material")))
        items.append(("PCA loadings (PC1)",
                      C.pca_loading_bar(pca["loadings"], path("load"), pc="PC1")))
        items.append(("PCA variable contribution",
                      C.pca_variable_contribution(pca["loadings"], path("pca_contrib"),
                                                  pcs=("PC1", "PC2"))))

        # ---- AI Research Assistant + conclusion ----
        insights = intelligence.ai_insights(summary, pred, redund, diff, learn,
                                            causal, lat, tgt)
        conclusion = intelligence.final_conclusion(summary, pred, learn, causal, tgt)

        STATE.intel_results = {"summary": summary, "pred": pred, "redund": redund,
                               "diff": diff, "learn": learn, "causal": causal,
                               "latent": lat, "target": tgt, "pca": pca, "cfg": cfg}
        STATE.intel_chart_items = items
        STATE.intel_insights = insights
        STATE.intel_conclusion = conclusion
        STATE.intel_status = (f"Done. Difficulty: {diff['label']} · best CV R²="
                              f"{learn['best_r2']:.3f}. See findings below.")
        STATE.prog_intel.finish(f"Done — difficulty {diff['label']}, "
                                f"best CV R² = {learn['best_r2']:.3f}.")
    except Exception as e:  # noqa: BLE001
        STATE.intel_error = f"{type(e).__name__}: {e}"
        STATE.intel_status = "Dataset Intelligence failed."
        STATE.prog_intel.fail(f"{type(e).__name__}: {e}")
    finally:
        STATE.is_intel_running = False


def _top_features_for_summary(n=5):
    """Display-friendly [(name, importance), ...] for the slideshow conclusion."""
    if not STATE.importances:
        return []
    agg = _aggregate_importance(STATE.importances, STATE.numeric_schema,
                                STATE.categorical_schema)
    ranked = sorted(agg.items(), key=lambda kv: -kv[1])[:n]
    return [(pretty(name), val) for name, val in ranked]


def _reset_chart_cache():
    """Wipe the transient chart-render cache (CHART_DIR) at startup so generated
    charts never accumulate on disk between sessions. Charts are only kept when
    the user explicitly exports them with the 'Save charts…' button."""
    exts = (".png", ".svg", ".pdf", ".txt", ".json")
    try:
        if os.path.isdir(CHART_DIR):
            for f in os.listdir(CHART_DIR):
                if f.lower().endswith(exts):
                    try:
                        os.remove(os.path.join(CHART_DIR, f))
                    except OSError:
                        pass
    except OSError:
        pass


def save_charts_to_folder(folder):
    """Copy every chart generated this session (diagnostics + latent + dataset
    intelligence) into ``folder`` with readable, ordered filenames. This is the
    explicit, user-controlled save — generation itself only writes a transient
    display cache."""
    import shutil

    sections = []
    if STATE.chart_items:
        sections.append(("diagnostics", STATE.chart_items))
    if STATE.lat_chart_items:
        sections.append(("latent", STATE.lat_chart_items))
    if STATE.intel_chart_items:
        sections.append(("intelligence", STATE.intel_chart_items))
    if not sections:
        STATE.save_charts_status = "Generate charts first, then save."
        return
    saved = 0
    try:
        os.makedirs(folder, exist_ok=True)
        for section, items in sections:
            for i, (title, rel) in enumerate(items, 1):
                src = os.path.abspath(rel)
                if not os.path.exists(src):
                    continue
                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(title)).strip("_") or "chart"
                dst = os.path.join(folder, f"{section}_{i:02d}_{safe}.png")
                shutil.copy2(src, dst)
                saved += 1
                note = os.path.splitext(src)[0] + ".txt"   # optional sidecar note
                if os.path.exists(note):
                    shutil.copy2(note, os.path.splitext(dst)[0] + ".txt")
        STATE.save_charts_status = f"Saved {saved} chart(s) to {folder}"
    except Exception as e:  # noqa: BLE001
        STATE.save_charts_status = f"Save failed: {e}"


def export_slideshow(path):
    """Build a narrated slideshow PDF of every chart generated this session.

    Gathers whichever chart bundles exist (Model Diagnostics, Latent Variables,
    Dataset Intelligence), adds a title slide, per-chart explanations, and an
    auto-synthesised conclusion built from the CV metrics and top features.
    """
    sections = []
    if STATE.chart_items:
        sections.append(("Model Diagnostics", list(STATE.chart_items)))
    if STATE.lat_chart_items:
        sections.append(("Latent Variables", list(STATE.lat_chart_items)))
    if STATE.intel_chart_items:
        sections.append(("Dataset Intelligence", list(STATE.intel_chart_items)))
    if not sections:
        STATE.slideshow_status = "Generate charts first (Charts tab)."
        return

    dataset = os.path.basename(STATE.data_path) if STATE.data_path else "(unsaved dataset)"
    targets = list(STATE.targets)
    meta_rows = [
        ("Dataset", dataset),
        ("Target(s)", ", ".join(targets) if targets else "n/a"),
        ("Training rows", STATE.metrics[0].get("n_rows", "n/a") if STATE.metrics else "n/a"),
        ("Features", len(STATE.feature_columns) if STATE.feature_columns else "n/a"),
        ("Charts", sum(len(items) for _, items in sections)),
        ("Generated", time.strftime("%Y-%m-%d %H:%M")),
    ]
    try:
        out = slideshow.build_slideshow(
            path, sections,
            title="Model Analysis Summary",
            subtitle="BioCarbon Screen — hard-carbon synthesis",
            meta_rows=meta_rows,
            metrics=STATE.metrics,
            top_features=_top_features_for_summary(),
            targets=targets,
        )
        STATE.slideshow_status = f"Saved slideshow to '{out}'."
    except Exception as e:  # noqa: BLE001
        STATE.slideshow_status = f"Slideshow export failed: {e}"


def intel_export_pdf():
    """One-click PDF report (Section 10)."""
    r = STATE.intel_results
    if not r:
        STATE.intel_status = "Run the analysis first."
        return
    try:
        out = intelligence.build_pdf_report(
            "intelligence_report.pdf", r["summary"], r["pred"], r["diff"],
            r["learn"], STATE.intel_conclusion, STATE.intel_insights,
            STATE.intel_chart_items)
        STATE.intel_status = f"Saved {out}."
    except Exception as e:  # noqa: BLE001
        STATE.intel_error = f"PDF export failed: {e}"


# =============================================================================
# PREDICTION HELPERS  (shared by single + batch)
# =============================================================================
def build_matrix(X_raw):
    """One-hot encode a raw feature frame and align it to the trained columns.

    Uses drop_first=False so a single-row prediction keeps the one present
    category (drop_first=True would drop the only dummy and silently ignore the
    input); reindexing to the trained columns already omits each baseline dummy.
    """
    prepared = pd.DataFrame(X_raw).copy()
    chemistry_columns = list(STATE.chemistry_schema.get("columns", {}))
    if STATE.chemistry_enabled and chemistry_columns:
        for source in chemistry_columns:
            if source not in prepared:
                if len(prepared) == 1:
                    prepared[source] = (STATE.chemistry_custom_values.get(source, "").strip()
                                        or STATE.chemistry_values.get(source, "Unknown"))
                else:
                    prepared[source] = "Unknown"
        # Prediction must use the deployable model's exact selected chemistry
        # schema; never auto-configure from a single candidate row.
        prepared = chemistry.ENGINE.transform_with_schema(
            prepared, STATE.chemistry_schema)

    # Apply the exact final-model schema after descriptor expansion.
    cols = {}
    for col, med in STATE.numeric_schema.items():
        series = prepared.get(col)
        if series is None:
            series = pd.Series([med] * len(prepared), index=prepared.index)
        cols[col] = coerce_numeric_series(series).fillna(med)
    for col in STATE.categorical_schema:
        series = prepared.get(col)
        if series is None:
            series = pd.Series(["Missing"] * len(prepared), index=prepared.index)
        cols[col] = normalize_categorical_series(series)
    X_enc = pd.get_dummies(pd.DataFrame(cols), drop_first=False)
    # Overlapping names (e.g. 'Electrolyte' vs 'Electrolyte_Additive') or odd cell
    # values can yield duplicate dummy columns; keep the first so reindex is safe.
    X_enc = X_enc.loc[:, ~X_enc.columns.duplicated()]
    return X_enc.reindex(columns=STATE.feature_columns, fill_value=0)


def _predict_2d(model, X):
    """Normalise scalar and multi-output estimator predictions for shared UI code."""
    pred = np.asarray(model.predict(X), dtype=float)
    return pred.reshape(-1, 1) if pred.ndim == 1 else pred


def predict_single():
    """Predict all targets from the current Predict-tab widget values."""
    STATE.single_pred = None
    STATE.predict_error = ""
    row = dict(STATE.numeric_values)
    for col, choices in STATE.categorical_schema.items():
        # A '1M NaOH'-style recipe field may have typed a level unseen in
        # training; it lives in screen_custom_category and wins over the combo.
        custom = STATE.screen_custom_category.get(col, "").strip()
        row[col] = custom or choices[STATE.category_index[col]]
    try:
        X_raw = pd.DataFrame([row])
        preds = _predict_2d(STATE.model, build_matrix(X_raw))[0]
        STATE.single_pred = list(zip(STATE.targets, [float(v) for v in preds]))
    except Exception as e:  # noqa: BLE001
        STATE.predict_error = f"Prediction failed: {e}"


def predict_batch(path, sheet=None):
    """Predict for every row of a spreadsheet and save the results."""
    STATE.batch_results = None
    STATE.batch_status = ""
    STATE.batch_error = ""
    try:
        raw = read_any(path, sheet=sheet)
        prepared = prepare_raw(raw, STATE.cfg["ids"], STATE.cfg["mixed"])

        preds = _predict_2d(STATE.model, build_matrix(prepared))
        out = raw.copy()
        for i, t in enumerate(STATE.targets):
            out[f"pred_{t}"] = preds[:, i]
        STATE.batch_results = out
        out.to_csv("predictions.csv", index=False)
        STATE.batch_status = f"Predicted {len(out)} row(s). Saved to 'predictions.csv'."
    except Exception as e:  # noqa: BLE001
        STATE.batch_error = f"Batch prediction failed: {e}"


def rebuild_screener():
    """(Re)build the screening engine from the currently trained model + data."""
    try:
        if STATE.model is None or STATE.X_train is None or STATE.y_train is None:
            STATE.screener = None
            return
        STATE.screener = screening.Screener(
            STATE.model, STATE.feature_columns, STATE.numeric_schema,
            STATE.categorical_schema, STATE.targets, STATE.X_train, STATE.y_train,
            cv_rmse=STATE.cv_rmse, cv_r2=STATE.cv_r2,
            display_labels=STATE.feature_labels,
            chemistry_schema=STATE.chemistry_schema,
            chemistry_originals=STATE.chemistry_originals,
        )
    except Exception as e:  # noqa: BLE001 - screening is optional; never crash training
        STATE.screener = None
        STATE.screen_error = f"Could not build screening engine: {e}"


def save_model():
    joblib.dump(
        {
            "model": STATE.model,
            "feature_columns": STATE.feature_columns,
            "numeric_schema": STATE.numeric_schema,
            "categorical_schema": STATE.categorical_schema,
            "targets": STATE.targets,
            "cfg": STATE.cfg,
            "validation_config": STATE.validation_config,
            "group_column": STATE.validation_config.get("group_column", ""),
            "metrics": STATE.metrics,
            "metric_distributions": STATE.metric_distributions,
            "feature_diagnostics": STATE.feature_diagnostics,
            "reporting": {"top_feature_count": STATE.top_feature_count,
                          "show_feature_ratio_warnings": STATE.show_feature_ratio_warnings},
            "oof_predictions": STATE.oof_predictions,
            "oof_prediction_std": STATE.oof_prediction_std,
            "oof_prediction_samples": STATE.oof_prediction_samples,
            "oof_truth": STATE.oof_truth,
            "preprocessing": STATE.preprocessing_statement,
            "chemistry_config": STATE.chemistry_config,
            "chemistry_schema": STATE.chemistry_schema,
            "chemistry_feature_diagnostics": STATE.chemistry_feature_diagnostics,
            "chemistry_originals": STATE.chemistry_originals,
            "chemistry_overrides": chemistry.ENGINE.generator.export_overrides(),
            "chemistry_enabled": STATE.chemistry_enabled,
            "chemistry_pubchem_enabled": STATE.chemistry_pubchem_enabled,
            # Training data + CV metrics let a reloaded model still screen
            # (applicability domain, similar experiments, uncertainty).
            "X_train": STATE.X_train,
            "y_train": STATE.y_train,
            "cv_rmse": STATE.cv_rmse,
            "cv_r2": STATE.cv_r2,
            "importances": STATE.importances,
            "summary": STATE.summary,
            "feature_labels": STATE.feature_labels,
        },
        MODEL_OUT,
    )


def _normalize_loaded_metrics(metrics, bundle):
    """Convert pre-validation tuple metrics so old saved models still render."""
    rows = list(metrics or [])
    if not rows or isinstance(rows[0], dict):
        return rows
    converted = []
    for row in rows:
        if len(row) < 5:
            continue
        target, train_r2, train_rmse, cv_r2, cv_rmse = row[:5]
        def summary(value):
            return {"mean": float(value), "std": 0.0, "lower": float(value),
                    "upper": float(value), "n": 1, "confidence_level": 0.95,
                    "interval_method": "legacy"}
        converted.append({
            "target": target, "train_r2": float(train_r2),
            "train_rmse": float(train_rmse),
            "cv": {"r2": summary(cv_r2), "rmse": summary(cv_rmse),
                   "mae": summary(float("nan"))},
            "pooled_oof": {"r2": float(cv_r2), "rmse": float(cv_rmse),
                           "mae": float("nan")},
            "n_rows": (len(bundle.get("X_train"))
                       if bundle.get("X_train") is not None else 0), "n_groups": 0,
            "n_splits": 5, "n_repeats": 1,
            "validation_method": "Legacy 5-fold CV",
        })
    return converted


def load_model():
    b = V.load_model_bundle_compat(joblib.load(MODEL_OUT))
    STATE.model = b["model"]
    STATE.feature_columns = b["feature_columns"]
    STATE.numeric_schema = b["numeric_schema"]
    STATE.categorical_schema = b["categorical_schema"]
    STATE.targets = b["targets"]
    STATE.cfg = b.get("cfg", {"ids": [], "mixed": ""})
    STATE.feature_labels = (b.get("feature_labels")
                            or mixed_feature_labels(STATE.cfg.get("mixed", "")))
    STATE.X_train = b.get("X_train")
    STATE.y_train = b.get("y_train")
    STATE.cv_rmse = b.get("cv_rmse", {})
    STATE.cv_r2 = b.get("cv_r2", {})
    STATE.metrics = _normalize_loaded_metrics(b.get("metrics", []), b)
    STATE.metric_distributions = b.get("metric_distributions", {})
    STATE.validation_config = V.normalize_validation_config(b.get("validation_config"))
    STATE.feature_diagnostics = b.get("feature_diagnostics", {})
    STATE.oof_predictions = b.get("oof_predictions")
    STATE.oof_prediction_std = b.get("oof_prediction_std")
    STATE.oof_prediction_samples = b.get("oof_prediction_samples")
    STATE.oof_truth = b.get("oof_truth")
    STATE.preprocessing_statement = b.get("preprocessing", "Legacy model; preprocessing scope not recorded")
    STATE.chemistry_enabled = bool(b.get("chemistry_enabled", False))
    STATE.chemistry_config = dict(b.get("chemistry_config") or {})
    STATE.chemistry_feature_diagnostics = dict(
        b.get("chemistry_feature_diagnostics") or {})
    loaded_mode = str(STATE.chemistry_config.get(
        "mode", "automatic" if STATE.chemistry_enabled else "off")).lower()
    modes = ["off", "automatic", "compact", "standard", "full", "custom"]
    STATE.chemistry_mode_idx = modes.index(loaded_mode) if loaded_mode in modes else 1
    STATE.chemistry_pubchem_enabled = bool(b.get("chemistry_pubchem_enabled", False))
    chemistry.ENGINE.generator.set_pubchem_enabled(STATE.chemistry_pubchem_enabled)
    chemistry.ENGINE.generator.clear_overrides()
    for chemical_name, values in (b.get("chemistry_overrides") or {}).items():
        chemistry.ENGINE.generator.set_override(chemical_name, values)
    _set_chemistry_state(
        b.get("chemistry_schema", {}), b.get("chemistry_originals"))
    methods = ["random_kfold", "group_kfold", "repeated_grouped_cv"]
    STATE.validation_method_idx = methods.index(STATE.validation_config["method"])
    STATE.group_column = b.get("group_column", STATE.validation_config.get("group_column", ""))
    STATE.n_splits = STATE.validation_config["n_splits"]
    STATE.n_repeats = STATE.validation_config["n_repeats"]
    STATE.confidence_level = STATE.validation_config["confidence_level"] * 100.0
    STATE.random_state = STATE.validation_config["random_state"]
    STATE.single_target_mode = len(STATE.targets) == 1
    STATE.single_target = STATE.targets[0] if len(STATE.targets) == 1 else ""
    if STATE.feature_diagnostics:
        STATE.top_feature_count = int(b.get("reporting", {}).get("top_feature_count", 20))
    STATE.importances = b.get("importances", [])
    STATE.summary = b.get("summary", "")
    STATE.numeric_values = dict(STATE.numeric_schema)
    STATE.category_index = {c: 0 for c in STATE.categorical_schema}
    STATE.screen_custom_category = {c: "" for c in STATE.categorical_schema}
    STATE.mixed_text = {}
    STATE.trained = True
    rebuild_screener()
    STATE.status = f"Loaded model from '{MODEL_OUT}'."


# =============================================================================
# SCREENING  (primary workflow — runs the full research-assistant analysis)
# =============================================================================
def _current_screen_raw():
    """Assemble the synthesis-condition dict from the Screen-tab widgets."""
    row = dict(STATE.numeric_values)
    for col, choices in STATE.categorical_schema.items():
        custom = STATE.screen_custom_category.get(col, "").strip()
        if custom:
            row[col] = custom
        else:
            idx = STATE.category_index.get(col, 0)
            row[col] = choices[idx] if choices else "Missing"
    original_chemicals = {}
    for source in STATE.chemistry_schema.get("columns", {}):
        value = (STATE.chemistry_custom_values.get(source, "").strip()
                 or STATE.chemistry_values.get(source, "Unknown"))
        row[source] = value
        original_chemicals[source] = value
    if original_chemicals:
        row.update(chemistry.ENGINE.expand_row(row, STATE.chemistry_schema))
        row.update(original_chemicals)  # reporting retains the original chemical names
    return row


def _apply_screen_recipe(raw):
    """Push one imported/raw recipe into the Screen-tab widgets."""
    for col, med in STATE.numeric_schema.items():
        try:
            v = pd.to_numeric(raw.get(col), errors="coerce")
            STATE.numeric_values[col] = float(v) if pd.notna(v) else float(med)
        except Exception:  # noqa: BLE001
            STATE.numeric_values[col] = float(med)

    for col, choices in STATE.categorical_schema.items():
        value = raw.get(col, "Missing")
        value = "Missing" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
        if value in choices:
            STATE.category_index[col] = choices.index(value)
            STATE.screen_custom_category[col] = ""
        else:
            STATE.category_index[col] = STATE.category_index.get(col, 0)
            STATE.screen_custom_category[col] = value
    STATE.mixed_text = {}                  # recompose '1M NaOH' texts from the recipe
    STATE.screen_dirty = True


def import_screen_recipe(path):
    """Import the first candidate recipe from CSV/Excel into the Screen tab."""
    try:
        raw = read_any(path, nrows=1)
        if raw.empty:
            raise ValueError("The file has no rows.")
        prepared = prepare_raw(raw, STATE.cfg.get("ids", []), STATE.cfg.get("mixed", ""))
        prepared = auto_coerce_numeric(prepared)
        _apply_screen_recipe(prepared.iloc[0].to_dict())
        STATE.screen_import_status = f"Imported first recipe from {path}"
    except Exception as e:  # noqa: BLE001
        STATE.screen_import_status = f"Recipe import failed: {e}"


def start_screen():
    """Run the full screening workflow on a background thread."""
    if STATE.is_screening or STATE.screener is None or not STATE.targets:
        return
    STATE.is_screening = True
    STATE.screen_error = ""
    STATE.screen_dirty = False
    target = STATE.targets[min(STATE.screen_target_idx, len(STATE.targets) - 1)]
    raw = _current_screen_raw()
    threading.Thread(target=_screen_worker, args=(raw, target), daemon=True).start()


def _screen_worker(raw, target):
    try:
        STATE.screen_result = STATE.screener.screen(raw, target)
    except Exception as e:  # noqa: BLE001
        STATE.screen_error = f"{type(e).__name__}: {e}"
        STATE.screen_result = None
    finally:
        STATE.is_screening = False


def export_screen_report(path, kind):
    """Write the current screening result to a PDF or Excel file."""
    if not STATE.screen_result:
        STATE.screen_export_msg = "Run a screen first."
        return
    try:
        metadata = {
            "validation_config": STATE.validation_config,
            "metrics": STATE.metrics,
            "metric_distributions": STATE.metric_distributions,
            "feature_diagnostics": STATE.feature_diagnostics,
            "preprocessing": STATE.preprocessing_statement,
            "chemistry_config": STATE.chemistry_config,
            "chemistry_schema": STATE.chemistry_schema,
            "chemistry_feature_diagnostics": STATE.chemistry_feature_diagnostics,
            "chemistry_overrides": chemistry.ENGINE.generator.export_overrides(),
        }
        if kind == "pdf":
            report.build_prediction_pdf(path, STATE.screen_result, STATE.screener,
                                        model_summary=STATE.summary,
                                        model_metadata=metadata)
        else:
            report.export_prediction_excel(path, STATE.screen_result, STATE.screener,
                                           model_metadata=metadata)
        STATE.screen_export_msg = f"Saved report to {path}"
    except Exception as e:  # noqa: BLE001
        STATE.screen_export_msg = f"Export failed: {e}"


# =============================================================================
# EXPERIMENT PRIORITIZATION  (rank candidate synthesis routes)
# =============================================================================
def _candidate_rows_from_file(path, sheet=None):
    """Read candidate recipes from a spreadsheet, aligned to the model schema."""
    raw = read_any(path, sheet=sheet)
    prepared = prepare_raw(raw, STATE.cfg.get("ids", []), STATE.cfg.get("mixed", ""))
    prepared = auto_coerce_numeric(prepared)
    # Apply the same 'not-done' (0 / "None") vs unknown (median / "Missing") rules.
    num_cols, cat_cols = {}, {}
    for c, med in STATE.numeric_schema.items():
        s = prepared.get(c)
        num_cols[c] = (coerce_numeric_series(s).fillna(med)
                       if s is not None else pd.Series([med] * len(prepared)))
    for c in STATE.categorical_schema:
        s = prepared.get(c)
        cat_cols[c] = (normalize_categorical_series(s)
                       if s is not None else pd.Series(["Missing"] * len(prepared)))
    rows = []
    for i in range(len(prepared)):
        rec = {c: float(num_cols[c].iloc[i]) for c in num_cols}
        rec.update({c: str(cat_cols[c].iloc[i]) for c in cat_cols})
        rows.append(rec)
    return rows


def start_prioritize(candidates=None):
    """Rank candidate synthesis routes on a background thread."""
    if STATE.is_prioritizing or STATE.screener is None or not STATE.targets:
        return
    STATE.is_prioritizing = True
    STATE.prio_error = ""
    STATE.prio_status = "Scoring candidates…"
    target = STATE.targets[min(STATE.prio_target_idx, len(STATE.targets) - 1)]
    src = STATE.prio_source_path
    sheet = (STATE.prio_source_sheet_names[STATE.prio_source_sheet_idx]
             if STATE.prio_source_sheet_names else None)
    threading.Thread(target=_prioritize_worker, args=(candidates, target, src, sheet),
                     daemon=True).start()


def _prioritize_worker(candidates, target, src, sheet=None):
    try:
        if candidates is None:
            if not src:
                raise ValueError("Choose a candidate spreadsheet, or optimize first.")
            candidates = _candidate_rows_from_file(src, sheet=sheet)
        if not candidates:
            raise ValueError("No candidate rows found.")
        weights = {"performance": STATE.prio_w_perf, "confidence": STATE.prio_w_conf,
                   "novelty": STATE.prio_w_novel, "similarity": STATE.prio_w_sim,
                   "feasibility": STATE.prio_w_feas}
        STATE.prio_df = screening.prioritize(STATE.screener, candidates, target, weights)
        STATE.prio_status = (f"Ranked {len(STATE.prio_df)} candidate(s) for "
                             f"{target}.")
    except Exception as e:  # noqa: BLE001
        STATE.prio_error = f"{type(e).__name__}: {e}"
        STATE.prio_status = "Prioritization failed."
    finally:
        STATE.is_prioritizing = False


# =============================================================================
# UI TABS
# =============================================================================
GREEN = (0.20, 0.80, 0.30, 1.0)
RED = (0.90, 0.30, 0.30, 1.0)
DIM = (0.6, 0.6, 0.6, 1.0)


def draw_train_tab():
    imgui.text("1) Choose your spreadsheet")
    if imgui.button("Choose file (.xlsx / .csv)..."):
        STATE.data_dialog = pfd.open_file(
            "Select dataset", filters=["Data", "*.xlsx *.xls *.csv", "All", "*"]
        )
    if STATE.data_dialog is not None and STATE.data_dialog.ready():
        res = STATE.data_dialog.result()
        if res:
            STATE.data_path = res[0]
            STATE.sheet_names = list_sheets(STATE.data_path)  # tabs in this workbook
            STATE.sheet_idx = 0
            _autofill_columns(STATE.data_path)  # auto-populate step 2
        STATE.data_dialog = None
    if STATE.data_path:
        imgui.same_line()
        imgui.text_colored(DIM, STATE.data_path)

    # Worksheet/tab picker — only shown when the workbook has more than one tab.
    if len(STATE.sheet_names) > 1:
        imgui.set_next_item_width(360)
        STATE.sheet_idx = min(STATE.sheet_idx, len(STATE.sheet_names) - 1)
        changed, STATE.sheet_idx = imgui.combo(
            "Sheet / tab", STATE.sheet_idx, STATE.sheet_names)
        if changed:
            _autofill_columns(STATE.data_path)  # re-detect columns for the new tab
            

    imgui.dummy(imgui.ImVec2(0, 6))
    imgui.text("2) Confirm the columns")
    imgui.same_line()
    if imgui.small_button("Auto-detect") and STATE.data_path:
        _autofill_columns(STATE.data_path)
    imgui.set_next_item_width(360)
    _, STATE.mixed_column = imgui.input_text("Messy columns to parse (comma-sep)", STATE.mixed_column)
    imgui.set_next_item_width(360)
    _, STATE.id_columns = imgui.input_text("ID columns to drop (comma-sep)", STATE.id_columns)
    imgui.set_next_item_width(360)
    _, STATE.target_columns = imgui.input_text("Target columns (comma-sep)", STATE.target_columns)

    _, STATE.standardize_units = imgui.checkbox(
        "Auto-standardize units", STATE.standardize_units)
    imgui.same_line()
    imgui.text_colored(DIM, "'2 h' -> 120 min, K -> C, '3.2%V H2SO4' -> mol/L; "
                            "off = use values exactly as in the sheet")

    chemistry_modes = [
        "Off", "Automatic - recommended", "Compact", "Standard", "Full", "Custom"
    ]
    imgui.set_next_item_width(260)
    changed, STATE.chemistry_mode_idx = imgui.combo(
        "Chemistry feature mode", STATE.chemistry_mode_idx, chemistry_modes)
    STATE.chemistry_enabled = STATE.chemistry_mode_idx != 0
    if changed:
        STATE.chemistry_full_risk_confirmed = False
        if STATE.data_path:
            scan_chemical_knowledge(silent=True)
    if STATE.chemistry_mode_idx == 5 and imgui.tree_node("Custom descriptor families"):
        for family in chemistry.DESCRIPTOR_FAMILIES:
            selected = bool(STATE.chemistry_custom_families.get(family, False))
            changed_family, selected = imgui.checkbox(
                family.replace("_", " ").title(), selected)
            if changed_family:
                STATE.chemistry_custom_families[family] = selected
        imgui.tree_pop()
    estimate = STATE.chemistry_estimate
    if estimate:
        ratio = estimate.get("groups_per_encoded_predictor", 0.0)
        risk = estimate.get("risk", "Caution")
        color = GREEN if risk == "Good" else (RED if risk == "High risk" else ORANGE)
        imgui.text_wrapped(
            f"Pre-training estimate: {estimate.get('original_predictors', 0)} original | "
            f"{estimate.get('estimated_numeric_descriptors', 0)} chemistry numeric | "
            f"{estimate.get('estimated_categorical_dummy_columns', 0)} chemistry dummies | "
            f"{estimate.get('estimated_total_encoded_predictors', 0)} encoded total")
        imgui.text_colored(
            color, f"{risk}: {ratio:.2f} independent groups per encoded predictor")
        if STATE.chemistry_mode_idx == 4 and ratio < 1:
            _, STATE.chemistry_full_risk_confirmed = imgui.checkbox(
                "I understand Full chemistry mode is high risk for this dataset",
                STATE.chemistry_full_risk_confirmed)

    imgui.dummy(imgui.ImVec2(0, 6))
    imgui.text("3) Validation settings")
    methods = ["Random KFold", "GroupKFold", "Repeated Grouped CV"]
    imgui.set_next_item_width(260)
    _, STATE.validation_method_idx = imgui.combo(
        "Validation method", STATE.validation_method_idx, methods)
    grouped = STATE.validation_method_idx in (1, 2)
    if grouped:
        columns = list(STATE.coltype_columns)
        if STATE.group_column and STATE.group_column not in columns:
            columns.append(STATE.group_column)
        if columns:
            group_idx = columns.index(STATE.group_column) if STATE.group_column in columns else 0
            imgui.set_next_item_width(260)
            changed, group_idx = imgui.combo("Grouping column", group_idx, columns)
            if changed or not STATE.group_column:
                STATE.group_column = columns[group_idx]
        else:
            imgui.text_colored(RED, "Load a dataset to choose the required grouping column.")
    else:
        imgui.text_colored(DIM, "Grouping column: not required for Random KFold")

    imgui.set_next_item_width(120)
    changed, value = imgui.input_int("Number of folds", int(STATE.n_splits))
    if changed:
        STATE.n_splits = max(2, value)
    if STATE.validation_method_idx == 2:
        imgui.set_next_item_width(120)
        changed, value = imgui.input_int("Number of repeats", int(STATE.n_repeats))
        if changed:
            STATE.n_repeats = max(1, value)
    else:
        imgui.text_colored(DIM, "Number of repeats: only used by Repeated Grouped CV")
    imgui.set_next_item_width(120)
    changed, value = imgui.input_float("Confidence level (%)", float(STATE.confidence_level))
    if changed:
        STATE.confidence_level = min(max(value, 50.0), 99.9)
    imgui.set_next_item_width(120)
    changed, value = imgui.input_int("Random seed", int(STATE.random_state))
    if changed:
        STATE.random_state = value

    changed, STATE.single_target_mode = imgui.checkbox(
        "Single-target publication mode", STATE.single_target_mode)
    if changed and STATE.single_target_mode:
        _sync_validation_column_defaults(STATE.coltype_columns)
    if STATE.single_target_mode:
        candidates = [c for c in STATE.coltype_columns
                      if STATE.coltype_role.get(c) == "target"]
        if not candidates:
            candidates = _split_cols(STATE.target_columns)
        if STATE.single_target and STATE.single_target not in candidates:
            candidates.append(STATE.single_target)
        if candidates:
            target_idx = candidates.index(STATE.single_target) if STATE.single_target in candidates else 0
            imgui.set_next_item_width(260)
            changed, target_idx = imgui.combo("Publication target", target_idx, candidates)
            if changed or not STATE.single_target:
                STATE.single_target = candidates[target_idx]
        else:
            imgui.text_colored(RED, "Choose exactly one target in the Column types tab.")

    imgui.set_next_item_width(120)
    changed, value = imgui.input_int(
        "Show only top N features in importance plots", int(STATE.top_feature_count))
    if changed:
        STATE.top_feature_count = max(1, value)

    imgui.dummy(imgui.ImVec2(0, 6))
    imgui.text("4) Train")
    imgui.text_colored(DIM, "Imputation and one-hot encoding are fitted inside every CV fold.")
    # Guard clicks while a run is in progress.
    imgui.begin_disabled(STATE.is_training)
    if imgui.button("Training…" if STATE.is_training else "Train model",
                    size=imgui.ImVec2(140, 0)):
        start_training()
    imgui.end_disabled()
    imgui.same_line()
    if STATE.trained and imgui.button("Save model", size=imgui.ImVec2(120, 0)):
        save_model()
        STATE.status = f"Saved to '{MODEL_OUT}'."

    # Live progress bar + processing log + status line.
    imgui.dummy(imgui.ImVec2(0, 4))
    draw_progress_panel(STATE.prog_train)
    imgui.text_colored(RED if STATE.train_error else DIM, STATE.status)
    if STATE.train_error:
        imgui.text_wrapped(STATE.train_error)

    # Results.coo
    if STATE.trained:
        imgui.separator()
        imgui.text_colored(GREEN, STATE.summary)

        imgui.dummy(imgui.ImVec2(0, 4))
        imgui.text("Per-target performance (fold uncertainty and pooled OOF diagnostics)")
        flags = (imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
                 | imgui.TableFlags_.scroll_x)
        if imgui.begin_table("metrics", 9, flags, outer_size=imgui.ImVec2(0, 180)):
            interval_label = f"{STATE.validation_config['confidence_level'] * 100:g}% interval"
            for h in ("Target", "CV R2 mean +/- SD", "R2 " + interval_label,
                      "CV RMSE mean +/- SD", "RMSE " + interval_label,
                      "CV MAE mean +/- SD", "MAE " + interval_label,
                      "Train R2", "Train RMSE"):
                imgui.table_setup_column(h)
            imgui.table_headers_row()
            for metric in STATE.metrics:
                name = metric["target"]
                summaries = metric["cv"]
                imgui.table_next_row()
                imgui.table_next_column(); imgui.text(name)
                for key in ("r2", "rmse", "mae"):
                    summary = summaries[key]
                    imgui.table_next_column()
                    imgui.text(f"{summary['mean']:.4f} +/- {summary['std']:.4f}")
                    imgui.table_next_column()
                    imgui.text(f"[{summary['lower']:.4f}, {summary['upper']:.4f}]")
                imgui.table_next_column(); imgui.text(f"{metric['train_r2']:.4f}")
                imgui.table_next_column(); imgui.text(f"{metric['train_rmse']:.4f}")
            imgui.end_table()

        if STATE.metrics:
            m = STATE.metrics[0]
            group_text = (str(m["n_groups"]) if STATE.validation_config["method"] != "random_kfold"
                          else "not applicable (rows are independent)")
            imgui.text_wrapped(
                f"Validation method: {m['validation_method']}  |  Rows: {m['n_rows']}  |  "
                f"Independent groups: {group_text}  |  Folds: {m['n_splits']}  |  "
                f"Repeats: {m['n_repeats']}"
            )
            if STATE.validation_config["method"] != "random_kfold":
                imgui.text_colored(
                    GREEN,
                    "Group integrity check: passed - zero groups overlap between train and validation."
                )
            for metric in STATE.metrics:
                pooled = metric["pooled_oof"]
                imgui.text_wrapped(
                    f"{metric['target']} pooled OOF: R2 {pooled['r2']:.4f}, "
                    f"RMSE {pooled['rmse']:.4f}, MAE {pooled['mae']:.4f}."
                )
            imgui.text_colored(DIM, "Intervals above describe cross-validation uncertainty, "
                               "not prediction intervals for individual recipes.")

        if STATE.feature_diagnostics:
            d = STATE.feature_diagnostics
            imgui.separator()
            imgui.text("Feature-count diagnostics")
            imgui.text_wrapped(
                f"Original selected predictors: {d['original_predictors']}  |  "
                f"Encoded predictors: {d['encoded_predictors']}  |  "
                f"Training rows: {d['training_rows']}  |  Independent groups: {d['independent_groups']}  |  "
                f"Rows per encoded predictor: {d['rows_per_encoded_predictor']:.2f}  |  "
                f"Groups per encoded predictor: {d['groups_per_encoded_predictor']:.2f}"
            )
            if STATE.show_feature_ratio_warnings:
                for warning in d.get("warnings", []):
                    imgui.text_colored(RED, warning)

        imgui.dummy(imgui.ImVec2(0, 4))
        imgui.text("Top 5 most influential features")
        if imgui.begin_table("imp", 2, flags):
            imgui.table_setup_column("Feature")
            imgui.table_setup_column("Importance")
            imgui.table_headers_row()
            for feat, imp in STATE.importances[:5]:
                imgui.table_next_row()
                imgui.table_next_column(); imgui.text(feature_label(feat))
                imgui.table_next_column(); imgui.text(f"{imp * 100:.2f}%")
            imgui.end_table()


def draw_predict_tab():
    if not STATE.trained:
        imgui.text_colored(DIM, "Train (or load) a model first — see the Train tab.")
        return

    imgui.text("Enter feature values")
    imgui.separator()

    if imgui.begin_child("inputs", imgui.ImVec2(0, 300)):
        draw_chemical_inputs("pred")
        hidden_chemistry = _chemistry_model_columns()
        # Parsed messy columns appear as ONE field each ('1M NaOH' style).
        consumed = draw_recipe_inputs("pred")
        for col in STATE.numeric_schema:
            if col in consumed or col in hidden_chemistry:
                continue
            changed, val = imgui.input_float(f"{feature_label(col)}##num_{col}",
                                             float(STATE.numeric_values[col]))
            if changed:
                STATE.numeric_values[col] = val
        for col, choices in STATE.categorical_schema.items():
            if col in consumed or col in hidden_chemistry:
                continue
            changed, idx = imgui.combo(f"{feature_label(col)}##cat_{col}",
                                       STATE.category_index[col], choices)
            if changed:
                STATE.category_index[col] = idx
    imgui.end_child()

    if imgui.button("Predict all targets", size=imgui.ImVec2(180, 0)):
        predict_single()

    if STATE.predict_error:
        imgui.text_colored(RED, STATE.predict_error)
    if STATE.single_pred is not None:
        imgui.separator()
        imgui.text_colored(GREEN, "Predicted targets:")
        flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
        if imgui.begin_table("pred", 2, flags):
            imgui.table_setup_column("Target")
            imgui.table_setup_column("Prediction")
            imgui.table_headers_row()
            for name, val in STATE.single_pred:
                imgui.table_next_row()
                imgui.table_next_column(); imgui.text(name)
                imgui.table_next_column(); imgui.text(f"{val:,.4f}")
            imgui.end_table()


def draw_batch_tab():
    if not STATE.trained:
        imgui.text_colored(DIM, "Train (or load) a model first — see the Train tab.")
        return

    imgui.text("Predict for a whole spreadsheet")
    imgui.text_wrapped(
        "Use a file with the same columns as your training data. Results are saved "
        "to 'predictions.csv' (6 new pred_* columns are appended)."
    )
    imgui.separator()

    if imgui.button("Choose file (.xlsx / .csv)..."):
        STATE.batch_dialog = pfd.open_file(
            "Select file to predict", filters=["Data", "*.xlsx *.xls *.csv", "All", "*"]
        )
    if STATE.batch_dialog is not None and STATE.batch_dialog.ready():
        res = STATE.batch_dialog.result()
        if res:
            STATE.batch_path = res[0]
            STATE.batch_sheet_names = list_sheets(STATE.batch_path)
            STATE.batch_sheet_idx = 0
            STATE.batch_results = None
            STATE.batch_status = ""
            STATE.batch_error = ""
        STATE.batch_dialog = None
    if STATE.batch_path:
        imgui.same_line()
        imgui.text_colored(DIM, STATE.batch_path)

    # Tab picker for multi-sheet batch files.
    batch_sheet = None
    if len(STATE.batch_sheet_names) > 1:
        imgui.set_next_item_width(360)
        STATE.batch_sheet_idx = min(STATE.batch_sheet_idx, len(STATE.batch_sheet_names) - 1)
        _, STATE.batch_sheet_idx = imgui.combo(
            "Sheet / tab", STATE.batch_sheet_idx, STATE.batch_sheet_names)
        batch_sheet = STATE.batch_sheet_names[STATE.batch_sheet_idx]

    if STATE.batch_path and imgui.button("Predict all rows", size=imgui.ImVec2(180, 0)):
        predict_batch(STATE.batch_path, sheet=batch_sheet)

    if STATE.batch_status:
        imgui.text_colored(GREEN, STATE.batch_status)
    if STATE.batch_error:
        imgui.text_colored(RED, STATE.batch_error)

    if STATE.batch_results is not None:
        imgui.separator()
        _draw_dataframe(STATE.batch_results.head(50))


def draw_coltypes_tab():
    imgui.text("Configure columns: role + Numeric/Categorical type")
    imgui.text_wrapped(
        "Scan the spreadsheet chosen in the Train tab, then set each column's Role "
        "(Feature / Target / ID / Messy) and its type. Numeric features are imputed "
        "and fed as numbers; categorical features are one-hot encoded. Roles here "
        "drive training — no need to type column names in the Train tab. Applied "
        "when you Train and when you run Compare models."
    )
    imgui.separator()

    # Load a JSON spec (dataset/target/targets/features.categorical/numerical/exclude).
    if imgui.button("Load JSON config...", size=imgui.ImVec2(170, 0)):
        STATE.json_dialog = pfd.open_file(
            "Select JSON config", filters=["JSON", "*.json", "All", "*"])
    if STATE.json_dialog is not None and STATE.json_dialog.ready():
        res = STATE.json_dialog.result()
        if res:
            try:
                apply_json_config(res[0])
            except Exception as e:  # noqa: BLE001
                STATE.coltype_status = f"JSON load failed: {e}"
        STATE.json_dialog = None
    imgui.same_line()
    imgui.text_colored(DIM, "sets target / features / exclude from a spec file")
    if imgui.button("Save JSON config...", size=imgui.ImVec2(170, 0)):
        STATE.json_save_dialog = pfd.save_file(
            "Save JSON config", "training_settings.json",
            filters=["JSON", "*.json"])
    if STATE.json_save_dialog is not None and STATE.json_save_dialog.ready():
        result = STATE.json_save_dialog.result()
        if result:
            try:
                path = result if str(result).lower().endswith(".json") else str(result) + ".json"
                export_json_config(path)
            except Exception as e:  # noqa: BLE001
                STATE.coltype_status = f"JSON save failed: {e}"
        STATE.json_save_dialog = None

    if not STATE.data_path and not STATE.coltype_columns:
        imgui.text_colored(DIM, "Choose a spreadsheet in the Train tab first, "
                                "or load a JSON config above.")
        return
    if STATE.data_path:
        imgui.text_colored(DIM, STATE.data_path)

    # Worksheet/tab picker — same selection as the Train tab (STATE.sheet_idx).
    if len(STATE.sheet_names) > 1:
        imgui.set_next_item_width(360)
        STATE.sheet_idx = min(STATE.sheet_idx, len(STATE.sheet_names) - 1)
        changed, STATE.sheet_idx = imgui.combo(
            "Sheet / tab", STATE.sheet_idx, STATE.sheet_names)
        if changed:
            scan_column_types()  # re-scan columns for the newly selected tab

    if imgui.button("Scan columns", size=imgui.ImVec2(130, 0)):
        scan_column_types()
    if STATE.coltype_columns:
        imgui.same_line()
        if imgui.button("Train model", size=imgui.ImVec2(130, 0)):
            sync_roles_to_cfg()
            start_training()

    imgui.text_colored(DIM, STATE.coltype_status)

    if not STATE.coltype_columns:
        return

    # ---- Type-to-filter + bulk set: the fast path ----------------------------
    # Filter narrows the list; the bulk buttons then act on just the matches, so
    # you can type e.g. "wt%" and set every matching column numeric in one click.
    imgui.set_next_item_width(260)
    _, STATE.coltype_filter = imgui.input_text("Filter columns", STATE.coltype_filter)
    q = STATE.coltype_filter.strip().lower()
    visible = [c for c in STATE.coltype_columns if q in c.lower()] if q else list(STATE.coltype_columns)

    scope = "filtered" if q else "all"
    imgui.same_line()
    imgui.text_colored(DIM, f"({len(visible)} shown)")
    imgui.text_colored(DIM, f"Set {scope} →")
    imgui.same_line()
    if imgui.small_button("Numeric"):
        for c in visible:
            STATE.coltype_map[c] = "numeric"
    imgui.same_line()
    if imgui.small_button("Categorical"):
        for c in visible:
            STATE.coltype_map[c] = "categorical"
    imgui.same_line()
    imgui.text_colored(DIM, "|  role:")
    for key, label in zip(ROLE_KEYS, ROLE_LABELS):
        imgui.same_line()
        if imgui.small_button(f"{label}##bulk_{key}"):
            for c in visible:
                STATE.coltype_role[c] = key
            sync_roles_to_cfg()

    # Live summary of the training config derived from the roles.
    targets = [c for c in STATE.coltype_columns if STATE.coltype_role.get(c) == "target"]
    excluded = [c for c in STATE.coltype_columns if STATE.coltype_role.get(c) == "exclude"]
    imgui.text_colored(GREEN if targets else RED,
                       "Targets: " + (", ".join(targets) if targets else "none — set at least one!"))
    if excluded:
        imgui.text_colored(DIM, f"Excluded ({len(excluded)}): " + ", ".join(excluded))

    imgui.separator()
    if imgui.begin_child("coltypes", imgui.ImVec2(0, 360)):
        flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
        if imgui.begin_table("ct", 4, flags):
            imgui.table_setup_column("Column")
            imgui.table_setup_column("Role")
            imgui.table_setup_column("Numeric")
            imgui.table_setup_column("Categorical")
            imgui.table_headers_row()
            for col in visible:
                kind = STATE.coltype_map.get(col, "categorical")
                role = STATE.coltype_role.get(col, "feature")
                imgui.table_next_row()
                imgui.table_next_column(); imgui.text(col)
                # Role dropdown (target/id/messy/feature). "##col" keeps IDs unique.
                imgui.table_next_column()
                imgui.set_next_item_width(-1)
                changed, idx = imgui.combo("##role_" + col,
                                           ROLE_KEYS.index(role), ROLE_LABELS)
                if changed:
                    STATE.coltype_role[col] = ROLE_KEYS[idx]
                    sync_roles_to_cfg()
                # Numeric / Categorical radios.
                imgui.table_next_column()
                if imgui.radio_button("##num_" + col, kind == "numeric"):
                    STATE.coltype_map[col] = "numeric"
                imgui.table_next_column()
                if imgui.radio_button("##cat_" + col, kind == "categorical"):
                    STATE.coltype_map[col] = "categorical"
            imgui.end_table()
    imgui.end_child()


def draw_compare_tab():
    imgui.text("Benchmark multiple architectures with the Train-tab validation settings")
    imgui.text_wrapped(
        "Uses the columns configured in the Train tab. Reports each model's average "
        "out-of-fold R2, RMSE, MAE, training R2 and prediction latency across every "
        "configured target. This can take a few minutes on full data."
    )
    imgui.separator()

    if not STATE.data_path:
        imgui.text_colored(DIM, "Choose a spreadsheet in the Train tab first.")
        return
    imgui.text_colored(DIM, STATE.data_path)

    live_cfg = _validation_config_from_state()
    summary = STATE.compare_validation_summary
    label = summary.get("label", V.validation_method_label(live_cfg))
    group_column = summary.get("group_column", live_cfg.get("group_column", ""))
    folds = summary.get("n_splits", live_cfg["n_splits"])
    repeats = summary.get("n_repeats", live_cfg["n_repeats"]
                      if live_cfg["method"] == "repeated_grouped_cv" else 1)
    imgui.text_colored(GREEN, f"Validation: {label}")
    if live_cfg["method"] != "random_kfold":
        imgui.text(f"Grouping column: {group_column or 'not selected'}")
    imgui.text(f"{folds} folds x {repeats} repeats")
    if summary:
        imgui.text(f"Independent groups: {summary['n_groups']}")

    imgui.begin_disabled(STATE.is_comparing)
    if imgui.button("Running…" if STATE.is_comparing else "Run model comparison",
                    size=imgui.ImVec2(220, 0)):
        start_comparison()
    imgui.end_disabled()

    # Live progress bar (fraction of models completed) + processing log.
    draw_progress_panel(STATE.prog_compare)
    imgui.text_colored(RED if STATE.compare_error else DIM, STATE.compare_status)
    if STATE.compare_error:
        imgui.text_wrapped(STATE.compare_error)
    if STATE.is_comparing and STATE.compare_current:
        imgui.text_colored(DIM, f"  … running {STATE.compare_current}")

    # Live ranked table (best R2 first). NaN = the model errored out.
    if STATE.compare_results:
        imgui.separator()
        imgui.text("Ranking (highest average CV R2 first)")
        flags = (imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
                 | imgui.TableFlags_.scroll_x)
        if imgui.begin_table("compare", 11, flags, outer_size=imgui.ImVec2(0, 260)):
            confidence = live_cfg["confidence_level"] * 100
            for h in ("#", "Model", "Mean CV R2", f"R2 {confidence:g}% interval",
                      "Mean CV RMSE", f"RMSE {confidence:g}% interval", "Mean CV MAE",
                      f"MAE {confidence:g}% interval", "Train R2", "Runtime",
                      "Prediction latency"):
                imgui.table_setup_column(h)
            imgui.table_headers_row()
            ranked = sorted(
                STATE.compare_results,
                key=lambda r: r.get("r2", {}).get("mean", -1e9),
                reverse=True,
            )
            for rank, row in enumerate(ranked, start=1):
                name = row["name"]
                r2s, rmses, maes = row.get("r2"), row.get("rmse"), row.get("mae")
                score = r2s["mean"] if r2s else float("nan")
                imgui.table_next_row()
                imgui.table_next_column(); imgui.text(str(rank))
                imgui.table_next_column()
                if rank == 1 and score == score:
                    imgui.text_colored(GREEN, name)  # current best
                else:
                    imgui.text(name)
                imgui.table_next_column()
                imgui.text_colored(
                    GREEN if score == score and score >= 0.5 else
                    (RED if score != score or score < 0 else DIM),
                    "FAILED" if score != score else f"{score:.4f}")
                imgui.table_next_column(); imgui.text(
                    "-" if not r2s else f"[{r2s['lower']:.3f}, {r2s['upper']:.3f}]")
                imgui.table_next_column(); imgui.text("-" if not rmses else f"{rmses['mean']:.3f}")
                imgui.table_next_column(); imgui.text(
                    "-" if not rmses else f"[{rmses['lower']:.3f}, {rmses['upper']:.3f}]")
                imgui.table_next_column(); imgui.text("-" if not maes else f"{maes['mean']:.3f}")
                imgui.table_next_column(); imgui.text(
                    "-" if not maes else f"[{maes['lower']:.3f}, {maes['upper']:.3f}]")
                imgui.table_next_column()
                train_r2 = row.get("train_r2", float("nan"))
                imgui.text("-" if train_r2 != train_r2 else f"{train_r2:.4f}")
                imgui.table_next_column()
                runtime = row.get("runtime", float("nan"))
                imgui.text("-" if runtime != runtime else f"{runtime:.2f} s")
                imgui.table_next_column()
                predict_ms = row.get("predict_ms", float("nan"))
                imgui.text("-" if predict_ms != predict_ms else f"{predict_ms:.2f} ms/row")
            imgui.end_table()
            imgui.text_colored(DIM, "Models are ranked by mean cross-validation R2. "
                               "A large Train-vs-CV R2 gap signals overfitting. "
                               "Predict = average latency per row.")
        if STATE.compare_chart_path:
            imgui.separator()
            imgui.text_colored(GREEN, "Model comparison figure")
            _show_chart_image(STATE.compare_chart_path)


def _draw_dataframe(df):
    """Render a DataFrame as a scrollable ImGui table (preview helper)."""
    flags = (
        imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
        | imgui.TableFlags_.scroll_x | imgui.TableFlags_.scroll_y
    )
    if imgui.begin_table("df", len(df.columns), flags, outer_size=imgui.ImVec2(0, 240)):
        for col in df.columns:
            imgui.table_setup_column(str(col))
        imgui.table_headers_row()
        for _, r in df.iterrows():
            imgui.table_next_row()
            for value in r:
                imgui.table_next_column()
                imgui.text(str(value))
        imgui.end_table()
         


def draw_optimize_tab():
    imgui.text("Recommend the recipe that maximises (or minimises) a target")
    imgui.text_wrapped(
        "Trains on CONTROLLABLE knobs only, then searches for the best recipe. "
        "Uses the file chosen in the Train tab.")
    imgui.separator()

    if not STATE.data_path:
        imgui.text_colored(DIM, "Choose a spreadsheet in the Train tab first.")
        return
    imgui.text_colored(DIM, STATE.data_path)

    # Column-type mode: derive the knobs + held-out columns from the Column
    # types roles, so there's nothing to hand-type.
    role = STATE.coltype_role
    has_roles = bool(STATE.coltype_columns)
    feats = [c for c in STATE.coltype_columns if role.get(c) == "feature"] if has_roles else []
    messy = [c for c in STATE.coltype_columns if role.get(c) == "messy"] if has_roles else []
    tgts = [c for c in STATE.coltype_columns if role.get(c) == "target"] if has_roles else []
    held = [c for c in STATE.coltype_columns
            if role.get(c) in ("target", "id", "exclude")] if has_roles else []

    if has_roles:
        imgui.text_colored(GREEN, f"Using Column-types roles: {len(feats)} feature column(s) "
                           f"+ {len(messy)} parsed messy column(s) as knobs.")
        imgui.text_colored(DIM, f"Automatically held out ({len(held)}): "
                           + ", ".join(held[:12]) + (" …" if len(held) > 12 else ""))
        # Target: pick from the columns tagged 'Target'.
        if tgts:
            if STATE.opt_target not in tgts:
                STATE.opt_target = tgts[0]
            ti = tgts.index(STATE.opt_target)
            imgui.set_next_item_width(360)
            changed, ti = imgui.combo("Target to optimise", ti, tgts)
            if changed:
                STATE.opt_target = tgts[ti]
        else:
            imgui.set_next_item_width(360)
            _, STATE.opt_target = imgui.input_text("Target to optimise", STATE.opt_target)
            imgui.text_colored(ORANGE, "Tip: tag your outcome column's role as 'Target' "
                               "in the Column types tab.")
    else:
        imgui.text_colored(ORANGE, "No Column-types roles set — using the manual exclusion "
                           "list below. Configure the Column types tab for automatic "
                           "knob selection.")
        if imgui.small_button("Scan Column types now"):
            scan_column_types()
        imgui.set_next_item_width(360)
        _, STATE.opt_target = imgui.input_text("Target to optimise", STATE.opt_target)

    imgui.set_next_item_width(360)
    _, STATE.opt_direction_idx = imgui.combo("Direction", STATE.opt_direction_idx,
                                             ["maximise", "minimise"])
    imgui.set_next_item_width(360)
    _, STATE.opt_fixed = imgui.input_text("Fixed knobs (name=value, comma-sep)", STATE.opt_fixed)

    if imgui.collapsing_header("Laboratory constraints"):
        imgui.text_wrapped(
            "One per line: Column <= value, Column >= value, Column == value, "
            "Column != value, Column IN [a, b, c], Column NOT IN [a, b, c]. "
            "Eliminates invalid candidates from the search itself.")
        imgui.text_wrapped(
            "Shortcuts — chemical class: 'NO STRONG ACID' / 'NO STRONG BASE' / "
            "'NO OXIDIZER' / 'NO REDUCING AGENT' / 'NO CHLORIDE' / 'NO FLUORIDE' / "
            "'NO SULFATE' (excludes that class from every chemistry role in use). "
            "Process: 'stages <= N' / 'steps <= N' (caps pyrolysis stages / optional "
            "processing steps in the recommended recipes).")
        _, STATE.opt_constraints = imgui.input_text_multiline(
            "##opt_constraints", STATE.opt_constraints, imgui.ImVec2(-1, 80))
        r = STATE.opt_result
        if r and r.get("constraint_notes"):
            for note in r["constraint_notes"]:
                imgui.text_colored(DIM, "• " + note)

    if imgui.collapsing_header("Multiple objectives (Pareto front)"):
        imgui.text_wrapped(
            "Optional: 'Column:maximize|minimize:weight' entries, comma- or "
            "newline-separated, e.g. 'LIB_1A:maximize:0.6, Py.2 temp. "
            "(oC):minimize:0.4'. Weights auto-normalise. When set, a SEPARATE "
            "search also runs for these objectives together and returns "
            "non-dominated trade-off recipes below — the single best recipe "
            "and Top-N list above always stay driven by the primary target alone.")
        _, STATE.opt_objectives = imgui.input_text_multiline(
            "##opt_objectives", STATE.opt_objectives, imgui.ImVec2(-1, 60))
        r = STATE.opt_result
        if r and r.get("objective_notes"):
            for note in r["objective_notes"]:
                imgui.text_colored((0.9, 0.7, 0.2, 1.0), "• " + note)

    if imgui.collapsing_header("Recommended-experiments settings (Top-N)",
                               imgui.TreeNodeFlags_.default_open):
        imgui.text_wrapped(
            "Beyond the single best recipe below, also ranks a diverse set of "
            "in-domain alternatives — each scored for predicted value, "
            "uncertainty, and similarity to real training experiments.")
        imgui.set_next_item_width(200)
        changed, v = imgui.input_int("How many to recommend", int(STATE.opt_top_n))
        if changed:
            STATE.opt_top_n = max(1, v)
        imgui.set_next_item_width(200)
        changed, v = imgui.slider_float(
            "Risk penalty (uncertainty + extrapolation)", float(STATE.opt_risk_lambda), 0.0, 3.0)
        if changed:
            STATE.opt_risk_lambda = v
        imgui.set_next_item_width(200)
        changed, v = imgui.slider_float(
            "Minimum applicability %", float(STATE.opt_min_applicability), 0.0, 100.0)
        if changed:
            STATE.opt_min_applicability = v
        imgui.text_colored(
            DIM, "Applicability compares a candidate to real training experiments; "
                "with many independent knobs, absolute values run low even for "
                "reasonable recipes — the ranking (not the raw %) is what matters.")

    # The manual exclusion list is only used as a fallback (no roles configured).
    if not has_roles:
        imgui.text_colored(DIM, "Excluded (measured / outcome) columns — one big list:")
        _, STATE.opt_excluded = imgui.input_text_multiline(
            "##excluded", STATE.opt_excluded, imgui.ImVec2(-1, 80))
    elif imgui.tree_node("Advanced: manual exclusions (ignored in column-type mode)"):
        _, STATE.opt_excluded = imgui.input_text_multiline(
            "##excluded", STATE.opt_excluded, imgui.ImVec2(-1, 80))
        imgui.tree_pop()

    imgui.begin_disabled(STATE.is_optimizing)
    if imgui.button("Running…" if STATE.is_optimizing else "Run optimization",
                    size=imgui.ImVec2(200, 0)):
        start_optimize()
    imgui.end_disabled()
    imgui.text_colored(RED if STATE.opt_error else DIM, STATE.opt_status)
    draw_progress_panel(STATE.prog_opt)
    if STATE.opt_error:
        imgui.text_wrapped(STATE.opt_error)

    r = STATE.opt_result
    if r is not None:
        labels = r.get("labels", {})
        def knob_label(name):
            return labels.get(name, name)

        imgui.separator()
        verdict = GREEN if r["predicted"] <= r["obs_max"] * 1.05 else (0.9, 0.7, 0.2, 1.0)
        imgui.text_colored(verdict,
                           f"Predicted {r['target']} = {r['predicted']:.0f}   "
                           f"(observed {r['obs_min']:.0f}–{r['obs_max']:.0f})")
        imgui.text_colored(DIM, f"Model 5-fold R² = {r['r2']:.2f} — treat as a hypothesis to test.")
        if r.get("optimized_over"):
            imgui.text_colored(DIM, "Search basis: " + r["optimized_over"] + ".")
        if r["edges"]:
            imgui.text_colored((0.9, 0.7, 0.2, 1.0),
                               "Extrapolation risk (knob at edge of data): "
                               + ", ".join(knob_label(e) for e in r["edges"]))

        imgui.dummy(imgui.ImVec2(0, 4))
        imgui.text("Recommended conditions")
        flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
        if imgui.begin_table("recipe", 2, flags, outer_size=imgui.ImVec2(0, 320)):
            imgui.table_setup_column("Knob")
            imgui.table_setup_column("Value")
            imgui.table_headers_row()
            for name, val in r["recipe"]:
                shown = knob_label(name)
                imgui.table_next_row()
                imgui.table_next_column()
                if name in r["fixed"]:
                    imgui.text_colored(DIM, shown + "  (fixed)")
                else:
                    imgui.text(shown)
                imgui.table_next_column()
                imgui.text(f"{val:g}" if isinstance(val, float) else str(val))
            imgui.end_table()

        if r.get("chemical_recommendations"):
            imgui.dummy(imgui.ImVec2(0, 4))
            imgui.text("Chemically feasible reagent mapping")
            imgui.text_colored(
                DIM,
                "The search operates on descriptors; these are the closest known reagents."
            )
            flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
            if imgui.begin_table("chemical_mapping", 4, flags):
                imgui.table_setup_column("Input")
                imgui.table_setup_column("Recommended")
                imgui.table_setup_column("Profile match")
                imgui.table_setup_column("Alternatives")
                imgui.table_headers_row()
                for item in r["chemical_recommendations"]:
                    alternatives = ", ".join(
                        f"{alt['name']} ({alt['score']:.2f})"
                        for alt in item.get("alternatives", [])
                    ) or "None"
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text(str(item["column"]))
                    imgui.table_next_column()
                    imgui.text(str(item["recommended"]))
                    imgui.table_next_column()
                    imgui.text(f"{item['similarity']:.2f}")
                    imgui.table_next_column()
                    imgui.text_wrapped(alternatives)
                imgui.end_table()

        top_recipes = r.get("top_recipes") or []
        if top_recipes:
            imgui.dummy(imgui.ImVec2(0, 8))
            imgui.separator()
            imgui.text(f"Recommended experiments (top {len(top_recipes)} of "
                      f"{r.get('n_candidates_considered', 0)} candidates considered)")
            imgui.text_colored(
                DIM, "Diverse, in-domain alternatives to the single best recipe above — "
                    "hover a row for the full reasoning. Risk/Applicability come from "
                    "similarity to real training experiments, not just the raw prediction.")
            flags = (imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
                     | imgui.TableFlags_.scroll_y)
            if imgui.begin_table("top_recipes", 9, flags, outer_size=imgui.ImVec2(0, 240)):
                imgui.table_setup_column("#")
                imgui.table_setup_column("Predicted")
                imgui.table_setup_column("95% interval")
                imgui.table_setup_column("Applicability")
                imgui.table_setup_column("Risk")
                imgui.table_setup_column("Similarity")
                imgui.table_setup_column("Strategy")
                imgui.table_setup_column("Green")
                imgui.table_setup_column("Reason")
                imgui.table_headers_row()
                risk_color = {"Low": GREEN, "Moderate": (0.9, 0.7, 0.2, 1.0), "High": RED}
                green_color = {"A": GREEN, "B": GREEN, "C": (0.9, 0.7, 0.2, 1.0),
                              "D": (0.9, 0.7, 0.2, 1.0), "F": RED}
                for cand in top_recipes:
                    imgui.table_next_row()
                    imgui.table_next_column(); imgui.text(str(cand["rank"]))
                    imgui.table_next_column(); imgui.text(f"{cand['predicted']:.1f}")
                    imgui.table_next_column()
                    imgui.text(f"{cand['lo']:.0f} – {cand['hi']:.0f}")
                    imgui.table_next_column()
                    imgui.text(f"{cand['applicability_pct']:.0f}%")
                    imgui.table_next_column()
                    imgui.text_colored(risk_color.get(cand["risk"], DIM), cand["risk"])
                    imgui.table_next_column()
                    sim = cand.get("similarity_pct")
                    imgui.text(f"{sim:.0f}%" if sim is not None else "-")
                    imgui.table_next_column()
                    imgui.text(cand.get("strategy", "-"))
                    if imgui.is_item_hovered() and cand.get("strategy_reason"):
                        imgui.set_tooltip(cand["strategy_reason"])
                    imgui.table_next_column()
                    sus = cand.get("sustainability")
                    if sus:
                        imgui.text_colored(green_color.get(sus["grade"], DIM),
                                          f"{sus['grade']} ({sus['score']:.0f})")
                        if imgui.is_item_hovered() and sus.get("deductions"):
                            imgui.set_tooltip("\n".join(
                                f"-{d['points']:.0f}: {d['reason']}" for d in sus["deductions"]))
                    else:
                        imgui.text("-")
                    imgui.table_next_column()
                    reason_text = " ".join(cand.get("reason", []))
                    imgui.text_wrapped(reason_text[:70] + ("…" if len(reason_text) > 70 else ""))
                    if imgui.is_item_hovered() and cand.get("reason"):
                        imgui.set_tooltip("\n".join(cand["reason"]))
                imgui.end_table()
            if imgui.tree_node("Recommended experiments: full recipes##top_recipe_details"):
                for cand in top_recipes:
                    label = (f"#{cand['rank']}: {cand.get('strategy', '')} — "
                            f"predicted {cand['predicted']:.1f}, "
                            f"{cand['risk']} risk, {cand['applicability_pct']:.0f}% applicability"
                            f"##top_recipe_{cand['rank']}")
                    if imgui.tree_node(label):
                        imgui.text_colored(DIM, cand.get("strategy_reason", ""))
                        imgui.text(f"Feasibility: {cand.get('feasibility', 'n/a')}  |  "
                                  f"Active-learning info gain: {cand.get('info_gain', 0):.0f}/100")
                        for line in cand.get("reason", []):
                            imgui.bullet_text(line)
                        reagents = cand.get("reagents") or []
                        if reagents:
                            imgui.text("Reagent cost / hazard (from your entered data):")
                            for rg in reagents:
                                cost = (f"${rg['cost_per_kg']:g}/kg" if rg["cost_per_kg"] is not None
                                       else (f"${rg['cost_per_liter']:g}/L"
                                             if rg["cost_per_liter"] is not None else "cost: not entered"))
                                haz = rg["hazard_class"]
                                corrosive = " · corrosive" if rg["corrosive"] else ""
                                imgui.bullet_text(f"{rg['reagent']}: {cost} · hazard: {haz}{corrosive}")
                        sus = cand.get("sustainability")
                        if sus:
                            imgui.text(f"Green Score: {sus['score']:.0f}/100 (grade {sus['grade']})")
                            for d in sus.get("deductions", []):
                                imgui.bullet_text(f"-{d['points']:.0f}: {d['reason']}")
                            if sus.get("unscored_reagents"):
                                imgui.text_colored(
                                    DIM, "No hazard data entered for: "
                                        + ", ".join(sus["unscored_reagents"]) + " (not scored).")
                        flags2 = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
                        if imgui.begin_table(f"top_recipe_detail_{cand['rank']}", 2, flags2):
                            imgui.table_setup_column("Knob")
                            imgui.table_setup_column("Value")
                            imgui.table_headers_row()
                            for name, val in cand["recipe"]:
                                shown = knob_label(name)
                                imgui.table_next_row()
                                imgui.table_next_column()
                                if name in cand["fixed"]:
                                    imgui.text_colored(DIM, shown + "  (fixed)")
                                else:
                                    imgui.text(shown)
                                imgui.table_next_column()
                                imgui.text(f"{val:g}" if isinstance(val, float) else str(val))
                            imgui.end_table()
                        imgui.tree_pop()
                imgui.tree_pop()

        batch_plan = r.get("batch_plan") or []
        if batch_plan:
            imgui.dummy(imgui.ImVec2(0, 8))
            imgui.separator()
            imgui.text(f"Balanced batch plan — {len(batch_plan)} experiment(s)")
            imgui.text_colored(
                DIM, "A portfolio to run together, not just the single best repeated — "
                    "one bucket per experiment strategy (safe, validation, novel chemistry, "
                    "gap-filling, high-risk-high-reward, ...).")
            summary = r.get("batch_plan_summary") or {}
            if summary:
                composition = ", ".join(f"{n} {label}" for label, n in summary.items())
                imgui.text_colored(DIM, "Composition: " + composition)
            flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
            if imgui.begin_table("batch_plan", 5, flags, outer_size=imgui.ImVec2(0, 160)):
                imgui.table_setup_column("#")
                imgui.table_setup_column("Predicted")
                imgui.table_setup_column("Strategy")
                imgui.table_setup_column("Feasibility")
                imgui.table_setup_column("Info gain")
                imgui.table_headers_row()
                for cand in batch_plan:
                    imgui.table_next_row()
                    imgui.table_next_column(); imgui.text(str(cand["batch_rank"]))
                    imgui.table_next_column(); imgui.text(f"{cand.get('predicted', float('nan')):.1f}")
                    imgui.table_next_column(); imgui.text(cand.get("strategy", "-"))
                    imgui.table_next_column(); imgui.text(cand.get("feasibility", "-"))
                    imgui.table_next_column(); imgui.text(f"{cand.get('info_gain', 0):.0f}")
                imgui.end_table()

        research_gaps = r.get("research_gaps") or []
        if research_gaps:
            imgui.dummy(imgui.ImVec2(0, 8))
            imgui.separator()
            if imgui.collapsing_header(f"Research gaps — {len(research_gaps)} under-studied "
                                       "area(s) in your training data"):
                imgui.text_colored(
                    DIM, "Regions of the dataset with little or no coverage — experiments "
                        "here would expand what the model has actually seen, beyond just "
                        "chasing the highest prediction.")
                for g in research_gaps:
                    imgui.bullet_text(g["description"])

        pareto_front = r.get("pareto_front") or []
        objectives_used = r.get("objectives_used") or []
        fronts_summary = r.get("pareto_fronts_summary") or []
        if pareto_front:
            imgui.dummy(imgui.ImVec2(0, 8))
            imgui.separator()
            obj_summary = ", ".join(
                f"{knob_label(o['column'])} ({o['direction']}, weight {o['weight']:.2f})"
                for o in objectives_used)
            imgui.text(f"Pareto front — {len(pareto_front)} non-dominated trade-off(s)")
            imgui.text_colored(DIM, "Objectives: " + obj_summary)
            imgui.text_colored(
                DIM, "No single recipe wins on every objective here — each row below "
                    "beats every other on at least one, and none beats it on all.")
            if len(fronts_summary) > 1:
                imgui.text_colored(
                    DIM, f"{fronts_summary[0]} candidate(s) on the optimal front "
                        f"(shown here, diversity-filtered to {len(pareto_front)}); "
                        + ", ".join(f"{n} on rank-{i+2}" for i, n in enumerate(fronts_summary[1:5]))
                        + " — close-but-not-quite trade-offs, for context.")
            obj_cols = [o["column"] for o in objectives_used]
            flags = (imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
                     | imgui.TableFlags_.scroll_y)
            n_cols = 4 + len(obj_cols)
            if imgui.begin_table("pareto", n_cols, flags, outer_size=imgui.ImVec2(0, 200)):
                imgui.table_setup_column("#")
                for col in obj_cols:
                    imgui.table_setup_column(knob_label(col))
                imgui.table_setup_column("Applicability")
                imgui.table_setup_column("Risk")
                imgui.table_setup_column("Green")
                imgui.table_headers_row()
                risk_color = {"Low": GREEN, "Moderate": (0.9, 0.7, 0.2, 1.0), "High": RED}
                green_color = {"A": GREEN, "B": GREEN, "C": (0.9, 0.7, 0.2, 1.0),
                              "D": (0.9, 0.7, 0.2, 1.0), "F": RED}
                for pt in pareto_front:
                    imgui.table_next_row()
                    imgui.table_next_column(); imgui.text(str(pt["rank"]))
                    for col in obj_cols:
                        imgui.table_next_column()
                        imgui.text(f"{pt['objectives'].get(col, float('nan')):.3g}")
                    imgui.table_next_column()
                    imgui.text(f"{pt['applicability_pct']:.0f}%")
                    imgui.table_next_column()
                    imgui.text_colored(risk_color.get(pt["risk"], DIM), pt["risk"])
                    imgui.table_next_column()
                    sus = pt.get("sustainability")
                    if sus:
                        imgui.text_colored(green_color.get(sus["grade"], DIM),
                                          f"{sus['grade']} ({sus['score']:.0f})")
                    else:
                        imgui.text("-")
                imgui.end_table()
            if imgui.tree_node("Pareto front: trade-offs and full recipes##pareto_details"):
                for pt in pareto_front:
                    obj_bits = ", ".join(f"{knob_label(c)}={v:.3g}"
                                         for c, v in pt["objectives"].items())
                    label = f"#{pt['rank']}: {obj_bits}##pareto_recipe_{pt['rank']}"
                    if imgui.tree_node(label):
                        if pt.get("advantages"):
                            imgui.text_colored(GREEN, "Advantages:")
                            for a in pt["advantages"]:
                                imgui.bullet_text(a)
                        if pt.get("disadvantages"):
                            imgui.text_colored((0.9, 0.7, 0.2, 1.0), "Disadvantages:")
                            for d in pt["disadvantages"]:
                                imgui.bullet_text(d)
                        if pt.get("tradeoff"):
                            imgui.text_wrapped("Trade-off: " + pt["tradeoff"])
                        if pt.get("crowding") is not None:
                            imgui.text_colored(
                                DIM, f"Uniqueness within this front: {pt['crowding']:.2f} "
                                    "(higher = more distinct trade-off, less redundant with "
                                    "its neighbours)")
                        sus = pt.get("sustainability")
                        if sus:
                            imgui.text(f"Green Score: {sus['score']:.0f}/100 (grade {sus['grade']})")
                            for d in sus.get("deductions", []):
                                imgui.bullet_text(f"-{d['points']:.0f}: {d['reason']}")
                        flags2 = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
                        if imgui.begin_table(f"pareto_detail_{pt['rank']}", 2, flags2):
                            imgui.table_setup_column("Knob")
                            imgui.table_setup_column("Value")
                            imgui.table_headers_row()
                            for name, val in pt["recipe"]:
                                shown = knob_label(name)
                                imgui.table_next_row()
                                imgui.table_next_column()
                                if name in pt["fixed"]:
                                    imgui.text_colored(DIM, shown + "  (fixed)")
                                else:
                                    imgui.text(shown)
                                imgui.table_next_column()
                                imgui.text(f"{val:g}" if isinstance(val, float) else str(val))
                            imgui.end_table()
                        imgui.tree_pop()
                imgui.tree_pop()

            if len(obj_cols) >= 2:
                imgui.dummy(imgui.ImVec2(0, 6))
                if imgui.tree_node("Pareto trade-off chart##pareto_chart"):
                    obj_labels = [knob_label(c) for c in obj_cols]
                    imgui.set_next_item_width(220)
                    changed, ia = imgui.combo(
                        "X axis", min(STATE.opt_pareto_chart_a, len(obj_cols) - 1), obj_labels)
                    if changed:
                        STATE.opt_pareto_chart_a = ia
                    imgui.set_next_item_width(220)
                    changed, ib = imgui.combo(
                        "Y axis", min(STATE.opt_pareto_chart_b, len(obj_cols) - 1), obj_labels)
                    if changed:
                        STATE.opt_pareto_chart_b = ib
                    if imgui.button("Generate chart"):
                        _generate_pareto_chart(pareto_front, objectives_used,
                                              STATE.opt_pareto_chart_a, STATE.opt_pareto_chart_b)
                    if STATE.opt_pareto_chart_error:
                        imgui.text_colored(RED, STATE.opt_pareto_chart_error)
                    if STATE.opt_pareto_chart_path:
                        _show_chart_image(STATE.opt_pareto_chart_path)
                    imgui.tree_pop()
        elif len(objectives_used) > 1:
            imgui.dummy(imgui.ImVec2(0, 8))
            imgui.text_colored(
                (0.9, 0.7, 0.2, 1.0),
                "No Pareto-optimal candidates survived the applicability gate — "
                "try lowering 'Minimum applicability %' above.")

        sensitivity = r.get("sensitivity") or []
        if sensitivity:
            imgui.dummy(imgui.ImVec2(0, 8))
            imgui.separator()
            imgui.text("Robustness — sensitivity of the best recipe to small changes")
            imgui.text_colored(
                DIM, "Each knob is swept a little either side of its recommended value, "
                    "holding everything else fixed, to show whether the prediction is "
                    "stable or needs unrealistic precision.")
            for s in sensitivity:
                tag_color = GREEN if s["robust"] else (0.9, 0.7, 0.2, 1.0)
                tag = "robust" if s["robust"] else "sensitive"
                imgui.text_colored(
                    tag_color,
                    f"{knob_label(s['knob'])}: {tag} — predicted {r['target']} varies "
                    f"{s['pct_range']:.1f}% across the sweep")
                sweep_text = "  ".join(f"{v:g}→{p:.0f}" for v, p in s["sweep"])
                imgui.text_colored(DIM, "    " + sweep_text)


def _generate_pareto_chart(pareto_front, objectives_used, idx_a, idx_b):
    """Render a 2D Pareto scatter for any pair of objectives from an
    already-computed Pareto front, reusing charts.py's existing
    pareto_front() chart maker (the same one the multi-target Charts tab
    uses) rather than duplicating plotting code."""
    STATE.opt_pareto_chart_error = ""
    STATE.opt_pareto_chart_path = None
    try:
        import charts as C
        obj_cols = [o["column"] for o in objectives_used]
        if idx_a >= len(obj_cols) or idx_b >= len(obj_cols):
            raise ValueError("axis selection out of range")
        col_a, col_b = obj_cols[idx_a], obj_cols[idx_b]
        dir_a = objectives_used[idx_a]["direction"]
        dir_b = objectives_used[idx_b]["direction"]
        df = pd.DataFrame([pt["objectives"] for pt in pareto_front])
        os.makedirs(CHART_DIR, exist_ok=True)
        STATE.charts_run += 1
        out = f"{CHART_DIR}/pareto_optimizer_{STATE.charts_run}.png"
        C.pareto_front(df, col_a, col_b, out, maximize_a=(dir_a == "maximise"),
                       maximize_b=(dir_b == "maximise"), title=f"{col_a} vs {col_b}")
        STATE.opt_pareto_chart_path = out
    except Exception as e:  # noqa: BLE001
        STATE.opt_pareto_chart_error = f"Chart generation failed: {e}"


def _show_chart_image(rel_path):
    """Display a generated PNG in-window, scaled to the panel; fall back to a
    button that opens it in the OS viewer if texture loading isn't available."""
    abs_path = os.path.abspath(rel_path)
    try:
        native = hello_imgui.image_size_from_asset(rel_path)  # assets folder = project dir
        avail = imgui.get_content_region_avail().x
        if native.x > 0 and avail > 0:
            scale = min(avail / native.x, 1.0)
            hello_imgui.image_from_asset(
                rel_path, imgui.ImVec2(native.x * scale, native.y * scale))
        else:
            raise RuntimeError("zero-size image")
    except Exception:  # noqa: BLE001 - texture path unavailable; offer to open the file
        imgui.text_colored(DIM, rel_path)
    root, _ = os.path.splitext(rel_path)
    txt_path = root + ".txt"
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                note = f.read().strip()
            if note:
                imgui.text_wrapped(note)
        except Exception:  # noqa: BLE001
            pass
    imgui.text_colored(DIM, "Exports:")
    for ext in (".png", ".svg", ".pdf"):
        p = root + ext
        if os.path.exists(p):
            imgui.same_line()
            if imgui.small_button(f"Open {ext[1:].upper()}##{p}"):
                try:
                    os.startfile(os.path.abspath(p))
                except Exception:  # noqa: BLE001
                    pass
    imgui.same_line()
    if imgui.small_button(f"Copy path##{rel_path}"):
        try:
            imgui.set_clipboard_text(abs_path)
        except Exception:  # noqa: BLE001
            pass
    if imgui.small_button(f"Open in viewer##{rel_path}"):
        try:
            os.startfile(abs_path)  # Windows
        except Exception:  # noqa: BLE001
            pass


def draw_bayesopt_tab():
    imgui.text("Suggest Experiments — automated Bayesian optimization")
    imgui.text_colored(
        DIM,
        "Fits a Gaussian-process surrogate that models both the predicted "
        "outcome and its own uncertainty, then proposes the next experiments to "
        "run by maximizing Expected Improvement — the experiments most likely to "
        "beat your current best. Ideal for planning the next round in the lab.")
    imgui.separator()

    if not STATE.data_path:
        imgui.text_colored(DIM, "Choose a spreadsheet in the Train tab first.")
        return
    imgui.text_colored(DIM, STATE.data_path)

    role = STATE.coltype_role
    has_roles = bool(STATE.coltype_columns)
    tgts = [c for c in STATE.coltype_columns
            if role.get(c) == "target"] if has_roles else []
    feats = [c for c in STATE.coltype_columns
             if role.get(c) == "feature"] if has_roles else []

    if has_roles:
        imgui.text_colored(GREEN, f"Using Column-types roles: {len(feats)} knob(s). "
                           "Constraints & fixed knobs are shared with the Optimize tab.")
        if tgts:
            if STATE.opt_target not in tgts:
                STATE.opt_target = tgts[0]
            ti = tgts.index(STATE.opt_target)
            imgui.set_next_item_width(360)
            changed, ti = imgui.combo("Target to optimise", ti, tgts)
            if changed:
                STATE.opt_target = tgts[ti]
        else:
            imgui.set_next_item_width(360)
            _, STATE.opt_target = imgui.input_text("Target to optimise", STATE.opt_target)
    else:
        imgui.text_colored(ORANGE, "No Column-types roles set — using the Optimize "
                           "tab's manual exclusion list and target.")
        imgui.set_next_item_width(360)
        _, STATE.opt_target = imgui.input_text("Target to optimise", STATE.opt_target)

    imgui.set_next_item_width(360)
    _, STATE.opt_direction_idx = imgui.combo("Direction", STATE.opt_direction_idx,
                                             ["maximise", "minimise"])
    imgui.set_next_item_width(200)
    changed, v = imgui.input_int("Experiments to propose", int(STATE.bo_batch))
    if changed:
        STATE.bo_batch = max(1, min(v, 20))
    imgui.set_next_item_width(200)
    changed, v = imgui.slider_float("Exploration (EI margin)", float(STATE.bo_xi),
                                    0.0, 0.5)
    if changed:
        STATE.bo_xi = v
    imgui.same_line()
    imgui.text_colored(DIM, "higher = bolder, more exploratory picks")

    imgui.begin_disabled(STATE.is_bayesopt)
    if imgui.button("Suggesting…" if STATE.is_bayesopt else "Suggest experiments",
                    size=imgui.ImVec2(220, 0)):
        start_bayesopt()
    imgui.end_disabled()
    imgui.same_line()
    imgui.text_colored(DIM, "Shares target / constraints / fixed knobs with the "
                       "Optimize tab.")
    imgui.text_colored(RED if STATE.bo_error else DIM, STATE.bo_status)
    draw_progress_panel(STATE.prog_bo)
    if STATE.bo_error:
        imgui.text_wrapped(STATE.bo_error)

    r = STATE.bo_result
    if r is None:
        return
    labels = r.get("labels", {})

    def knob_label(name):
        return labels.get(name, name)

    imgui.separator()
    unit_dir = "maximise" if r["direction"] == "maximise" else "minimise"
    imgui.text_colored(GREEN,
                       f"{len(r['proposals'])} experiment(s) proposed to {unit_dir} "
                       f"{r['target']}.")
    imgui.text_colored(DIM, f"Best observed so far: {r['y_best']:.0f}   "
                       f"(range {r['obs_min']:.0f}–{r['obs_max']:.0f}). "
                       f"Surrogate cross-check R² = {r['r2']:.2f}.")
    for note in r.get("notes", []):
        imgui.text_colored(ORANGE, "• " + note)

    for p in r["proposals"]:
        header = (f"#{p['rank']}   predicted {p['predicted']:.0f} "
                  f"± {p['sigma']:.0f}    Expected Improvement {p['ei']:.2f}"
                  f"##bo{p['rank']}")
        flags = (imgui.TreeNodeFlags_.default_open if p["rank"] == 1 else 0)
        if imgui.collapsing_header(header, flags):
            imgui.text_colored(
                DIM, f"95% predictive interval: {p['lo']:.0f} – {p['hi']:.0f} "
                     f"{r['target']}.")
            if p["edges"]:
                imgui.text_colored(ORANGE, "Extrapolation risk (knob at edge of "
                                   "data): " + ", ".join(knob_label(e)
                                                         for e in p["edges"]))
            tflags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
            if imgui.begin_table(f"bo_recipe_{p['rank']}", 2, tflags,
                                 outer_size=imgui.ImVec2(0, 300)):
                imgui.table_setup_column("Knob")
                imgui.table_setup_column("Value")
                imgui.table_headers_row()
                for name, val in p["recipe"]:
                    imgui.table_next_row()
                    imgui.table_next_column()
                    if name in p["fixed"]:
                        imgui.text_colored(DIM, knob_label(name) + "  (fixed)")
                    else:
                        imgui.text(knob_label(name))
                    imgui.table_next_column()
                    imgui.text(f"{val:g}" if isinstance(val, float) else str(val))
                imgui.end_table()
            if p.get("chemical_recommendations"):
                imgui.text_colored(DIM, "Closest known reagents (search runs on "
                                   "descriptors):")
                cflags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
                if imgui.begin_table(f"bo_chem_{p['rank']}", 3, cflags):
                    imgui.table_setup_column("Input")
                    imgui.table_setup_column("Recommended")
                    imgui.table_setup_column("Profile match")
                    imgui.table_headers_row()
                    for item in p["chemical_recommendations"]:
                        imgui.table_next_row()
                        imgui.table_next_column()
                        imgui.text(str(item["column"]))
                        imgui.table_next_column()
                        imgui.text(str(item["recommended"]))
                        imgui.table_next_column()
                        imgui.text(f"{item['similarity']:.2f}")
                    imgui.end_table()


def draw_charts_tab():
    imgui.text("Publication-quality diagnostic charts for the trained model")
    imgui.text_wrapped(
        "Generates a research chart bundle: Pearson/Spearman correlations, mutual "
        "information, target distribution, model performance, actual-vs-predicted, "
        "residual diagnostics, grouped feature importance, SHAP, optimization "
        "surface and Pareto front. Every figure is exported as PNG, SVG and PDF.")
    imgui.separator()

    if not STATE.trained:
        imgui.text_colored(DIM, "Train (or load) a model first — see the Train tab.")
        return

    targets = list(STATE.targets)
    numeric_feats = list(STATE.numeric_schema.keys())

    # Selectors (clamped so a shrinking list can't crash the combo).
    def clamp(i, seq):
        return min(max(i, 0), len(seq) - 1) if seq else 0

    if targets:
        imgui.set_next_item_width(260)
        STATE.charts_target_idx = clamp(STATE.charts_target_idx, targets)
        _, STATE.charts_target_idx = imgui.combo(
            "SHAP / heatmap target", STATE.charts_target_idx, targets)
    if len(targets) >= 2:
        imgui.set_next_item_width(200)
        STATE.charts_pareto_a = clamp(STATE.charts_pareto_a, targets)
        _, STATE.charts_pareto_a = imgui.combo("Pareto X", STATE.charts_pareto_a, targets)
        imgui.same_line()
        imgui.set_next_item_width(200)
        STATE.charts_pareto_b = clamp(STATE.charts_pareto_b, targets)
        _, STATE.charts_pareto_b = imgui.combo("Pareto Y", STATE.charts_pareto_b, targets)
    if len(numeric_feats) >= 2:
        imgui.set_next_item_width(200)
        STATE.charts_featx_idx = clamp(STATE.charts_featx_idx, numeric_feats)
        _, STATE.charts_featx_idx = imgui.combo("Heatmap X", STATE.charts_featx_idx, numeric_feats)
        imgui.same_line()
        imgui.set_next_item_width(200)
        STATE.charts_featy_idx = clamp(STATE.charts_featy_idx, numeric_feats)
        _, STATE.charts_featy_idx = imgui.combo("Heatmap Y", STATE.charts_featy_idx, numeric_feats)
    imgui.set_next_item_width(180)
    changed, value = imgui.input_int(
        "Top features in importance plots", int(STATE.top_feature_count))
    if changed:
        STATE.top_feature_count = max(1, value)

    # Poll the async slideshow save dialog.
    if STATE.slideshow_dialog is not None and STATE.slideshow_dialog.ready():
        r = STATE.slideshow_dialog.result()
        if r:
            export_slideshow(r)
        STATE.slideshow_dialog = None
    # Poll the async "Save charts…" folder picker.
    if STATE.save_charts_dialog is not None and STATE.save_charts_dialog.ready():
        folder = STATE.save_charts_dialog.result()
        if folder:
            save_charts_to_folder(folder)
        STATE.save_charts_dialog = None

    imgui.dummy(imgui.ImVec2(0, 4))
    imgui.begin_disabled(STATE.is_charting)
    if imgui.button("Generating…" if STATE.is_charting else "Generate charts",
                    size=imgui.ImVec2(200, 0)):
        start_charts()
    imgui.end_disabled()
    have_charts = bool(STATE.chart_items or STATE.lat_chart_items or STATE.intel_chart_items)
    if have_charts:
        imgui.same_line()
        if imgui.button("Save charts…", size=imgui.ImVec2(140, 0)):
            STATE.save_charts_status = ""
            STATE.save_charts_dialog = pfd.select_folder(
                "Choose a folder to save the charts")
        imgui.same_line()
        if imgui.button("Create slideshow summary", size=imgui.ImVec2(220, 0)):
            STATE.slideshow_dialog = pfd.save_file(
                "Save slideshow summary", "BioCarbon_summary_slideshow.pdf",
                filters=["PDF", "*.pdf"])
        imgui.same_line()
        imgui.text_colored(DIM, "Charts are only kept on disk when you save them.")
    draw_progress_panel(STATE.prog_charts)
    imgui.text_colored(RED if STATE.charts_error else DIM, STATE.charts_status)
    if STATE.charts_error:
        imgui.text_wrapped(STATE.charts_error)
    if STATE.slideshow_status:
        imgui.text_colored(GREEN if "Saved" in STATE.slideshow_status else RED,
                           STATE.slideshow_status)
    if STATE.save_charts_status:
        imgui.text_colored(GREEN if "Saved" in STATE.save_charts_status else RED,
                           STATE.save_charts_status)

    if STATE.chart_items:
        imgui.separator()
        if imgui.begin_child("chart_scroll", imgui.ImVec2(0, 0)):
            for title, rel in STATE.chart_items:
                imgui.text_colored(GREEN, title)
                _show_chart_image(rel)
                imgui.dummy(imgui.ImVec2(0, 8))
        imgui.end_child()


def draw_latent_tab():
    imgui.text("Latent Variables — engineered indices + learned PCA/PLS components")
    imgui.text_wrapped(
        "Builds interpretable engineered indices and learned latent variables, then "
        "compares three leakage-safe pipelines (original / latent / both) with 5-fold "
        "CV. All preprocessing is fit inside each fold. Uses the file & tab from the "
        "Train tab.")
    imgui.separator()

    if not STATE.data_path:
        imgui.text_colored(DIM, "Choose a spreadsheet in the Train tab first.")
        return
    imgui.text_colored(DIM, STATE.data_path)

    if len(STATE.sheet_names) > 1:
        imgui.set_next_item_width(360)
        STATE.sheet_idx = min(STATE.sheet_idx, len(STATE.sheet_names) - 1)
        changed, STATE.sheet_idx = imgui.combo("Sheet / tab", STATE.sheet_idx, STATE.sheet_names)
        if changed:
            STATE.lat_columns = []  # force a rescan for the new tab

    if imgui.button("Scan columns", size=imgui.ImVec2(130, 0)):
        lat_scan_columns()
    imgui.same_line()
    if imgui.small_button("Reset to recommended defaults"):
        STATE.lat_categorical = ", ".join(latent.DEFAULT_CATEGORICAL)
        STATE.lat_numerical = ", ".join(latent.DEFAULT_NUMERICAL)
        if STATE.lat_columns:
            tgt = (STATE.lat_columns[STATE.lat_target_idx]
                   if 0 <= STATE.lat_target_idx < len(STATE.lat_columns) else "")
            STATE.lat_excluded = ", ".join(latent.default_excluded(tgt, STATE.lat_columns))

    if not STATE.lat_columns:
        imgui.text_colored(DIM, STATE.lat_status)
        return

    # --- Selectors ---
    imgui.set_next_item_width(360)
    STATE.lat_target_idx = min(STATE.lat_target_idx, len(STATE.lat_columns) - 1)
    _, STATE.lat_target_idx = imgui.combo("Target column", STATE.lat_target_idx, STATE.lat_columns)
    imgui.set_next_item_width(520)
    _, STATE.lat_categorical = imgui.input_text("Categorical features", STATE.lat_categorical)
    imgui.set_next_item_width(520)
    _, STATE.lat_numerical = imgui.input_text("Numerical features", STATE.lat_numerical)
    imgui.set_next_item_width(520)
    _, STATE.lat_excluded = imgui.input_text("Excluded / ID columns", STATE.lat_excluded)

    # --- Engineered-index options ---
    imgui.separator()
    imgui.text("Engineered index options")
    imgui.set_next_item_width(300)
    _, STATE.lat_chem_w = imgui.input_float3("Chemical weights (pretreat, post, additive)",
                                             STATE.lat_chem_w)
    _, STATE.lat_biomass_pca = imgui.checkbox("Biomass index uses PCA-1 (else equal-weight mean)",
                                              STATE.lat_biomass_pca)

    # --- Component counts ---
    imgui.set_next_item_width(300)
    _, STATE.lat_pca_components = imgui.slider_int("PCA components", STATE.lat_pca_components, 2, 10)
    imgui.set_next_item_width(300)
    _, STATE.lat_pls_components = imgui.slider_int("PLS components", STATE.lat_pls_components, 1, 10)

    imgui.dummy(imgui.ImVec2(0, 4))
    if imgui.button("Run analysis", size=imgui.ImVec2(200, 0)):
        start_latent()
    imgui.text_colored(RED if STATE.lat_error else DIM, STATE.lat_status)
    if STATE.lat_error:
        imgui.text_wrapped(STATE.lat_error)

    # --- Formulas (requirement 7: display formulas) ---
    if imgui.collapsing_header("Engineered index formulas"):
        for name in latent.ENGINEERED_INDEX_NAMES:
            imgui.text_colored(GREEN, name)
            imgui.text_wrapped("   " + latent.ENGINEERED_FORMULAS[name])

    # --- PLS metrics ---
    if STATE.lat_pls_result:
        p = STATE.lat_pls_result
        imgui.separator()
        imgui.text(f"PLS (k={p['n_components']}, 5-fold CV):  "
                   f"R²={p['r2_mean']:.3f}±{p['r2_std']:.3f}   "
                   f"RMSE={p['rmse_mean']:.3f}±{p['rmse_std']:.3f}   "
                   f"MAE={p['mae_mean']:.3f}±{p['mae_std']:.3f}")

    # --- Comparison table ---
    if STATE.lat_compare_result:
        _draw_comparison_table(STATE.lat_compare_result)

    # --- Export buttons ---
    if STATE.lat_last:
        imgui.dummy(imgui.ImVec2(0, 4))
        imgui.text("Export:")
        imgui.same_line()
        if imgui.small_button("rows+latents .xlsx"):
            lat_export("xlsx")
        imgui.same_line()
        if imgui.small_button("PCA loadings .csv"):
            lat_export("loadings")
        imgui.same_line()
        if imgui.small_button("CV comparison .csv"):
            lat_export("comparison")
        imgui.same_line()
        if imgui.small_button("pipeline .joblib"):
            lat_export("pipeline")

    # --- Charts ---
    if STATE.lat_chart_items:
        imgui.separator()
        if imgui.begin_child("lat_charts", imgui.ImVec2(0, 0)):
            for title, rel in STATE.lat_chart_items:
                imgui.text_colored(GREEN, title)
                _show_chart_image(rel)
                imgui.dummy(imgui.ImVec2(0, 8))
        imgui.end_child()


def _draw_comparison_table(comp):
    """Render the A/B/C x model CV results as a table."""
    imgui.dummy(imgui.ImVec2(0, 4))
    imgui.text("Pipeline comparison (mean ± std, 5-fold CV)")
    flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
    if imgui.begin_table("latcmp", 5, flags):
        for h in ("Pipeline", "Model", "R2", "RMSE", "MAE"):
            imgui.table_setup_column(h)
        imgui.table_headers_row()
        labels = {"A": "A: original", "B": "B: latent", "C": "C: both"}
        for v in ("A", "B", "C"):
            for m, sc in comp.get(v, {}).items():
                imgui.table_next_row()
                imgui.table_next_column(); imgui.text(labels[v])
                imgui.table_next_column(); imgui.text(m)
                if "error" in sc:
                    imgui.table_next_column(); imgui.text_colored(RED, "error")
                    imgui.table_next_column(); imgui.text("-")
                    imgui.table_next_column(); imgui.text("-")
                    continue
                imgui.table_next_column()
                imgui.text_colored(GREEN if sc["r2_mean"] >= 0.5 else DIM,
                                   f"{sc['r2_mean']:.3f}±{sc['r2_std']:.3f}")
                imgui.table_next_column(); imgui.text(f"{sc['rmse_mean']:.2f}±{sc['rmse_std']:.2f}")
                imgui.table_next_column(); imgui.text(f"{sc['mae_mean']:.2f}±{sc['mae_std']:.2f}")
        imgui.end_table()


def draw_intelligence_tab():
    imgui.text("Dataset Intelligence — can this dataset predict the target, and why / why not?")
    imgui.text_wrapped(
        "Reuses your Train / Column types selections (file, tab, target, features, "
        "exclusions). Evaluates predictability, redundancy, difficulty, learnability, "
        "causal structure, latent contribution, target shape and PCA — then explains "
        "the result in plain language. Runs in the background.")
    imgui.separator()

    if not STATE.data_path:
        imgui.text_colored(DIM, "Choose a spreadsheet in the Train tab first.")
        return
    imgui.text_colored(DIM, STATE.data_path
                       + (f"  ·  tab: {_current_sheet()}" if _current_sheet() else ""))

    targets = _configured_targets()
    if not targets:
        imgui.text_colored(RED, "No target configured. Set one in the Train / Column types tab.")
        return
    imgui.set_next_item_width(360)
    STATE.intel_target_idx = min(STATE.intel_target_idx, len(targets) - 1)
    _, STATE.intel_target_idx = imgui.combo("Target to analyse", STATE.intel_target_idx, targets)
    imgui.set_next_item_width(300)
    _, STATE.intel_pca_components = imgui.slider_int("PCA components", STATE.intel_pca_components, 2, 10)

    imgui.dummy(imgui.ImVec2(0, 4))
    imgui.begin_disabled(STATE.is_intel_running)
    if imgui.button("Analysing…" if STATE.is_intel_running else "Run analysis",
                    size=imgui.ImVec2(200, 0)):
        start_intelligence()
    imgui.end_disabled()
    if STATE.intel_results:
        imgui.same_line()
        if imgui.button("Export PDF report", size=imgui.ImVec2(180, 0)):
            intel_export_pdf()
    draw_progress_panel(STATE.prog_intel)
    imgui.text_colored(RED if STATE.intel_error else DIM, STATE.intel_status)
    if STATE.intel_error:
        imgui.text_wrapped(STATE.intel_error)

    r = STATE.intel_results
    if not r:
        return

    # ---- AI Research Assistant panel ----
    if STATE.intel_insights and imgui.collapsing_header("AI Research Assistant",
                                                        imgui.TreeNodeFlags_.default_open):
        for line in STATE.intel_insights:
            imgui.bullet()
            imgui.same_line()
            imgui.text_wrapped(line)

    # ---- Section 1: summary ----
    if imgui.collapsing_header("1 · Dataset summary", imgui.TreeNodeFlags_.default_open):
        s = r["summary"]
        _kv_table("intel_sum", [
            ("Samples", str(s["n_samples"])),
            ("Features", f"{s['n_features']} ({s['n_numerical']} num, {s['n_categorical']} cat)"),
            ("Missing overall", f"{s['missing_pct_overall']}%"),
            ("Duplicate rows", str(s["duplicate_rows"])),
            ("Duplicate conditions", str(s["duplicate_conditions"])),
            ("Target std", f"{s['target_std']:.4g}"),
            ("Target range", f"{s['target_min']:.4g} – {s['target_max']:.4g}"),
        ])

    # ---- Section 4: difficulty ----
    if imgui.collapsing_header("4 · Prediction difficulty", imgui.TreeNodeFlags_.default_open):
        d = r["diff"]
        color = {"Easy": GREEN, "Moderate": (0.9, 0.7, 0.2, 1.0),
                 "Hard": (0.95, 0.5, 0.2, 1.0), "Very Hard": RED}.get(d["label"], DIM)
        imgui.text_colored(color, f"Difficulty: {d['label']}   "
                           f"(estimated achievable CV R²  {d['est_r2_low']:.2f}–{d['est_r2_high']:.2f})")
        imgui.text_wrapped(d["explanation"])

    # ---- Section 5: learnability ----
    if imgui.collapsing_header("5 · Learnability (5-fold CV)"):
        _draw_learnability_table(r["learn"])

    # ---- Section 3: redundancy ----
    if imgui.collapsing_header("3 · Feature redundancy"):
        red = r["redund"]
        if red["suggest_remove"]:
            imgui.text_wrapped("Suggested to drop: " + ", ".join(red["suggest_remove"][:20]))
        if red["high_corr_pairs"]:
            imgui.text_wrapped("Highly correlated pairs: "
                               + ", ".join(f"{a}~{b} ({r_:.2f})"
                                           for a, b, r_ in red["high_corr_pairs"][:8]))
        if red["near_zero_variance"]:
            imgui.text_colored(DIM, "Near-zero variance: " + ", ".join(red["near_zero_variance"]))

    # ---- Final conclusion ----
    if imgui.collapsing_header("Final conclusion", imgui.TreeNodeFlags_.default_open):
        imgui.text_wrapped(STATE.intel_conclusion)

    # ---- Charts ----
    if STATE.intel_chart_items:
        imgui.separator()
        if imgui.begin_child("intel_charts", imgui.ImVec2(0, 0)):
            for title, rel in STATE.intel_chart_items:
                imgui.text_colored(GREEN, title)
                _show_chart_image(rel)
                imgui.dummy(imgui.ImVec2(0, 8))
        imgui.end_child()


def _kv_table(tid, rows):
    """Render a two-column key/value table."""
    flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
    if imgui.begin_table(tid, 2, flags):
        imgui.table_setup_column("Metric")
        imgui.table_setup_column("Value")
        imgui.table_headers_row()
        for k, v in rows:
            imgui.table_next_row()
            imgui.table_next_column(); imgui.text(k)
            imgui.table_next_column(); imgui.text(v)
        imgui.end_table()


def _draw_learnability_table(learn):
    """CV metrics per model, with the insufficiency warning."""
    flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
    if imgui.begin_table("intel_learn", 5, flags):
        for h in ("Model", "R2", "RMSE", "MAE", "kind"):
            imgui.table_setup_column(h)
        imgui.table_headers_row()
        for name, sc in learn["results"].items():
            imgui.table_next_row()
            imgui.table_next_column(); imgui.text(name)
            if "error" in sc:
                imgui.table_next_column(); imgui.text_colored(RED, "error")
                imgui.table_next_column(); imgui.text("-")
                imgui.table_next_column(); imgui.text("-")
                imgui.table_next_column(); imgui.text(sc.get("kind", ""))
                continue
            imgui.table_next_column()
            imgui.text_colored(GREEN if sc["r2_mean"] >= 0.3 else RED,
                               f"{sc['r2_mean']:.3f}±{sc['r2_std']:.3f}")
            imgui.table_next_column(); imgui.text(f"{sc['rmse_mean']:.2f}")
            imgui.table_next_column(); imgui.text(f"{sc['mae_mean']:.2f}")
            imgui.table_next_column(); imgui.text(sc.get("kind", ""))
        imgui.end_table()
    if learn.get("insufficient"):
        imgui.text_colored(RED, "This dataset may not contain enough predictive "
                           "information for the selected target.")


ORANGE = (0.95, 0.6, 0.2, 1.0)
BLUE = (0.45, 0.7, 0.95, 1.0)


def _screen_inputs_panel():
    """Left panel: synthesis-condition widgets shared with the What-if workflow."""
    imgui.text_colored(BLUE, "Synthesis conditions")
    imgui.same_line()
    imgui.text_colored(DIM, "(edit to run a what-if)")
    if imgui.begin_child("screen_inputs", imgui.ImVec2(0, 0)):
        draw_chemical_inputs("screen", mark_dirty=True)
        hidden_chemistry = _chemistry_model_columns()
        # Parsed messy columns appear as ONE field each ('1M NaOH' style).
        consumed = draw_recipe_inputs("scr", mark_dirty=True)
        for col in STATE.numeric_schema:
            if col in consumed or col in hidden_chemistry:
                continue
            changed, val = imgui.input_float(f"{feature_label(col)}##snum_{col}",
                                             float(STATE.numeric_values[col]))
            if changed:
                STATE.numeric_values[col] = val
                STATE.screen_dirty = True
        for col, choices in STATE.categorical_schema.items():
            if col in consumed or col in hidden_chemistry:
                continue
            STATE.category_index[col] = min(STATE.category_index.get(col, 0),
                                            max(len(choices) - 1, 0))
            changed, idx = imgui.combo(f"{feature_label(col)}##scat_{col}",
                                       STATE.category_index[col], choices)
            if changed:
                STATE.category_index[col] = idx
                STATE.screen_dirty = True
            current = STATE.screen_custom_category.get(col, "")
            imgui.set_next_item_width(-1)
            changed, custom = imgui.input_text(
                f"Other / unknown {feature_label(col)}##soth_{col}", current)
            if changed:
                STATE.screen_custom_category[col] = custom
                STATE.screen_dirty = True
    imgui.end_child()


def _draw_similar_table(res, scr):
    target = res["target"]
    sims = res["similar"]
    if not sims:
        return
    # Show the most influential synthesis variables as context columns; the
    # parts of a parsed messy column collapse into ONE '1M NaOH'-style column.
    cond_cols = set(sims[0]["conditions"])
    groups = units.recipe_groups(cond_cols)
    member = {col: s for s, g in groups.items() for col in g.values()}
    disp, seen = [], set()
    for f, _ in scr.importance:
        if f not in cond_cols:
            continue
        gs = member.get(f)
        if gs is not None:
            if gs not in seen:
                seen.add(gs)
                disp.append(("group", groups[gs]))
        else:
            disp.append(("col", f))
        if len(disp) == 4:
            break
    flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
    ncol = 2 + len(disp)
    if imgui.begin_table("sim", ncol, flags):
        imgui.table_setup_column("Similar")
        imgui.table_setup_column("Measured")
        for kind, d in disp:
            imgui.table_setup_column(group_base_label(d) if kind == "group" else pretty(d))
        imgui.table_headers_row()
        for s in sims:
            imgui.table_next_row()
            imgui.table_next_column()
            sim = s["similarity"]
            imgui.text_colored(GREEN if sim >= 70 else (ORANGE if sim >= 40 else DIM),
                               f"{sim:.0f}%")
            imgui.table_next_column()
            imgui.text(f"{s['measured'][target]:.1f}")
            for kind, d in disp:
                imgui.table_next_column()
                if kind == "group":
                    imgui.text(units.compose_group(d, s["conditions"].get)[:20])
                else:
                    v = s["conditions"][d]
                    imgui.text(f"{v:g}" if isinstance(v, float) else str(v)[:16])
        imgui.end_table()


def _draw_screen_result(res, scr):
    """Right panel: the full research-assistant read-out for one recipe."""
    rec = res["recommendation"]
    pred = res["prediction"]
    ad = res["applicability"]
    mq = res["model_quality"]
    target = res["target"]
    tname = pretty(target)

    if imgui.begin_child("screen_results", imgui.ImVec2(0, 0)):
        # --- Verdict banner ---
        imgui.text_colored(rec["color"], f"  {rec['verdict'].upper()}  ")
        imgui.same_line()
        imgui.text_colored(DIM, f"priority score {rec['score']:.2f}")
        imgui.separator()

        # --- Prediction + interval ---
        imgui.text_colored(BLUE, "Estimated performance (prediction, not measurement)")
        imgui.text(f"{tname}:  ")
        imgui.same_line()
        imgui.text_colored(GREEN, f"{pred['mean']:.1f}")
        imgui.same_line()
        imgui.text_colored(DIM, f"  95% interval  [{pred['lo']:.1f} , {pred['hi']:.1f}]")
        conf = rec["confidence"]
        cc = {"High": GREEN, "Moderate": (0.8, 0.8, 0.3, 1.0),
              "Low": ORANGE, "Very low": RED}.get(conf, DIM)
        imgui.text("Confidence: ")
        imgui.same_line(); imgui.text_colored(cc, conf)
        imgui.same_line()
        imgui.text_colored(DIM, f"   ± {pred['expected_error']:.1f} expected error (CV RMSE)")
        rk = rec["ranking"]["percentile"]
        imgui.text_colored(DIM, f"Ranks higher than {rk:.0f}% of the "
                           f"{mq['n_train']} experiments in the dataset.")

        # --- Applicability domain ---
        imgui.dummy(imgui.ImVec2(0, 3))
        imgui.text_colored(BLUE, "Applicability domain")
        imgui.text_colored(GREEN if ad["in_domain"] else RED,
                           "  " + ("● " + ad["label"]))
        if res["ood"]:
            for f in res["ood"]:
                imgui.text_colored(RED, "  ⚠ " + f)
        if res["missing"]:
            imgui.text_colored(ORANGE, "  ⚠ Unspecified: " + ", ".join(res["missing"][:6]))

        if res.get("chemistry"):
            imgui.dummy(imgui.ImVec2(0, 3))
            imgui.text_colored(BLUE, "Chemical descriptor evidence")
            for evidence in res["chemistry"]:
                imgui.text_wrapped(
                    f"  {evidence['column']}: {evidence['original']} → {evidence['summary']} "
                    f"Descriptor confidence {evidence['descriptor_confidence']:.2f}."
                )
                nearest = ", ".join(
                    f"{item['name']} {item['score']:.2f}" for item in evidence["similarities"])
                imgui.text_colored(DIM, f"    Similarity to known chemicals: {nearest}")
                descriptors = evidence.get("descriptors", {})
                central = []
                for key in ("Strong acid", "Strong base", "pKa", "Molecular weight",
                            "Contains chloride", "Hydroxide", "Oxidation tendency"):
                    value = descriptors.get(key)
                    if isinstance(value, float) and not np.isfinite(value):
                        continue
                    central.append(f"{key}={value}")
                if central:
                    imgui.text_colored(DIM, "    " + "  |  ".join(central))
                if not evidence["exactly_observed"]:
                    imgui.text_colored(
                        ORANGE,
                        "    Prediction confidence reduced because this exact chemical "
                        "was not observed during training."
                    )

        # --- Model quality ---
        cvr2 = mq["cv_r2"]
        imgui.dummy(imgui.ImVec2(0, 3))
        imgui.text_colored(BLUE, "Model quality")
        q = "n/a" if cvr2 is None else f"{cvr2:.3f}"
        imgui.text_colored(DIM, f"  Cross-validated R² = {q}   ·   "
                           f"{mq['n_train']} rows · {mq['n_features']} features")

        # --- Why (reasons) ---
        imgui.dummy(imgui.ImVec2(0, 3))
        imgui.text_colored(BLUE, "Why this recommendation")
        for r in rec["reasons"]:
            col = RED if r.startswith("Warning") or r.startswith("Outside") else None
            if col:
                imgui.text_colored(col, "  • " + r)
            else:
                imgui.text_wrapped("  • " + r)

        # --- Feature contributions ---
        contribs = res["contributions"]["contributions"]
        if contribs:
            imgui.dummy(imgui.ImVec2(0, 3))
            imgui.text_colored(BLUE, "What drove this prediction")
            imgui.text_colored(DIM, "  green raises · red lowers predicted "
                               + tname)
            maxabs = max(abs(v) for _, v in contribs) or 1.0
            for name, v in contribs[:8]:
                bar = "█" * int(round(10 * abs(v) / maxabs))
                imgui.text_colored(GREEN if v >= 0 else RED,
                                   f"  {('+' if v>=0 else '-')}{abs(v):7.1f} {bar}")
                imgui.same_line()
                imgui.text(" " + pretty(name))

        # --- What-if sensitivity ---
        effects = res.get("effect_summary", [])
        if effects:
            imgui.dummy(imgui.ImVec2(0, 3))
            imgui.text_colored(BLUE, "What-if sensitivity")
            for e in effects[:6]:
                imgui.text_wrapped("  - " + e["summary"])
            flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
            if imgui.begin_table("sens", 3, flags):
                for h in ("Variable", "Swing", "Direction"):
                    imgui.table_setup_column(h)
                imgui.table_headers_row()
                for e in effects[:6]:
                    imgui.table_next_row()
                    imgui.table_next_column(); imgui.text(pretty(e["feature"]))
                    imgui.table_next_column(); imgui.text(f"{e['swing']:.1f}")
                    imgui.table_next_column(); imgui.text(str(e["direction"]))
                imgui.end_table()

        # --- Similar experiments ---
        imgui.dummy(imgui.ImVec2(0, 3))
        imgui.text_colored(BLUE, "Most similar real experiments")
        _draw_similar_table(res, scr)

        # --- Report export ---
        imgui.dummy(imgui.ImVec2(0, 6))
        imgui.separator()
        if imgui.button("Export PDF report"):
            STATE.screen_pdf_dialog = pfd.save_file(
                "Save screening report", "BioCarbon_screening_report.pdf",
                filters=["PDF", "*.pdf"])
        imgui.same_line()
        if imgui.button("Export Excel report"):
            STATE.screen_xlsx_dialog = pfd.save_file(
                "Save screening report", "BioCarbon_screening_report.xlsx",
                filters=["Excel", "*.xlsx"])
        if STATE.screen_export_msg:
            imgui.text_colored(DIM, STATE.screen_export_msg)
    imgui.end_child()


def draw_screen_tab():
    """PRIMARY WORKFLOW: enter conditions → predict, quantify, explain, recommend."""
    if not STATE.trained or STATE.screener is None:
        imgui.text_colored(DIM, "Train (or load) a model first — see the Train tab. "
                           "The screening engine is built automatically after training.")
        return

    # Poll async report-save dialogs.
    if STATE.screen_pdf_dialog is not None and STATE.screen_pdf_dialog.ready():
        r = STATE.screen_pdf_dialog.result()
        if r:
            export_screen_report(r, "pdf")
        STATE.screen_pdf_dialog = None
    if STATE.screen_xlsx_dialog is not None and STATE.screen_xlsx_dialog.ready():
        r = STATE.screen_xlsx_dialog.result()
        if r:
            export_screen_report(r, "xlsx")
        STATE.screen_xlsx_dialog = None
    if STATE.screen_import_dialog is not None and STATE.screen_import_dialog.ready():
        r = STATE.screen_import_dialog.result()
        if r:
            import_screen_recipe(r[0])
        STATE.screen_import_dialog = None

    # Target selector + run controls.
    imgui.set_next_item_width(320)
    STATE.screen_target_idx = min(STATE.screen_target_idx, len(STATE.targets) - 1)
    _, STATE.screen_target_idx = imgui.combo(
        "Performance target to screen", STATE.screen_target_idx, STATE.targets)
    if imgui.button("Import recipe (.xlsx / .csv)...", size=imgui.ImVec2(220, 0)):
        STATE.screen_import_dialog = pfd.open_file(
            "Import synthesis recipe", filters=["Data", "*.xlsx *.xls *.csv", "All", "*"])
    if STATE.screen_import_status:
        imgui.same_line()
        imgui.text_colored(DIM, STATE.screen_import_status)

    if imgui.button("Screen this recipe", size=imgui.ImVec2(170, 0)):
        start_screen()
    imgui.same_line()
    _, STATE.screen_live = imgui.checkbox("Live what-if", STATE.screen_live)
    imgui.same_line()
    if STATE.is_screening:
        imgui.text_colored(ORANGE, "analyzing…")
    elif STATE.screen_dirty:
        imgui.text_colored(DIM, "inputs changed — press Screen")

    # Live mode: re-run automatically once the previous run finishes.
    if STATE.screen_live and STATE.screen_dirty and not STATE.is_screening:
        start_screen()

    if STATE.screen_error:
        imgui.text_colored(RED, STATE.screen_error)
    imgui.separator()

    # Two-panel layout: inputs on the left, screening read-out on the right.
    avail = imgui.get_content_region_avail()
    left_w = max(280.0, avail.x * 0.38)
    if imgui.begin_child("screen_left", imgui.ImVec2(left_w, 0)):
        _screen_inputs_panel()
    imgui.end_child()
    imgui.same_line()
    if STATE.screen_result is not None:
        _draw_screen_result(STATE.screen_result, STATE.screener)
    else:
        if imgui.begin_child("screen_hint", imgui.ImVec2(0, 0)):
            imgui.text_colored(DIM, "Set the synthesis conditions, pick a target, "
                               "then press \"Screen this recipe\".")
            imgui.text_wrapped(
                "You'll get an estimated value with a prediction interval and "
                "confidence, an applicability-domain check, the most similar real "
                "experiments, a plain-language recommendation, and the synthesis "
                "variables that drove the prediction.")
        imgui.end_child()


def draw_priority_tab():
    """Rank candidate synthesis routes for experimental prioritization."""
    if not STATE.trained or STATE.screener is None:
        imgui.text_colored(DIM, "Train (or load) a model first — see the Train tab.")
        return

    # Poll async dialogs.
    if STATE.prio_source_dialog is not None and STATE.prio_source_dialog.ready():
        r = STATE.prio_source_dialog.result()
        if r:
            STATE.prio_source_path = r[0]
            STATE.prio_source_sheet_names = list_sheets(STATE.prio_source_path)
            STATE.prio_source_sheet_idx = 0
        STATE.prio_source_dialog = None
    if STATE.prio_save_dialog is not None and STATE.prio_save_dialog.ready():
        r = STATE.prio_save_dialog.result()
        if r and STATE.prio_df is not None:
            try:
                if r.lower().endswith(".xlsx"):
                    STATE.prio_df.to_excel(r, index=False)
                else:
                    STATE.prio_df.to_csv(r, index=False)
                STATE.prio_status = f"Saved ranked list to {r}"
            except Exception as e:  # noqa: BLE001
                STATE.prio_error = f"Save failed: {e}"
        STATE.prio_save_dialog = None

    imgui.text_wrapped(
        "Rank many candidate synthesis routes so you know which to run first. "
        "The score blends predicted performance, confidence, novelty, similarity "
        "to strong experiments and experimental feasibility.")
    imgui.separator()

    imgui.set_next_item_width(320)
    STATE.prio_target_idx = min(STATE.prio_target_idx, len(STATE.targets) - 1)
    _, STATE.prio_target_idx = imgui.combo(
        "Target to rank by", STATE.prio_target_idx, STATE.targets)

    # Weight sliders.
    imgui.text_colored(DIM, "Scoring weights")
    _, STATE.prio_w_perf = imgui.slider_float("Performance", STATE.prio_w_perf, 0.0, 1.0)
    _, STATE.prio_w_conf = imgui.slider_float("Confidence", STATE.prio_w_conf, 0.0, 1.0)
    _, STATE.prio_w_novel = imgui.slider_float("Novelty", STATE.prio_w_novel, 0.0, 1.0)
    _, STATE.prio_w_sim = imgui.slider_float("Similarity", STATE.prio_w_sim, 0.0, 1.0)
    _, STATE.prio_w_feas = imgui.slider_float("Feasibility", STATE.prio_w_feas, 0.0, 1.0)

    imgui.dummy(imgui.ImVec2(0, 4))
    imgui.text("Candidate source")
    if imgui.button("Choose spreadsheet (.xlsx / .csv)..."):
        STATE.prio_source_dialog = pfd.open_file(
            "Select candidate recipes", filters=["Data", "*.xlsx *.xls *.csv", "All", "*"])
    if STATE.prio_source_path:
        imgui.same_line()
        imgui.text_colored(DIM, STATE.prio_source_path)
    if len(STATE.prio_source_sheet_names) > 1:
        imgui.set_next_item_width(360)
        STATE.prio_source_sheet_idx = min(STATE.prio_source_sheet_idx,
                                          len(STATE.prio_source_sheet_names) - 1)
        _, STATE.prio_source_sheet_idx = imgui.combo(
            "Candidate sheet / tab", STATE.prio_source_sheet_idx,
            STATE.prio_source_sheet_names)

    if imgui.button("Rank candidates from file", size=imgui.ImVec2(210, 0)):
        start_prioritize()
    imgui.same_line()
    if imgui.button("Rank the training experiments"):
        # Re-rank every real recipe the model has seen (a quick sanity ranking).
        cands = STATE.screener.train_raw.to_dict("records")
        start_prioritize(candidates=cands)
    if STATE.opt_result is not None:
        imgui.same_line()
        if imgui.button("Rank optimizer recipe"):
            start_prioritize(candidates=[dict(STATE.opt_result["recipe"])])

    imgui.text_colored(RED if STATE.prio_error else DIM,
                       STATE.prio_error or STATE.prio_status)

    if STATE.prio_df is not None and len(STATE.prio_df):
        if imgui.button("Export ranked list..."):
            STATE.prio_save_dialog = pfd.save_file(
                "Save ranked list", "BioCarbon_priority_list.xlsx",
                filters=["Excel", "*.xlsx", "CSV", "*.csv"])
        imgui.separator()
        # Show the ranking (headline columns only; full detail is exported).
        cols = ["rank", "predicted", "interval_low", "interval_high", "confidence",
                "in_domain", "novelty_pct", "top_similarity_pct", "feasibility",
                "verdict", "priority_score"]
        cols = [c for c in cols if c in STATE.prio_df.columns]
        flags = (imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
                 | imgui.TableFlags_.scroll_y | imgui.TableFlags_.scroll_x)
        if imgui.begin_table("prio", len(cols), flags, outer_size=imgui.ImVec2(0, 320)):
            for c in cols:
                imgui.table_setup_column(pretty(c))
            imgui.table_headers_row()
            for _, r in STATE.prio_df.iterrows():
                imgui.table_next_row()
                for c in cols:
                    imgui.table_next_column()
                    v = r[c]
                    if c == "verdict":
                        imgui.text_colored(screening.VERDICTS.get(v, DIM), str(v))
                    elif isinstance(v, float):
                        imgui.text(f"{v:g}")
                    else:
                        imgui.text(str(v))
            imgui.end_table()


def _refresh_chemical_knowledge():
    values = []
    for info in STATE.chemistry_schema.get("columns", {}).values():
        values.extend(info.get("observed_chemicals", []))
    STATE.chemical_knowledge = chemistry.ENGINE.knowledge_for_values(values)


def draw_chemical_knowledge_tab():
    imgui.text("Chemical Knowledge Engine")
    imgui.text_wrapped(
        "Original reagent names are retained for reporting, while model training uses "
        "formula, functional-class, elemental, physicochemical and optional RDKit descriptors."
    )
    imgui.separator()
    changed_enabled, STATE.chemistry_enabled = imgui.checkbox(
        "Enable chemistry-aware feature engineering", STATE.chemistry_enabled)
    if changed_enabled:
        if not STATE.chemistry_enabled:
            STATE.chemistry_mode_idx = 0
        elif STATE.chemistry_mode_idx == 0:
            STATE.chemistry_mode_idx = 1
    changed, pubchem_enabled = imgui.checkbox(
        "Allow PubChem lookup for unrecognized names (internet)",
        STATE.chemistry_pubchem_enabled)
    if changed:
        STATE.chemistry_pubchem_enabled = pubchem_enabled
        chemistry.ENGINE.generator.set_pubchem_enabled(pubchem_enabled)
        _refresh_chemical_knowledge()
    imgui.text_colored(
        DIM,
        "PubChem is optional and cached; formula and lookup fallbacks always work offline."
    )
    imgui.text_colored(
        GREEN if chemistry.ENGINE.generator.rdkit.available else DIM,
        "RDKit: available (molecular descriptors + Morgan fingerprints)"
        if chemistry.ENGINE.generator.rdkit.available
        else "RDKit: unavailable - internal lookup and formula heuristics remain active"
    )
    if imgui.button("Auto configure chemistry", size=imgui.ImVec2(210, 0)):
        scan_chemical_knowledge()
    imgui.same_line()
    imgui.text_colored(DIM, STATE.chemistry_status)

    estimate = STATE.chemistry_estimate
    if estimate:
        interactions = len(STATE.chemistry_schema.get("interactions", []))
        labels_retained = bool(STATE.chemistry_schema.get("retain_original_labels", False))
        imgui.text_wrapped(
            f"Independent groups: {estimate.get('independent_groups', 0)}  |  "
            f"Chemical columns detected: {len(STATE.chemistry_schema.get('columns', {}))}  |  "
            f"Candidate descriptors: {STATE.chemistry_schema.get('candidate_descriptor_count', 0)}")
        imgui.text_wrapped(
            f"Selected chemistry features: {estimate.get('chemistry_features', 0)}  |  "
            f"Selected interactions: {interactions}  |  "
            f"Original chemical labels retained: {'Yes' if labels_retained else 'No'}  |  "
            f"Estimated total encoded predictors: "
            f"{estimate.get('estimated_total_encoded_predictors', 0)}")
        for reason in STATE.chemistry_schema.get("rationale", []):
            imgui.bullet_text(str(reason))
        omitted = STATE.chemistry_schema.get("omitted_interactions", [])
        if omitted and imgui.tree_node("Omitted interaction details"):
            for feature, reason in omitted:
                imgui.bullet_text(f"{chemistry.descriptor_display_name(feature)}: {reason}")
            imgui.tree_pop()

    detections = list(STATE.chemistry_detection)
    if detections and imgui.tree_node("Chemical-column detection review"):
        for item in detections:
            column = item["column"]
            automatic = float(item["confidence"]) >= .70
            selected = bool(STATE.chemistry_column_overrides.get(column, automatic))
            changed_column, selected = imgui.checkbox(
                f"{column} ({float(item['confidence']):.2f})##chem_detect_{column}", selected)
            if changed_column:
                STATE.chemistry_column_overrides[column] = selected
            imgui.same_line()
            imgui.text_colored(DIM, str(item.get("reason", "")))
        imgui.text_colored(DIM, "Click Auto configure chemistry to apply detection overrides.")
        imgui.tree_pop()
    if not STATE.chemical_knowledge:
        imgui.text_wrapped(
            "No chemicals are listed yet. Load a dataset and scan it, or train once "
            "to populate the exact chemistry schema used by the model."
        )
        return

    imgui.text("Original chemical  →  Expanded descriptor summary")
    flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg | imgui.TableFlags_.scroll_y
    if imgui.begin_table("chemical_overview", 6, flags, outer_size=imgui.ImVec2(0, 190)):
        for header in ("Original", "Resolved as", "Class", "MW", "pKa", "Confidence"):
            imgui.table_setup_column(header)
        imgui.table_headers_row()
        for item in STATE.chemical_knowledge:
            descriptor = item["descriptor"]
            imgui.table_next_row()
            imgui.table_next_column(); imgui.text(item["original"])
            imgui.table_next_column(); imgui.text(descriptor.canonical_name)
            imgui.table_next_column(); imgui.text(descriptor.categorical["ChemicalClass"])
            mw = descriptor.numeric["MolecularWeight"]
            imgui.table_next_column(); imgui.text("n/a" if not np.isfinite(mw) else f"{mw:.3f}")
            pka = descriptor.numeric["pKa"]
            imgui.table_next_column(); imgui.text("n/a" if not np.isfinite(pka) else f"{pka:g}")
            imgui.table_next_column(); imgui.text(f"{descriptor.confidence:.2f}")
        imgui.end_table()

    if imgui.collapsing_header("Reagent cost & hazard (your own data, optional)"):
        imgui.text_wrapped(
            "Nothing here is guessed — enter real prices/hazard ratings from your own "
            "supplier sheets or MSDS data if you want the Optimize tab to show cost and "
            "hazard information for recommended experiments. Left blank = 'not entered', "
            "shown honestly as such rather than defaulted to zero or 'safe'. Saved to "
            f"'{cost_model.ENGINE.path}'.")
        flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
        if imgui.begin_table("cost_hazard", 5, flags, outer_size=imgui.ImVec2(0, 180)):
            imgui.table_setup_column("Chemical")
            imgui.table_setup_column("$ / kg")
            imgui.table_setup_column("$ / L")
            imgui.table_setup_column("Hazard class")
            imgui.table_setup_column("Corrosive")
            imgui.table_headers_row()
            any_dirty = False
            for item in STATE.chemical_knowledge:
                name = item["descriptor"].canonical_name
                entry = cost_model.ENGINE.get(name) or cost_model.ReagentCostEntry(name=name)
                row_dirty = False
                imgui.table_next_row()
                imgui.table_next_column(); imgui.text(name)
                imgui.table_next_column()
                imgui.set_next_item_width(-1)
                changed, v = imgui.input_float(
                    f"##cost_kg_{name}", entry.cost_per_kg or 0.0, format="%.2f")
                if changed:
                    entry.cost_per_kg = v if v > 0 else None
                    row_dirty = True
                imgui.table_next_column()
                imgui.set_next_item_width(-1)
                changed, v = imgui.input_float(
                    f"##cost_l_{name}", entry.cost_per_liter or 0.0, format="%.2f")
                if changed:
                    entry.cost_per_liter = v if v > 0 else None
                    row_dirty = True
                imgui.table_next_column()
                imgui.set_next_item_width(-1)
                idx = (cost_model.HAZARD_LEVELS.index(entry.hazard_class)
                      if entry.hazard_class in cost_model.HAZARD_LEVELS else 0)
                changed, idx = imgui.combo(f"##hazard_{name}", idx, cost_model.HAZARD_LEVELS)
                if changed:
                    entry.hazard_class = cost_model.HAZARD_LEVELS[idx]
                    row_dirty = True
                imgui.table_next_column()
                changed, v = imgui.checkbox(f"##corrosive_{name}", entry.corrosive)
                if changed:
                    entry.corrosive = v
                    row_dirty = True
                if row_dirty:
                    cost_model.ENGINE.set(name, entry)
                    any_dirty = True
            imgui.end_table()
            if any_dirty:
                try:
                    cost_model.ENGINE.save()
                except OSError:
                    pass

    names = [item["original"] for item in STATE.chemical_knowledge]
    STATE.chemistry_selected_idx = min(STATE.chemistry_selected_idx, len(names) - 1)
    imgui.set_next_item_width(280)
    changed, STATE.chemistry_selected_idx = imgui.combo(
        "Inspect / edit chemical", STATE.chemistry_selected_idx, names)
    selected = STATE.chemical_knowledge[STATE.chemistry_selected_idx]
    descriptor = chemistry.ENGINE.generator.describe(selected["original"])
    similarities = chemistry.ENGINE.generator.similarities(selected["original"], top=3)
    imgui.text_wrapped(
        f"{descriptor.categorical['ChemicalClass']} · source: {descriptor.source} · "
        f"descriptor confidence {descriptor.confidence:.2f}"
    )
    imgui.text("Nearest known chemistry: " + ", ".join(
        f"{item.name} ({item.score:.2f})" for item in similarities))
    if imgui.small_button("Reset manual edits for this chemical"):
        chemistry.ENGINE.generator.clear_override(selected["original"])
        _refresh_chemical_knowledge()
        descriptor = chemistry.ENGINE.generator.describe(selected["original"])

    central_numeric = [
        "Is_Acid", "Is_Base", "Is_Strong_Acid", "Is_Weak_Acid",
        "Is_Strong_Base", "Is_Weak_Base", "Is_Oxidizer", "Is_Reducing_Agent",
        "Is_Chelating_Agent", "Is_Transition_Metal_Salt", "Contains_Hydroxide",
        "Contains_Chloride", "Contains_Sulfate", "Contains_Nitrate",
        "pKa", "pKb", "MolecularWeight", "EstimatedIonicCharge",
        "EstimatedAcidity", "EstimatedBasicity", "EstimatedOxidationTendency",
        "EstimatedReductionTendency", "WaterSoluble", "Organic", "Inorganic",
    ]
    if imgui.begin_child("chemical_descriptor_editor", imgui.ImVec2(0, 0)):
        imgui.text_colored(DIM, "Manual edits are cached and included in settings/model exports.")
        overrides = dict(chemistry.ENGINE.generator.overrides.get(
            chemistry.normalize_chemical_key(selected["original"]), {}))
        for key in central_numeric:
            current = float(descriptor.numeric.get(key, math.nan))
            display = current if np.isfinite(current) else 0.0
            imgui.set_next_item_width(180)
            changed, value = imgui.input_float(chemistry.descriptor_display_name(key), display)
            if changed:
                overrides[key] = float(value)
                chemistry.ENGINE.generator.set_override(selected["original"], overrides)
                _refresh_chemical_knowledge()
                descriptor = chemistry.ENGINE.generator.describe(selected["original"])
        for key in ("ChemicalClass", "Metal", "Halogen", "AnionType", "CationType",
                    "IonicStrengthClass", "Corrosiveness"):
            imgui.set_next_item_width(260)
            changed, value = imgui.input_text(
                chemistry.descriptor_display_name(key), descriptor.categorical.get(key, "None"))
            if changed:
                overrides[key] = value
                chemistry.ENGINE.generator.set_override(selected["original"], overrides)
                _refresh_chemical_knowledge()
                descriptor = chemistry.ENGINE.generator.describe(selected["original"])
    imgui.end_child()


def gui():
    """Top-level GUI callback — called every frame by imgui-bundle."""
    style = imgui.get_style()
    # -- Title bar --------------------------------------------------------
    imgui.push_style_color(imgui.Col_.text, _rgba(_PALETTE["accent"]))
    imgui.text("BioCarbon Screen")
    imgui.pop_style_color()
    imgui.same_line()
    imgui.push_style_color(imgui.Col_.text, _rgba(_PALETTE["text_dim"]))
    imgui.text("    AI-assisted hard-carbon synthesis screening")
    imgui.pop_style_color()

    # Model status + reuse-a-saved-model button, right-aligned.
    have_saved = os.path.exists(MODEL_OUT)
    show_load = not STATE.trained and have_saved
    if STATE.trained:
        status, status_col = "Model ready", _PALETTE["accent"]
    elif have_saved:
        status, status_col = "Saved model on disk", _PALETTE["text_dim"]
    else:
        status, status_col = "No model trained yet", _PALETTE["text_dim"]
    right_w = imgui.calc_text_size(status).x
    if show_load:
        load_label = f"Load {MODEL_OUT}"
        right_w += (imgui.calc_text_size(load_label).x
                    + style.frame_padding.x * 2 + style.item_spacing.x)
    imgui.same_line(max(0.0, imgui.get_window_width() - right_w
                        - style.window_padding.x - 4))
    imgui.push_style_color(imgui.Col_.text, _rgba(status_col))
    imgui.text(status)
    imgui.pop_style_color()
    if show_load:
        imgui.same_line()
        if imgui.small_button(f"Load {MODEL_OUT}"):
            try:
                load_model()
            except Exception as e:  # noqa: BLE001
                STATE.train_error = f"Could not load model: {e}"
    imgui.separator()

    if imgui.begin_tab_bar("tabs"):
        # PRIMARY WORKFLOW first — the research-assistant screening view.
        if imgui.begin_tab_item("Screen")[0]:
            draw_screen_tab()
            imgui.end_tab_item()
        if imgui.begin_tab_item("Prioritize")[0]:
            draw_priority_tab()
            imgui.end_tab_item()
        if imgui.begin_tab_item("Train")[0]:
            draw_train_tab()
            imgui.end_tab_item()
        if imgui.begin_tab_item("Column types")[0]:
            draw_coltypes_tab()
            imgui.end_tab_item()
        if imgui.begin_tab_item("Chemical Knowledge")[0]:
            draw_chemical_knowledge_tab()
            imgui.end_tab_item()
        if imgui.begin_tab_item("Compare models")[0]:
            draw_compare_tab()
            imgui.end_tab_item()
        if imgui.begin_tab_item("Predict")[0]:
            draw_predict_tab()
            imgui.end_tab_item()
        if imgui.begin_tab_item("Batch predict")[0]:
            draw_batch_tab()
            imgui.end_tab_item()
        if imgui.begin_tab_item("Optimize")[0]:
            draw_optimize_tab()
            imgui.end_tab_item()
        if imgui.begin_tab_item("Suggest Experiments")[0]:
            draw_bayesopt_tab()
            imgui.end_tab_item()
        if imgui.begin_tab_item("Charts")[0]:
            draw_charts_tab()
            imgui.end_tab_item()
        if imgui.begin_tab_item("Latent Variables")[0]:
            draw_latent_tab()
            imgui.end_tab_item()
        if imgui.begin_tab_item("Dataset Intelligence")[0]:
            draw_intelligence_tab()
            imgui.end_tab_item()
        imgui.end_tab_bar()


# =============================================================================
# APPEARANCE  —  professional "light / clean" theme
# =============================================================================
# A calm, report-friendly palette: white panels on a soft grey canvas, a single
# blue accent reserved for selection / active states, thin borders and gently
# rounded corners. Applied once at startup via the runner's setup callback.

_PALETTE = {
    "canvas":        "#f4f5f8",  # app background
    "surface":       "#ffffff",  # panels / popups / selected tab
    "surface_alt":   "#eef1f6",  # alt table rows / hovered controls
    "input":         "#ffffff",  # text inputs, combos, sliders track
    "input_hover":   "#eef1f6",
    "input_active":  "#e6ebf4",
    "border":        "#d8dce4",
    "border_strong": "#c4cad4",
    "text":          "#1c2331",
    "text_dim":      "#6b7482",
    "accent":        "#2563eb",  # blue-600
    "accent_hover":  "#3b82f6",
    "accent_active": "#1d4ed8",
    "tab_idle":      "#e7eaf1",
    "scroll_grab":   "#c4cad4",
    "scroll_hover":  "#a9b1be",
}


def _rgba(hex_str, a=1.0):
    """'#rrggbb' -> imgui.ImVec4 (linear passthrough, alpha overridable)."""
    h = hex_str.lstrip("#")
    return imgui.ImVec4(
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
        a,
    )


def apply_professional_theme():
    """Setup callback: geometry + colours for a clean light UI."""
    style = imgui.get_style()

    # -- geometry -----------------------------------------------------------
    style.window_padding = imgui.ImVec2(16, 14)
    style.frame_padding = imgui.ImVec2(11, 6)
    style.cell_padding = imgui.ImVec2(9, 6)
    style.item_spacing = imgui.ImVec2(10, 9)
    style.item_inner_spacing = imgui.ImVec2(8, 6)
    style.indent_spacing = 22
    style.scrollbar_size = 13
    style.grab_min_size = 11

    style.window_border_size = 1
    style.child_border_size = 1
    style.popup_border_size = 1
    style.frame_border_size = 1
    style.tab_bar_border_size = 1 if hasattr(style, "tab_bar_border_size") else 0

    style.window_rounding = 9
    style.child_rounding = 9
    style.frame_rounding = 6
    style.popup_rounding = 7
    style.scrollbar_rounding = 9
    style.grab_rounding = 6
    style.tab_rounding = 6

    style.window_title_align = imgui.ImVec2(0.0, 0.5)
    if hasattr(imgui, "Dir"):
        style.window_menu_button_position = imgui.Dir.none
    style.anti_aliased_lines = True
    style.anti_aliased_fill = True

    # -- colours ------------------------------------------------------------
    p = _PALETTE
    C = imgui.Col_
    accent = _rgba(p["accent"])
    accent_h = _rgba(p["accent_hover"])
    accent_a = _rgba(p["accent_active"])

    def s(col, hex_or_vec, a=1.0):
        style.set_color_(col, hex_or_vec if isinstance(hex_or_vec, imgui.ImVec4)
                         else _rgba(hex_or_vec, a))

    s(C.text, p["text"])
    s(C.text_disabled, p["text_dim"])
    s(C.window_bg, p["canvas"])
    s(C.child_bg, p["surface"])
    s(C.popup_bg, p["surface"])
    s(C.border, p["border"])
    s(C.border_shadow, imgui.ImVec4(0, 0, 0, 0))

    s(C.frame_bg, p["input"])
    s(C.frame_bg_hovered, p["input_hover"])
    s(C.frame_bg_active, p["input_active"])

    s(C.title_bg, p["surface"])
    s(C.title_bg_active, p["surface"])
    s(C.title_bg_collapsed, p["surface"])
    s(C.menu_bar_bg, p["surface"])

    s(C.scrollbar_bg, imgui.ImVec4(0, 0, 0, 0))
    s(C.scrollbar_grab, p["scroll_grab"])
    s(C.scrollbar_grab_hovered, p["scroll_hover"])
    s(C.scrollbar_grab_active, p["border_strong"])

    s(C.check_mark, accent)
    s(C.slider_grab, accent)
    s(C.slider_grab_active, accent_a)

    # Buttons stay neutral; accent is reserved for state, not chrome.
    s(C.button, p["surface_alt"])
    s(C.button_hovered, _rgba(p["accent"], 0.16))
    s(C.button_active, _rgba(p["accent"], 0.28))

    # Selection surfaces (selectable / tree / collapsing header) read blue-tinted.
    s(C.header, _rgba(p["accent"], 0.16))
    s(C.header_hovered, _rgba(p["accent"], 0.24))
    s(C.header_active, _rgba(p["accent"], 0.32))

    s(C.separator, p["border"])
    s(C.separator_hovered, accent_h)
    s(C.separator_active, accent)

    s(C.resize_grip, _rgba(p["accent"], 0.0))
    s(C.resize_grip_hovered, _rgba(p["accent"], 0.30))
    s(C.resize_grip_active, _rgba(p["accent"], 0.55))

    # Tabs — flat segmented look, white when selected with a blue overline.
    s(C.tab, p["tab_idle"])
    s(C.tab_hovered, _rgba(p["accent"], 0.20))
    if hasattr(C, "tab_selected"):
        s(C.tab_selected, p["surface"])
        s(C.tab_dimmed, p["tab_idle"])
        s(C.tab_dimmed_selected, p["surface_alt"])
    if hasattr(C, "tab_selected_overline"):
        s(C.tab_selected_overline, accent)
        s(C.tab_dimmed_selected_overline, _rgba(p["accent"], 0.0))

    s(C.text_selected_bg, _rgba(p["accent"], 0.22))
    s(C.nav_highlight, accent)
    s(C.drag_drop_target, accent_h)
    s(C.plot_lines, accent)
    s(C.plot_lines_hovered, accent_a)
    s(C.plot_histogram, accent)
    s(C.plot_histogram_hovered, accent_a)

    # Tables
    s(C.table_header_bg, p["surface_alt"])
    s(C.table_border_strong, p["border_strong"])
    s(C.table_border_light, p["border"])
    s(C.table_row_bg, imgui.ImVec4(0, 0, 0, 0))
    s(C.table_row_bg_alt, _rgba(p["surface_alt"], 0.55))


_UI_FONT_SIZE = 16.5


def _find_roboto():
    """Locate a Roboto TTF in the frozen bundle, the assets folder, or the
    imgui_bundle install. Returns None if none is present."""
    names = ("fonts/Roboto/Roboto-Medium.ttf", "fonts/Roboto/Roboto-Regular.ttf")
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(getattr(sys, "_MEIPASS", None))
    roots.append(_base_dir())
    try:
        import imgui_bundle as _ib
        roots.append(os.path.join(os.path.dirname(_ib.__file__), "assets"))
    except Exception:  # noqa: BLE001
        pass
    for root in roots:
        if not root:
            continue
        for n in names:
            p = os.path.join(root, *n.split("/"))
            if os.path.isfile(p):
                return p
    return None


def _load_fonts():
    """Font-loading callback: use Roboto for a clean, modern look. Best-effort —
    any failure silently falls back to Dear ImGui's built-in font."""
    try:
        path = _find_roboto()
        if not path:
            return
        io = imgui.get_io()
        try:
            factor = hello_imgui.dpi_font_loading_factor()
        except Exception:  # noqa: BLE001
            factor = 1.0
        font = io.fonts.add_font_from_file_ttf(path, round(_UI_FONT_SIZE * factor))
        if font is not None:
            io.font_default = font
    except Exception:  # noqa: BLE001
        pass


def _base_dir():
    """Writable, stable base folder for assets/model/charts.

    Frozen (PyInstaller one-folder): the directory that holds the .exe.
    Source run: the project directory next to this file.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def main():
    base = _base_dir()
    # When frozen, anchor relative paths ("charts/", "model.joblib", *.ini) to the
    # folder next to the .exe so writes and asset lookups resolve to one place.
    if getattr(sys, "frozen", False):
        try:
            os.chdir(base)
        except OSError:
            pass
    # image_from_asset resolves paths relative to the assets folder.
    hello_imgui.set_assets_folder(base)
    # Charts are a transient display cache — clear any left over from prior runs
    # so nothing accumulates on disk; the user keeps charts via "Save charts…".
    _reset_chart_cache()

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "BioCarbon Screen"
    runner_params.app_window_params.window_geometry.size = [1180, 860]
    runner_params.app_window_params.restore_previous_geometry = True
    runner_params.imgui_window_params.show_menu_bar = False
    runner_params.imgui_window_params.background_color = _rgba(_PALETTE["canvas"])
    runner_params.callbacks.setup_imgui_style = apply_professional_theme
    runner_params.callbacks.load_additional_fonts = _load_fonts
    runner_params.callbacks.show_gui = gui

    immapp.run(runner_params)


if __name__ == "__main__":
    main()
