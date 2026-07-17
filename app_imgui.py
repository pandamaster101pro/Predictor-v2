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
    (Dependencies are also auto-installed on first run.)

The trained model can be saved to / loaded from 'model.joblib'.
==============================================================
"""

import sys
import subprocess
import threading


# -----------------------------------------------------------------------------
# Auto-install missing dependencies BEFORE importing them.
# -----------------------------------------------------------------------------
def auto_bootstrap():
    """Install any missing packages this script needs (wheels only)."""
    dependencies = {
        "imgui_bundle": "imgui-bundle",
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "openpyxl": "openpyxl",   # lets pandas read .xlsx files
        "joblib": "joblib",
        # Gradient-boosting libraries for the "Compare models" tab.
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "scipy": "scipy",         # differential evolution for the "Optimize" tab
        "matplotlib": "matplotlib",  # chart rendering for the "Charts" tab
        "shap": "shap",           # SHAP summary / dependence plots (Charts tab)
        "reportlab": "reportlab",  # PDF screening reports
    }
    missing = []
    for import_name, pip_name in dependencies.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[!] Missing modules found: {missing}")
        print("[*] Installing dependencies, please wait...")
        try:
            # --only-binary=:all: forces prebuilt wheels (imgui-bundle won't
            # compile from source on Windows without a C++ toolchain).
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--only-binary=:all:", *missing]
            )
            print("[OK] All packages installed successfully!\n" + "=" * 40)
        except subprocess.CalledProcessError as e:
            print(f"[X] Auto-installation failed: {e}")
            sys.exit(1)


auto_bootstrap()

import os
import re
import json
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
        tpat = re.compile(r"^\d+_\d+p?\d*(a|ma)$", re.I)
        targets = [c for c in cols
                   if tpat.match(c) and pd.api.types.is_numeric_dtype(df[c])]

    # --- id columns ---
    ids = [c for c in ["serial_orig", "serial_clean"] if c in cols]
    if not ids:
        for c in cols:
            if df[c].dtype == object:
                uniq_ratio = df[c].nunique(dropna=True) / n
                name_hint = "serial" in c.lower() or c.lower() in {"id", "index"} \
                    or c.lower().endswith("_id")
                if uniq_ratio > 0.9 and (name_hint or uniq_ratio >= 0.98):
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


def prepare_raw(df, ids, mixed):
    """
    Drop id columns and parse+drop every messy column into its own anonymized
    A/B/C triple. `mixed` may be a single name or a list of names.
    """
    df = df.drop(columns=ids, errors="ignore")

    # Accept either a string (one column) or a list (several).
    if isinstance(mixed, str):
        mixed = [mixed] if mixed else []
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


def build_Xy(cfg, notes=None):
    """
    Full preprocessing shared by training and the model-comparison benchmark:
    drop empty columns, drop rows with no target, drop blank/'Unknown'-only rows,
    prune sparse/high-cardinality features, impute, one-hot encode.

    `notes` (optional list) collects human-readable messages about what was dropped.
    Returns (X_encoded, y, numeric_schema, categorical_schema).
    """
    def log(msg):
        if notes is not None:
            notes.append(msg)

    df = read_any(cfg["data_path"], sheet=cfg.get("sheet"))
    df = prepare_raw(df, cfg["ids"], cfg["mixed"])
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

    targets = cfg["targets"]
    missing = [c for c in targets if c not in df.columns]
    if missing:
        raise ValueError(f"Target columns not found: {missing}")

    # Targets: coerce to numeric, then DROP rows with no label (never impute y).
    for col in targets:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=targets).reset_index(drop=True)
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
    df = df[frac >= MIN_INFORMATIVE_FRAC].reset_index(drop=True)
    log(f"Dropped {n1 - len(df)} blank/'Unknown'-only rows -> {len(df)} rows.")

    X = df[feat_only].copy()
    y = df[targets]

    # Feature selection: drop too-sparse and too-high-cardinality columns.
    dropped = 0
    for col in list(X.columns):
        if X[col].isna().mean() > MAX_MISSING_FRAC:
            X = X.drop(columns=col); dropped += 1
        elif not pd.api.types.is_numeric_dtype(X[col]) and \
                X[col].dropna().astype(str).nunique() > MAX_CATEGORIES:
            X = X.drop(columns=col); dropped += 1
    log(f"Feature selection: kept {X.shape[1]} columns (dropped {dropped} noisy).")

    # Impute the survivors and record the schema (for predict-tab widgets).
    numeric_schema, categorical_schema = {}, {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            med = X[col].median()
            med = 0.0 if pd.isna(med) else float(med)
            X[col] = X[col].fillna(med)
            numeric_schema[col] = med
        else:
            # 'not-done' -> explicit "None" category; unknown -> "Missing".
            X[col] = normalize_categorical_series(X[col])
            categorical_schema[col] = sorted(X[col].unique().tolist())

    X_enc = pd.get_dummies(X, drop_first=True)
    return X_enc, y, numeric_schema, categorical_schema


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
def run_capacity_optimization(data_path, target, excluded, fixed, direction,
                              min_support=3, random_state=RANDOM_STATE, sheet=None,
                              features=None, col_types=None, ids=None, mixed=None):
    """
    Train XGBoost to predict `target` from CONTROLLABLE inputs only, then use
    differential evolution to find the input recipe with the best predicted
    target. `fixed` pins chosen knobs (e.g. a test current density).

    Which columns are controllable is driven by the Column-types configuration
    when available: pass ``features`` (the columns tagged role='Feature') and
    ``col_types`` ({col: 'numeric'|'categorical'}). Everything else — targets,
    ids, excluded and measured-after-synthesis outcomes — is held out
    automatically. Without ``features`` it falls back to "everything except the
    target and the `excluded` list".

    Returns a result dict for the UI.
    """
    from xgboost import XGBRegressor

    df = read_any(data_path, sheet=sheet)
    # Drop id columns and split any messy columns into anonymized A/B/C features,
    # so the optimizer's knobs line up with the training pipeline.
    df = prepare_raw(df, ids or [], mixed or [])
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in the file.")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).reset_index(drop=True)

    col_types = col_types or {}
    generated = ("numeric_feature_", "group_label_", "text_modifier_")
    if features:
        # COLUMN-TYPE MODE: optimise only over role='Feature' columns (plus any
        # features generated from parsed messy columns). Targets / ids / excluded
        # / measured-outcome columns are automatically held out.
        feat_set = set(features)
        controllable = [c for c in df.columns
                        if c != target and (c in feat_set or c.startswith(generated))]
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
    numeric_cols, cat_choices = [], {}
    for c in controllable:
        kind = col_types.get(c)
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

    X_enc = pd.get_dummies(X_raw, drop_first=False)
    feat_cols = X_enc.columns.tolist()
    col_pos = {c: i for i, c in enumerate(feat_cols)}
    n_feat = len(feat_cols)

    model = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=random_state, n_jobs=-1, verbosity=0)
    cv_r2 = float(r2_score(y, cross_val_predict(
        model, X_enc.astype(float).values, y,
        cv=KFold(5, shuffle=True, random_state=random_state))))
    model.fit(X_enc.astype(float).values, y)

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
    for c in numeric_cols:
        if c in fixed:
            continue
        lo, hi = np.percentile(X_raw[c], 1), np.percentile(X_raw[c], 99)
        if lo == hi:
            hi = lo + 1e-6
        bounds.append((lo, hi)); integrality.append(False)
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
        return x

    def objective(vec):
        return sign * float(model.predict(vectorize(vec).reshape(1, -1))[0])

    result = differential_evolution(
        objective, bounds, integrality=integrality, seed=random_state,
        popsize=15, maxiter=80, tol=1e-4, mutation=(0.5, 1.0),
        recombination=0.9, polish=False, updating="immediate")

    # Decode the winning recipe (all controllable knobs, incl. fixed ones).
    recipe = dict(fixed)
    edges = []
    for v, s in zip(result.x, spec):
        if s[0] == "num":
            recipe[s[1]] = round(float(v), 4)
            lo, hi = np.percentile(X_raw[s[1]], 1), np.percentile(X_raw[s[1]], 99)
            span = (hi - lo) or 1.0
            if abs(v - lo) < 0.02 * span or abs(v - hi) < 0.02 * span:
                edges.append(s[1])
        else:
            i = max(0, min(int(round(v)), len(cat_choices[s[1]]) - 1))
            recipe[s[1]] = cat_choices[s[1]][i]
    predicted = float(model.predict(vectorize(result.x).reshape(1, -1))[0])

    ordered = [(c, recipe[c]) for c in controllable]
    return dict(target=target, r2=cv_r2, predicted=predicted,
                obs_min=float(y.min()), obs_max=float(y.max()),
                recipe=ordered, fixed=set(fixed), edges=edges,
                n_rows=len(df), n_knobs=len(controllable),
                mode=("column-type" if features else "manual"),
                n_numeric=len(numeric_cols), n_categorical=len(cat_choices))


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

        # --- Worksheet/tab selection for multi-sheet .xlsx files ---
        self.sheet_names = []              # tabs in the chosen data file ([] = single/CSV)
        self.sheet_idx = 0                 # which tab is selected
        self.batch_sheet_names = []        # tabs in the batch-predict file
        self.batch_sheet_idx = 0

        # --- File dialogs (async) ---
        self.data_dialog = None
        self.batch_dialog = None
        self.json_dialog = None

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
        self.metrics = []                  # [(target, tr_r2, tr_rmse, cv_r2, cv_rmse), ...]
        self.importances = []             # [(feature, importance), ...]
        self.summary = ""

        # --- Predict tab widget values ---
        self.numeric_values = {}           # {col: float}
        self.category_index = {}           # {col: chosen index}
        self.single_pred = None            # [(target, value), ...]
        self.predict_error = ""

        # --- Batch tab ---
        self.batch_path = ""
        self.batch_status = ""
        self.batch_error = ""
        self.batch_results = None

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
        n_num = sum(1 for v in new_map.values() if v == "numeric")
        n_tgt = sum(1 for v in new_role.values() if v == "target")
        STATE.coltype_status = (
            f"Scanned {len(cols)} column(s) · {n_num} numeric / "
            f"{len(cols) - n_num} categorical · {n_tgt} target(s). Adjust below, then Train."
        )
    except Exception as e:  # noqa: BLE001
        STATE.coltype_status = f"Scan failed: {e}"


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
        spec = json.load(f)

    feats = spec.get("features", {}) or {}
    categorical = [str(c) for c in feats.get("categorical", [])]
    numerical = [str(c) for c in feats.get("numerical", [])]
    targets = [str(c) for c in spec.get("targets", [])]
    if not targets and spec.get("target"):
        targets = [str(spec["target"])]
    excluded = [str(c) for c in spec.get("exclude", [])]

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

    # Make sure every JSON-named column is present in the table + config, keeping
    # any already-scanned order first, then appending anything new.
    referenced = numerical + categorical + targets + excluded
    seen = set(STATE.coltype_columns)
    for c in referenced:
        if c not in seen:
            STATE.coltype_columns.append(c)
            seen.add(c)
            STATE.coltype_map.setdefault(c, "categorical")

    sync_roles_to_cfg()
    tab = f" · tab '{dataset}'" if dataset and dataset in STATE.sheet_names else ""
    STATE.coltype_status = (
        f"Loaded JSON: {len(targets)} target(s), {len(numerical)} numeric + "
        f"{len(categorical)} categorical feature(s), {len(excluded)} excluded{tab}."
    )


def start_training():
    """Kick off training on a background thread so the UI stays responsive."""
    if STATE.is_training or not STATE.data_path:
        return
    STATE.is_training = True
    STATE.trained = False
    STATE.train_error = ""
    STATE.progress = 0.0
    STATE.status = "Starting…"

    # Snapshot config so the thread doesn't read fields being edited mid-run.
    cfg = dict(
        data_path=STATE.data_path,
        ids=_split_cols(STATE.id_columns),
        mixed=_split_cols(STATE.mixed_column),   # list: supports several messy columns
        targets=_split_cols(STATE.target_columns),
        col_types=dict(STATE.coltype_map),       # manual numeric/categorical overrides
        exclude=_split_cols(STATE.exclude_columns),  # cols dropped from training
        sheet=_current_sheet(),                  # selected worksheet/tab
    )
    threading.Thread(target=_train_worker, args=(cfg,), daemon=True).start()


def _train_worker(cfg):
    try:
        STATE.status = "Loading & cleaning data…"
        STATE.progress = 0.2
        targets = cfg["targets"]

        # Shared cleaning: drop empty cols, no-target rows, blank rows, prune features.
        clean_notes = []
        X_enc, y, numeric_schema, categorical_schema = build_Xy(cfg, notes=clean_notes)
        X_enc = X_enc.astype(float)
        feature_columns = X_enc.columns.tolist()

        # No rows survived cleaning — explain WHY instead of the cryptic split error.
        if len(X_enc) < 5:
            raise ValueError(
                f"Only {len(X_enc)} usable row(s) after cleaning — can't train. "
                f"Usually the Target column is wrong or not numeric (every row whose "
                f"target isn't a number is dropped). Check the Target column(s): "
                f"{', '.join(targets)}.  " + "  ".join(clean_notes)
            )

        # The winning architecture: tuned MultiOutputRegressor(ExtraTrees).
        model = MultiOutputRegressor(
            ExtraTreesRegressor(
                n_estimators=300, max_depth=12, min_samples_split=4,
                max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1,
            )
        )

        # EVALUATION: 5-fold out-of-fold cross-validation. Every row is predicted
        # by a model that never saw it, giving a stable generalization estimate
        # (matches the Compare-models tab) instead of one volatile hold-out split.
        STATE.status = "Evaluating with 5-fold cross-validation…"
        STATE.progress = 0.5
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        pred_cv = cross_val_predict(model, X_enc, y, cv=cv, n_jobs=-1)

        # Fit the DEPLOYABLE model on ALL rows for the Predict tab + in-sample fit.
        STATE.status = "Fitting final model on all rows…"
        STATE.progress = 0.9
        model.fit(X_enc, y)
        pred_tr = model.predict(X_enc)   # in-sample (training-fit) predictions
        importances = np.mean(
            [est.feature_importances_ for est in model.estimators_], axis=0
        )
        imp_series = pd.Series(importances, index=X_enc.columns).sort_values(
            ascending=False
        )

        # Per-target metrics: in-sample train fit vs 5-fold cross-validation.
        # A large train-vs-CV gap still flags overfitting.
        metrics = []
        for i, col in enumerate(targets):
            tr_r2 = r2_score(y.iloc[:, i], pred_tr[:, i])
            tr_rmse = float(np.sqrt(mean_squared_error(y.iloc[:, i], pred_tr[:, i])))
            cv_r2 = r2_score(y.iloc[:, i], pred_cv[:, i])
            cv_rmse = float(np.sqrt(mean_squared_error(y.iloc[:, i], pred_cv[:, i])))
            metrics.append((col, tr_r2, tr_rmse, cv_r2, cv_rmse))

        # Publish results to the UI.
        STATE.model = model
        STATE.feature_columns = feature_columns
        STATE.numeric_schema = numeric_schema
        STATE.categorical_schema = categorical_schema
        STATE.targets = targets
        STATE.cfg = {"ids": cfg["ids"], "mixed": cfg["mixed"]}
        STATE.metrics = metrics
        STATE.importances = list(imp_series.items())

        # Keep the training data + CV metrics for the screening engine
        # (applicability domain, similar experiments, uncertainty intervals).
        STATE.X_train = X_enc
        STATE.y_train = y.reset_index(drop=True)
        STATE.cv_rmse = {col: m[4] for col, m in zip(targets, metrics)}
        STATE.cv_r2 = {col: m[3] for col, m in zip(targets, metrics)}
        rebuild_screener()
        mean_tr = float(np.mean([m[1] for m in metrics]))
        mean_cv = float(np.mean([m[3] for m in metrics]))
        STATE.summary = (
            f"ExtraTrees · 5-fold CV on {len(X_enc)} rows.  "
            f"Train R2 = {mean_tr:.3f}  ·  CV R2 = {mean_cv:.3f}  ·  "
            f"{len(feature_columns)} encoded features.  "
            + "  ".join(clean_notes)
        )

        # Seed the predict-tab widgets with sensible defaults.
        STATE.numeric_values = dict(numeric_schema)
        STATE.category_index = {c: 0 for c in categorical_schema}
        STATE.screen_custom_category = {c: "" for c in categorical_schema}
        STATE.single_pred = None
        STATE.predict_error = ""

        STATE.trained = True
        STATE.status = "Done. See metrics below, then use the Predict tab."
        STATE.progress = 1.0
    except Exception as e:  # noqa: BLE001
        STATE.train_error = f"{type(e).__name__}: {e}"
        STATE.status = "Training failed."
        STATE.progress = 0.0
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

    cfg = dict(
        data_path=STATE.data_path,
        ids=_split_cols(STATE.id_columns),
        mixed=_split_cols(STATE.mixed_column),
        targets=_split_cols(STATE.target_columns),
        col_types=dict(STATE.coltype_map),       # manual numeric/categorical overrides
        exclude=_split_cols(STATE.exclude_columns),  # cols dropped from training
        sheet=_current_sheet(),                  # selected worksheet/tab
    )
    threading.Thread(target=_compare_worker, args=(cfg,), daemon=True).start()


def _compare_worker(cfg):
    try:
        X, y, _, _ = build_Xy(cfg)
        X = X.astype(float)  # get_dummies yields bool columns; models want floats
        models = build_models()
        STATE.compare_total = len(models)
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for name, model in models.items():
            STATE.compare_current = name
            STATE.compare_status = f"Evaluating {name}  ({STATE.compare_done + 1}/{STATE.compare_total})…"
            t0 = time.time()
            score = rmse = mae = train_r2 = float("nan")
            predict_ms = float("nan")
            try:
                oof = cross_val_predict(model, X, y, cv=kf)
                score = float(r2_score(y, oof, multioutput="uniform_average"))
                rmse = float(np.sqrt(mean_squared_error(y, oof)))
                mae = float(np.mean(np.abs(y.values - oof)))
                cv_secs = time.time() - t0
                # Train R² (in-sample) + a timed prediction pass for latency.
                model.fit(X, y)
                train_r2 = float(r2_score(y, model.predict(X),
                                          multioutput="uniform_average"))
                tp = time.time()
                model.predict(X)
                predict_ms = 1000.0 * (time.time() - tp) / max(len(X), 1)
            except Exception as e:  # noqa: BLE001 - one bad model shouldn't stop the rest
                cv_secs = time.time() - t0
                print(f"[compare] {name} failed: {e}")
            STATE.compare_results.append(
                (name, score, cv_secs, rmse, mae, train_r2, predict_ms))
            STATE.compare_done += 1

        STATE.compare_current = ""
        try:
            import charts as C
            os.makedirs(CHART_DIR, exist_ok=True)
            STATE.compare_run += 1
            STATE.compare_chart_path = C.model_comparison_plot(
                STATE.compare_results,
                f"{CHART_DIR}/compare_models_{STATE.compare_run}.png")
        except Exception as e:  # noqa: BLE001
            print(f"[compare chart] failed: {e}")
        STATE.compare_status = f"Done. Benchmarked {STATE.compare_total} architectures."
    except Exception as e:  # noqa: BLE001
        STATE.compare_error = f"{type(e).__name__}: {e}"
        STATE.compare_status = "Comparison failed."
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
    features = col_types = None
    if STATE.coltype_columns:
        features = [c for c in STATE.coltype_columns
                    if STATE.coltype_role.get(c) == "feature"]
        col_types = dict(STATE.coltype_map)

    cfg = dict(
        data_path=STATE.data_path,
        target=STATE.opt_target.strip(),
        excluded=_split_cols(STATE.opt_excluded),
        fixed=_parse_fixed(STATE.opt_fixed),
        direction="maximise" if STATE.opt_direction_idx == 0 else "minimise",
        sheet=_current_sheet(),                  # selected worksheet/tab
        features=features,
        col_types=col_types,
        ids=_split_cols(STATE.id_columns),
        mixed=_split_cols(STATE.mixed_column),
    )
    threading.Thread(target=_optimize_worker, args=(cfg,), daemon=True).start()


def _optimize_worker(cfg):
    try:
        STATE.opt_status = "Training + searching (this can take a minute)…"
        res = run_capacity_optimization(
            cfg["data_path"], cfg["target"], cfg["excluded"],
            cfg["fixed"], cfg["direction"], sheet=cfg.get("sheet"),
            features=cfg.get("features"), col_types=cfg.get("col_types"),
            ids=cfg.get("ids"), mixed=cfg.get("mixed"))
        STATE.opt_result = res
        src = ("column-type roles" if res.get("mode") == "column-type"
               else "manual exclusions")
        STATE.opt_status = (
            f"Done ({src}). Model R²={res['r2']:.2f} on {res['n_rows']} rows, "
            f"{res['n_knobs']} knobs "
            f"({res.get('n_numeric', 0)} numeric / {res.get('n_categorical', 0)} categorical).")
    except Exception as e:  # noqa: BLE001
        STATE.opt_error = f"{type(e).__name__}: {e}"
        STATE.opt_status = "Optimization failed."
    finally:
        STATE.is_optimizing = False


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

    cfg = dict(
        data_path=STATE.data_path,
        ids=_split_cols(STATE.id_columns),
        mixed=_split_cols(STATE.mixed_column),
        targets=_split_cols(STATE.target_columns) or list(STATE.targets),
        col_types=dict(STATE.coltype_map),
        exclude=_split_cols(STATE.exclude_columns),
        sheet=_current_sheet(),                  # selected worksheet/tab
    )
    numeric_feats = list(STATE.numeric_schema.keys())

    def pick(idx):
        return numeric_feats[idx] if 0 <= idx < len(numeric_feats) else None

    opts = dict(
        target_idx=STATE.charts_target_idx,
        pareto_a=STATE.charts_pareto_a,
        pareto_b=STATE.charts_pareto_b,
        featx=pick(STATE.charts_featx_idx),
        featy=pick(STATE.charts_featy_idx),
        imp_top=[5, 10, 20, 0][min(max(STATE.charts_imp_top_idx, 0), 3)],
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

        STATE.charts_status = "Loading & cleaning data…"
        X_enc, y, numeric_schema, categorical_schema = build_Xy(cfg)
        X_enc = X_enc.astype(float)
        numeric_feats = list(numeric_schema.keys())

        # Correlation and target-signal analysis.
        STATE.charts_status = "1/13 · correlation analysis…"
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
        STATE.charts_status = "6/13 · out-of-fold predictions…"
        oof_model = MultiOutputRegressor(ExtraTreesRegressor(
            n_estimators=300, max_depth=12, min_samples_split=4,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1))
        oof = cross_val_predict(
            oof_model, X_enc, y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
        r2_by = {t: float(r2_score(y.iloc[:, i], oof[:, i])) for i, t in enumerate(targets)}

        items.append(("Predicted vs actual",
                      C.predicted_vs_actual(y.values, oof, targets, path("pva"), r2_by)))
        STATE.charts_status = "7/13 · residual diagnostics…"
        items.append(("Residual plot",
                      C.residual_plot(y.values, oof, targets, path("resid"))))
        items.append(("Residual distribution",
                      C.residual_distribution(y.values, oof, targets, path("resid_dist"))))

        # 4. Feature importance (folded back to source columns).
        STATE.charts_status = "9/13 · feature importance…"
        imp_src = _aggregate_importance(STATE.importances, numeric_schema, categorical_schema)
        top = opts.get("imp_top", 20)
        items.append(("Feature importance", C.feature_importance(
            imp_src, path("imp"), top=top,
            title="Grouped feature importance"
            + (" (all features)" if not top else f" (top {top})"))))

        # 5 & 6. SHAP on the trained single-target estimator.
        Xs_shap = X_enc.reindex(columns=STATE.feature_columns, fill_value=0)
        est = STATE.model.estimators_[ti] if hasattr(STATE.model, "estimators_") else STATE.model
        STATE.charts_status = "10/13 · SHAP summary (can be slow)…"
        items.append(("SHAP summary",
                      C.shap_summary(est, Xs_shap, path("shap_sum"), targets[ti])))
        STATE.charts_status = "11/13 · SHAP dependence…"
        items.append(("SHAP dependence",
                      C.shap_dependence(est, Xs_shap, path("shap_dep"), targets[ti])))

        # 7. Optimization heatmap — sweep two numeric knobs through the model.
        STATE.charts_status = "12/13 · optimization heatmap…"
        fx, fy = opts.get("featx"), opts.get("featy")
        if fx not in numeric_feats or fy not in numeric_feats or fx == fy:
            ranked = [f for f in sorted(imp_src, key=lambda k: -imp_src[k])
                      if f in numeric_feats]
            picks = (ranked or numeric_feats)[:2]
            fx = fx if fx in numeric_feats else (picks[0] if picks else None)
            fy = fy if fy in numeric_feats and fy != fx else (
                picks[1] if len(picks) > 1 else None)

        def predict_fn(raw_df):
            return STATE.model.predict(build_matrix(raw_df))

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
        STATE.charts_status = "13/13 · Pareto front…"
        if len(targets) >= 2:
            a = targets[min(max(opts["pareto_a"], 0), len(targets) - 1)]
            b = targets[min(max(opts["pareto_b"], 0), len(targets) - 1)]
            items.append(("Pareto front", C.pareto_front(y, a, b, path("pareto"))))
        else:
            items.append(("Pareto front", C._placeholder(
                path("pareto"), "Pareto front", "Need at least two targets.")))

        STATE.chart_items = items
        STATE.charts_status = f"Done. Generated {len(items)} charts."
    except Exception as e:  # noqa: BLE001
        STATE.charts_error = f"{type(e).__name__}: {e}"
        STATE.charts_status = "Chart generation failed."
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

        def prog(msg):
            STATE.intel_status = msg

        prog("Loading data…")
        df = read_any(STATE.data_path, sheet=opts["sheet"])
        if cfg.target not in df.columns:
            raise ValueError(f"Target '{cfg.target}' not found in the sheet.")
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
    except Exception as e:  # noqa: BLE001
        STATE.intel_error = f"{type(e).__name__}: {e}"
        STATE.intel_status = "Dataset Intelligence failed."
    finally:
        STATE.is_intel_running = False


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
    X_enc = pd.get_dummies(X_raw, drop_first=False)
    # Overlapping names (e.g. 'Electrolyte' vs 'Electrolyte_Additive') or odd cell
    # values can yield duplicate dummy columns; keep the first so reindex is safe.
    X_enc = X_enc.loc[:, ~X_enc.columns.duplicated()]
    return X_enc.reindex(columns=STATE.feature_columns, fill_value=0)


def predict_single():
    """Predict all targets from the current Predict-tab widget values."""
    STATE.single_pred = None
    STATE.predict_error = ""
    row = dict(STATE.numeric_values)
    for col, choices in STATE.categorical_schema.items():
        row[col] = choices[STATE.category_index[col]]
    try:
        X_raw = pd.DataFrame([row])
        preds = STATE.model.predict(build_matrix(X_raw))[0]
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

        # Rebuild the raw feature frame using the training schema + the same
        # 'not-done' (-> 0 / "None") vs unknown (-> impute / "Missing") semantics.
        cols = {}
        for col, med in STATE.numeric_schema.items():
            series = prepared.get(col)
            if series is None:
                series = pd.Series([med] * len(prepared))
            cols[col] = coerce_numeric_series(series).fillna(med)
        for col in STATE.categorical_schema:
            series = prepared.get(col)
            if series is None:
                series = pd.Series(["Missing"] * len(prepared))
            cols[col] = normalize_categorical_series(series)
        X_raw = pd.DataFrame(cols)

        preds = STATE.model.predict(build_matrix(X_raw))
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
            # Training data + CV metrics let a reloaded model still screen
            # (applicability domain, similar experiments, uncertainty).
            "X_train": STATE.X_train,
            "y_train": STATE.y_train,
            "cv_rmse": STATE.cv_rmse,
            "cv_r2": STATE.cv_r2,
            "metrics": STATE.metrics,
            "importances": STATE.importances,
            "summary": STATE.summary,
        },
        MODEL_OUT,
    )


def load_model():
    b = joblib.load(MODEL_OUT)
    STATE.model = b["model"]
    STATE.feature_columns = b["feature_columns"]
    STATE.numeric_schema = b["numeric_schema"]
    STATE.categorical_schema = b["categorical_schema"]
    STATE.targets = b["targets"]
    STATE.cfg = b.get("cfg", {"ids": [], "mixed": ""})
    STATE.X_train = b.get("X_train")
    STATE.y_train = b.get("y_train")
    STATE.cv_rmse = b.get("cv_rmse", {})
    STATE.cv_r2 = b.get("cv_r2", {})
    STATE.metrics = b.get("metrics", [])
    STATE.importances = b.get("importances", [])
    STATE.summary = b.get("summary", "")
    STATE.numeric_values = dict(STATE.numeric_schema)
    STATE.category_index = {c: 0 for c in STATE.categorical_schema}
    STATE.screen_custom_category = {c: "" for c in STATE.categorical_schema}
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
        if kind == "pdf":
            report.build_prediction_pdf(path, STATE.screen_result, STATE.screener,
                                        model_summary=STATE.summary)
        else:
            report.export_prediction_excel(path, STATE.screen_result, STATE.screener)
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

    imgui.dummy(imgui.ImVec2(0, 6))
    imgui.text("3) Train")
    imgui.text_colored(DIM, "Evaluated with 5-fold cross-validation (stable on small data).")
    # Guard clicks while a run is in progress.
    if imgui.button("Train model", size=imgui.ImVec2(140, 0)):
        start_training()
    imgui.same_line()
    if STATE.trained and imgui.button("Save model", size=imgui.ImVec2(120, 0)):
        save_model()
        STATE.status = f"Saved to '{MODEL_OUT}'."

    # Progress + status line.
    imgui.dummy(imgui.ImVec2(0, 4))
    imgui.progress_bar(STATE.progress, imgui.ImVec2(-1, 0))
    imgui.text_colored(RED if STATE.train_error else DIM, STATE.status)
    if STATE.train_error:
        imgui.text_wrapped(STATE.train_error)

    # Results.
    if STATE.trained:
        imgui.separator()
        imgui.text_colored(GREEN, STATE.summary)

        imgui.dummy(imgui.ImVec2(0, 4))
        imgui.text("Per-target performance (in-sample train vs 5-fold CV)")
        flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
        if imgui.begin_table("metrics", 5, flags):
            for h in ("Target", "Train R2", "Train RMSE", "CV R2", "CV RMSE"):
                imgui.table_setup_column(h)
            imgui.table_headers_row()
            for name, tr_r2, tr_rmse, cv_r2, cv_rmse in STATE.metrics:
                imgui.table_next_row()
                imgui.table_next_column(); imgui.text(name)
                imgui.table_next_column(); imgui.text(f"{tr_r2:.4f}")
                imgui.table_next_column(); imgui.text(f"{tr_rmse:.4f}")
                # Colour CV R2 by quality; a big train-vs-CV gap flags overfitting.
                imgui.table_next_column()
                imgui.text_colored(GREEN if cv_r2 >= 0.5 else (RED if cv_r2 < 0 else DIM),
                                   f"{cv_r2:.4f}")
                imgui.table_next_column(); imgui.text(f"{cv_rmse:.4f}")
            imgui.end_table()

        imgui.dummy(imgui.ImVec2(0, 4))
        imgui.text("Top 5 most influential features")
        if imgui.begin_table("imp", 2, flags):
            imgui.table_setup_column("Feature")
            imgui.table_setup_column("Importance")
            imgui.table_headers_row()
            for feat, imp in STATE.importances[:5]:
                imgui.table_next_row()
                imgui.table_next_column(); imgui.text(str(feat))
                imgui.table_next_column(); imgui.text(f"{imp * 100:.2f}%")
            imgui.end_table()


def draw_predict_tab():
    if not STATE.trained:
        imgui.text_colored(DIM, "Train (or load) a model first — see the Train tab.")
        return

    imgui.text("Enter feature values")
    imgui.separator()

    if imgui.begin_child("inputs", imgui.ImVec2(0, 300)):
        for col in STATE.numeric_schema:
            changed, val = imgui.input_float(col, float(STATE.numeric_values[col]))
            if changed:
                STATE.numeric_values[col] = val
        for col, choices in STATE.categorical_schema.items():
            changed, idx = imgui.combo(col, STATE.category_index[col], choices)
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
    imgui.text("Benchmark multiple architectures with 5-fold cross-validation")
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

    if imgui.button("Run 5-fold comparison", size=imgui.ImVec2(220, 0)):
        start_comparison()

    # Progress: fraction of models completed.
    frac = (STATE.compare_done / STATE.compare_total) if STATE.compare_total else 0.0
    imgui.progress_bar(frac, imgui.ImVec2(-1, 0),
                       f"{STATE.compare_done}/{STATE.compare_total}" if STATE.compare_total else "")
    imgui.text_colored(RED if STATE.compare_error else DIM, STATE.compare_status)
    if STATE.compare_error:
        imgui.text_wrapped(STATE.compare_error)
    if STATE.is_comparing and STATE.compare_current:
        imgui.text_colored(DIM, f"  … running {STATE.compare_current}")

    # Live ranked table (best R2 first). NaN = the model errored out.
    if STATE.compare_results:
        imgui.separator()
        imgui.text("Ranking (highest average CV R2 first)")
        flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
        if imgui.begin_table("compare", 7, flags):
            for h in ("#", "Model", "CV R2", "CV RMSE", "CV MAE",
                      "Train R2", "Predict"):
                imgui.table_setup_column(h)
            imgui.table_headers_row()
            ranked = sorted(
                STATE.compare_results,
                key=lambda r: (r[1] if r[1] == r[1] else -1e9),  # NaN sinks to bottom
                reverse=True,
            )
            for rank, row in enumerate(ranked, start=1):
                name, score, secs, rmse, mae, train_r2, predict_ms = row
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
                imgui.table_next_column(); imgui.text("-" if rmse != rmse else f"{rmse:.3f}")
                imgui.table_next_column(); imgui.text("-" if mae != mae else f"{mae:.3f}")
                imgui.table_next_column()
                imgui.text("-" if train_r2 != train_r2 else f"{train_r2:.4f}")
                imgui.table_next_column()
                imgui.text("-" if predict_ms != predict_ms else f"{predict_ms:.2f} ms/row")
            imgui.end_table()
            imgui.text_colored(DIM, "CV = 5-fold cross-validation (out-of-fold). "
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
    tgts = [c for c in STATE.coltype_columns if role.get(c) == "target"] if has_roles else []
    held = [c for c in STATE.coltype_columns
            if role.get(c) in ("target", "id", "exclude", "messy")] if has_roles else []

    if has_roles:
        imgui.text_colored(GREEN, f"Using Column-types roles: {len(feats)} feature knob(s) "
                           f"to optimize.")
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
        imgui.set_next_item_width(360)
        _, STATE.opt_target = imgui.input_text("Target to optimise", STATE.opt_target)

    imgui.set_next_item_width(360)
    _, STATE.opt_direction_idx = imgui.combo("Direction", STATE.opt_direction_idx,
                                             ["maximise", "minimise"])
    imgui.set_next_item_width(360)
    _, STATE.opt_fixed = imgui.input_text("Fixed knobs (col=value, comma-sep)", STATE.opt_fixed)

    # The manual exclusion list is only used as a fallback (no roles configured).
    if not has_roles:
        imgui.text_colored(DIM, "Excluded (measured / outcome) columns — one big list:")
        _, STATE.opt_excluded = imgui.input_text_multiline(
            "##excluded", STATE.opt_excluded, imgui.ImVec2(-1, 80))
    elif imgui.tree_node("Advanced: manual exclusions (ignored in column-type mode)"):
        _, STATE.opt_excluded = imgui.input_text_multiline(
            "##excluded", STATE.opt_excluded, imgui.ImVec2(-1, 80))
        imgui.tree_pop()

    if imgui.button("Run optimization", size=imgui.ImVec2(200, 0)):
        start_optimize()
    imgui.text_colored(RED if STATE.opt_error else DIM, STATE.opt_status)
    if STATE.opt_error:
        imgui.text_wrapped(STATE.opt_error)

    r = STATE.opt_result
    if r is not None:
        imgui.separator()
        verdict = GREEN if r["predicted"] <= r["obs_max"] * 1.05 else (0.9, 0.7, 0.2, 1.0)
        imgui.text_colored(verdict,
                           f"Predicted {r['target']} = {r['predicted']:.0f}   "
                           f"(observed {r['obs_min']:.0f}–{r['obs_max']:.0f})")
        imgui.text_colored(DIM, f"Model 5-fold R² = {r['r2']:.2f} — treat as a hypothesis to test.")
        if r["edges"]:
            imgui.text_colored((0.9, 0.7, 0.2, 1.0),
                               "Extrapolation risk (knob at edge of data): " + ", ".join(r["edges"]))

        imgui.dummy(imgui.ImVec2(0, 4))
        imgui.text("Recommended conditions")
        flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
        if imgui.begin_table("recipe", 2, flags, outer_size=imgui.ImVec2(0, 320)):
            imgui.table_setup_column("Knob")
            imgui.table_setup_column("Value")
            imgui.table_headers_row()
            for name, val in r["recipe"]:
                imgui.table_next_row()
                imgui.table_next_column()
                if name in r["fixed"]:
                    imgui.text_colored(DIM, name + "  (fixed)")
                else:
                    imgui.text(name)
                imgui.table_next_column()
                imgui.text(f"{val:g}" if isinstance(val, float) else str(val))
            imgui.end_table()


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
    _, STATE.charts_imp_top_idx = imgui.combo(
        "Feature importance", STATE.charts_imp_top_idx,
        ["Top 5", "Top 10", "Top 20", "All features"])

    imgui.dummy(imgui.ImVec2(0, 4))
    if imgui.button("Generate charts", size=imgui.ImVec2(200, 0)):
        start_charts()
    imgui.text_colored(RED if STATE.charts_error else DIM, STATE.charts_status)
    if STATE.charts_error:
        imgui.text_wrapped(STATE.charts_error)

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
    if imgui.button("Run analysis", size=imgui.ImVec2(200, 0)):
        start_intelligence()
    if STATE.intel_results:
        imgui.same_line()
        if imgui.button("Export PDF report", size=imgui.ImVec2(180, 0)):
            intel_export_pdf()
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
        for col in STATE.numeric_schema:
            changed, val = imgui.input_float(col, float(STATE.numeric_values[col]))
            if changed:
                STATE.numeric_values[col] = val
                STATE.screen_dirty = True
        for col, choices in STATE.categorical_schema.items():
            STATE.category_index[col] = min(STATE.category_index.get(col, 0),
                                            max(len(choices) - 1, 0))
            changed, idx = imgui.combo(col, STATE.category_index[col], choices)
            if changed:
                STATE.category_index[col] = idx
                STATE.screen_dirty = True
            current = STATE.screen_custom_category.get(col, "")
            imgui.set_next_item_width(-1)
            changed, custom = imgui.input_text(f"Other / unknown {col}", current)
            if changed:
                STATE.screen_custom_category[col] = custom
                STATE.screen_dirty = True
    imgui.end_child()


def _draw_similar_table(res, scr):
    target = res["target"]
    sims = res["similar"]
    if not sims:
        return
    # Show the most influential synthesis variables as context columns.
    disp = [f for f, _ in scr.importance[:4] if f in sims[0]["conditions"]]
    flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
    ncol = 2 + len(disp)
    if imgui.begin_table("sim", ncol, flags):
        imgui.table_setup_column("Similar")
        imgui.table_setup_column("Measured")
        for d in disp:
            imgui.table_setup_column(screening._pretty(d))
        imgui.table_headers_row()
        for s in sims:
            imgui.table_next_row()
            imgui.table_next_column()
            sim = s["similarity"]
            imgui.text_colored(GREEN if sim >= 70 else (ORANGE if sim >= 40 else DIM),
                               f"{sim:.0f}%")
            imgui.table_next_column()
            imgui.text(f"{s['measured'][target]:.1f}")
            for d in disp:
                imgui.table_next_column()
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
    tname = screening._pretty(target)

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
                imgui.text(" " + screening._pretty(name))

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
                    imgui.table_next_column(); imgui.text(screening._pretty(e["feature"]))
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
                imgui.table_setup_column(screening._pretty(c))
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


def gui():
    """Top-level GUI callback — called every frame by imgui-bundle."""
    imgui.text("🌿 BioCarbon Screen — AI-assisted hard-carbon synthesis screening")
    imgui.same_line()
    # Offer to reuse a previously-saved model.
    if not STATE.trained and os.path.exists(MODEL_OUT):
        if imgui.small_button(f"Load {MODEL_OUT}"):
            try:
                load_model()
            except Exception as e:  # noqa: BLE001
                STATE.train_error = f"Could not load model: {e}"
    imgui.separator()

    if imgui.begin_tab_bar("tabs"):
        # PRIMARY WORKFLOW first — the research-assistant screening view.
        if imgui.begin_tab_item("🔬 Screen")[0]:
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


def main():
    # image_from_asset resolves paths relative to the assets folder; point it at
    # the project directory so "charts/xxx.png" loads.
    hello_imgui.set_assets_folder(os.path.dirname(os.path.abspath(__file__)))
    immapp.run(
        gui_function=gui,
        window_title="BioCarbon Screen",
        window_size=(1080, 820),
    )


if __name__ == "__main__":
    main()
