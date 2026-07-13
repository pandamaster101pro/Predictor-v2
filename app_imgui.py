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
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
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

MODEL_OUT = "model.joblib"
RANDOM_STATE = 42

# --- Data-cleaning knobs (mirror train_model.py) ---
BLANK_TOKENS = {"", "unknown", "nan", "none", "na", "missing", "missing or unspecified"}
MIN_INFORMATIVE_FRAC = 0.10   # drop a row if <10% of its feature cells carry real info
MAX_MISSING_FRAC = 0.60       # drop a feature if missing in >60% of the kept rows
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
            df[col] = pd.to_numeric(df[col], errors="coerce")
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
        hits = sum(0 if ((v != v) or str(v).strip().lower() in BLANK_TOKENS) else 1
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
            X[col] = X[col].astype(str).replace(
                {"nan": "Missing", "None": "Missing"}
            ).fillna("Missing")
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
                              min_support=3, random_state=RANDOM_STATE, sheet=None):
    """
    Train XGBoost to predict `target` from CONTROLLABLE inputs only (everything
    except `target` and the `excluded` measured/outcome columns), then use
    differential evolution to find the input recipe with the best predicted
    target. `fixed` pins chosen knobs (e.g. a test current density).

    Returns a result dict for the UI.
    """
    from xgboost import XGBRegressor

    df = read_any(data_path, sheet=sheet)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in the file.")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).reset_index(drop=True)

    excluded = set(excluded) | {target}
    controllable = [c for c in df.columns if c not in excluded]
    fixed = {k: v for k, v in fixed.items() if k in controllable}
    y = df[target].values

    # Split knobs into numeric ranges and categorical choices.
    X_raw = df[controllable].copy()
    numeric_cols, cat_choices = [], {}
    for c in controllable:
        if pd.api.types.is_numeric_dtype(X_raw[c]):
            X_raw[c] = X_raw[c].fillna(X_raw[c].median())
            numeric_cols.append(c)
        else:
            X_raw[c] = X_raw[c].astype(str).fillna("NA")
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
                n_rows=len(df), n_knobs=len(controllable))


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
        self.test_size = 0.2               # held-out test fraction for train/test eval
        self.metrics = []                  # [(target, tr_r2, tr_rmse, te_r2, te_rmse), ...]
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
        self.is_charting = False
        self.charts_status = "Train (or load) a model, then Generate charts."
        self.charts_error = ""
        self.chart_items = []              # [(title, rel_path), ...] rendered PNGs
        self.charts_run = 0                # bumped each run -> unique names (texture cache)


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
        test_size=float(STATE.test_size),        # held-out fraction for train/test eval
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

        # EVALUATION: hold out a random test split, fit on the TRAINING rows only,
        # then score both the training rows and the held-out test rows. Comparing
        # the two shows the train-vs-test gap (a large gap flags overfitting).
        test_size = float(cfg.get("test_size", 0.2))
        pct_tr = int(round((1 - test_size) * 100))
        pct_te = int(round(test_size * 100))
        STATE.status = f"Evaluating with a {pct_tr}/{pct_te} train/test split…"
        STATE.progress = 0.55
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_enc, y, test_size=test_size, random_state=RANDOM_STATE
        )
        model.fit(X_tr, y_tr)
        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)
        metrics = []
        for i, col in enumerate(targets):
            tr_r2 = r2_score(y_tr.iloc[:, i], pred_tr[:, i])
            tr_rmse = float(np.sqrt(mean_squared_error(y_tr.iloc[:, i], pred_tr[:, i])))
            te_r2 = r2_score(y_te.iloc[:, i], pred_te[:, i])
            te_rmse = float(np.sqrt(mean_squared_error(y_te.iloc[:, i], pred_te[:, i])))
            metrics.append((col, tr_r2, tr_rmse, te_r2, te_rmse))

        # Fit the DEPLOYABLE model on ALL rows (train+test) for the Predict tab.
        STATE.status = "Fitting final model on all rows…"
        STATE.progress = 0.9
        model.fit(X_enc, y)
        importances = np.mean(
            [est.feature_importances_ for est in model.estimators_], axis=0
        )
        imp_series = pd.Series(importances, index=X_enc.columns).sort_values(
            ascending=False
        )

        # Publish results to the UI.
        STATE.model = model
        STATE.feature_columns = feature_columns
        STATE.numeric_schema = numeric_schema
        STATE.categorical_schema = categorical_schema
        STATE.targets = targets
        STATE.cfg = {"ids": cfg["ids"], "mixed": cfg["mixed"]}
        STATE.metrics = metrics
        STATE.importances = list(imp_series.items())
        mean_tr = float(np.mean([m[1] for m in metrics]))
        mean_te = float(np.mean([m[3] for m in metrics]))
        STATE.summary = (
            f"ExtraTrees · {len(X_tr)} train / {len(X_te)} test rows.  "
            f"Train R2 = {mean_tr:.3f}  ·  Test R2 = {mean_te:.3f}  ·  "
            f"{len(feature_columns)} encoded features.  "
            + "  ".join(clean_notes)
        )

        # Seed the predict-tab widgets with sensible defaults.
        STATE.numeric_values = dict(numeric_schema)
        STATE.category_index = {c: 0 for c in categorical_schema}
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
            try:
                oof = cross_val_predict(model, X, y, cv=kf)
                score = float(r2_score(y, oof, multioutput="uniform_average"))
            except Exception as e:  # noqa: BLE001 - one bad model shouldn't stop the rest
                score = float("nan")
                print(f"[compare] {name} failed: {e}")
            STATE.compare_results.append((name, score, time.time() - t0))
            STATE.compare_done += 1

        STATE.compare_current = ""
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

    cfg = dict(
        data_path=STATE.data_path,
        target=STATE.opt_target.strip(),
        excluded=_split_cols(STATE.opt_excluded),
        fixed=_parse_fixed(STATE.opt_fixed),
        direction="maximise" if STATE.opt_direction_idx == 0 else "minimise",
        sheet=_current_sheet(),                  # selected worksheet/tab
    )
    threading.Thread(target=_optimize_worker, args=(cfg,), daemon=True).start()


def _optimize_worker(cfg):
    try:
        STATE.opt_status = "Training + searching (this can take a minute)…"
        res = run_capacity_optimization(
            cfg["data_path"], cfg["target"], cfg["excluded"],
            cfg["fixed"], cfg["direction"], sheet=cfg.get("sheet"))
        STATE.opt_result = res
        STATE.opt_status = (
            f"Done. Model R²={res['r2']:.2f} on {res['n_rows']} rows, "
            f"{res['n_knobs']} controllable knobs.")
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

        # 1. Correlation heatmap — numeric features + targets.
        STATE.charts_status = "1/8 · correlation heatmap…"
        corr_df = pd.concat(
            [X_enc[numeric_feats].reset_index(drop=True), y.reset_index(drop=True)],
            axis=1,
        )
        items.append(("Correlation heatmap", C.correlation_heatmap(corr_df, path("corr"))))

        # Out-of-fold predictions (shared by charts 2 & 3), same model as training.
        STATE.charts_status = "2/8 · out-of-fold predictions…"
        oof_model = MultiOutputRegressor(ExtraTreesRegressor(
            n_estimators=300, max_depth=12, min_samples_split=4,
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1))
        oof = cross_val_predict(
            oof_model, X_enc, y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))
        r2_by = {t: float(r2_score(y.iloc[:, i], oof[:, i])) for i, t in enumerate(targets)}

        items.append(("Predicted vs actual",
                      C.predicted_vs_actual(y.values, oof, targets, path("pva"), r2_by)))
        STATE.charts_status = "3/8 · residual plot…"
        items.append(("Residual plot",
                      C.residual_plot(y.values, oof, targets, path("resid"))))

        # 4. Feature importance (folded back to source columns).
        STATE.charts_status = "4/8 · feature importance…"
        imp_src = _aggregate_importance(STATE.importances, numeric_schema, categorical_schema)
        items.append(("Feature importance", C.feature_importance(imp_src, path("imp"))))

        # 5 & 6. SHAP on the trained single-target estimator.
        Xs_shap = X_enc.reindex(columns=STATE.feature_columns, fill_value=0)
        est = STATE.model.estimators_[ti] if hasattr(STATE.model, "estimators_") else STATE.model
        STATE.charts_status = "5/8 · SHAP summary (can be slow)…"
        items.append(("SHAP summary",
                      C.shap_summary(est, Xs_shap, path("shap_sum"), targets[ti])))
        STATE.charts_status = "6/8 · SHAP dependence…"
        items.append(("SHAP dependence",
                      C.shap_dependence(est, Xs_shap, path("shap_dep"), targets[ti])))

        # 7. Optimization heatmap — sweep two numeric knobs through the model.
        STATE.charts_status = "7/8 · optimization heatmap…"
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

        # 8. Pareto front — trade-off between two targets (both maximised).
        STATE.charts_status = "8/8 · Pareto front…"
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
# PREDICTION HELPERS  (shared by single + batch)
# =============================================================================
def build_matrix(X_raw):
    """One-hot encode a raw feature frame and align it to the trained columns."""
    X_enc = pd.get_dummies(X_raw, drop_first=True)
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

        # Rebuild the raw feature frame using the training schema (impute the same way).
        cols = {}
        for col, med in STATE.numeric_schema.items():
            series = pd.to_numeric(prepared.get(col), errors="coerce")
            cols[col] = (series if series is not None else pd.Series([med] * len(prepared))).fillna(med)
        for col in STATE.categorical_schema:
            series = prepared.get(col)
            if series is None:
                series = pd.Series(["Missing"] * len(prepared))
            cols[col] = series.astype(str).replace(
                {"nan": "Missing", "None": "Missing"}
            ).fillna("Missing")
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


def save_model():
    joblib.dump(
        {
            "model": STATE.model,
            "feature_columns": STATE.feature_columns,
            "numeric_schema": STATE.numeric_schema,
            "categorical_schema": STATE.categorical_schema,
            "targets": STATE.targets,
            "cfg": STATE.cfg,
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
    STATE.numeric_values = dict(STATE.numeric_schema)
    STATE.category_index = {c: 0 for c in STATE.categorical_schema}
    STATE.trained = True
    STATE.status = f"Loaded model from '{MODEL_OUT}'."


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
    imgui.set_next_item_width(360)
    changed, pct = imgui.slider_int("Test split %", int(round(STATE.test_size * 100)), 5, 50)
    if changed:
        STATE.test_size = pct / 100.0
    imgui.same_line()
    imgui.text_colored(DIM, f"({100 - pct}% train / {pct}% test)")
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
        imgui.text("Per-target performance (train vs held-out test)")
        flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
        if imgui.begin_table("metrics", 5, flags):
            for h in ("Target", "Train R2", "Train RMSE", "Test R2", "Test RMSE"):
                imgui.table_setup_column(h)
            imgui.table_headers_row()
            for name, tr_r2, tr_rmse, te_r2, te_rmse in STATE.metrics:
                imgui.table_next_row()
                imgui.table_next_column(); imgui.text(name)
                imgui.table_next_column(); imgui.text(f"{tr_r2:.4f}")
                imgui.table_next_column(); imgui.text(f"{tr_rmse:.4f}")
                # Colour test R2 by quality; a big train-test gap shows here.
                imgui.table_next_column()
                imgui.text_colored(GREEN if te_r2 >= 0.5 else (RED if te_r2 < 0 else DIM),
                                   f"{te_r2:.4f}")
                imgui.table_next_column(); imgui.text(f"{te_rmse:.4f}")
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
    imgui.text("Benchmark 7 architectures with 5-fold cross-validation")
    imgui.text_wrapped(
        "Uses the columns configured in the Train tab. Reports each model's average "
        "out-of-fold R2 across all 6 targets. This can take a few minutes on full data."
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
        imgui.text("Ranking (highest average OOF R2 first)")
        flags = imgui.TableFlags_.borders | imgui.TableFlags_.row_bg
        if imgui.begin_table("compare", 4, flags):
            for h in ("#", "Model", "Avg OOF R2", "Time"):
                imgui.table_setup_column(h)
            imgui.table_headers_row()
            ranked = sorted(
                STATE.compare_results,
                key=lambda r: (r[1] if r[1] == r[1] else -1e9),  # NaN sinks to bottom
                reverse=True,
            )
            for rank, (name, score, secs) in enumerate(ranked, start=1):
                imgui.table_next_row()
                imgui.table_next_column(); imgui.text(str(rank))
                imgui.table_next_column()
                # Highlight the current best in green.
                if rank == 1 and score == score:
                    imgui.text_colored(GREEN, name)
                else:
                    imgui.text(name)
                imgui.table_next_column()
                imgui.text("FAILED" if score != score else f"{score:.4f}")
                imgui.table_next_column(); imgui.text(f"{secs:.1f}s")
            imgui.end_table()


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
        "Trains on CONTROLLABLE knobs only (everything except the target and the "
        "excluded measured/outcome columns), then searches for the best recipe. "
        "Uses the file chosen in the Train tab.")
    imgui.separator()

    if not STATE.data_path:
        imgui.text_colored(DIM, "Choose a spreadsheet in the Train tab first.")
        return
    imgui.text_colored(DIM, STATE.data_path)

    imgui.set_next_item_width(360)
    _, STATE.opt_target = imgui.input_text("Target to optimise", STATE.opt_target)
    imgui.set_next_item_width(360)
    _, STATE.opt_direction_idx = imgui.combo("Direction", STATE.opt_direction_idx,
                                             ["maximise", "minimise"])
    imgui.set_next_item_width(360)
    _, STATE.opt_fixed = imgui.input_text("Fixed knobs (col=value, comma-sep)", STATE.opt_fixed)
    imgui.text_colored(DIM, "Excluded (measured / outcome) columns — one big list:")
    _, STATE.opt_excluded = imgui.input_text_multiline(
        "##excluded", STATE.opt_excluded, imgui.ImVec2(-1, 80))

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
    if imgui.small_button(f"Open in viewer##{rel_path}"):
        try:
            os.startfile(abs_path)  # Windows
        except Exception:  # noqa: BLE001
            pass


def draw_charts_tab():
    imgui.text("Diagnostic charts for the trained model")
    imgui.text_wrapped(
        "Generates 8 plots: correlation heatmap, predicted-vs-actual, residuals, "
        "feature importance, SHAP summary & dependence, an optimization heatmap, "
        "and a Pareto front. Uses the trained model and its data.")
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


def gui():
    """Top-level GUI callback — called every frame by imgui-bundle."""
    imgui.text("🌲 ExtraTrees Multi-Target Trainer & Predictor")
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
        imgui.end_tab_bar()


def main():
    # image_from_asset resolves paths relative to the assets folder; point it at
    # the project directory so "charts/xxx.png" loads.
    hello_imgui.set_assets_folder(os.path.dirname(os.path.abspath(__file__)))
    immapp.run(
        gui_function=gui,
        window_title="RF Trainer & Predictor",
        window_size=(860, 760),
    )


if __name__ == "__main__":
    main()
