"""
train_model.py  —  Tuned ExtraTrees pipeline (scikit-learn)
===========================================================

Focuses on the winning architecture — MultiOutputRegressor(ExtraTreesRegressor)
— with RobustScaler and tuned hyperparameters, evaluated by 5-fold CV on 6
numerical "current" targets.

Pipeline (matches the documented steps below):
  1. Load Excel + drop unique string identifier columns.
  2. Parse one messy combined column into 3 anonymized feature columns via regex.
  3. Drop rows with no target label; impute feature gaps (numeric -> median, text -> 'Missing').
  4. One-hot encode categoricals with pd.get_dummies(drop_first=True).
  5. RobustScaler on the numeric feature columns (scale by IQR, outlier-safe).
  6. 5-fold CV of the tuned ExtraTrees model; report average out-of-fold R²
     across all 6 targets plus the top-5 feature importances.

Model: MultiOutputRegressor(ExtraTreesRegressor(
           n_estimators=300, max_depth=12, min_samples_split=4,
           max_features='sqrt'))

-----------------------------------------------------------------------------
HOW TO RUN
----------
    pip install pandas numpy scikit-learn openpyxl
    python train_model.py --data path/to/your_spreadsheet.xlsx

  (Runs fully offline; install the packages above yourself first.)
=============================================================================
"""

import sys
import importlib.util
import argparse

# Force UTF-8 output so prints never crash on non-UTF-8 consoles (e.g. Windows GBK).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# -----------------------------------------------------------------------------
# Dependency check (offline — never installs anything or touches the network).
# -----------------------------------------------------------------------------
def check_dependencies():
    """Exit with a clear message if any required package is missing."""
    dependencies = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "openpyxl": "openpyxl",   # needed by pandas to read .xlsx files
    }
    missing = [pip for mod, pip in dependencies.items()
               if importlib.util.find_spec(mod) is None]
    if missing:
        print("[X] Missing required packages: " + " ".join(missing))
        print("    This script runs fully offline; install them yourself, then re-run:")
        print("        pip install " + " ".join(missing))
        sys.exit(1)


check_dependencies()

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import chemistry_features as chemistry


# =============================================================================
# CONFIGURATION  —  EDIT THESE TO MATCH YOUR SPREADSHEET
# =============================================================================

# Default spreadsheet path (override on the command line with --data).
DATA_PATH = r"C:/Users/28jay/Downloads/LIB_1A_one_big_wide_chart.xlsx"

# The single messy/combined column to parse+anonymize (percent + text, e.g. '2wt% nahco3').
MIXED_COLUMN = "Additive_1_original"

# Unique identifiers to drop immediately (prevents the model memorizing rows).
ID_COLUMNS = ["Record_ID"]

# The numerical capacity target(s) to predict. This wide chart has a single target.
TARGET_COLUMNS = [
    "LIB_1A",
]

RANDOM_STATE = 42  # fixed seed for reproducible splits/forest

# --- Row filtering (STEP 3) ---
# Values that count as "no information" when deciding if a row is worth keeping.
BLANK_TOKENS = {"", "unknown", "nan", "none", "na", "missing", "missing or unspecified"}
# Drop a row if fewer than this fraction of its feature cells carry real info.
MIN_INFORMATIVE_FRAC = 0.10

# --- Feature-selection thresholds (STEP 3b) ---
MAX_MISSING_FRAC = 0.60   # drop a feature if missing in >60% of the kept rows
MAX_CATEGORIES = 25       # drop a text feature with more unique values than this


# =============================================================================
# STEP 2 (helper): Regex parser for the messy combined column
# =============================================================================
def parse_mixed_value(value):
    """
    Split one messy cell into three ANONYMIZED parts.

    Examples (input -> (numeric_feature_A, group_label_B, text_modifier_C)):
        "15% mn"             -> (15.0, "MN",   "None")
        "bio oil"            -> (0.0,  "None", "bio oil")
        "5%zn(znso4·7h2o)"   -> (5.0,  "ZN",   "znso4·7h2o")
        NaN / "" / "None"    -> (0.0,  "None", "None")

    Returns a tuple (float, str, str). Never raises on odd input.
    """
    # --- Safely handle missing / placeholder values ---
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0, "None", "None"
    s = str(value).strip()
    if s.lower() in {"", "none", "nan", "missing", "na"}:
        return 0.0, "None", "None"

    s_low = s.lower()

    # --- numeric_feature_A: first number/percentage in the string (else 0.0) ---
    num_match = re.search(r"\d+(?:\.\d+)?", s_low)
    numeric_feature_A = float(num_match.group()) if num_match else 0.0

    # --- text_modifier_C: prefer text inside parentheses if present ---
    paren_match = re.search(r"\(([^)]*)\)", s_low)

    # --- group_label_B: short alphabetic code tied to the number (e.g. "5%zn" -> ZN) ---
    code_match = re.search(r"\d+\s*%?\s*([a-z]{1,4})", s_low)
    group_label_B = code_match.group(1).upper() if code_match else "None"

    if paren_match:
        # Contents isolated from inside the parentheses.
        text_modifier_C = paren_match.group(1).strip() or "None"
    else:
        # No parentheses: strip out the number, '%', and the code token; whatever
        # readable text remains becomes the modifier (e.g. "bio oil").
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


# =============================================================================
# WINNING MODEL  —  tuned, RobustScaled MultiOutputRegressor(ExtraTrees)
# =============================================================================
def build_extratrees():
    """The winning architecture with hyperparameters tuned for ~1,200 rows."""
    return MultiOutputRegressor(
        ExtraTreesRegressor(
            n_estimators=300,       # more trees -> lower variance
            max_depth=12,           # deeper than 5 to capture finer interactions
            min_samples_split=4,    # don't split on tiny noise blocks
            max_features="sqrt",    # force tree diversity (ExtraTrees' strength)
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )


def run_extratrees(X, y):
    """
    Cross-validate the tuned ExtraTrees pipeline (X is already RobustScaled) and
    report the average out-of-fold R² plus the top-5 feature importances.
    """
    # Trees need plain floats (get_dummies yields bool dummy columns).
    X = X.astype(float)
    model = build_extratrees()

    # Identify "recipes": rows with an identical feature vector are replicates of
    # the same experiment. We group by them so replicates never straddle the
    # train/test split — otherwise the model just memorises a recipe from its
    # training copies and the R² is inflated (leakage).
    recipe = pd.factorize(X.round(6).astype(str).agg("|".join, axis=1))[0]
    n_recipes = len(np.unique(recipe))
    print(f"      {len(X)} rows span {n_recipes} unique recipe(s) "
          f"(~{len(X) / max(n_recipes, 1):.1f} replicate(s) each).")

    # --- (a) Random 5-fold: optimistic — a recipe can appear in train AND test.
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_rand = cross_val_predict(model, X, y, cv=kf)
    r2_rand = r2_score(y, oof_rand, multioutput="uniform_average")

    # --- (b) Grouped 5-fold: HONEST — every test recipe is unseen in training.
    n_splits = min(5, n_recipes)
    gkf = GroupKFold(n_splits=n_splits)
    oof_grp = cross_val_predict(model, X, y, cv=gkf, groups=recipe)
    per_target = r2_score(y, oof_grp, multioutput="raw_values")
    r2_grp = r2_score(y, oof_grp, multioutput="uniform_average")

    print("\nPer-target R² — grouped CV (predicting an UNSEEN recipe):")
    print(f"  {'Target':<12} {'OOF R2':>10}")
    print("  " + "-" * 24)
    for col, r2 in zip(TARGET_COLUMNS, per_target):
        print(f"  {col:<12} {r2:>10.4f}")

    ntgt = len(TARGET_COLUMNS)
    lbl = f"({ntgt} target{'s' if ntgt != 1 else ''})"
    print("\n" + "=" * 56)
    print(f"  Random  5-fold OOF R² {lbl:<12} : {r2_rand:>8.4f}  (optimistic/leaky)")
    print(f"  Grouped 5-fold OOF R² {lbl:<12} : {r2_grp:>8.4f}  (HONEST: new recipe)")
    print(f"  {'Target to beat':<34} : {'0.4400':>8}")
    verdict = "PASSED" if r2_grp > 0.44 else "below target (honest score)"
    print(f"  {verdict:^54}")
    print("=" * 56)
    avg_r2 = r2_grp

    # --- Feature importances (fit once on the full scaled data) ---
    # cross_val_predict doesn't expose fitted models, so fit on all rows to read
    # importances, averaged across the 6 per-target sub-estimators.
    model.fit(X, y)
    importances = np.mean(
        [est.feature_importances_ for est in model.estimators_], axis=0
    )
    top = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    print("\nTop 5 most influential features:")
    for rank, (feat, imp) in enumerate(top.head(5).items(), start=1):
        shown = chemistry.descriptor_display_name(str(feat))
        print(f"  {rank}. {shown:<30} {imp * 100:6.2f}%")

    return avg_r2


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main(data_path):
    # -------------------------------------------------------------------------
    # STEP 1: Data loading & initial drop of unique identifier columns
    # -------------------------------------------------------------------------
    print(f"[1/6] Loading spreadsheet: {data_path}")
    df = pd.read_excel(data_path)
    print(f"      Loaded {df.shape[0]} rows × {df.shape[1]} columns.")

    # Drop unique string IDs so the model can't memorize individual rows.
    df = df.drop(columns=ID_COLUMNS, errors="ignore")

    # Drop columns that are entirely empty — they carry no signal.
    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        print(f"      Dropped {len(empty_cols)} completely-empty column(s).")

    # -------------------------------------------------------------------------
    # STEP 2: Anonymous parsing of the sensitive mixed text column
    # -------------------------------------------------------------------------
    if MIXED_COLUMN in df.columns:
        print(f"[2/6] Parsing messy column '{MIXED_COLUMN}' into anonymized features…")
        # .apply() runs the parser on every cell; result_type='expand' turns the
        # returned 3-tuples into 3 separate columns in one shot.
        parsed = df[MIXED_COLUMN].apply(parse_mixed_value).apply(pd.Series)
        parsed.columns = ["numeric_feature_A", "group_label_B", "text_modifier_C"]

        # Unpack the anonymized columns into the DataFrame.
        df["numeric_feature_A"] = parsed["numeric_feature_A"].astype(float)
        df["group_label_B"] = parsed["group_label_B"].astype(str)
        df["text_modifier_C"] = parsed["text_modifier_C"].astype(str)

        # Drop the original messy column immediately (privacy + no duplication).
        df = df.drop(columns=[MIXED_COLUMN])
    else:
        print(f"[2/6] WARNING: mixed column '{MIXED_COLUMN}' not found — skipping parse. "
              f"Edit MIXED_COLUMN at the top of this script.")

    # Confirm the target columns exist before continuing.
    missing_targets = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing_targets:
        raise ValueError(
            f"These target columns are missing from the spreadsheet: {missing_targets}. "
            f"Check TARGET_COLUMNS at the top of the script."
        )

    # -------------------------------------------------------------------------
    # STEP 3: Target cleanup (drop rows with no label) + feature imputation
    # -------------------------------------------------------------------------
    print("[3/6] Cleaning targets and imputing feature gaps…")

    # 3a. Targets: coerce stray text to NaN, then DROP rows missing any target.
    #     We never impute the target itself — a fabricated label would corrupt
    #     both training and the R² we report. Only real, observed rows are used.
    for col in TARGET_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    print(f"      Kept {len(df)} of {before} rows that have observed target value(s).")

    # 3a-2. DROP rows with (almost) no recorded features. Some rows carry a
    #       capacity value but every input is blank/'Unknown'/'Missing' — the
    #       model can't tell them apart, so they only add noise (and cap R²).
    feat_only = [c for c in df.columns if c not in TARGET_COLUMNS]

    def _informative_frac(row):
        hits = 0
        for v in row:
            s = str(v).strip().lower()
            if not ((v != v) or s in BLANK_TOKENS):
                hits += 1
        return hits / len(feat_only) if feat_only else 0.0

    info_frac = df[feat_only].apply(_informative_frac, axis=1)
    before = len(df)
    df = df[info_frac >= MIN_INFORMATIVE_FRAC].reset_index(drop=True)
    print(f"      Kept {len(df)} of {before} rows with recorded features "
          f"(dropped {before - len(df)} blank/'Unknown'-only rows).")

    # Separate features (X) from targets (y).
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMNS].copy()
    chemistry_config = chemistry.ENGINE.auto_configure(
        X, requested_mode="automatic")
    chemistry_expansion = chemistry.ENGINE.transform(X, config=chemistry_config)
    X = chemistry_expansion.frame
    if chemistry_expansion.metadata.get("columns"):
        print(f"      Chemistry engine: expanded "
              f"{len(chemistry_expansion.metadata['columns'])} chemical column(s) into "
              f"{chemistry_expansion.metadata['descriptor_feature_count']} descriptors.")

    # 3b. FEATURE SELECTION — prune columns that add noise instead of signal:
    #       * too sparse:      missing in more than MAX_MISSING_FRAC of the rows.
    #       * too high-cardinality text: >MAX_CATEGORIES unique values, which would
    #         explode into hundreds of near-useless one-hot columns (free-text
    #         names/formulas/notes). Low-cardinality categoricals are kept.
    dropped = []
    n = len(X)
    for col in list(X.columns):
        miss_frac = X[col].isna().mean()
        if miss_frac > MAX_MISSING_FRAC:
            dropped.append((col, f"{miss_frac:.0%} missing"))
            X = X.drop(columns=col)
            continue
        if not pd.api.types.is_numeric_dtype(X[col]):
            card = X[col].dropna().astype(str).nunique()
            if card > MAX_CATEGORIES:
                dropped.append((col, f"{card} categories"))
                X = X.drop(columns=col)
    print(f"      Feature selection: kept {X.shape[1]}, dropped {len(dropped)} "
          f"noisy column(s) (sparse or high-cardinality).")

    # 3c. Numeric feature columns -> median; text/categorical columns -> 'Missing'.
    #     Track the continuous numeric columns so we can RobustScale them later
    #     (their names survive one-hot encoding unchanged).
    numeric_cols = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            med = X[col].median()
            if pd.isna(med):          # column is entirely empty -> no usable median
                med = 0.0
            X[col] = X[col].fillna(med)
            numeric_cols.append(col)
        else:
            X[col] = X[col].astype(str).fillna("Missing").replace(
                {"nan": "Missing", "None": "Missing"}
            )

    # -------------------------------------------------------------------------
    # STEP 4: Categorical encoding via one-hot (drop_first avoids collinearity)
    # -------------------------------------------------------------------------
    print("[4/6] One-hot encoding categorical/text columns…")
    X = pd.get_dummies(X, drop_first=True)
    print(f"      Feature matrix after encoding: {X.shape[1]} columns.")

    # -------------------------------------------------------------------------
    # STEP 5: Robust scaling of the numeric features (before CV).
    #   RobustScaler centres on the median and scales by the IQR, so a handful of
    #   extreme experimental outliers can't dominate the feature ranges. The 0/1
    #   one-hot columns are left untouched — only the continuous numerics scale.
    # -------------------------------------------------------------------------
    print(f"[5/6] Robust-scaling {len(numeric_cols)} numeric feature(s) by IQR…")
    scaler = RobustScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # -------------------------------------------------------------------------
    # STEP 6: 5-fold CV with the tuned, scaled ExtraTrees pipeline + report
    # -------------------------------------------------------------------------
    print("[6/6] 5-fold CV with tuned MultiOutput(ExtraTrees)…")
    run_extratrees(X, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multi-output regression model.")
    parser.add_argument(
        "--data", default=DATA_PATH,
        help=f"Path to the Excel spreadsheet (default: {DATA_PATH})",
    )
    args = parser.parse_args()
    main(args.data)
