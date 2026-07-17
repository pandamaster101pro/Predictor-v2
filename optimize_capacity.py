"""
optimize_capacity.py  —  Recommend experimental conditions that maximise capacity.
==================================================================================

Trains a model to predict reversible capacity from ONLY controllable inputs
(precursor, pyrolysis, atmosphere, activation, additives, test window), then
searches that input space for the recipe with the highest predicted capacity.

Measured-after-synthesis properties (lignin content, surface area, d-spacing,
pore volume, carbon yield, …) are deliberately EXCLUDED — you can't set them as
knobs, so an optimiser must not use them.

HOW TO RUN
----------
    pip install pandas numpy scikit-learn xgboost scipy openpyxl
    python optimize_capacity.py
    (runs fully offline; install the packages above yourself first)
==================================================================================
"""

import sys, importlib.util

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def check_dependencies():
    """Offline — never installs anything; exit if a required package is missing."""
    deps = {"pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn",
            "xgboost": "xgboost", "scipy": "scipy", "openpyxl": "openpyxl"}
    missing = [p for m, p in deps.items() if importlib.util.find_spec(m) is None]
    if missing:
        print("[X] Missing required packages: " + " ".join(missing))
        print("    Runs fully offline; install them yourself, then re-run:")
        print("        pip install " + " ".join(missing))
        sys.exit(1)


check_dependencies()

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
from scipy.optimize import differential_evolution

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = r"C:/Users/28jay/Downloads/lignin_hard_carbon_dataset_fixed.xlsx"
TARGET = "Reversible_Capacity_mAh_per_g"
DIRECTION = "maximise"           # or "minimise"

# Columns that are MEASURED after synthesis (cannot be dialled in) -> excluded.
MEASURED = [
    "Lignin_Purity_wt%", "Ash_Content_wt%", "Sulfur_Content_wt%",
    "d002_Angstrom", "La_nm", "Lc_nm", "ID_IG_Ratio", "BET_Surface_Area_m2_per_g",
    "Total_Pore_Volume_cm3_per_g", "Micropore_Fraction", "Closed_Pore_Fraction",
    "True_Density_g_per_cm3", "Carbon_Yield_wt%",
]
# Other performance metrics (outcomes, not inputs) -> excluded.
OUTCOMES = [
    "Plateau_Capacity_mAh_per_g", "Slope_Capacity_mAh_per_g", "ICE_%",
    "Rate_Cap_Retention_%", "Cycle_Retention_100cyc_%", "Avg_Sodiation_Voltage_V",
]
DROP_IDS = ["Sample_ID"]

MIN_SUPPORT = 3          # a categorical option must appear >=this many times to be recommended
RANDOM_STATE = 42


def build_model():
    return XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )


def main():
    # ---- Load & select controllable features -------------------------------
    df = pd.read_excel(DATA_PATH).drop(columns=DROP_IDS, errors="ignore")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    controllable = [c for c in df.columns
                    if c not in MEASURED + OUTCOMES + [TARGET]]
    print(f"Rows: {len(df)}   Controllable knobs: {len(controllable)}")

    X_raw = df[controllable].copy()
    y = df[TARGET].values

    # Split knobs into numeric (continuous ranges) and categorical (choices).
    numeric_cols, cat_choices = [], {}
    for c in controllable:
        if pd.api.types.is_numeric_dtype(X_raw[c]):
            X_raw[c] = X_raw[c].fillna(X_raw[c].median())
            numeric_cols.append(c)
        else:
            X_raw[c] = X_raw[c].astype(str).fillna("NA")
            # keep only well-supported categories so we don't recommend a one-off
            counts = X_raw[c].value_counts()
            cat_choices[c] = sorted(counts[counts >= MIN_SUPPORT].index.tolist()) \
                or sorted(counts.index.tolist())

    # ---- Fit the predictive model + report honest CV R² --------------------
    X_enc = pd.get_dummies(X_raw, drop_first=False)
    feat_cols = X_enc.columns.tolist()
    col_pos = {c: i for i, c in enumerate(feat_cols)}
    model = build_model()
    cv_r2 = r2_score(y, cross_val_predict(model, X_enc.astype(float).values, y,
                                          cv=KFold(5, shuffle=True, random_state=RANDOM_STATE)))
    model.fit(X_enc.astype(float).values, y)
    print(f"Model: XGBoost on controllable knobs only  ->  5-fold test R² = {cv_r2:.3f}", flush=True)
    print(f"Observed {TARGET}: min={y.min():.0f}  mean={y.mean():.0f}  max={y.max():.0f}\n", flush=True)

    # ---- Build the search space -------------------------------------------
    # numeric: bounded by the 1st–99th percentile (stay inside real experience)
    bounds, integrality, spec = [], [], []
    for c in numeric_cols:
        lo, hi = np.percentile(X_raw[c], 1), np.percentile(X_raw[c], 99)
        if lo == hi:
            hi = lo + 1e-6
        bounds.append((lo, hi)); integrality.append(False)
        spec.append(("num", c, col_pos.get(c)))
    for c, choices in cat_choices.items():
        bounds.append((0, len(choices) - 1)); integrality.append(True)
        # precompute the encoded-column index for each choice (fast lookup)
        idxs = [col_pos.get(f"{c}_{ch}") for ch in choices]
        spec.append(("cat", c, idxs))

    n_feat = len(feat_cols)
    sign = -1.0 if DIRECTION == "maximise" else 1.0

    def vectorize(vec):
        """Build the encoded feature row directly (no DataFrame / get_dummies)."""
        x = np.zeros(n_feat, dtype=float)
        for v, s in zip(vec, spec):
            if s[0] == "num":
                if s[2] is not None:
                    x[s[2]] = v
            else:
                i = int(round(v)); i = max(0, min(i, len(s[2]) - 1))
                pos = s[2][i]
                if pos is not None:
                    x[pos] = 1.0
        return x

    def objective(vec):
        return sign * float(model.predict(vectorize(vec).reshape(1, -1))[0])

    def decode_row(vec):
        row = {}
        for v, s in zip(vec, spec):
            if s[0] == "num":
                row[s[1]] = float(v)
            else:
                i = int(round(v)); i = max(0, min(i, len(cat_choices[s[1]]) - 1))
                row[s[1]] = cat_choices[s[1]][i]
        return row

    # ---- Optimise ----------------------------------------------------------
    print("Searching for the best recipe (differential evolution)…", flush=True)
    result = differential_evolution(
        objective, bounds, integrality=integrality,
        seed=RANDOM_STATE, popsize=15, maxiter=80, tol=1e-4,
        mutation=(0.5, 1.0), recombination=0.9, polish=False, updating="immediate",
    )
    best = decode_row(result.x)
    best_row = pd.DataFrame([best])
    # Predict directly from the winning vector (unambiguous, no sign juggling).
    best_cap = float(model.predict(vectorize(result.x).reshape(1, -1))[0])

    # ---- Report ------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  RECOMMENDED CONDITIONS to {DIRECTION} {TARGET}")
    print("=" * 60)
    for c in controllable:
        val = best_row.iloc[0][c]
        if c in numeric_cols:
            print(f"  {c:<28} = {float(val):.3g}")
        else:
            print(f"  {c:<28} = {val}")
    print("-" * 60)
    print(f"  PREDICTED {TARGET} = {best_cap:.0f} mAh/g")
    print(f"  (best ever observed in data       = {y.max():.0f} mAh/g)")
    print(f"  Model 5-fold R² = {cv_r2:.2f}  -> treat as a hypothesis to test, "
          f"not a guarantee.")
    print("=" * 60)

    # Flag any numeric knob sitting at a search boundary (wants to extrapolate).
    edge = []
    for v, s in zip(result.x, spec):
        if s[0] == "num":
            c = s[1]
            lo, hi = np.percentile(X_raw[c], 1), np.percentile(X_raw[c], 99)
            span = (hi - lo) or 1.0
            if abs(v - lo) < 0.02 * span or abs(v - hi) < 0.02 * span:
                edge.append(c)
    if edge:
        print("  Note: these knobs hit the edge of the observed range "
              f"(extrapolation risk): {', '.join(edge)}")


if __name__ == "__main__":
    main()
