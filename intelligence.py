"""
intelligence.py  —  Dataset Intelligence engine for the Predictor app.
=====================================================================

Answers *why prediction succeeds or fails* for a chosen target, rather than
chasing accuracy. GUI-free so it is unit-testable head-lessly. Heavily reuses
:mod:`latent` for leakage-safe preprocessing / config / PCA and :mod:`charts`
for plotting.

Ten analysis sections (all pure functions returning plain dicts / DataFrames):

  1. dataset_summary          6. causal_structure
  2. predictability           7. latent_analysis
  3. redundancy (+VIF)        8. target_analysis
  4. difficulty_score         9. (PCA reused from latent.fit_pca)
  5. learnability            10. build_pdf_report

Plus :func:`final_conclusion` and :func:`ai_insights` which turn the numbers
into plain-language findings for the AI Research Assistant panel.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

import latent as L

RANDOM_STATE = 42

# Structural / characterization descriptors that typically unlock capacity
# prediction. Detected by case-insensitive substring so column naming can vary.
STRUCTURAL_HINTS = [
    "bet", "surface_area", "d002", "id_ig", "raman", "pore",
    "crystallite", "la_nm", "lc_nm", "true_density", "micropore",
]
STRUCTURAL_SUGGESTIONS = ("BET surface area, d002 spacing, Raman ID/IG ratio, "
                          "pore volume, or crystallite size (La/Lc)")


# =============================================================================
# SECTION 1 — DATASET SUMMARY
# =============================================================================
def dataset_summary(df: pd.DataFrame, config: L.LatentConfig) -> dict:
    """Shape, missingness, duplicates and target dispersion for the dataset.

    Returns scalar stats plus a per-column ``missing`` DataFrame
    (column, dtype, missing_count, missing_pct).
    """
    cat, num = config.feature_columns(list(df.columns))
    feat_cols = cat + num
    y = pd.to_numeric(df[config.target], errors="coerce") if config.target in df.columns \
        else pd.Series(dtype=float)
    y_valid = y.dropna()

    # Per-column missingness (over all columns, not just features).
    miss_rows = []
    for c in df.columns:
        n_missing = int(df[c].isna().sum())
        miss_rows.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "missing_count": n_missing,
            "missing_pct": round(100.0 * n_missing / max(len(df), 1), 2),
        })
    missing_df = pd.DataFrame(miss_rows).sort_values("missing_pct", ascending=False)

    # Duplicate experimental conditions = duplicate rows across the FEATURE set
    # (ignoring the target / ids), which flags repeated recipes.
    dup_conditions = int(df[feat_cols].duplicated().sum()) if feat_cols else 0

    return {
        "n_samples": int(len(df)),
        "n_features": len(feat_cols),
        "n_categorical": len(cat),
        "n_numerical": len(num),
        "categorical": cat,
        "numerical": num,
        "total_missing": int(df.isna().sum().sum()),
        "missing_pct_overall": round(100.0 * df.isna().sum().sum()
                                     / max(df.size, 1), 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_conditions": dup_conditions,
        "target": config.target,
        "target_n_valid": int(len(y_valid)),
        "target_variance": float(y_valid.var(ddof=1)) if len(y_valid) > 1 else 0.0,
        "target_std": float(y_valid.std(ddof=1)) if len(y_valid) > 1 else 0.0,
        "target_min": float(y_valid.min()) if len(y_valid) else float("nan"),
        "target_max": float(y_valid.max()) if len(y_valid) else float("nan"),
        "target_range": float(y_valid.max() - y_valid.min()) if len(y_valid) else 0.0,
        "missing_table": missing_df,
    }


# =============================================================================
# SECTION 2 — PREDICTABILITY ANALYSIS
# =============================================================================
def _distance_correlation(x: np.ndarray, y: np.ndarray, max_n: int = 800) -> float:
    """Székely distance correlation (0 = independent). O(n²); sub-samples if large."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    if n > max_n:
        idx = np.random.default_rng(RANDOM_STATE).choice(n, max_n, replace=False)
        x, y = x[idx], y[idx]
        n = max_n
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(0)[None, :] - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0)[None, :] - b.mean(1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvarx = (A * A).mean(); dvary = (B * B).mean()
    denom = np.sqrt(dvarx * dvary)
    return float(np.sqrt(max(dcov2, 0.0) / denom)) if denom > 0 else 0.0


def _correlation_ratio(codes: np.ndarray, y: np.ndarray) -> float:
    """Correlation ratio (eta) between a categorical (integer codes) and numeric y."""
    y = np.asarray(y, dtype=float)
    total_var = y.var()
    if total_var == 0:
        return 0.0
    ss_between = 0.0
    grand = y.mean()
    for c in np.unique(codes):
        grp = y[codes == c]
        ss_between += len(grp) * (grp.mean() - grand) ** 2
    return float(np.sqrt(ss_between / (total_var * len(y))))


def predictability(df: pd.DataFrame, config: L.LatentConfig) -> dict:
    """Rank every feature by its association with the target.

    Numeric features report |Pearson|, |Spearman|, mutual information and distance
    correlation. Categorical features report mutual information and correlation
    ratio (eta). Ranked by mutual information (comparable across both types).
    """
    cat, num = config.feature_columns(list(df.columns))
    y_all = pd.to_numeric(df[config.target], errors="coerce")
    mask = y_all.notna()
    y = y_all[mask].to_numpy(dtype=float)

    rows = []
    for feat in num:
        x_ser = pd.to_numeric(df[feat], errors="coerce")[mask]
        med = x_ser.median()
        x = x_ser.fillna(0.0 if pd.isna(med) else med).to_numpy(dtype=float)
        # A column that is empty (all-NaN) for the target rows, or constant,
        # carries no signal — score it 0 instead of feeding NaN to sklearn.
        if not np.isfinite(x).all() or np.std(x) == 0:
            pear = spear = mi = dcor = 0.0
        else:
            pear = abs(float(np.corrcoef(x, y)[0, 1]))
            spear = abs(float(pd.Series(x).corr(pd.Series(y), method="spearman")))
            mi = float(mutual_info_regression(x.reshape(-1, 1), y,
                                              discrete_features=False,
                                              random_state=RANDOM_STATE)[0])
            dcor = _distance_correlation(x, y)
        rows.append({"feature": feat, "type": "numeric", "pearson": pear,
                     "spearman": spear, "mutual_info": mi, "distance_corr": dcor})

    for feat in cat:
        codes = pd.factorize(df[feat].astype(str)[mask])[0]
        if len(np.unique(codes)) < 2:
            mi = eta = 0.0
        else:
            mi = float(mutual_info_regression(codes.reshape(-1, 1), y,
                                              discrete_features=True,
                                              random_state=RANDOM_STATE)[0])
            eta = _correlation_ratio(codes, y)
        rows.append({"feature": feat, "type": "categorical", "pearson": np.nan,
                     "spearman": np.nan, "mutual_info": mi, "distance_corr": eta})

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("mutual_info", ascending=False).reset_index(drop=True)
    num_pear = table.loc[table["type"] == "numeric", "pearson"]
    return {
        "table": table,
        "max_pearson": float(num_pear.max()) if len(num_pear) else 0.0,
        "max_spearman": float(table["spearman"].max(skipna=True)) if not table.empty else 0.0,
        "mean_mutual_info": float(table["mutual_info"].mean()) if not table.empty else 0.0,
        "top_feature": table.iloc[0]["feature"] if not table.empty else None,
    }


# =============================================================================
# SECTION 3 — FEATURE REDUNDANCY (+ VIF)
# =============================================================================
def redundancy(df: pd.DataFrame, config: L.LatentConfig,
               corr_threshold: float = 0.9, vif_threshold: float = 10.0) -> dict:
    """Correlation matrix, VIF, near-zero-variance and highly-correlated pairs,
    with automatic removal / merge suggestions."""
    _cat, num = config.feature_columns(list(df.columns))
    X = df[num].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    X = X.loc[:, X.std(ddof=0) > 0]  # drop constant cols for a valid corr/VIF

    corr = X.corr().fillna(0.0) if X.shape[1] else pd.DataFrame()

    # Highly correlated pairs (upper triangle).
    pairs = []
    cols = list(X.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= corr_threshold:
                pairs.append((cols[i], cols[j], float(r)))
    pairs.sort(key=lambda t: -abs(t[2]))

    # Near-zero variance (relative to mean scale) among original numeric cols.
    nzv = []
    for c in num:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.std(ddof=0) == 0 or s.nunique(dropna=True) <= 1:
            nzv.append(c)

    vif = _compute_vif(X)

    high_vif = [c for c, v in vif.items() if np.isfinite(v) and v > vif_threshold]
    remove = sorted(set(nzv) | {b for _a, b, _r in pairs})  # drop the 2nd of each pair
    merge = [(a, b) for a, b, _r in pairs]

    return {
        "corr_matrix": corr,
        "vif": vif,
        "high_corr_pairs": pairs,
        "near_zero_variance": nzv,
        "high_vif": high_vif,
        "suggest_remove": remove,
        "suggest_merge": merge,
    }


def _compute_vif(X: pd.DataFrame) -> dict[str, float]:
    """Variance Inflation Factor per column via R² of regressing it on the rest.

    VIF = 1 / (1 - R²). No statsmodels dependency — uses LinearRegression.
    """
    vif: dict[str, float] = {}
    cols = list(X.columns)
    if len(cols) < 2:
        return {c: 1.0 for c in cols}
    Xv = X.to_numpy(dtype=float)
    for i, c in enumerate(cols):
        y = Xv[:, i]
        others = np.delete(Xv, i, axis=1)
        try:
            r2 = LinearRegression().fit(others, y).score(others, y)
            vif[c] = float(1.0 / (1.0 - r2)) if r2 < 1 - 1e-12 else float("inf")
        except Exception:  # noqa: BLE001
            vif[c] = float("nan")
    return vif


# =============================================================================
# SECTION 5 — LEARNABILITY REPORT  (must be computed before difficulty)
# =============================================================================
def _learnability_models() -> dict:
    """Linear + tree models; XGBoost added only when installed."""
    models = {
        "LinearRegression": ("linear", LinearRegression()),
        "RandomForest": ("tree", RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        "ExtraTrees": ("tree", ExtraTreesRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = ("tree", XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0))
    except ImportError:
        pass
    return models


def learnability(df: pd.DataFrame, config: L.LatentConfig, n_splits: int = 5,
                 progress: Callable[[str], None] | None = None) -> dict:
    """5-fold CV of linear + tree models with identical folds, leakage-safe.

    Reports mean/std R², RMSE, MAE. Flags if the best model fails to clear
    ``CV R² > 0.3``.
    """
    cat, num = config.feature_columns(list(df.columns))
    X = df[cat + num]
    y = pd.to_numeric(df[config.target], errors="coerce")
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    results: dict[str, dict] = {}
    for name, (kind, model) in _learnability_models().items():
        if progress:
            progress(f"CV · {name}…")
        # Linear models need scaling; trees don't.
        prep = (L.make_scaled_preprocessor(cat, num) if kind == "linear"
                else L.make_tree_preprocessor(cat, num))
        pipe = Pipeline([("prep", prep), ("model", model)])
        try:
            results[name] = {**L._cv_scores(pipe, X, y, cv), "kind": kind}
        except Exception as exc:  # noqa: BLE001
            results[name] = {"error": f"{type(exc).__name__}: {exc}", "kind": kind}

    def best(kind: str) -> tuple[str | None, float]:
        pool = {n: s for n, s in results.items()
                if s.get("kind") == kind and "r2_mean" in s}
        if not pool:
            return None, float("-inf")
        name = max(pool, key=lambda n: pool[n]["r2_mean"])
        return name, pool[name]["r2_mean"]

    best_lin_name, best_lin_r2 = best("linear")
    best_tree_name, best_tree_r2 = best("tree")
    best_overall = max(best_lin_r2, best_tree_r2)
    return {
        "results": results,
        "best_linear": (best_lin_name, best_lin_r2),
        "best_tree": (best_tree_name, best_tree_r2),
        "best_r2": best_overall,
        "insufficient": best_overall <= 0.3,
    }


# =============================================================================
# SECTION 4 — DATASET DIFFICULTY SCORE
# =============================================================================
def difficulty_score(summary: dict, pred: dict, learn: dict | None = None) -> dict:
    """Heuristic difficulty from sample/feature counts, missingness, target
    dispersion, association strength, duplicates and (if available) best CV R².

    Returns a 0-100 difficulty, a label, an estimated achievable CV R² range and
    a human explanation.
    """
    n = summary["n_samples"]
    p = summary["n_features"]
    max_corr = pred["max_pearson"]
    avg_mi = pred["mean_mutual_info"]
    miss = summary["missing_pct_overall"] / 100.0
    dup_rate = summary["duplicate_conditions"] / max(n, 1)

    # Sub-scores in [0,1], higher = harder.
    s_samples = np.clip(1.0 - n / 300.0, 0, 1)          # <300 rows gets harder
    s_ratio = np.clip((p / max(n, 1)) / 0.2, 0, 1)      # many features per row
    s_missing = np.clip(miss / 0.3, 0, 1)
    s_corr = np.clip(1.0 - max_corr / 0.6, 0, 1)        # weak top correlation
    s_mi = np.clip(1.0 - avg_mi / 0.3, 0, 1)
    s_dup = np.clip(dup_rate / 0.3, 0, 1)
    difficulty = float(100 * np.mean([s_samples, s_ratio, s_missing,
                                      s_corr, s_mi, s_dup]))

    # Estimated achievable CV R²: anchor to the best measured model if available,
    # else infer a ceiling from the strongest correlation.
    if learn is not None and np.isfinite(learn["best_r2"]):
        center = max(learn["best_r2"], max_corr ** 2)
    else:
        center = max_corr ** 2
    lo = max(-0.1, center - 0.10)
    hi = min(0.95, center + 0.10)

    if difficulty < 30:
        label = "Easy"
    elif difficulty < 55:
        label = "Moderate"
    elif difficulty < 75:
        label = "Hard"
    else:
        label = "Very Hard"

    reasons = []
    if n < 150:
        reasons.append(f"only {n} samples")
    if max_corr < 0.3:
        reasons.append(f"weak top correlation ({max_corr:.2f})")
    if avg_mi < 0.05:
        reasons.append("low average mutual information")
    if s_ratio > 0.5:
        reasons.append(f"high feature-to-sample ratio ({p}/{n})")
    if miss > 0.1:
        reasons.append(f"{miss*100:.0f}% missing values")
    if dup_rate > 0.1:
        reasons.append(f"{dup_rate*100:.0f}% duplicate conditions")
    explanation = ("Difficulty driven by " + ", ".join(reasons) + "."
                   if reasons else "No major difficulty drivers detected.")

    return {
        "difficulty": difficulty,
        "label": label,
        "est_r2_low": lo,
        "est_r2_high": hi,
        "explanation": explanation,
    }


# =============================================================================
# SECTION 6 — CAUSAL STRUCTURE
# =============================================================================
def causal_structure(df: pd.DataFrame) -> dict:
    """Detect whether structural/characterization descriptors are present so the
    causal diagram can dash the 'Carbon Structure' intermediate when they're not."""
    cols_low = [c.lower() for c in df.columns]
    present = sorted({h for h in STRUCTURAL_HINTS
                      if any(h in c for c in cols_low)})
    return {
        "structure_present": bool(present),
        "structural_columns": present,
        "message": ("" if present else "Intermediate variables unavailable."),
    }


# =============================================================================
# SECTION 7 — LATENT ANALYSIS
# =============================================================================
def latent_analysis(df: pd.DataFrame, config: L.LatentConfig) -> dict:
    """Associate each engineered latent index with the target.

    Reuses :func:`latent.compute_engineered_latents`. Reports Pearson correlation,
    mutual information and (when SHAP is installed) mean |SHAP| importance.
    """
    eng = L.compute_engineered_latents(df, config.chem_weights, config.biomass_method)
    y_all = pd.to_numeric(df[config.target], errors="coerce")
    mask = y_all.notna()
    y = y_all[mask].to_numpy(dtype=float)
    E = eng.loc[mask]

    corr, mi = {}, {}
    for col in eng.columns:
        x = E[col].to_numpy(dtype=float)
        if not np.isfinite(x).all():          # never feed NaN/inf to sklearn
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if np.std(x) == 0:
            corr[col], mi[col] = 0.0, 0.0
            continue
        corr[col] = float(np.corrcoef(x, y)[0, 1])
        mi[col] = float(mutual_info_regression(x.reshape(-1, 1), y,
                                               random_state=RANDOM_STATE)[0])

    shap_importance = _latent_shap(E, y, list(eng.columns))
    total_mi = sum(mi.values()) or 1.0
    contribution = {k: v / total_mi for k, v in mi.items()}
    return {
        "latents": eng,
        "correlation": corr,
        "mutual_info": mi,
        "shap_importance": shap_importance,
        "contribution": contribution,
    }


def _latent_shap(E: pd.DataFrame, y: np.ndarray, names: list[str]) -> dict:
    """Mean |SHAP| per latent from a quick ExtraTrees fit; {} if SHAP missing."""
    try:
        import shap
    except ImportError:
        return {}
    try:
        model = ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE,
                                    n_jobs=-1).fit(E.to_numpy(dtype=float), y)
        sv = shap.TreeExplainer(model).shap_values(E.to_numpy(dtype=float))
        sv = np.asarray(sv)
        return {names[i]: float(np.abs(sv[:, i]).mean()) for i in range(len(names))}
    except Exception:  # noqa: BLE001
        return {}


# =============================================================================
# SECTION 8 — TARGET ANALYSIS
# =============================================================================
def target_analysis(df: pd.DataFrame, config: L.LatentConfig) -> dict:
    """Distribution shape, normality, outliers and a transform recommendation."""
    from scipy import stats

    y = pd.to_numeric(df[config.target], errors="coerce").dropna().to_numpy(dtype=float)
    if len(y) < 8:
        return {"values": y, "n": len(y), "note": "Too few values for distribution tests."}

    skew = float(stats.skew(y))
    kurt = float(stats.kurtosis(y))  # excess kurtosis
    # D'Agostino normality for larger n, Shapiro for small.
    if len(y) >= 20:
        stat, pval = stats.normaltest(y)
        test = "D'Agostino"
    else:
        stat, pval = stats.shapiro(y)
        test = "Shapiro"

    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = int(((y < lo) | (y > hi)).sum())

    recommend_log = bool(abs(skew) > 1.0 and (y > 0).all())
    return {
        "values": y, "n": len(y),
        "skewness": skew, "kurtosis": kurt,
        "normality_test": test, "normality_stat": float(stat), "normality_p": float(pval),
        "is_normal": bool(pval > 0.05),
        "n_outliers": n_outliers,
        "recommend_log": recommend_log,
    }


# =============================================================================
# AI RESEARCH ASSISTANT  —  plain-language interpretation
# =============================================================================
def ai_insights(summary: dict, pred: dict, redund: dict, diff: dict,
                learn: dict, causal: dict, latent_res: dict,
                target: dict) -> list[str]:
    """Turn every section's numbers into short interpretive statements."""
    out: list[str] = []
    top = pred.get("top_feature")
    if top:
        out.append(f"{top} appears to dominate prediction "
                   f"(highest mutual information with {summary['target']}).")

    # Weak latent influence.
    mi = latent_res.get("mutual_info", {})
    if mi:
        weakest = min(mi, key=mi.get)
        out.append(f"{weakest.replace('_', ' ')} has weak influence on the target.")

    if len(redund.get("high_corr_pairs", [])) >= 3 or len(redund.get("high_vif", [])) >= 3:
        out.append("Feature redundancy is high — several features carry overlapping information.")

    ratio = summary["n_features"] / max(summary["n_samples"], 1)
    if ratio > 0.15 or learn.get("best_r2", 0) < 0.1:
        out.append("Dataset appears underdetermined for this target.")

    if not causal.get("structure_present"):
        out.append("Structural descriptors are likely missing.")
        out.append(f"Collect {STRUCTURAL_SUGGESTIONS} to improve prediction.")

    if learn.get("insufficient"):
        out.append("This dataset may not contain enough predictive information "
                   "for the selected target.")

    if target.get("recommend_log"):
        out.append(f"Target is highly skewed (skew={target['skewness']:.2f}); "
                   "a log transform is recommended.")

    out.append(f"Estimated achievable CV R²: {diff['est_r2_low']:.2f}–{diff['est_r2_high']:.2f} "
               f"({diff['label']} problem).")
    return out


def final_conclusion(summary: dict, pred: dict, learn: dict, causal: dict,
                     target: dict) -> str:
    """One-paragraph verdict, templated from the computed statistics."""
    n = summary["n_samples"]
    n_num = summary["n_numerical"]
    max_corr = pred["max_pearson"]
    best_r2 = learn.get("best_r2", float("nan"))
    parts = [
        f"The current dataset contains {n} usable samples and {n_num} numerical features.",
        f"The strongest correlation with {summary['target']} is only {max_corr:.2f}.",
    ]
    if learn.get("insufficient"):
        parts.append(f"Cross-validation R² stays at or below 0.3 (best {best_r2:.2f}) "
                     "across multiple algorithms.")
        parts.append("This indicates that the current synthesis variables alone do not "
                     f"sufficiently explain {summary['target']}.")
    else:
        parts.append(f"Cross-validation reaches R² ≈ {best_r2:.2f}, so the synthesis "
                     "variables carry meaningful predictive signal.")
    if not causal.get("structure_present"):
        parts.append(f"Additional structural descriptors such as {STRUCTURAL_SUGGESTIONS} "
                     "are likely required for accurate prediction.")
    if target.get("recommend_log"):
        parts.append("The target is right-skewed; modelling log(target) may help.")
    return " ".join(parts)


# =============================================================================
# SECTION 10 — PDF REPORT
# =============================================================================
def build_pdf_report(out_path: str, summary: dict, pred: dict, diff: dict,
                     learn: dict, conclusion: str, insights: list[str],
                     chart_paths: list[tuple[str, str]]) -> str:
    """Assemble a multi-page PDF: text pages + every generated chart PNG.

    ``chart_paths`` is ``[(title, png_path), ...]``. Uses matplotlib's PdfPages,
    so no extra dependency is required.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import os

    def text_page(pdf, title, lines):
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.text(0.08, 0.94, title, fontsize=16, fontweight="bold")
        y = 0.88
        for ln in lines:
            wrapped = _wrap(ln, 95)
            for w in wrapped:
                fig.text(0.08, y, w, fontsize=10, va="top")
                y -= 0.028
            y -= 0.006
            if y < 0.06:
                pdf.savefig(fig); plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69)); y = 0.94
        pdf.savefig(fig); plt.close(fig)

    with PdfPages(out_path) as pdf:
        text_page(pdf, "Dataset Intelligence Report", [
            f"Target: {summary['target']}",
            f"Samples: {summary['n_samples']}    Features: {summary['n_features']} "
            f"({summary['n_numerical']} numeric, {summary['n_categorical']} categorical)",
            f"Missing overall: {summary['missing_pct_overall']}%    "
            f"Duplicate conditions: {summary['duplicate_conditions']}",
            f"Target std/range: {summary['target_std']:.3g} / {summary['target_range']:.3g}",
            "",
            f"Prediction difficulty: {diff['label']}  "
            f"(estimated achievable CV R² {diff['est_r2_low']:.2f}–{diff['est_r2_high']:.2f})",
            diff["explanation"],
            f"Best CV R²: {learn.get('best_r2', float('nan')):.3f}",
            "",
            "Top features (by mutual information):",
            *[f"  {r.feature}: MI={r.mutual_info:.3f}"
              for r in pred["table"].head(8).itertuples()],
        ])
        text_page(pdf, "AI Research Assistant — Findings", insights)
        text_page(pdf, "Final Conclusion", [conclusion])
        # One image page per chart.
        for title, path in chart_paths:
            if not path or not os.path.exists(path):
                continue
            img = plt.imread(path)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.95, title, fontsize=13, fontweight="bold")
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.86]); ax.axis("off")
            ax.imshow(img)
            pdf.savefig(fig); plt.close(fig)
    return out_path


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word-wrap for the PDF text pages."""
    words = str(text).split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur); cur = w
    lines.append(cur)
    return lines
