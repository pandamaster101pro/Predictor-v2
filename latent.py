"""
latent.py  —  Latent-variable engine for the Predictor app.
============================================================

Pure, GUI-free logic so it can be unit-tested head-lessly. Two families of
latent variables are built here:

A. **Interpretable engineered indices** — transparent formulas (Thermal Severity,
   Chemical Treatment, Biomass Composition, Process Complexity). See
   ``ENGINEERED_FORMULAS`` for the human-readable formula strings shown in the UI.

B. **Learned latent variables** — PCA and PLSRegression (optional autoencoder,
   gated on sample count).

Everything that feeds cross-validation is **leakage-safe**: imputation, encoding,
scaling, PCA and PLS are wrapped in sklearn ``Pipeline`` / ``ColumnTransformer``
objects and fit *inside* each CV training fold via :func:`compare_pipelines`.
The full-dataset fits (:func:`fit_pca`, :func:`compute_engineered_latents`) exist
only for display/export and are clearly separated from the evaluation path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# --- Recommended defaults (requirements 3-5) ---------------------------------
DEFAULT_CATEGORICAL: list[str] = [
    "Material", "Pretreat_family", "Post_treat_family",
    "Atmosphere1", "Atmosphere2", "Fiber_sample",
]
DEFAULT_NUMERICAL: list[str] = [
    "Py1_temp_C", "Py1_time_min", "Py2_temp_C", "Py2_time_min",
    "Cellulose_avg_pct", "Hemicellulose_avg_pct", "Lignin_avg_pct",
    "Has_pretreat", "Has_post_treat", "Has_any_additive",
    "Two_step_pyrolysis", "Total_py_time_min", "Max_temp_C",
]
# Capacity targets: the selected one is the label; the rest are always excluded
# so they can never leak in as features.
CAPACITY_TARGETS: list[str] = [
    "LIB_1A_mean", "LIB_0_1A_mean", "LIB_30mA_mean",
    "SIB_1A_mean", "SIB_0_1A_mean", "SIB_30mA_mean",
]
DEFAULT_ALWAYS_EXCLUDE: list[str] = ["Condition_ID", "Replicate_rows"]

ENGINEERED_INDEX_NAMES = [
    "Thermal_Severity_Index", "Chemical_Treatment_Index",
    "Biomass_Composition_Index", "Process_Complexity_Index",
]

# Human-readable formulas surfaced in the UI (requirement 7: display formulas).
ENGINEERED_FORMULAS: dict[str, str] = {
    "Thermal_Severity_Index":
        "mean( z(Max_temp_C), z(Total_py_time_min), z(Two_step_pyrolysis) )",
    "Chemical_Treatment_Index":
        "w1*Has_pretreat + w2*Has_post_treat + w3*Has_any_additive  (weights editable)",
    "Biomass_Composition_Index":
        "mean( z(Cellulose), z(Hemicellulose), z(Lignin) )  OR  PCA-1 of the standardized trio",
    "Process_Complexity_Index":
        "Has_pretreat + Has_post_treat + Has_any_additive + Two_step_pyrolysis"
        " + (1 if Atmosphere1 != Atmosphere2 else 0)",
}


@dataclass
class LatentConfig:
    """User selections for the Latent Variables tab."""
    target: str
    categorical: list[str] = field(default_factory=lambda: list(DEFAULT_CATEGORICAL))
    numerical: list[str] = field(default_factory=lambda: list(DEFAULT_NUMERICAL))
    excluded: list[str] = field(default_factory=list)
    chem_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    biomass_method: str = "mean"  # "mean" or "pca"

    def feature_columns(self, available: list[str]) -> tuple[list[str], list[str]]:
        """Return (categorical, numerical) restricted to columns that exist and are
        NOT excluded / a capacity target / the label — the leakage-safe feature set."""
        blocked = set(self.excluded) | set(CAPACITY_TARGETS) | {self.target}
        cat = [c for c in self.categorical if c in available and c not in blocked]
        num = [c for c in self.numerical if c in available and c not in blocked]
        return cat, num


def default_excluded(target: str, available: list[str]) -> list[str]:
    """Requirement 5: Condition_ID, Replicate_rows, and every capacity target
    except the selected one — filtered to columns that actually exist."""
    cols = set(DEFAULT_ALWAYS_EXCLUDE)
    cols |= {t for t in CAPACITY_TARGETS if t != target}
    return [c for c in available if c in cols]


# =============================================================================
# STANDARDIZATION HELPERS
# =============================================================================
def _zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Z-score with a safe zero-variance fallback (returns zeros)."""
    if std is None or std == 0 or np.isnan(std):
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Numeric view of a column; missing column -> all-NaN series of right length."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


# =============================================================================
# A. ENGINEERED LATENT INDICES  (leakage-safe transformer)
# =============================================================================
class EngineeredLatents(BaseEstimator, TransformerMixin):
    """
    Compute the four interpretable indices. ``fit`` learns the median (for
    imputation) and mean/std (for standardization) of each required column from
    the *training* rows only, so it is safe inside a CV fold.

    Output column order == :data:`ENGINEERED_INDEX_NAMES`.
    """

    _STD_COLS = ["Max_temp_C", "Total_py_time_min", "Two_step_pyrolysis",
                 "Cellulose_avg_pct", "Hemicellulose_avg_pct", "Lignin_avg_pct"]
    _FLAG_COLS = ["Has_pretreat", "Has_post_treat", "Has_any_additive",
                  "Two_step_pyrolysis"]

    def __init__(self, weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 biomass_method: str = "mean") -> None:
        self.weights = weights
        self.biomass_method = biomass_method

    def fit(self, X: pd.DataFrame, y: object = None) -> "EngineeredLatents":
        X = _ensure_df(X)
        self.medians_: dict[str, float] = {}
        self.means_: dict[str, float] = {}
        self.stds_: dict[str, float] = {}
        for c in self._STD_COLS + self._FLAG_COLS:
            s = _col(X, c)
            med = s.median()
            self.medians_[c] = 0.0 if pd.isna(med) else float(med)
            filled = s.fillna(self.medians_[c])
            self.means_[c] = float(filled.mean())
            self.stds_[c] = float(filled.std(ddof=0))
        # Optional PCA for the biomass trio (fit on standardized training values).
        self.biomass_pca_: PCA | None = None
        if self.biomass_method == "pca":
            trio = self._standardized_trio(X)
            self.biomass_pca_ = PCA(n_components=1, random_state=RANDOM_STATE).fit(trio)
        return self

    # -- helpers ----------------------------------------------------------------
    def _impute_std(self, X: pd.DataFrame, col: str) -> np.ndarray:
        s = _col(X, col).fillna(self.medians_[col]).to_numpy(dtype=float)
        return _zscore(s, self.means_[col], self.stds_[col])

    def _impute_flag(self, X: pd.DataFrame, col: str) -> np.ndarray:
        return _col(X, col).fillna(self.medians_[col]).to_numpy(dtype=float)

    def _standardized_trio(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            self._impute_std(X, "Cellulose_avg_pct"),
            self._impute_std(X, "Hemicellulose_avg_pct"),
            self._impute_std(X, "Lignin_avg_pct"),
        ])

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = _ensure_df(X)
        # Thermal Severity
        thermal = np.mean(np.column_stack([
            self._impute_std(X, "Max_temp_C"),
            self._impute_std(X, "Total_py_time_min"),
            self._impute_std(X, "Two_step_pyrolysis"),
        ]), axis=1)

        # Chemical Treatment (weighted sum of flags)
        w1, w2, w3 = self.weights
        chemical = (w1 * self._impute_flag(X, "Has_pretreat")
                    + w2 * self._impute_flag(X, "Has_post_treat")
                    + w3 * self._impute_flag(X, "Has_any_additive"))

        # Biomass Composition
        trio = self._standardized_trio(X)
        if self.biomass_method == "pca" and self.biomass_pca_ is not None:
            biomass = self.biomass_pca_.transform(trio)[:, 0]
        else:
            biomass = trio.mean(axis=1)

        # Process Complexity (flag sum + atmosphere-change bump)
        complexity = (self._impute_flag(X, "Has_pretreat")
                      + self._impute_flag(X, "Has_post_treat")
                      + self._impute_flag(X, "Has_any_additive")
                      + self._impute_flag(X, "Two_step_pyrolysis"))
        atmo_change = _atmosphere_change(X)
        complexity = complexity + atmo_change

        return np.column_stack([thermal, chemical, biomass, complexity])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(ENGINEERED_INDEX_NAMES, dtype=object)


def _atmosphere_change(X: pd.DataFrame) -> np.ndarray:
    """1.0 where Atmosphere1 differs from Atmosphere2, else 0.0 (NaN-safe)."""
    a1 = X["Atmosphere1"].astype(str) if "Atmosphere1" in X.columns else pd.Series("", index=X.index)
    a2 = X["Atmosphere2"].astype(str) if "Atmosphere2" in X.columns else pd.Series("", index=X.index)
    return (a1.to_numpy() != a2.to_numpy()).astype(float)


def _ensure_df(X: object) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def compute_engineered_latents(
    df: pd.DataFrame,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    biomass_method: str = "mean",
) -> pd.DataFrame:
    """Full-dataset engineered indices for **display / export** (not CV).

    Missing numeric inputs are median-imputed; optional Py2 blanks never drop a
    row. Returns a DataFrame indexed like ``df``.
    """
    tf = EngineeredLatents(weights=weights, biomass_method=biomass_method).fit(df)
    out = tf.transform(df)
    return pd.DataFrame(out, columns=ENGINEERED_INDEX_NAMES, index=df.index)


# =============================================================================
# PREPROCESSING BUILDERS  (leakage-safe; used inside pipelines)
# =============================================================================
def _onehot() -> OneHotEncoder:
    # sparse_output kw name changed across sklearn versions; guard both.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - old sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _as_str_array(a):
    """Cast every cell to str so OneHotEncoder never sees a mixed int/str column.

    A messy column can hold both numbers and text (e.g. 900 and 'Ar'); the
    encoder rejects that ("input must be uniformly strings or numbers"). Runs
    AFTER imputation, so there are no NaNs left to turn into the string 'nan'.
    Module-level (not a lambda) so the pipeline stays picklable under n_jobs.
    """
    return np.asarray(a, dtype=object).astype(str)


def make_tree_preprocessor(categorical: list[str], numerical: list[str]) -> ColumnTransformer:
    """Impute (median / 'Missing') + one-hot. No scaling — trees don't need it."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numerical),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("tostr", FunctionTransformer(_as_str_array,
                                          feature_names_out="one-to-one")),
            ("onehot", _onehot()),
        ]), categorical),
    ], remainder="drop")


def make_scaled_preprocessor(categorical: list[str], numerical: list[str]) -> Pipeline:
    """Impute + one-hot + **standardize everything** (for PCA / PLS)."""
    return Pipeline([
        ("impute_encode", make_tree_preprocessor(categorical, numerical)),
        ("scale", StandardScaler()),
    ])


def make_latent_features(config: LatentConfig, categorical: list[str],
                         numerical: list[str], n_pca: int) -> FeatureUnion:
    """FeatureUnion of engineered indices + PCA scores (both fit in-fold)."""
    transformers: list[tuple[str, object]] = [
        ("engineered", EngineeredLatents(config.chem_weights, config.biomass_method)),
    ]
    if n_pca and n_pca > 0:
        transformers.append(("pca", Pipeline([
            ("prep", make_scaled_preprocessor(categorical, numerical)),
            ("pca", PCA(n_components=n_pca, random_state=RANDOM_STATE)),
        ])))
    return FeatureUnion(transformers)


# =============================================================================
# COMPONENT-COUNT LIMITS  (requirements 8 & 9)
# =============================================================================
def max_pca_components(n_samples: int, n_features: int) -> int:
    """PCA can extract at most min(n_samples, n_features) components."""
    return max(1, min(int(n_samples), int(n_features)))


def clamp_pca_components(requested: int, n_samples: int, n_features: int) -> int:
    """Clamp a requested PCA component count into [2, max] (UI offers 2-10)."""
    hi = max_pca_components(n_samples, n_features)
    return int(max(2, min(requested, max(2, hi))))


def max_pls_components(n_samples: int, n_features: int, n_cv_splits: int = 5) -> int:
    """PLS components are bounded by features and by the smallest CV train fold."""
    train_rows = int(n_samples) - int(np.ceil(n_samples / n_cv_splits))
    return max(1, min(int(n_features), max(1, train_rows - 1)))


def clamp_pls_components(requested: int, n_samples: int, n_features: int,
                         n_cv_splits: int = 5) -> int:
    """Clamp a requested PLS component count into [1, max] (UI offers 1-10)."""
    hi = max_pls_components(n_samples, n_features, n_cv_splits)
    return int(max(1, min(requested, max(1, hi))))


# =============================================================================
# PCA  (full-dataset fit for display / export)
# =============================================================================
def fit_pca(df: pd.DataFrame, config: LatentConfig, n_components: int) -> dict:
    """Fit PCA on the whole dataset for visualization/export only.

    Returns explained-variance ratios, cumulative variance, a loadings DataFrame
    (feature x PC) and a scores DataFrame (row x PC).
    """
    cat, num = config.feature_columns(list(df.columns))
    prep = make_scaled_preprocessor(cat, num)
    X = prep.fit_transform(df)
    n_comp = clamp_pca_components(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(X)
    scores = pca.transform(X)

    feat_names = _feature_names_from_prep(prep, cat, num)
    pc_cols = [f"PC{i + 1}" for i in range(n_comp)]
    loadings = pd.DataFrame(pca.components_.T, index=feat_names, columns=pc_cols)
    scores_df = pd.DataFrame(scores, columns=pc_cols, index=df.index)
    return {
        "n_components": n_comp,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        "loadings": loadings,
        "scores": scores_df,
        "feature_names": feat_names,
    }


def _feature_names_from_prep(prep: Pipeline, cat: list[str], num: list[str]) -> list[str]:
    """Best-effort feature names out of the impute+encode+scale preprocessor."""
    try:
        ct: ColumnTransformer = prep.named_steps["impute_encode"]
        names = list(ct.get_feature_names_out())
        # Strip the "num__"/"cat__" prefixes ColumnTransformer adds.
        return [n.split("__", 1)[-1] for n in names]
    except Exception:  # pragma: no cover - fallback
        return num + cat


# =============================================================================
# PLS  (leakage-safe 5-fold CV on the selected target)
# =============================================================================
def evaluate_pls(df: pd.DataFrame, config: LatentConfig, n_components: int,
                 n_splits: int = 5) -> dict:
    """5-fold CV of a leakage-safe PLS pipeline. Reports mean/std R², RMSE, MAE."""
    cat, num = config.feature_columns(list(df.columns))
    X = df[cat + num]
    y = pd.to_numeric(df[config.target], errors="coerce")
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    n_feat_est = len(num) + sum(max(1, df[c].astype(str).nunique()) for c in cat)
    k = clamp_pls_components(n_components, len(y), n_feat_est, n_splits)

    pipe = Pipeline([
        ("prep", make_scaled_preprocessor(cat, num)),
        ("pls", PLSRegression(n_components=k)),
    ])
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = _cv_scores(pipe, X, y, cv)
    scores["n_components"] = k
    return scores


# =============================================================================
# PIPELINE COMPARISON  A / B / C  (requirement 10, leakage-safe)
# =============================================================================
def _models() -> dict[str, BaseEstimator]:
    """ExtraTrees always; XGBoost when installed (never crash if missing)."""
    models: dict[str, BaseEstimator] = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    except ImportError:
        pass
    return models


def _feature_builder(variant: str, config: LatentConfig, cat: list[str],
                     num: list[str], n_pca: int) -> object:
    """Preprocessing/features for pipeline A (original), B (latent), C (both)."""
    if variant == "A":
        return make_tree_preprocessor(cat, num)
    if variant == "B":
        return make_latent_features(config, cat, num, n_pca)
    if variant == "C":
        return FeatureUnion([
            ("original", make_tree_preprocessor(cat, num)),
            ("latent", make_latent_features(config, cat, num, n_pca)),
        ])
    raise ValueError(f"Unknown variant {variant!r}")


def compare_pipelines(df: pd.DataFrame, config: LatentConfig, n_pca: int = 3,
                      n_splits: int = 5,
                      progress: Callable[[str], None] | None = None) -> dict:
    """
    Compare pipelines A/B/C x {ExtraTrees, XGBoost} with the SAME CV folds.
    All preprocessing is fit inside each fold (no leakage). Returns nested dict:
    ``{variant: {model: {r2_mean, r2_std, rmse_mean, ...}}}``.
    """
    cat, num = config.feature_columns(list(df.columns))
    X = df[cat + num]
    y = pd.to_numeric(df[config.target], errors="coerce")
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    models = _models()
    results: dict[str, dict] = {}
    for variant in ("A", "B", "C"):
        results[variant] = {}
        for mname, model in models.items():
            if progress:
                progress(f"Pipeline {variant} · {mname}…")
            k = clamp_pca_components(n_pca, len(y), max(2, len(num))) if variant != "A" else 0
            features = _feature_builder(variant, config, cat, num, k)
            pipe = Pipeline([("features", features), ("model", clone(model))])
            try:
                results[variant][mname] = _cv_scores(pipe, X, y, cv)
            except Exception as exc:  # noqa: BLE001 - one bad combo shouldn't stop others
                results[variant][mname] = {"error": f"{type(exc).__name__}: {exc}"}
    results["_meta"] = {"n_rows": int(len(y)), "n_cat": len(cat), "n_num": len(num),
                        "models": list(models.keys())}
    return results


def oof_predict(df: pd.DataFrame, config: LatentConfig, variant: str = "C",
                n_pca: int = 3, n_splits: int = 5,
                model_name: str = "ExtraTrees") -> tuple[np.ndarray, np.ndarray, str]:
    """Leakage-safe out-of-fold predictions for actual-vs-predicted / residual charts."""
    cat, num = config.feature_columns(list(df.columns))
    X = df[cat + num]
    y = pd.to_numeric(df[config.target], errors="coerce")
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    models = _models()
    used = model_name if model_name in models else next(iter(models))
    model = models[used]
    k = clamp_pca_components(n_pca, len(y), max(2, len(num))) if variant != "A" else 0
    features = _feature_builder(variant, config, cat, num, k)
    pipe = Pipeline([("features", features), ("model", clone(model))])
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(pipe, X, y, cv=cv)
    return y.to_numpy(), np.asarray(pred).ravel(), used


def fit_full_pipeline(df: pd.DataFrame, config: LatentConfig, variant: str = "C",
                      n_pca: int = 3, model_name: str = "ExtraTrees") -> Pipeline:
    """Fit a deployable pipeline on ALL rows (for joblib export)."""
    cat, num = config.feature_columns(list(df.columns))
    X = df[cat + num]
    y = pd.to_numeric(df[config.target], errors="coerce")
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    models = _models()
    used = model_name if model_name in models else next(iter(models))
    model = models[used]
    k = clamp_pca_components(n_pca, len(y), max(2, len(num))) if variant != "A" else 0
    features = _feature_builder(variant, config, cat, num, k)
    pipe = Pipeline([("features", features), ("model", clone(model))])
    pipe.fit(X, y)
    return pipe


# =============================================================================
# SHARED CV SCORER
# =============================================================================
def _cv_scores(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series,
               cv: KFold) -> dict:
    """Run cross_validate and fold the three metrics into mean/std."""
    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }
    cvres = cross_validate(estimator, X, y, cv=cv, scoring=scoring,
                           return_train_score=False, n_jobs=None)
    out: dict[str, float] = {}
    for key in ("r2", "rmse", "mae"):
        vals = cvres[f"test_{key}"]
        if key != "r2":
            vals = -vals  # neg_* scorers -> positive error
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


# =============================================================================
# EXPORT HELPERS
# =============================================================================
def build_export_frame(df: pd.DataFrame, config: LatentConfig,
                       pca_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """Original rows + engineered indices (+ PCA scores) for XLSX export.

    Experimental values in ``df`` are copied verbatim — never modified.
    """
    eng = compute_engineered_latents(df, config.chem_weights, config.biomass_method)
    parts = [df.reset_index(drop=True), eng.reset_index(drop=True)]
    if pca_scores is not None:
        parts.append(pca_scores.reset_index(drop=True))
    return pd.concat(parts, axis=1)


def comparison_to_frame(results: dict) -> pd.DataFrame:
    """Flatten :func:`compare_pipelines` output into a tidy CSV-ready table."""
    rows = []
    labels = {"A": "A: original", "B": "B: latent", "C": "C: original+latent"}
    for variant in ("A", "B", "C"):
        for mname, sc in results.get(variant, {}).items():
            if "error" in sc:
                rows.append({"pipeline": labels[variant], "model": mname,
                             "error": sc["error"]})
                continue
            rows.append({
                "pipeline": labels[variant], "model": mname,
                "R2_mean": sc["r2_mean"], "R2_std": sc["r2_std"],
                "RMSE_mean": sc["rmse_mean"], "RMSE_std": sc["rmse_std"],
                "MAE_mean": sc["mae_mean"], "MAE_std": sc["mae_std"],
            })
    return pd.DataFrame(rows)


def autoencoder_available(n_samples: int) -> tuple[bool, str]:
    """Requirement 7B: autoencoder only when >=500 samples AND a DL backend exists."""
    if n_samples < 500:
        return False, f"Autoencoder needs >=500 samples (have {n_samples})."
    try:
        import torch  # noqa: F401
        return True, "PyTorch backend available."
    except ImportError:
        return False, "Autoencoder needs an optional deep-learning backend (torch) — not installed."
