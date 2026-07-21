"""Reusable, leakage-safe cross-validation helpers for the desktop trainer.

The public functions in this module deliberately accept pandas objects so row
indices, targets, and grouping labels stay aligned throughout data cleaning and
evaluation.  Preprocessing is fitted separately inside every validation fold.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_VALIDATION_CONFIG = {
    "method": "random_kfold",
    "group_column": "",
    "n_splits": 5,
    "n_repeats": 10,
    "confidence_level": 0.95,
    "random_state": 42,
    "interval_method": "percentile",
}

VALIDATION_METHOD_LABELS = {
    "random_kfold": "Random KFold",
    "group_kfold": "GroupKFold",
    "repeated_grouped_cv": "Repeated Grouped CV",
}


def normalize_validation_config(config: Mapping | None = None) -> dict:
    """Return validated settings while accepting old/missing configuration keys."""
    out = dict(DEFAULT_VALIDATION_CONFIG)
    if config:
        out.update({k: v for k, v in dict(config).items() if v is not None})
    aliases = {
        "random": "random_kfold",
        "kfold": "random_kfold",
        "grouped": "group_kfold",
        "groupkfold": "group_kfold",
        "repeated_grouped": "repeated_grouped_cv",
    }
    out["method"] = aliases.get(str(out["method"]).strip().lower(),
                                str(out["method"]).strip().lower())
    if out["method"] not in VALIDATION_METHOD_LABELS:
        out["method"] = DEFAULT_VALIDATION_CONFIG["method"]
    out["group_column"] = str(out.get("group_column") or "")
    out["n_splits"] = max(2, int(out.get("n_splits", 5)))
    out["n_repeats"] = max(1, int(out.get("n_repeats", 10)))
    level = float(out.get("confidence_level", 0.95))
    if level > 1.0:
        level /= 100.0
    out["confidence_level"] = min(max(level, 0.50), 0.999)
    out["random_state"] = int(out.get("random_state", 42))
    if out.get("interval_method") not in {"percentile", "mean"}:
        out["interval_method"] = "percentile"
    return out


def validation_method_label(config_or_method: Mapping | str) -> str:
    method = (normalize_validation_config(config_or_method).get("method")
              if isinstance(config_or_method, Mapping) else str(config_or_method))
    return VALIDATION_METHOD_LABELS.get(method, method)


def _aligned_groups(groups: Sequence, n_rows: int) -> pd.Series:
    if groups is None:
        raise ValueError("A grouping column is required for grouped validation.")
    result = pd.Series(groups).reset_index(drop=True)
    if len(result) != n_rows:
        raise ValueError(
            f"Grouping labels are not aligned with the training rows "
            f"({len(result)} labels for {n_rows} rows)."
        )
    if result.isna().any():
        raise ValueError("The grouping column contains missing values after cleaning.")
    return result


def generate_repeated_group_splits(
    groups: Sequence,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield deterministic, row-count-balanced repeated grouped CV splits.

    Unique group labels are shuffled independently for each repeat and assigned
    greedily to the fold currently containing the fewest rows.  A group is
    therefore indivisible and can never leak between train and validation rows.
    """
    g = pd.Series(groups).reset_index(drop=True)
    if g.isna().any():
        raise ValueError("Grouping labels cannot contain missing values.")
    n_splits = int(n_splits)
    n_repeats = int(n_repeats)
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("Use at least 2 folds and 1 repeat.")
    counts = g.value_counts(sort=False)
    if len(counts) < n_splits:
        raise ValueError(
            f"Number of unique groups ({len(counts)}) is smaller than the "
            f"number of folds ({n_splits})."
        )

    labels = counts.index.to_numpy(dtype=object)
    for repeat in range(n_repeats):
        rng = np.random.RandomState(int(random_state) + repeat)
        shuffled = labels[rng.permutation(len(labels))]
        # Descending-size placement improves balance; the random tie-breaker
        # preserves genuinely different group allocations between repeats.
        tie_order = {label: pos for pos, label in enumerate(shuffled)}
        ordered = sorted(shuffled, key=lambda label: (-int(counts[label]),
                                                       tie_order[label]))
        fold_groups: list[list[object]] = [[] for _ in range(n_splits)]
        fold_rows = np.zeros(n_splits, dtype=int)
        fold_ties = rng.permutation(n_splits).tolist()
        for label in ordered:
            smallest = int(min(range(n_splits),
                               key=lambda f: (fold_rows[f], fold_ties.index(f))))
            fold_groups[smallest].append(label)
            fold_rows[smallest] += int(counts[label])
            # Rotate equal-size tie priority so one fold does not always win.
            fold_ties = fold_ties[1:] + fold_ties[:1]

        all_idx = np.arange(len(g), dtype=int)
        for assigned in fold_groups:
            valid_mask = g.isin(assigned).to_numpy()
            valid_idx = all_idx[valid_mask]
            train_idx = all_idx[~valid_mask]
            yield train_idx, valid_idx


def repeated_group_kfold_splits(
    groups: Sequence,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Compatibility alias using the name shown in the UI specification."""
    return generate_repeated_group_splits(groups, n_splits, n_repeats, random_state)


def make_cv_splits(
    n_rows: int,
    config: Mapping | None = None,
    groups: Sequence | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create concrete CV splits for one of the three supported methods."""
    cfg = normalize_validation_config(config)
    n_rows = int(n_rows)
    if n_rows < cfg["n_splits"]:
        raise ValueError(
            f"Only {n_rows} usable rows are available for {cfg['n_splits']} folds."
        )
    if cfg["method"] == "random_kfold":
        cv = KFold(n_splits=cfg["n_splits"], shuffle=True,
                   random_state=cfg["random_state"])
        return list(cv.split(np.arange(n_rows)))

    aligned = _aligned_groups(groups, n_rows)
    n_groups = int(aligned.nunique())
    if n_groups < cfg["n_splits"]:
        raise ValueError(
            f"Number of unique groups ({n_groups}) is smaller than the "
            f"number of folds ({cfg['n_splits']})."
        )
    if cfg["method"] == "group_kfold":
        cv = GroupKFold(n_splits=cfg["n_splits"])
        return list(cv.split(np.arange(n_rows), groups=aligned))
    return list(generate_repeated_group_splits(
        aligned, cfg["n_splits"], cfg["n_repeats"], cfg["random_state"]
    ))


def validate_group_integrity(
    splits: Iterable[tuple[Sequence[int], Sequence[int]]], groups: Sequence
) -> bool:
    """Raise on group leakage or split rows; return True when every split is safe."""
    g = pd.Series(groups).reset_index(drop=True)
    for split_no, (train_idx, valid_idx) in enumerate(splits, start=1):
        train_idx = np.asarray(train_idx, dtype=int)
        valid_idx = np.asarray(valid_idx, dtype=int)
        if np.intersect1d(train_idx, valid_idx).size:
            raise ValueError(f"Split {split_no} contains the same row in train and validation.")
        overlap = set(g.iloc[train_idx]) & set(g.iloc[valid_idx])
        if overlap:
            sample = sorted(map(str, overlap))[:3]
            raise ValueError(
                f"Group leakage in split {split_no}; group(s) occur on both sides: {sample}."
            )
    return True


def calculate_metric_summary(
    scores: Sequence[float],
    confidence_level: float = 0.95,
    interval_method: str = "percentile",
) -> dict:
    """Summarise fold/repeat scores with percentile or t-based uncertainty."""
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": np.nan, "std": np.nan, "lower": np.nan, "upper": np.nan,
                "n": 0, "confidence_level": confidence_level,
                "interval_method": interval_method}
    level = float(confidence_level)
    if level > 1:
        level /= 100.0
    alpha = 1.0 - level
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if interval_method == "mean" and values.size > 1:
        try:
            from scipy.stats import t
            critical = float(t.ppf(1.0 - alpha / 2.0, values.size - 1))
        except Exception:  # pragma: no cover - scipy is optional to this module
            critical = 1.96
        half = critical * std / np.sqrt(values.size)
        lower, upper = mean - half, mean + half
    else:
        lower, upper = np.percentile(values, [100 * alpha / 2.0,
                                              100 * (1.0 - alpha / 2.0)])
    return {"mean": mean, "std": std, "lower": float(lower), "upper": float(upper),
            "n": int(values.size), "confidence_level": level,
            "interval_method": interval_method}


def _dense_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


def build_preprocessor(
    numeric_columns: Sequence[str], categorical_columns: Sequence[str]
) -> ColumnTransformer:
    """Build fold-local imputation and encoding with unknown-category support."""
    num = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _dense_one_hot_encoder()),
    ])
    return ColumnTransformer(
        [("num", num, list(numeric_columns)),
         ("cat", cat, list(categorical_columns))],
        remainder="drop",
    )


def _target_frame(y, target_names: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(y, pd.DataFrame):
        result = y.reset_index(drop=True).copy()
    elif isinstance(y, pd.Series):
        result = y.reset_index(drop=True).to_frame(name=y.name or "target")
    else:
        arr = np.asarray(y)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        names = list(target_names or [f"target_{i + 1}" for i in range(arr.shape[1])])
        result = pd.DataFrame(arr, columns=names)
    if target_names:
        result.columns = list(target_names)
    return result


def _fit_y(y_frame: pd.DataFrame):
    """Give single-target estimators a Series and multi-target estimators a frame."""
    return y_frame.iloc[:, 0] if y_frame.shape[1] == 1 else y_frame


def _prediction_2d(pred, n_targets: int) -> np.ndarray:
    arr = np.asarray(pred, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] != n_targets:
        raise ValueError(f"Estimator returned {arr.shape[1]} targets; expected {n_targets}.")
    return arr


def _metric_triplet(y_true, y_pred) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return r2, rmse, mae


def evaluate_model_cv(
    estimator,
    X: pd.DataFrame,
    y,
    validation_config: Mapping | None = None,
    groups: Sequence | None = None,
    numeric_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
) -> dict:
    """Evaluate an estimator and return pooled OOF and fold-level uncertainty.

    Every fold owns a fresh :class:`ColumnTransformer`; imputation and category
    discovery therefore never see validation rows.  Repeated validation retains
    all row predictions and returns their mean and standard deviation.
    """
    cfg = normalize_validation_config(validation_config)
    X = pd.DataFrame(X).reset_index(drop=True).copy()
    y_frame = _target_frame(y)
    if len(X) != len(y_frame):
        raise ValueError(f"X and y are not aligned ({len(X)} vs {len(y_frame)} rows).")
    numeric_columns = list(numeric_columns if numeric_columns is not None
                           else X.select_dtypes(include=[np.number, "bool"]).columns)
    categorical_columns = list(categorical_columns if categorical_columns is not None
                               else [c for c in X.columns if c not in numeric_columns])
    missing = [c for c in numeric_columns + categorical_columns if c not in X.columns]
    if missing:
        raise ValueError(f"Preprocessing columns not found in X: {missing}")

    splits = make_cv_splits(len(X), cfg, groups)
    if cfg["method"] != "random_kfold":
        validate_group_integrity(splits, _aligned_groups(groups, len(X)))

    n_targets = y_frame.shape[1]
    pred_sum = np.zeros((len(X), n_targets), dtype=float)
    pred_sumsq = np.zeros_like(pred_sum)
    pred_count = np.zeros(len(X), dtype=int)
    prediction_samples: list[list[np.ndarray]] = [[] for _ in range(len(X))]
    distributions = {
        target: {"r2": [], "rmse": [], "mae": []} for target in y_frame.columns
    }

    for train_idx, valid_idx in splits:
        fold_model = Pipeline([
            ("preprocess", build_preprocessor(numeric_columns, categorical_columns)),
            ("model", clone(estimator)),
        ])
        y_train = _fit_y(y_frame.iloc[train_idx])
        fold_model.fit(X.iloc[train_idx], y_train)
        pred = _prediction_2d(fold_model.predict(X.iloc[valid_idx]), n_targets)
        pred_sum[valid_idx] += pred
        pred_sumsq[valid_idx] += pred ** 2
        pred_count[valid_idx] += 1
        for row_index, row_prediction in zip(valid_idx, pred):
            prediction_samples[int(row_index)].append(np.asarray(row_prediction).copy())
        for i, target in enumerate(y_frame.columns):
            r2, rmse, mae = _metric_triplet(y_frame.iloc[valid_idx, i], pred[:, i])
            distributions[target]["r2"].append(r2)
            distributions[target]["rmse"].append(rmse)
            distributions[target]["mae"].append(mae)

    if np.any(pred_count == 0):
        missing_rows = np.flatnonzero(pred_count == 0)[:10].tolist()
        raise ValueError(f"OOF predictions do not cover every row; missing {missing_rows}.")
    oof = pred_sum / pred_count[:, None]
    variance = np.maximum(pred_sumsq / pred_count[:, None] - oof ** 2, 0.0)
    oof_std = np.sqrt(variance)

    summaries, pooled = {}, {}
    for i, target in enumerate(y_frame.columns):
        summaries[target] = {
            metric: calculate_metric_summary(
                scores, cfg["confidence_level"], cfg["interval_method"]
            )
            for metric, scores in distributions[target].items()
        }
        r2, rmse, mae = _metric_triplet(y_frame.iloc[:, i], oof[:, i])
        pooled[target] = {"r2": r2, "rmse": rmse, "mae": mae}

    return {
        "validation_config": cfg,
        "splits": splits,
        "oof_predictions": pd.DataFrame(oof, columns=y_frame.columns),
        "oof_prediction_std": pd.DataFrame(oof_std, columns=y_frame.columns),
        "oof_prediction_count": pd.Series(pred_count, name="prediction_count"),
        "oof_prediction_samples": [np.vstack(samples) for samples in prediction_samples],
        "metric_distributions": distributions,
        "metric_summaries": summaries,
        "pooled_metrics": pooled,
        "preprocessing": "SimpleImputer and OneHotEncoder fitted within each fold",
    }


def feature_diagnostics(
    original_predictors: int,
    encoded_predictors: int,
    training_rows: int,
    independent_groups: int | None = None,
) -> dict:
    """Calculate effective sample-size ratios and publication warnings."""
    p = max(int(encoded_predictors), 1)
    n = int(training_rows)
    g = int(independent_groups) if independent_groups is not None else n
    rows_per = n / p
    groups_per = g / p
    warnings = []
    if groups_per < 5:
        warnings.append("Warning: fewer than 5 independent groups per encoded predictor. High overfitting risk.")
    if rows_per < 10:
        warnings.append("Caution: fewer than 10 rows per encoded predictor. Consider reducing features or combining rare categories.")
    return {
        "original_predictors": int(original_predictors),
        "encoded_predictors": int(encoded_predictors),
        "training_rows": n,
        "independent_groups": g,
        "rows_per_encoded_predictor": float(rows_per),
        "groups_per_encoded_predictor": float(groups_per),
        "warnings": warnings,
    }


def load_settings_compat(spec: Mapping | None) -> dict:
    """Add new settings sections to an older JSON settings dictionary."""
    result = dict(spec or {})
    result["validation"] = normalize_validation_config(result.get("validation"))
    training = dict(result.get("training") or {})
    training.setdefault("single_target_mode", False)
    training.setdefault("target", result.get("target", ""))
    result["training"] = training
    reporting = dict(result.get("reporting") or {})
    reporting.setdefault("top_feature_count", 20)
    reporting.setdefault("show_feature_ratio_warnings", True)
    result["reporting"] = reporting
    return result


def load_model_bundle_compat(loaded) -> dict:
    """Normalise old raw estimators and older bundle dictionaries."""
    if isinstance(loaded, Mapping) and "model" in loaded:
        bundle = dict(loaded)
    else:
        bundle = {"model": loaded}
    bundle.setdefault("feature_columns", list(getattr(bundle["model"], "feature_names_in_", [])))
    bundle.setdefault("numeric_schema", {})
    bundle.setdefault("categorical_schema", {})
    bundle.setdefault("targets", [])
    bundle["validation_config"] = normalize_validation_config(bundle.get("validation_config"))
    bundle.setdefault("group_column", bundle["validation_config"].get("group_column", ""))
    bundle.setdefault("metrics", [])
    bundle.setdefault("metric_distributions", {})
    bundle.setdefault("feature_diagnostics", {})
    bundle.setdefault("oof_prediction_samples", None)
    return bundle
