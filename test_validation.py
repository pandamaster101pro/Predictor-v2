"""Synthetic regression tests for publication-quality grouped validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

import validation as V


def synthetic_frame(n_groups=12, rows_per_group=5, seed=7):
    rng = np.random.RandomState(seed)
    groups = np.repeat([f"condition_{i:02d}" for i in range(n_groups)], rows_per_group)
    n = len(groups)
    x = rng.normal(size=n)
    cat = np.where(np.arange(n) % 3 == 0, "A", "B")
    group_effect = np.repeat(rng.normal(scale=0.4, size=n_groups), rows_per_group)
    return pd.DataFrame({
        "Condition_ID": groups,
        "Sample_ID": [f"sample_{i}" for i in range(n)],
        "temperature": x,
        "route": cat,
        "SIB_0_1A": 3.0 * x + group_effect + rng.normal(scale=0.15, size=n),
        "SIB_1A": 1.5 * x - group_effect + rng.normal(scale=0.2, size=n),
    })


def grouped_config(**overrides):
    cfg = {
        "method": "repeated_grouped_cv", "group_column": "Condition_ID",
        "n_splits": 4, "n_repeats": 3, "confidence_level": 0.95,
        "random_state": 42,
    }
    cfg.update(overrides)
    return cfg


def split_signature(splits, groups):
    g = pd.Series(groups).reset_index(drop=True)
    return [tuple(sorted(g.iloc[test].unique())) for _, test in splits]


def test_no_group_overlap_and_each_group_stays_together():
    df = synthetic_frame()
    splits = V.make_cv_splits(len(df), grouped_config(), df["Condition_ID"])
    assert V.validate_group_integrity(splits, df["Condition_ID"])
    for train, valid in splits:
        assert not (set(df.iloc[train]["Condition_ID"]) & set(df.iloc[valid]["Condition_ID"]))


def test_repeated_splits_reproducible_and_repeats_differ():
    df = synthetic_frame()
    a = V.make_cv_splits(len(df), grouped_config(), df["Condition_ID"])
    b = V.make_cv_splits(len(df), grouped_config(), df["Condition_ID"])
    assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
               for x, y in zip(a, b))
    signatures = split_signature(a, df["Condition_ID"])
    per_repeat = [set(signatures[i:i + 4]) for i in range(0, len(signatures), 4)]
    assert len({tuple(sorted(value)) for value in per_repeat}) > 1


def test_unique_group_count_must_cover_folds():
    groups = pd.Series(["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="unique groups"):
        V.make_cv_splits(4, grouped_config(n_splits=3), groups)


def test_confidence_intervals_percentile_and_mean_options():
    scores = [1.0, 2.0, 3.0, 4.0]
    percentile = V.calculate_metric_summary(scores, 0.95, "percentile")
    assert percentile["mean"] == pytest.approx(2.5)
    assert percentile["std"] == pytest.approx(np.std(scores, ddof=1))
    assert percentile["lower"] == pytest.approx(np.percentile(scores, 2.5))
    mean_based = V.calculate_metric_summary(scores, 0.95, "mean")
    assert mean_based["lower"] < mean_based["mean"] < mean_based["upper"]


def test_grouped_oof_covers_every_row_and_single_target_uses_scalar_estimator():
    df = synthetic_frame()
    X = df[["temperature", "route"]]
    y = df["SIB_0_1A"]
    result = V.evaluate_model_cv(
        ExtraTreesRegressor(n_estimators=20, random_state=1), X, y,
        grouped_config(), groups=df["Condition_ID"],
        numeric_columns=["temperature"], categorical_columns=["route"],
    )
    assert result["oof_predictions"].shape == (len(df), 1)
    assert (result["oof_prediction_count"] == 3).all()
    assert all(sample.shape == (3, 1) for sample in result["oof_prediction_samples"])
    assert np.isfinite(result["oof_predictions"].to_numpy()).all()
    assert (result["oof_prediction_std"].to_numpy() >= 0).all()


def test_multi_target_mode_still_works():
    df = synthetic_frame()
    X = df[["temperature", "route"]]
    y = df[["SIB_0_1A", "SIB_1A"]]
    model = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=15, random_state=2))
    result = V.evaluate_model_cv(
        model, X, y, grouped_config(n_repeats=2), groups=df["Condition_ID"],
        numeric_columns=["temperature"], categorical_columns=["route"],
    )
    assert list(result["oof_predictions"].columns) == list(y.columns)
    assert set(result["pooled_metrics"]) == set(y.columns)


def test_target_missing_filter_keeps_groups_aligned_and_blocks_leakage_columns(tmp_path):
    import app_imgui as app

    df = synthetic_frame()
    df.loc[3, "SIB_0_1A"] = np.nan
    path = tmp_path / "grouped.csv"
    df.to_csv(path, index=False)
    cfg = {
        "data_path": str(path), "sheet": None, "ids": ["Sample_ID"], "mixed": [],
        "targets": ["SIB_0_1A"], "all_target_columns": ["SIB_0_1A", "SIB_1A"],
        "single_target_mode": True, "col_types": {}, "exclude": [],
        "standardize_units": False, "validation": grouped_config(),
    }
    data = app.build_training_data(cfg)
    expected_groups = df.loc[df["SIB_0_1A"].notna(), "Condition_ID"].reset_index(drop=True)
    pd.testing.assert_series_equal(data["groups"], expected_groups, check_names=False)
    forbidden = {"Condition_ID", "Sample_ID", "SIB_0_1A", "SIB_1A"}
    assert not (forbidden & set(data["X_raw"].columns))
    assert len(data["X_raw"]) == len(data["y"]) == len(data["groups"])


def test_app_single_target_training_uses_normal_regressor(tmp_path, monkeypatch):
    import app_imgui as app

    df = synthetic_frame()
    path = tmp_path / "single_target.csv"
    df.to_csv(path, index=False)

    def small_forest(**kwargs):
        kwargs["n_estimators"] = 8
        kwargs["n_jobs"] = 1
        return ExtraTreesRegressor(**kwargs)

    monkeypatch.setattr(app, "ExtraTreesRegressor", small_forest)
    app.STATE = app.AppState()
    cfg = {
        "data_path": str(path), "sheet": None, "ids": ["Sample_ID"], "mixed": [],
        "targets": ["SIB_0_1A"], "all_target_columns": ["SIB_0_1A", "SIB_1A"],
        "single_target_mode": True, "col_types": {}, "exclude": [],
        "standardize_units": False,
        "validation": grouped_config(n_repeats=2),
    }
    app._train_worker(cfg)
    assert app.STATE.trained, app.STATE.train_error
    assert not isinstance(app.STATE.model, MultiOutputRegressor)
    assert app.STATE.screener is not None
    assert app.STATE.metrics[0]["target"] == "SIB_0_1A"
    assert all(sample.shape == (2, 1) for sample in app.STATE.oof_prediction_samples)


def test_old_settings_and_saved_models_load():
    settings = V.load_settings_compat({"target": "old_target"})
    assert settings["validation"]["method"] == "random_kfold"
    assert settings["training"]["target"] == "old_target"
    assert settings["reporting"]["top_feature_count"] == 20
    estimator = Ridge()
    bundle = V.load_model_bundle_compat(estimator)
    assert bundle["model"] is estimator
    assert bundle["validation_config"]["n_splits"] == 5
    assert bundle["metric_distributions"] == {}
