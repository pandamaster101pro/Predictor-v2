"""
test_latent.py  —  Unit tests for the latent-variable engine.

Run:  python -m pytest test_latent.py -q
"""

import numpy as np
import pandas as pd
import pytest

import latent as L


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Material": rng.choice(["Wood", "Grass", "Straw"], n),
        "Pretreat_family": rng.choice(["None", "Acid", "Base"], n),
        "Post_treat_family": rng.choice(["None", "Wash"], n),
        "Atmosphere1": rng.choice(["N2", "Ar"], n),
        "Atmosphere2": rng.choice(["N2", "Ar"], n),
        "Fiber_sample": rng.choice(["F1", "F2"], n),
        "Py1_temp_C": rng.uniform(400, 900, n),
        "Py1_time_min": rng.uniform(10, 120, n),
        "Py2_temp_C": rng.uniform(800, 1400, n),
        "Py2_time_min": rng.uniform(10, 120, n),
        "Cellulose_avg_pct": rng.uniform(20, 50, n),
        "Hemicellulose_avg_pct": rng.uniform(10, 35, n),
        "Lignin_avg_pct": rng.uniform(15, 40, n),
        "Has_pretreat": rng.integers(0, 2, n),
        "Has_post_treat": rng.integers(0, 2, n),
        "Has_any_additive": rng.integers(0, 2, n),
        "Two_step_pyrolysis": rng.integers(0, 2, n),
        "Total_py_time_min": rng.uniform(20, 240, n),
        "Max_temp_C": rng.uniform(800, 1500, n),
        "Condition_ID": [f"C{i}" for i in range(n)],
        "Replicate_rows": rng.integers(1, 4, n),
    })
    # Six capacity targets, correlated with a couple of drivers.
    base = 0.15 * df["Max_temp_C"] + 2.0 * df["Cellulose_avg_pct"]
    for t in L.CAPACITY_TARGETS:
        df[t] = base + rng.normal(0, 20, n)
    return df


# ---------------------------------------------------------------------------
# 1. Engineered index formulas
# ---------------------------------------------------------------------------
def test_thermal_severity_is_standardized_mean():
    df = make_df()
    eng = L.compute_engineered_latents(df)
    # Standardized-average index should be ~zero-mean across the dataset.
    assert abs(eng["Thermal_Severity_Index"].mean()) < 1e-6
    assert eng.shape[1] == 4
    assert list(eng.columns) == L.ENGINEERED_INDEX_NAMES


def test_chemical_treatment_weights_applied():
    df = make_df()
    base = L.compute_engineered_latents(df, weights=(1, 1, 1))["Chemical_Treatment_Index"]
    weighted = L.compute_engineered_latents(df, weights=(2, 0, 0))["Chemical_Treatment_Index"]
    # weights (2,0,0) -> exactly 2 * Has_pretreat
    expected = 2.0 * df["Has_pretreat"].to_numpy()
    assert np.allclose(weighted.to_numpy(), expected)
    # base is the simple sum of the three flags
    simple = (df["Has_pretreat"] + df["Has_post_treat"] + df["Has_any_additive"]).to_numpy()
    assert np.allclose(base.to_numpy(), simple)


def test_process_complexity_atmosphere_bump():
    df = make_df(n=4)
    df.loc[:, ["Has_pretreat", "Has_post_treat", "Has_any_additive",
               "Two_step_pyrolysis"]] = 0
    df.loc[:, "Atmosphere1"] = ["N2", "N2", "Ar", "Ar"]
    df.loc[:, "Atmosphere2"] = ["N2", "Ar", "Ar", "N2"]  # differ on rows 1 and 3
    eng = L.compute_engineered_latents(df)
    assert list(eng["Process_Complexity_Index"]) == [0.0, 1.0, 0.0, 1.0]


def test_biomass_pca_method_runs():
    df = make_df()
    mean_idx = L.compute_engineered_latents(df, biomass_method="mean")["Biomass_Composition_Index"]
    pca_idx = L.compute_engineered_latents(df, biomass_method="pca")["Biomass_Composition_Index"]
    assert len(mean_idx) == len(df) and len(pca_idx) == len(df)
    assert np.isfinite(pca_idx.to_numpy()).all()


# ---------------------------------------------------------------------------
# 2. Missing-value handling
# ---------------------------------------------------------------------------
def test_missing_numeric_median_imputed_no_row_drop():
    df = make_df(n=50)
    # Blank the optional Py2 fields for half the rows.
    df.loc[:24, "Py2_temp_C"] = np.nan
    df.loc[:24, "Py2_time_min"] = np.nan
    df.loc[:10, "Max_temp_C"] = np.nan
    eng = L.compute_engineered_latents(df)
    # No rows dropped despite blanks.
    assert len(eng) == len(df)
    assert np.isfinite(eng.to_numpy()).all()


def test_missing_categorical_filled_missing_in_pipeline():
    df = make_df(n=60)
    df.loc[:5, "Material"] = np.nan
    cfg = L.LatentConfig(target=L.CAPACITY_TARGETS[0])
    cfg.excluded = L.default_excluded(cfg.target, list(df.columns))
    # Should train without error (NaN category -> "Missing" -> one-hot).
    res = L.evaluate_pls(df, cfg, n_components=2)
    assert np.isfinite(res["r2_mean"])


# ---------------------------------------------------------------------------
# 3. Target exclusion / leakage
# ---------------------------------------------------------------------------
def test_other_capacity_targets_excluded_from_features():
    df = make_df()
    target = L.CAPACITY_TARGETS[0]
    cfg = L.LatentConfig(target=target,
                         numerical=L.DEFAULT_NUMERICAL + L.CAPACITY_TARGETS)
    cat, num = cfg.feature_columns(list(df.columns))
    # Neither the selected target nor any other capacity column may be a feature.
    for t in L.CAPACITY_TARGETS:
        assert t not in cat and t not in num


def test_default_excluded_lists_other_targets():
    df = make_df()
    target = L.CAPACITY_TARGETS[2]
    excl = L.default_excluded(target, list(df.columns))
    assert "Condition_ID" in excl and "Replicate_rows" in excl
    assert target not in excl
    for t in L.CAPACITY_TARGETS:
        if t != target:
            assert t in excl


def test_leakage_safe_cv_beats_are_not_perfect():
    """A leakage-safe pipeline on noisy data must not score a suspicious ~1.0 R²."""
    df = make_df()
    cfg = L.LatentConfig(target=L.CAPACITY_TARGETS[0])
    cfg.excluded = L.default_excluded(cfg.target, list(df.columns))
    res = L.compare_pipelines(df, cfg, n_pca=3)
    for variant in ("A", "B", "C"):
        for mname, sc in res[variant].items():
            if "error" in sc:
                continue
            assert sc["r2_mean"] < 0.999, f"{variant}/{mname} looks like leakage"
            assert np.isfinite(sc["rmse_mean"]) and sc["rmse_mean"] > 0


# ---------------------------------------------------------------------------
# 4. PCA / PLS component limits
# ---------------------------------------------------------------------------
def test_pca_component_clamp():
    assert L.clamp_pca_components(3, 100, 20) == 3
    assert L.clamp_pca_components(50, 100, 8) == 8      # capped by n_features
    assert L.clamp_pca_components(1, 100, 20) == 2      # UI floor of 2
    assert L.clamp_pca_components(5, 4, 20) == 4        # capped by n_samples


def test_pls_component_clamp():
    assert L.clamp_pls_components(3, 200, 30) == 3
    assert L.clamp_pls_components(50, 200, 5) == 5      # capped by n_features
    # Capped by the smallest training fold.
    hi = L.max_pls_components(10, 30, n_cv_splits=5)
    assert L.clamp_pls_components(9, 10, 30, 5) == hi
    assert L.clamp_pls_components(0, 200, 30) == 1      # floor of 1


def test_pca_fit_reports_variance_and_loadings():
    df = make_df()
    cfg = L.LatentConfig(target=L.CAPACITY_TARGETS[0])
    cfg.excluded = L.default_excluded(cfg.target, list(df.columns))
    out = L.fit_pca(df, cfg, n_components=4)
    evr = out["explained_variance_ratio"]
    assert len(evr) == out["n_components"] == 4
    assert np.all(evr >= 0) and evr.sum() <= 1.0 + 1e-9
    assert np.allclose(out["cumulative_variance"], np.cumsum(evr))
    assert out["loadings"].shape[1] == 4
    assert len(out["scores"]) == len(df)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
