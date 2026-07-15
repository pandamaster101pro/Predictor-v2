"""
test_intelligence.py  —  Unit tests for the Dataset Intelligence engine.

Run:  python -m pytest test_intelligence.py -q
"""

import numpy as np
import pandas as pd
import pytest

import latent as L
import intelligence as I
from test_latent import make_df  # reuse the synthetic dataset fixture


def cfg_for(df, target=None):
    target = target or L.CAPACITY_TARGETS[0]
    c = L.LatentConfig(target=target)
    c.excluded = L.default_excluded(target, list(df.columns))
    return c


# ---------------------------------------------------------------------------
# Section 1 — summary
# ---------------------------------------------------------------------------
def test_summary_counts_and_missing():
    df = make_df(n=100)
    df.loc[:9, "Py2_temp_C"] = np.nan
    cfg = cfg_for(df)
    s = I.dataset_summary(df, cfg)
    assert s["n_samples"] == 100
    assert s["n_categorical"] == len(cfg.feature_columns(list(df.columns))[0])
    assert s["target_std"] > 0 and s["target_range"] > 0
    # Missing table has a row per column and the blanked column shows ~10%.
    row = s["missing_table"].set_index("column").loc["Py2_temp_C"]
    assert row["missing_count"] == 10 and abs(row["missing_pct"] - 10.0) < 1e-6


def test_summary_excludes_other_targets_from_feature_count():
    df = make_df()
    cfg = cfg_for(df, L.CAPACITY_TARGETS[0])
    s = I.dataset_summary(df, cfg)
    for t in L.CAPACITY_TARGETS:
        assert t not in s["categorical"] and t not in s["numerical"]


# ---------------------------------------------------------------------------
# Section 2 — predictability
# ---------------------------------------------------------------------------
def test_predictability_ranks_true_driver_high():
    # Target built mostly from Max_temp_C + Cellulose -> those should rank top.
    df = make_df(n=200)
    cfg = cfg_for(df)
    p = I.predictability(df, cfg)
    assert not p["table"].empty
    top3 = set(p["table"].head(3)["feature"])
    assert ("Max_temp_C" in top3) or ("Cellulose_avg_pct" in top3)
    assert 0.0 <= p["max_pearson"] <= 1.0


def test_distance_correlation_detects_dependence():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y = x ** 2  # nonlinear dependence: Pearson ~0, dCor > 0
    assert I._distance_correlation(x, y) > 0.3


# ---------------------------------------------------------------------------
# Section 3 — redundancy / VIF
# ---------------------------------------------------------------------------
def test_vif_flags_collinear_feature():
    df = make_df(n=150)
    # Make a near-duplicate of an existing numeric feature.
    df["Max_temp_C_copy"] = df["Max_temp_C"] + np.random.default_rng(0).normal(0, 1e-3, len(df))
    cfg = cfg_for(df)
    cfg.numerical = cfg.numerical + ["Max_temp_C_copy"]
    r = I.redundancy(df, cfg)
    pair_cols = {a for a, _b, _r in r["high_corr_pairs"]} | {b for _a, b, _r in r["high_corr_pairs"]}
    assert "Max_temp_C" in pair_cols and "Max_temp_C_copy" in pair_cols
    # The collinear pair should inflate VIF well past the threshold.
    assert max(r["vif"].get("Max_temp_C", 0), r["vif"].get("Max_temp_C_copy", 0)) > 10


def test_near_zero_variance_detected():
    df = make_df(n=80)
    df["constant_col"] = 5.0
    cfg = cfg_for(df)
    cfg.numerical = cfg.numerical + ["constant_col"]
    r = I.redundancy(df, cfg)
    assert "constant_col" in r["near_zero_variance"]


# ---------------------------------------------------------------------------
# Section 5 — learnability (leakage-safe)
# ---------------------------------------------------------------------------
def test_learnability_runs_all_models_leakage_safe():
    df = make_df(n=200)
    cfg = cfg_for(df)
    out = I.learnability(df, cfg)
    assert "LinearRegression" in out["results"] and "ExtraTrees" in out["results"]
    for name, sc in out["results"].items():
        if "error" in sc:
            continue
        assert sc["r2_mean"] < 0.999, f"{name} suspiciously perfect (leakage?)"
    assert out["best_linear"][0] is not None and out["best_tree"][0] is not None


def test_learnability_insufficient_flag_on_noise():
    rng = np.random.default_rng(0)
    df = make_df(n=120)
    df[L.CAPACITY_TARGETS[0]] = rng.normal(size=len(df))  # pure noise target
    cfg = cfg_for(df)
    out = I.learnability(df, cfg)
    assert out["insufficient"] is True


# ---------------------------------------------------------------------------
# Section 4 — difficulty score
# ---------------------------------------------------------------------------
def test_difficulty_label_and_estimate():
    df = make_df(n=200)
    cfg = cfg_for(df)
    s = I.dataset_summary(df, cfg)
    p = I.predictability(df, cfg)
    learn = I.learnability(df, cfg)
    d = I.difficulty_score(s, p, learn)
    assert d["label"] in ("Easy", "Moderate", "Hard", "Very Hard")
    assert d["est_r2_low"] <= d["est_r2_high"]
    assert 0 <= d["difficulty"] <= 100


# ---------------------------------------------------------------------------
# Section 6 — causal structure detection
# ---------------------------------------------------------------------------
def test_causal_structure_absent_then_present():
    df = make_df(n=50)
    assert I.causal_structure(df)["structure_present"] is False
    df["BET_Surface_Area_m2_per_g"] = 100.0
    out = I.causal_structure(df)
    assert out["structure_present"] is True and out["message"] == ""


# ---------------------------------------------------------------------------
# Section 8 — target analysis
# ---------------------------------------------------------------------------
def test_target_analysis_recommends_log_on_skew():
    rng = np.random.default_rng(0)
    df = make_df(n=200)
    df[L.CAPACITY_TARGETS[0]] = rng.lognormal(mean=1.0, sigma=1.0, size=len(df))
    cfg = cfg_for(df)
    t = I.target_analysis(df, cfg)
    assert t["skewness"] > 1.0 and t["recommend_log"] is True


# ---------------------------------------------------------------------------
# Section 10 + assistant — text generation & PDF
# ---------------------------------------------------------------------------
def test_conclusion_and_insights_text():
    df = make_df(n=120)
    cfg = cfg_for(df)
    s = I.dataset_summary(df, cfg)
    p = I.predictability(df, cfg)
    r = I.redundancy(df, cfg)
    learn = I.learnability(df, cfg)
    causal = I.causal_structure(df)
    lat = I.latent_analysis(df, cfg)
    tgt = I.target_analysis(df, cfg)
    d = I.difficulty_score(s, p, learn)
    concl = I.final_conclusion(s, p, learn, causal, tgt)
    ins = I.ai_insights(s, p, r, d, learn, causal, lat, tgt)
    assert cfg.target in concl and isinstance(ins, list) and len(ins) >= 3
    # No structural descriptors in the synthetic data -> recommendation present.
    assert any("Structural descriptors" in m for m in ins)


def test_pdf_report_written(tmp_path):
    df = make_df(n=120)
    cfg = cfg_for(df)
    s = I.dataset_summary(df, cfg)
    p = I.predictability(df, cfg)
    learn = I.learnability(df, cfg)
    d = I.difficulty_score(s, p, learn)
    causal = I.causal_structure(df)
    tgt = I.target_analysis(df, cfg)
    concl = I.final_conclusion(s, p, learn, causal, tgt)
    ins = I.ai_insights(s, p, I.redundancy(df, cfg), d, learn, causal,
                        I.latent_analysis(df, cfg), tgt)
    out = str(tmp_path / "report.pdf")
    path = I.build_pdf_report(out, s, p, d, learn, concl, ins, chart_paths=[])
    import os
    assert os.path.exists(path) and os.path.getsize(path) > 1000


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
