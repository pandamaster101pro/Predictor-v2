"""test_planner.py — experiment strategy labels, info gain, feasibility, batch plan.

Run:  python -m pytest test_planner.py -q
"""

import pytest

import planner as P


def _cand(**kw):
    base = {"recipe": [], "risk": "Moderate", "applicability_pct": 50.0,
           "similarity_pct": 50.0, "utility": 0.0, "edges": [],
           "chemical_recommendations": []}
    base.update(kw)
    return base


# ---- experiment_strategy ----------------------------------------------------
def test_highest_utility_is_best_candidate():
    pool = [_cand(utility=10.0), _cand(utility=5.0), _cand(utility=1.0)]
    label, reason = P.experiment_strategy(pool[0], pool)
    assert label == "Best candidate"


def test_high_similarity_is_validation_experiment():
    pool = [_cand(utility=10.0), _cand(utility=1.0, similarity_pct=97.0)]
    label, _ = P.experiment_strategy(pool[1], pool)
    assert label == "Validation experiment"


def test_low_risk_high_applicability_is_safe_improvement():
    pool = [_cand(utility=10.0), _cand(utility=1.0, risk="Low", applicability_pct=90.0,
                  similarity_pct=50.0)]
    label, _ = P.experiment_strategy(pool[1], pool)
    assert label == "Safe improvement"


def test_low_similarity_chemistry_is_novel_chemistry():
    pool = [_cand(utility=10.0),
           _cand(utility=1.0, risk="Moderate", applicability_pct=50.0, similarity_pct=50.0,
                 chemical_recommendations=[{"recommended": "HBr", "similarity": 0.4}])]
    label, reason = P.experiment_strategy(pool[1], pool)
    assert label == "Novel chemistry"
    assert "HBr" in reason


def test_edges_present_is_boundary_exploration():
    pool = [_cand(utility=10.0),
           _cand(utility=1.0, risk="Moderate", applicability_pct=50.0, similarity_pct=50.0,
                 edges=["Temperature"])]
    label, reason = P.experiment_strategy(pool[1], pool)
    assert label == "Boundary exploration"
    assert "Temperature" in reason


def test_best_among_risky_is_high_risk_high_reward():
    pool = [_cand(utility=10.0, risk="Low"),
           _cand(utility=5.0, risk="High", applicability_pct=50.0, similarity_pct=50.0),
           _cand(utility=1.0, risk="High", applicability_pct=50.0, similarity_pct=50.0)]
    label, _ = P.experiment_strategy(pool[1], pool)
    assert label == "High-risk high-reward"


def test_far_from_domain_is_gap_filling():
    pool = [_cand(utility=10.0),
           _cand(utility=1.0, risk="Moderate", applicability_pct=10.0, similarity_pct=50.0)]
    label, _ = P.experiment_strategy(pool[1], pool)
    assert label == "Gap-filling experiment"


def test_fallback_label():
    pool = [_cand(utility=10.0),
           _cand(utility=1.0, risk="Moderate", applicability_pct=60.0, similarity_pct=60.0)]
    label, _ = P.experiment_strategy(pool[1], pool)
    assert label == "Worth exploring"


# ---- information_gain_score -------------------------------------------------
def test_info_gain_high_when_novel_and_dissimilar():
    c = _cand(applicability_pct=5.0, similarity_pct=5.0)
    assert P.information_gain_score(c) > 80


def test_info_gain_low_when_typical_and_similar():
    c = _cand(applicability_pct=95.0, similarity_pct=95.0)
    assert P.information_gain_score(c) < 10


def test_info_gain_falls_back_without_similarity():
    c = _cand(applicability_pct=20.0, similarity_pct=None)
    assert P.information_gain_score(c) == pytest.approx(80.0)


def test_info_gain_bounded_0_100():
    for a, s in [(0, 0), (100, 100), (150, -50)]:
        v = P.information_gain_score(_cand(applicability_pct=a, similarity_pct=s))
        assert 0.0 <= v <= 100.0


# ---- feasibility_score --------------------------------------------------------
def test_feasibility_easy_for_minimal_recipe():
    recipe = [("Temperature", 900), ("Py.1 temp. (oC)", 900), ("Pretreat 1", "None")]
    label, reasons = P.feasibility_score(recipe)
    assert label == "Easy"


def test_feasibility_harder_with_more_steps_and_stages():
    recipe = [
        ("Pretreat 1", "1M NaOH"), ("Pretreat 2", "1M KOH"),
        ("Post-treat", "HCl wash"), ("Additive 1", "ZnCl2"),
        ("Py.1 temp. (oC)", 900), ("Py.2 temp. (oC)", 950), ("Py.3 temp. (oC)", 1000),
    ]
    label, reasons = P.feasibility_score(recipe)
    assert label == "Difficult"
    assert any("stage" in r for r in reasons)


def test_feasibility_edges_increase_difficulty():
    recipe = [("Pretreat 1", "1M NaOH"), ("Py.1 temp. (oC)", 900)]
    easy_label, _ = P.feasibility_score(recipe, edges=[])
    hard_label, hard_reasons = P.feasibility_score(recipe, edges=["Temperature"])
    assert any("edge of tested conditions" in r for r in hard_reasons)


# ---- build_batch_plan ---------------------------------------------------------
def test_batch_plan_returns_requested_size_when_pool_large_enough():
    pool = [_cand(utility=float(i), risk=("Low" if i % 2 == 0 else "High"),
                  applicability_pct=float(50 + i), similarity_pct=float(50 + i))
           for i in range(20)]
    plan = P.build_batch_plan(pool, batch_size=10)
    assert len(plan) == 10
    for i, c in enumerate(plan, start=1):
        assert c["batch_rank"] == i
    # every returned candidate got enriched
    assert all("strategy" in c and "info_gain" in c and "feasibility" in c for c in plan)


def test_batch_plan_never_exceeds_pool_size():
    pool = [_cand(utility=float(i)) for i in range(3)]
    plan = P.build_batch_plan(pool, batch_size=10)
    assert len(plan) == 3


def test_batch_plan_empty_pool():
    assert P.build_batch_plan([], batch_size=5) == []


def test_batch_plan_does_not_mutate_input():
    pool = [_cand(utility=1.0)]
    P.build_batch_plan(pool, batch_size=1)
    assert "strategy" not in pool[0]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
