"""test_sustainability.py — Green Score heuristic.

Run:  python -m pytest test_sustainability.py -q
"""

import pandas as pd
import pytest

import sustainability as S
from cost_model import CostDatabase, ReagentCostEntry


def _engine(**entries):
    db = CostDatabase(path="__unused__.json")
    for name, kwargs in entries.items():
        db.set(name, ReagentCostEntry(name=name, **kwargs))
    return db


# ---- grade_for ------------------------------------------------------------------
def test_grade_bands():
    assert S.grade_for(95) == "A"
    assert S.grade_for(75) == "B"
    assert S.grade_for(55) == "C"
    assert S.grade_for(35) == "D"
    assert S.grade_for(10) == "F"


# ---- estimate_temperature_percentile ---------------------------------------------
def test_temperature_percentile_high_value():
    X_raw = pd.DataFrame({"Py.1 temp. (oC)": list(range(600, 1000, 10))})
    recipe = {"Py.1 temp. (oC)": 990}
    pct = S.estimate_temperature_percentile(recipe, X_raw, ["Py.1 temp. (oC)"])
    assert pct > 90


def test_temperature_percentile_none_when_no_temp_knob():
    X_raw = pd.DataFrame({"Molarity": [1, 2, 3]})
    assert S.estimate_temperature_percentile({"Molarity": 2}, X_raw, ["Molarity"]) is None


# ---- green_score: hazard ----------------------------------------------------------
def test_green_score_no_hazard_data_is_unscored_not_penalized():
    result = S.green_score({"Py.1 temp. (oC)": 700}, ["Unknown Chemical X"],
                           cost_engine=_engine())
    assert result["score"] == 100.0
    assert "Unknown Chemical X" in result["unscored_reagents"]


def test_green_score_penalizes_high_hazard():
    engine = _engine(NaOH={"hazard_class": "High"})
    result = S.green_score({"Py.1 temp. (oC)": 700}, ["NaOH"], cost_engine=engine)
    assert result["score"] == 70.0
    assert any("High" in d["reason"] for d in result["deductions"])


def test_green_score_penalizes_corrosive_separately():
    engine = _engine(HCl={"hazard_class": "Moderate", "corrosive": True})
    result = S.green_score({}, ["HCl"], cost_engine=engine)
    assert result["score"] == 100.0 - 15 - 10


def test_green_score_uses_worst_hazard_among_reagents():
    engine = _engine(A={"hazard_class": "Low"}, B={"hazard_class": "Severe"})
    result = S.green_score({}, ["A", "B"], cost_engine=engine)
    assert result["score"] == 50.0


# ---- green_score: process complexity ----------------------------------------------
def test_green_score_penalizes_extra_stages_and_steps():
    recipe = {"Pretreat 1": "1M NaOH", "Py.1 temp. (oC)": 900, "Py.2 temp. (oC)": 950}
    result = S.green_score(recipe, [], cost_engine=_engine())
    # 1 extra stage (2 stages -> (2-1)*8=8) + 1 step (1*4=4)
    assert result["score"] == 100.0 - 8 - 4


def test_green_score_single_stage_no_steps_no_process_penalty():
    recipe = {"Py.1 temp. (oC)": 900}
    result = S.green_score(recipe, [], cost_engine=_engine())
    assert result["score"] == 100.0


# ---- green_score: temperature -----------------------------------------------------
def test_green_score_penalizes_high_temperature_percentile():
    result = S.green_score({}, [], temperature_percentile=95.0, cost_engine=_engine())
    assert result["score"] == 100.0 - 15


def test_green_score_no_penalty_for_low_temperature_percentile():
    result = S.green_score({}, [], temperature_percentile=40.0, cost_engine=_engine())
    assert result["score"] == 100.0


# ---- green_score: bounds and structure ---------------------------------------------
def test_green_score_never_below_zero():
    engine = _engine(X={"hazard_class": "Severe", "corrosive": True})
    recipe = {"Pretreat 1": "a", "Pretreat 2": "b", "Post-treat": "c", "Additive 1": "d",
             "Py.1 temp. (oC)": 900, "Py.2 temp. (oC)": 950, "Py.3 temp. (oC)": 1000}
    result = S.green_score(recipe, ["X"], temperature_percentile=99.0, cost_engine=engine)
    assert result["score"] >= 0.0


def test_green_score_returns_grade_and_deductions_list():
    result = S.green_score({}, [], cost_engine=_engine())
    assert result["grade"] == "A"
    assert isinstance(result["deductions"], list)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
