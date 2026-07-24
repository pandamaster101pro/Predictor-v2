"""test_tradeoffs.py — Pareto front advantage/disadvantage explanations.

Run:  python -m pytest test_tradeoffs.py -q
"""

import pytest

import tradeoffs as T

OBJECTIVES = [
    {"column": "Capacity", "direction": "maximise", "weight": 0.6},
    {"column": "Temperature", "direction": "minimise", "weight": 0.4},
]


def _cand(capacity, temperature):
    return {"objectives": {"Capacity": capacity, "Temperature": temperature}}


def test_single_candidate_front_has_no_comparison():
    front = [_cand(300, 900)]
    out = T.explain(front, OBJECTIVES)
    assert out[0]["advantages"] == []
    assert out[0]["disadvantages"] == []
    assert "no trade-off" in out[0]["tradeoff"]


def test_empty_front_returns_empty_list():
    assert T.explain([], OBJECTIVES) == []


def test_best_capacity_candidate_gets_advantage():
    front = [_cand(350, 950), _cand(300, 800)]
    out = T.explain(front, OBJECTIVES)
    assert any("Highest Capacity" in a for a in out[0]["advantages"])


def test_lowest_temperature_candidate_gets_advantage_for_minimise_objective():
    front = [_cand(350, 950), _cand(300, 800)]
    out = T.explain(front, OBJECTIVES)
    # out[1] has the lower temperature -> "Lowest Temperature" is an advantage
    # (direction is minimise, so lowest = best).
    assert any("Lowest Temperature" in a for a in out[1]["advantages"])


def test_worst_on_an_objective_gets_disadvantage():
    front = [_cand(350, 950), _cand(300, 800)]
    out = T.explain(front, OBJECTIVES)
    # out[0] has the highest temperature -> worst on a minimise objective.
    assert any("Highest Temperature" in d for d in out[0]["disadvantages"])


def test_tradeoff_sentence_combines_advantage_and_disadvantage():
    front = [_cand(350, 950), _cand(300, 800)]
    out = T.explain(front, OBJECTIVES)
    assert "Highest Capacity" in out[0]["tradeoff"]
    assert "but" in out[0]["tradeoff"]


def test_tied_objective_produces_no_advantage_or_disadvantage_for_it():
    front = [_cand(300, 900), _cand(350, 900)]
    out = T.explain(front, OBJECTIVES)
    assert not any("Temperature" in a for a in out[0]["advantages"] + out[0]["disadvantages"])
    assert not any("Temperature" in a for a in out[1]["advantages"] + out[1]["disadvantages"])


def test_does_not_mutate_input():
    front = [_cand(350, 950), _cand(300, 800)]
    T.explain(front, OBJECTIVES)
    assert "advantages" not in front[0]


def test_three_way_front_best_gets_no_disadvantage_on_that_objective():
    front = [_cand(400, 1000), _cand(300, 900), _cand(200, 800)]
    out = T.explain(front, OBJECTIVES)
    assert not out[0]["disadvantages"] or "Capacity" not in out[0]["disadvantages"][0]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
