"""test_pareto.py — dominance / non-dominated sorting / crowding distance.

Run:  python -m pytest test_pareto.py -q
"""

import math

import pytest

import pareto as PA


# ---- dominates --------------------------------------------------------------
def test_dominates_strictly_better_on_one_equal_on_other():
    assert PA.dominates([10, 5], [8, 5], ["maximise", "maximise"])


def test_dominates_false_when_worse_on_one():
    assert not PA.dominates([10, 3], [8, 5], ["maximise", "maximise"])


def test_dominates_false_when_identical():
    assert not PA.dominates([5, 5], [5, 5], ["maximise", "maximise"])


def test_dominates_respects_minimise_direction():
    # Lower cost is better: [1, 5] beats [2, 5] when minimising cost.
    assert PA.dominates([1, 5], [2, 5], ["minimise", "maximise"])
    assert not PA.dominates([2, 5], [1, 5], ["minimise", "maximise"])


# ---- non_dominated_sort -------------------------------------------------------
def test_non_dominated_sort_two_fronts():
    # A=(10,10) and B=(8,15) each win on one objective -> mutually
    # non-dominating, both front 0. C=(5,5) loses to both on every
    # objective -> dominated, front 1.
    points = [(10, 10), (8, 15), (5, 5)]
    fronts = PA.non_dominated_sort(points, ["maximise", "maximise"])
    assert set(fronts[0]) == {0, 1}
    assert fronts[1] == [2]


def test_non_dominated_sort_all_tied_single_front():
    points = [(5, 5), (5, 5), (5, 5)]
    fronts = PA.non_dominated_sort(points, ["maximise", "maximise"])
    assert len(fronts) == 1
    assert set(fronts[0]) == {0, 1, 2}


def test_non_dominated_sort_empty():
    assert PA.non_dominated_sort([], ["maximise"]) == []


def test_non_dominated_sort_single_point():
    fronts = PA.non_dominated_sort([(1, 2)], ["maximise", "minimise"])
    assert fronts == [[0]]


def test_non_dominated_sort_strict_chain():
    # Each point strictly dominates the next -> each gets its own front.
    points = [(3, 3), (2, 2), (1, 1)]
    fronts = PA.non_dominated_sort(points, ["maximise", "maximise"])
    assert [f[0] for f in fronts] == [0, 1, 2]


# ---- pareto_front -------------------------------------------------------------
def test_pareto_front_matches_first_front():
    points = [(10, 10), (10, 1), (5, 5)]
    directions = ["maximise", "maximise"]
    assert set(PA.pareto_front(points, directions)) == set(
        PA.non_dominated_sort(points, directions)[0])


def test_pareto_front_empty_input():
    assert PA.pareto_front([], ["maximise"]) == []


# ---- crowding_distance ---------------------------------------------------------
def test_crowding_distance_boundary_points_infinite():
    front = [(0, 10), (5, 5), (10, 0)]
    d = PA.crowding_distance(front)
    assert d[0] == float("inf")
    assert d[2] == float("inf")
    assert math.isfinite(d[1])
    assert d[1] > 0


def test_crowding_distance_two_points_all_infinite():
    d = PA.crowding_distance([(0, 0), (1, 1)])
    assert d == [float("inf"), float("inf")]


def test_crowding_distance_empty():
    assert PA.crowding_distance([]) == []


def test_crowding_distance_tightly_clustered_point_has_smaller_distance():
    front = [(0, 0), (1, 1), (2, 2), (10, 10)]
    d = PA.crowding_distance(front)
    # index 1 (value 1) is tightly wedged between neighbours 0 and 2;
    # index 2 (value 2) has a large gap to its other neighbour at 10 ->
    # more "elbow room" -> larger crowding distance.
    assert d[1] < d[2]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
