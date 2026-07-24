"""
pareto.py — Multi-objective dominance over plain numeric candidate values.

Pure, headless functions with no coupling to the optimizer's search/recipe
machinery: every function here takes raw per-objective value arrays and a
list of "maximise"/"minimise" directions, nothing else. app_imgui.py's
Pareto search pass (``_pareto_front_candidates``) uses these instead of an
inline dominance loop, so the algorithm is independently testable and named
the way the multi-objective planning spec asked for:

  * dominates            — does candidate A beat candidate B outright
  * non_dominated_sort    — partition every candidate into successive
                            Pareto fronts (front 0 = optimal, front 1 =
                            optimal once front 0 is removed, ...), not just
                            the first front
  * pareto_front          — convenience: just front 0's indices
  * crowding_distance     — NSGA-II spacing metric within one front, for
                            picking a spread of trade-offs rather than a
                            cluster of near-duplicates
"""

from __future__ import annotations

import numpy as np


def _to_scores(values, directions):
    """(n, m) array where every column is oriented "higher is better"."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    sign = np.array([1.0 if d == "maximise" else -1.0 for d in directions])
    return values * sign


def dominates(a, b, directions) -> bool:
    """True if candidate `a` Pareto-dominates `b`: at least as good as `b`
    on every objective and strictly better on at least one. `a`/`b` are
    length-m raw value sequences; `directions` is the matching length-m
    list of "maximise"/"minimise"."""
    sa = _to_scores(a, directions)[0]
    sb = _to_scores(b, directions)[0]
    return bool(np.all(sa >= sb) and np.any(sa > sb))


def non_dominated_sort(points, directions) -> list:
    """Partition `points` (n x m raw values) into successive Pareto fronts
    — the standard NSGA-II fast-non-dominated-sort. Returns a list of
    fronts, each a list of original indices into `points`, front 0 first.
    Points tied exactly on every objective are mutually non-dominating and
    land in the same front rather than one arbitrarily excluding the other.
    """
    n = len(points)
    if n == 0:
        return []
    scores = _to_scores(points, directions)
    dominated_sets = [[] for _ in range(n)]
    domination_count = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(scores[i] >= scores[j]) and np.any(scores[i] > scores[j]):
                dominated_sets[i].append(j)
            elif np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                domination_count[i] += 1

    fronts = []
    current = [i for i in range(n) if domination_count[i] == 0]
    while current:
        fronts.append(current)
        nxt = []
        for i in current:
            for j in dominated_sets[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def pareto_front(points, directions) -> list:
    """Indices of just the first (fully non-dominated) front. Convenience
    wrapper over ``non_dominated_sort`` for callers that only need the
    optimal set, not every rank."""
    fronts = non_dominated_sort(points, directions)
    return fronts[0] if fronts else []


def crowding_distance(front_points) -> list:
    """NSGA-II crowding distance for each point in ONE front — how much
    "elbow room" a point has from its neighbours in objective space.
    Direction-agnostic (it measures spacing along each raw axis, not
    dominance), so unlike the other functions here it takes no
    `directions` argument. `front_points` is (k, m) raw values for just
    this front. Boundary points (extreme on any objective) score
    ``inf`` — always worth keeping. Returns a length-k list of floats.
    """
    pts = np.asarray(front_points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return []
    k, m = pts.shape
    if k <= 2:
        return [float("inf")] * k
    distance = np.zeros(k)
    for obj_i in range(m):
        order = np.argsort(pts[:, obj_i])
        distance[order[0]] = float("inf")
        distance[order[-1]] = float("inf")
        lo, hi = pts[order[0], obj_i], pts[order[-1], obj_i]
        span = (hi - lo) or 1.0
        for rank in range(1, k - 1):
            prev_v = pts[order[rank - 1], obj_i]
            next_v = pts[order[rank + 1], obj_i]
            if np.isinf(distance[order[rank]]):
                continue
            distance[order[rank]] += (next_v - prev_v) / span
    return distance.tolist()
