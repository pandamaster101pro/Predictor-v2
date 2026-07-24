"""
tradeoffs.py — "Why would I pick THIS Pareto point over the others" explanations.

Pure comparison over a Pareto front's already-computed objective values (no
new model, no new search): for each candidate, which objectives is it
BEST on among its front (advantage), which is it WORST on (disadvantage),
and a one-line synthesis of the two (tradeoff) — the "Tradeoff Explorer"
the multi-objective planning spec asked for.
"""

from __future__ import annotations


def _direction_word(direction: str, is_best: bool) -> str:
    if direction == "maximise":
        return "Highest" if is_best else "Lowest"
    return "Lowest" if is_best else "Highest"


def _tradeoff_sentence(advantages, disadvantages) -> str:
    if advantages and disadvantages:
        return f"{advantages[0]}, but {disadvantages[0][0].lower()}{disadvantages[0][1:]}."
    if advantages:
        return f"{advantages[0]} — no clear downside within this front."
    if disadvantages:
        return f"{disadvantages[0]} — no standout advantage within this front."
    return "Squarely in the middle of the pack on every objective in this front."


def explain(front: list, objectives: list) -> list:
    """Attach ``advantages``/``disadvantages``/``tradeoff`` to a COPY of
    every candidate in ``front`` (a Pareto front from
    ``app_imgui._pareto_front_candidates``, each with an ``"objectives":
    {column: value}`` dict), without mutating the originals — same
    contract as ``planner.enrich_candidates``.

    A front of fewer than 2 candidates has nothing to compare against, so
    every candidate gets empty advantage/disadvantage lists and a note
    saying so, rather than a misleading "best at everything".
    """
    if not front:
        return []
    enriched = [dict(c) for c in front]
    for c in enriched:
        c["advantages"] = []
        c["disadvantages"] = []
    if len(front) < 2:
        for c in enriched:
            c["tradeoff"] = "Only one Pareto-optimal option found — no trade-off to compare."
        return enriched

    for obj in objectives:
        col, direction = obj["column"], obj["direction"]
        values = [c["objectives"].get(col) for c in front]
        valid = [v for v in values if v is not None]
        if not valid:
            continue
        raw_max, raw_min = max(valid), min(valid)
        if raw_max == raw_min:
            continue    # every candidate ties on this objective -> not a differentiator
        best = raw_max if direction == "maximise" else raw_min
        worst = raw_min if direction == "maximise" else raw_max
        for c, v in zip(enriched, values):
            if v is None:
                continue
            if v == best:
                c["advantages"].append(f"{_direction_word(direction, True)} {col} ({v:.3g})")
            elif v == worst:
                c["disadvantages"].append(f"{_direction_word(direction, False)} {col} ({v:.3g})")

    for c in enriched:
        c["tradeoff"] = _tradeoff_sentence(c["advantages"], c["disadvantages"])
    return enriched
