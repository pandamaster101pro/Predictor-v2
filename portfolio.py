"""
portfolio.py — A balanced experiment portfolio, one bucket per strategy label.

planner.build_batch_plan already does this at a coarse 4-bucket resolution
(safe / moderate / exploratory / high_risk — see planner.py's
_STRATEGY_BUCKET). This module buckets by the exact strategy label
planner.experiment_strategy assigns instead, matching the finer-grained
composition the multi-objective planning spec asked for, e.g. "3 safe
improvements, 2 validation experiments, 2 novel-chemistry, 2 gap-filling,
1 high-risk high-reward" out of a 10-experiment batch. planner.py's
build_batch_plan is left as-is (nothing else in the app depended on the
finer split); this is a separate, swappable portfolio builder over the
same underlying enrich_candidates data.
"""

from __future__ import annotations

import planner

# Illustrative default composition (a 10-experiment batch would look
# roughly like the spec's own example above) — override with `portions=`
# for a different mix. A label mapped to 0 is never targeted directly but
# can still fill remaining slots as backfill.
DEFAULT_STRATEGY_PORTIONS = {
    "Best candidate": 0.1,
    "Safe improvement": 0.2,
    "Validation experiment": 0.2,
    "Novel chemistry": 0.15,
    "Boundary exploration": 0.05,
    "Gap-filling experiment": 0.2,
    "High-risk high-reward": 0.1,
    "Worth exploring": 0.0,
}


def build_portfolio(pool: list, size: int = 10, portions: dict | None = None) -> list:
    """A balanced experiment portfolio bucketed by exact strategy label
    (see ``DEFAULT_STRATEGY_PORTIONS``) rather than a coarse risk-tier
    grouping. Each returned candidate carries strategy/info-gain/
    feasibility (from ``planner.enrich_candidates``) plus ``batch_rank``.

    Falls back to backfilling remaining slots from the overall utility-
    ranked pool if a bucket runs short, so the plan is never smaller than
    ``min(size, len(pool))`` just because one strategy was thin or absent.
    """
    if not pool:
        return []
    portions = portions or DEFAULT_STRATEGY_PORTIONS
    enriched = planner.enrich_candidates(pool)

    buckets: dict[str, list] = {}
    for c in enriched:
        buckets.setdefault(c["strategy"], []).append(c)
    for members in buckets.values():
        members.sort(key=lambda c: c.get("utility", 0.0), reverse=True)

    targets = {b: max(1, round(size * frac)) for b, frac in portions.items() if frac > 0}
    plan, used_ids = [], set()
    for b, target in targets.items():
        for c in buckets.get(b, [])[:target]:
            if id(c) not in used_ids:
                plan.append(c)
                used_ids.add(id(c))

    remaining = sorted((c for c in enriched if id(c) not in used_ids),
                       key=lambda c: c.get("utility", 0.0), reverse=True)
    for c in remaining:
        if len(plan) >= size:
            break
        plan.append(c)
        used_ids.add(id(c))

    plan.sort(key=lambda c: c.get("utility", 0.0), reverse=True)
    plan = plan[:size]
    for i, c in enumerate(plan, start=1):
        c["batch_rank"] = i
    return plan


def summarize(plan: list) -> dict:
    """``{"Safe improvement": 3, "Validation experiment": 2, ...}`` — the
    portfolio's actual composition, for a one-line summary like the spec's
    own illustrative example."""
    counts: dict[str, int] = {}
    for c in plan:
        label = c.get("strategy", "Unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts
