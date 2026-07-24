"""test_portfolio.py — strategy-matched granular portfolio builder.

Run:  python -m pytest test_portfolio.py -q
"""

import pytest

import portfolio as PF


def _cand(**kw):
    base = {"recipe": [], "risk": "Moderate", "applicability_pct": 50.0,
           "similarity_pct": 50.0, "utility": 0.0, "edges": [],
           "chemical_recommendations": []}
    base.update(kw)
    return base


def test_empty_pool_returns_empty():
    assert PF.build_portfolio([], size=5) == []


def test_never_exceeds_pool_size():
    pool = [_cand(utility=float(i)) for i in range(3)]
    plan = PF.build_portfolio(pool, size=10)
    assert len(plan) == 3


def test_returns_requested_size_when_pool_large_and_diverse():
    # A mix engineered to hit several distinct strategy labels.
    pool = [_cand(utility=100.0)]  # -> Best candidate
    pool += [_cand(utility=float(50 - i), risk="Low", applicability_pct=90.0,
                   similarity_pct=50.0) for i in range(4)]  # -> Safe improvement
    pool += [_cand(utility=float(10 - i), similarity_pct=97.0) for i in range(3)]  # -> Validation
    pool += [_cand(utility=float(5 - i), applicability_pct=10.0, similarity_pct=50.0)
            for i in range(4)]  # -> Gap-filling
    plan = PF.build_portfolio(pool, size=10)
    assert len(plan) == 10
    for i, c in enumerate(plan, start=1):
        assert c["batch_rank"] == i
    assert all("strategy" in c and "info_gain" in c and "feasibility" in c for c in plan)


def test_does_not_mutate_input_pool():
    pool = [_cand(utility=1.0)]
    PF.build_portfolio(pool, size=1)
    assert "strategy" not in pool[0]


def test_backfills_when_a_bucket_is_empty():
    # Every candidate is a "Best candidate" tie (utility all equal) so most
    # strategy buckets are empty; the portfolio should still fill via backfill.
    pool = [_cand(utility=5.0) for _ in range(6)]
    plan = PF.build_portfolio(pool, size=6)
    assert len(plan) == 6


def test_summarize_counts_by_strategy_label():
    pool = [_cand(utility=100.0)]
    pool += [_cand(utility=float(10 - i), similarity_pct=97.0) for i in range(3)]
    plan = PF.build_portfolio(pool, size=4)
    counts = PF.summarize(plan)
    assert sum(counts.values()) == len(plan)
    assert "Best candidate" in counts


def test_summarize_empty_plan():
    assert PF.summarize([]) == {}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
