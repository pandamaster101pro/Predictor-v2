"""Tests for the self-contained Bayesian optimizer (bayesopt.py)."""
import numpy as np
import pytest
from scipy.stats import norm

import bayesopt


# ---------------------------------------------------------------------------
# Expected Improvement
# ---------------------------------------------------------------------------
def test_ei_zero_when_no_uncertainty():
    ei = bayesopt.expected_improvement(np.array([5.0]), np.array([0.0]),
                                       best_f=1.0, xi=0.0)
    assert ei[0] == 0.0


def test_ei_matches_closed_form():
    mu, sigma, best, xi = 2.0, 1.5, 1.0, 0.01
    z = (mu - best - xi) / sigma
    expected = (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    got = bayesopt.expected_improvement(np.array([mu]), np.array([sigma]),
                                        best_f=best, xi=xi)[0]
    assert got == pytest.approx(expected, rel=1e-9)


def test_ei_never_negative():
    mu = np.array([-10.0, 0.0, 10.0])
    sigma = np.array([1.0, 2.0, 0.5])
    ei = bayesopt.expected_improvement(mu, sigma, best_f=0.0, xi=0.0)
    assert np.all(ei >= 0.0)


def test_ei_rewards_uncertainty_at_equal_mean():
    # Same mean as the incumbent: the more uncertain point has higher EI.
    ei = bayesopt.expected_improvement(np.array([1.0, 1.0]),
                                       np.array([0.1, 2.0]), best_f=1.0, xi=0.0)
    assert ei[1] > ei[0]


# ---------------------------------------------------------------------------
# Surrogate
# ---------------------------------------------------------------------------
def _quadratic_data(n=25, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3, 3, size=(n, 1))
    y = -(X[:, 0] ** 2)                      # peak at x = 0
    return X, y


def test_surrogate_recovers_simple_function():
    X, y = _quadratic_data()
    surr = bayesopt.fit_surrogate(X, y, direction="maximize", random_state=0)
    # Prediction near the true optimum should beat a point far from it.
    mu_peak, _ = surr.predict(np.array([[0.0]]))
    mu_far, _ = surr.predict(np.array([[3.0]]))
    assert mu_peak[0] > mu_far[0]


def test_surrogate_direction_sign():
    X, y = _quadratic_data()
    smax = bayesopt.fit_surrogate(X, y, direction="maximize")
    smin = bayesopt.fit_surrogate(X, y, direction="minimize")
    assert smax.sign == 1.0 and smin.sign == -1.0
    # y_best is in internal space: for minimize it is -min(y) = -(most negative)
    assert smax.y_best == pytest.approx(float(np.max(y)))
    assert smin.y_best == pytest.approx(float(np.max(-y)))


def test_surrogate_predict_original_units():
    X, y = _quadratic_data()
    surr = bayesopt.fit_surrogate(X, y, direction="maximize")
    mu_orig, sigma = surr.predict_original(np.array([[0.0]]))
    # near the peak the original-unit mean should be close to 0 (true max value)
    assert mu_orig[0] == pytest.approx(0.0, abs=1.0)
    assert sigma[0] >= 0.0


def test_fit_requires_min_observations():
    with pytest.raises(ValueError):
        bayesopt.fit_surrogate(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))


def test_pseudo_observation_grows_dataset():
    X, y = _quadratic_data()
    surr = bayesopt.fit_surrogate(X, y, direction="maximize")
    aug = surr.with_pseudo_observation(np.array([0.5]), 0.0)
    assert aug.X_enc.shape[0] == surr.X_enc.shape[0] + 1
    assert aug.y_internal.shape[0] == surr.y_internal.shape[0] + 1


# ---------------------------------------------------------------------------
# Batch proposal
# ---------------------------------------------------------------------------
def _identity(v):
    return np.asarray(v, dtype=float)


def test_propose_batch_returns_q_points_in_bounds():
    X, y = _quadratic_data()
    surr = bayesopt.fit_surrogate(X, y, direction="maximize", random_state=1)
    bounds = [(-3.0, 3.0)]
    props = bayesopt.propose_batch(surr, bounds, _identity, q=4, random_state=1)
    assert len(props) == 4
    for p in props:
        assert -3.0 <= p.x[0] <= 3.0
        assert p.ei >= 0.0
        assert p.sigma >= 0.0


def test_propose_batch_points_are_distinct():
    # Kriging-Believer should spread the batch rather than stack duplicates.
    X, y = _quadratic_data(n=30, seed=2)
    surr = bayesopt.fit_surrogate(X, y, direction="maximize", random_state=2)
    props = bayesopt.propose_batch(surr, [(-3.0, 3.0)], _identity, q=5,
                                   random_state=2)
    xs = np.array([p.x[0] for p in props])
    assert len(np.unique(np.round(xs, 3))) >= 2


def test_integrality_respected():
    rng = np.random.RandomState(3)
    X = rng.uniform(0, 5, size=(20, 1))
    y = -(X[:, 0] - 3.0) ** 2
    surr = bayesopt.fit_surrogate(X, y, direction="maximize", random_state=3)
    props = bayesopt.propose_batch(surr, [(0.0, 5.0)], _identity, q=3,
                                   integrality=[True], random_state=3)
    for p in props:
        assert float(p.x[0]) == pytest.approx(round(float(p.x[0])), abs=1e-9)


def test_suggest_experiments_targets_optimum_region():
    # End-to-end: a 2D bowl with optimum at (1, -1); best proposal should land
    # nearer the optimum than a typical random point in the box.
    rng = np.random.RandomState(4)
    X = rng.uniform(-3, 3, size=(40, 2))
    y = -((X[:, 0] - 1.0) ** 2 + (X[:, 1] + 1.0) ** 2)
    bounds = [(-3.0, 3.0), (-3.0, 3.0)]
    surr, props = bayesopt.suggest_experiments(
        X, y, bounds, _identity, direction="maximize", q=5, random_state=4)
    best = max(props, key=lambda p: p.mean)
    dist = np.hypot(best.x[0] - 1.0, best.x[1] + 1.0)
    assert dist < 2.5            # comfortably inside the good region


def test_minimize_direction_finds_low_values():
    rng = np.random.RandomState(5)
    X = rng.uniform(-3, 3, size=(30, 1))
    y = (X[:, 0] - 2.0) ** 2      # minimum at x = 2
    surr, props = bayesopt.suggest_experiments(
        X, y, [(-3.0, 3.0)], _identity, direction="minimize", q=5,
        random_state=5)
    best = min(props, key=lambda p: p.mean)
    assert abs(best.x[0] - 2.0) < 1.5


def test_empty_batch_for_zero_q():
    X, y = _quadratic_data()
    surr = bayesopt.fit_surrogate(X, y)
    assert bayesopt.propose_batch(surr, [(-3.0, 3.0)], _identity, q=0) == []
