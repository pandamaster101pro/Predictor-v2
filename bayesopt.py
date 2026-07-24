"""
Bayesian optimization for experiment suggestion.
=================================================

Self-contained and fully offline: uses only scikit-learn's
``GaussianProcessRegressor``, scipy and numpy — no torch / botorch / skopt.

The workflow this supports is *active learning* / sequential experimental
design. Given the experiments run so far — ``X`` (synthesis conditions, already
numerically encoded) and ``y`` (a measured outcome such as reversible
capacity) — it:

  1. fits a Gaussian-process (GP) surrogate that predicts the outcome AND its
     own uncertainty everywhere in the search space, then
  2. proposes the next batch of experiments to run by maximizing an
     *acquisition function* (Expected Improvement), which deliberately balances
     exploiting conditions the model already thinks are good against exploring
     conditions the model is unsure about.

This is different from ordinary "optimize the surrogate" search: EI values a
candidate by how much it is *expected to improve on the best result so far*,
accounting for uncertainty, so the suggestions are the most informative next
experiments rather than just the model's current best guess.

Everything works in an internal *maximization* convention. For a minimization
objective the outcome is negated on the way in and results are reported back in
the caller's original units, so callers never have to think about the flip.

The acquisition function is optimized over the *search space* (a compact vector
of numeric knobs and categorical indices, some marked integer-valued), while the
GP itself lives in the *encoded feature space* (e.g. one-hot expanded). The
caller supplies a ``to_encoded`` mapping between the two, so this module stays
agnostic to how features are engineered.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, Matern, WhiteKernel,
)
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Acquisition function
# -----------------------------------------------------------------------------
def expected_improvement(mu, sigma, best_f, xi=0.01):
    """Expected Improvement over ``best_f`` for a MAXIMIZATION objective.

    ``mu`` / ``sigma`` are the surrogate's posterior mean and standard
    deviation (arrays), ``best_f`` the best outcome observed so far, ``xi`` a
    small non-negative exploration margin (larger => more exploratory).

    EI(x) = (mu - best - xi) * Phi(z) + sigma * phi(z),  z = (mu-best-xi)/sigma
    and EI = 0 wherever sigma collapses to 0 (no uncertainty, no expected gain).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = mu - best_f - xi
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(sigma > 1e-12, ei, 0.0)
    return np.maximum(ei, 0.0)


# -----------------------------------------------------------------------------
# Surrogate model
# -----------------------------------------------------------------------------
@dataclass
class Surrogate:
    """A fitted GP surrogate over the ENCODED feature space, plus the bookkeeping
    Bayesian optimization needs (feature scaler, best observed value, and the
    maximize/minimize sign so results read back in the caller's units)."""

    gp: GaussianProcessRegressor
    scaler: StandardScaler
    X_enc: np.ndarray          # observed encoded rows (internal orientation)
    y_internal: np.ndarray     # observed outcomes in internal (maximize) space
    sign: float                # +1 maximize, -1 minimize (applied to raw y)

    @property
    def y_best(self) -> float:
        """Best observed outcome in the internal maximization space."""
        return float(np.max(self.y_internal))

    def predict(self, X_enc, return_std=True):
        """Posterior mean (and std) in the internal maximization space."""
        Xs = self.scaler.transform(np.atleast_2d(np.asarray(X_enc, dtype=float)))
        if return_std:
            mu, sigma = self.gp.predict(Xs, return_std=True)
            return mu, sigma
        return self.gp.predict(Xs)

    def predict_original(self, X_enc):
        """Posterior mean/std reported in the caller's ORIGINAL units."""
        mu, sigma = self.predict(X_enc, return_std=True)
        return self.sign * mu, sigma

    def with_pseudo_observation(self, x_enc, y_internal_value) -> "Surrogate":
        """Return a NEW surrogate conditioned on one extra (hallucinated) point,
        reusing the already-learned kernel hyper-parameters (no re-optimization).

        This is the "Kriging Believer" trick for batch proposal: after picking a
        candidate we pretend we already ran it and observed the model's own mean
        there, so the next acquisition maximization is steered away from it and
        toward genuinely different, still-informative conditions."""
        X_aug = np.vstack([self.X_enc, np.atleast_2d(np.asarray(x_enc, float))])
        y_aug = np.append(self.y_internal, float(y_internal_value))
        Xs = self.scaler.transform(X_aug)
        gp2 = GaussianProcessRegressor(
            kernel=self.gp.kernel_,      # fitted kernel, fixed
            optimizer=None,              # do NOT re-optimize — cheap conditioning
            normalize_y=True,
            alpha=self.gp.alpha,
            random_state=None,
        )
        gp2.fit(Xs, y_aug)
        return Surrogate(gp=gp2, scaler=self.scaler, X_enc=X_aug,
                         y_internal=y_aug, sign=self.sign)


def fit_surrogate(X_enc, y, direction="maximize", random_state=42,
                  n_restarts=3, alpha=1e-6) -> Surrogate:
    """Fit a GP surrogate on encoded observations.

    ``X_enc`` : (n, d) numeric feature matrix (already one-hot / descriptor
    encoded). ``y`` : (n,) outcomes in the caller's units. ``direction`` :
    ``"maximize"`` or ``"minimize"``. A Matern(nu=2.5) kernel with a learned
    scale and a WhiteKernel noise term is used — a robust default for smooth-ish
    physical responses that also tolerates noisy replicate measurements.
    """
    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be 'maximize' or 'minimize'")
    X_enc = np.atleast_2d(np.asarray(X_enc, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if X_enc.shape[0] != y.shape[0]:
        raise ValueError("X_enc and y must have the same number of rows")
    if X_enc.shape[0] < 3:
        raise ValueError("need at least 3 observations to fit a GP surrogate")

    sign = 1.0 if direction == "maximize" else -1.0
    y_internal = sign * y

    scaler = StandardScaler().fit(X_enc)
    Xs = scaler.transform(X_enc)

    # Isotropic Matern (robust with modest data + many one-hot dims) + learned
    # homoscedastic noise. Bounds keep the optimizer well-conditioned.
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, alpha=alpha,
        n_restarts_optimizer=n_restarts, random_state=random_state,
    )
    gp.fit(Xs, y_internal)
    return Surrogate(gp=gp, scaler=scaler, X_enc=X_enc,
                     y_internal=y_internal, sign=sign)


# -----------------------------------------------------------------------------
# Acquisition optimization + batch proposal
# -----------------------------------------------------------------------------
@dataclass
class Proposal:
    """One proposed experiment."""

    x: np.ndarray            # search-space vector (numeric knobs + cat indices)
    mean: float              # predicted outcome in the caller's ORIGINAL units
    sigma: float             # predictive std (uncertainty), original-unit scale
    ei: float                # Expected Improvement at proposal time


def _maximize_acquisition(acq_neg, bounds, integrality, random_state, de_kwargs):
    """Maximize an acquisition (given as its negative, to be minimized) over the
    search space with differential evolution, honoring integer dimensions."""
    result = differential_evolution(
        acq_neg, bounds, integrality=integrality, seed=random_state, **de_kwargs)
    return np.asarray(result.x, dtype=float), -float(result.fun)


def propose_batch(
    surrogate: Surrogate,
    bounds: Sequence[Tuple[float, float]],
    to_encoded: Callable[[np.ndarray], np.ndarray],
    q: int = 5,
    integrality: Sequence[bool] | None = None,
    xi: float = 0.01,
    random_state: int = 42,
    de_kwargs: dict | None = None,
    callback: Callable[[int, int], None] | None = None,
) -> List[Proposal]:
    """Propose ``q`` experiments by iterated Expected-Improvement maximization.

    ``bounds`` are the per-dimension (lo, hi) limits of the SEARCH space;
    ``integrality`` marks integer dimensions (e.g. categorical indices);
    ``to_encoded`` maps a search vector to the encoded feature vector the GP
    consumes. Batch diversity comes from the Kriging-Believer update: each
    accepted candidate is folded into the surrogate as a pseudo-observation
    before the next one is chosen.
    """
    if q < 1:
        return []
    n_dims = len(bounds)
    if integrality is None:
        integrality = [False] * n_dims
    integrality = list(integrality)
    de_defaults = dict(popsize=15, maxiter=60, tol=1e-4, mutation=(0.5, 1.0),
                       recombination=0.9, polish=False)
    if de_kwargs:
        de_defaults.update(de_kwargs)

    proposals: List[Proposal] = []
    work = surrogate
    seen: List[np.ndarray] = []
    for i in range(q):
        if callback is not None:
            callback(i, q)
        best_f = work.y_best

        def acq_neg(search_vec, _work=work, _best=best_f):
            x_enc = np.asarray(to_encoded(search_vec), dtype=float).reshape(1, -1)
            mu, sigma = _work.predict(x_enc, return_std=True)
            return -expected_improvement(mu, sigma, _best, xi)[0]

        x_star, ei_star = _maximize_acquisition(
            acq_neg, bounds, integrality, random_state + i, de_defaults)

        x_enc_star = np.asarray(to_encoded(x_star), dtype=float).reshape(1, -1)
        mu_int, sigma = work.predict(x_enc_star, return_std=True)
        mean_original = float(surrogate.sign * mu_int[0])
        proposals.append(Proposal(x=x_star, mean=mean_original,
                                  sigma=float(sigma[0]), ei=float(ei_star)))
        seen.append(x_star)

        # Kriging Believer: condition on the hallucinated mean before the next
        # pick so the batch spreads out instead of stacking on one point.
        if i < q - 1:
            work = work.with_pseudo_observation(x_enc_star[0], float(mu_int[0]))
    return proposals


def suggest_experiments(
    X_enc, y, bounds, to_encoded,
    direction="maximize", q=5, integrality=None, xi=0.01,
    random_state=42, n_restarts=3, de_kwargs=None,
) -> Tuple[Surrogate, List[Proposal]]:
    """Convenience end-to-end call: fit the surrogate then propose ``q``
    experiments. Returns ``(surrogate, proposals)``."""
    surrogate = fit_surrogate(X_enc, y, direction=direction,
                              random_state=random_state, n_restarts=n_restarts)
    proposals = propose_batch(
        surrogate, bounds, to_encoded, q=q, integrality=integrality, xi=xi,
        random_state=random_state, de_kwargs=de_kwargs)
    return surrogate, proposals
