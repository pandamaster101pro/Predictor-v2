"""
planner.py — Turns a ranked candidate pool into research-decision support.

Pure, headless functions over the candidate dicts the optimizer already
produces (``app_imgui._rank_top_recipes`` / ``_pareto_front_candidates``):
no new model, no new search — just enrichment and a balanced-portfolio
selection over data already computed.

  * experiment_strategy    — categorize one candidate ("Safe improvement",
                              "High-risk high-reward", ...) relative to its peers
  * information_gain_score — 0-100 "how much would this experiment teach the
                              model", from the novelty/dissimilarity signals
                              already computed (no fabricated epistemic term —
                              see the docstring for why)
  * feasibility_score      — "Easy"/"Moderate"/"Difficult" from objectively
                              countable recipe structure (steps, stages,
                              boundary conditions), never a guessed equipment cost
  * build_batch_plan       — a balanced portfolio (some safe, some exploratory)
                              instead of `n` near-identical "best" recipes
"""

from __future__ import annotations

from constraint_engine import count_stages, count_steps


def experiment_strategy(candidate: dict, pool: list) -> tuple:
    """Categorize one candidate relative to the pool it came from.

    ``pool`` is the full Top-N or Pareto list this candidate is a member of
    — needed for relative labels like "the best one" or "the strongest
    among the risky ones", which only make sense compared to peers. Returns
    ``(label, reason)``.
    """
    utility = candidate.get("utility", 0.0)
    applicability = candidate.get("applicability_pct", 0.0)
    risk = candidate.get("risk", "Moderate")
    similarity = candidate.get("similarity_pct")
    edges = candidate.get("edges") or []
    chem_recs = candidate.get("chemical_recommendations") or []

    utilities = [c.get("utility", 0.0) for c in pool] or [utility]
    if utility >= max(utilities) - 1e-9:
        return "Best candidate", "Highest overall score among the recommended experiments."

    if similarity is not None and similarity >= 95.0:
        return ("Validation experiment",
               f"Nearly identical to a real training experiment ({similarity:.0f}% match) "
               "— a good check that the model's prediction here is trustworthy.")

    if risk == "Low" and applicability >= 80.0:
        return ("Safe improvement",
               "Low risk and well inside the training domain — a reliable next step.")

    if chem_recs and any(c.get("similarity", 1.0) < 0.7 for c in chem_recs):
        low = min(chem_recs, key=lambda c: c.get("similarity", 1.0))
        return ("Novel chemistry",
               f"Uses {low.get('recommended', 'a reagent')}, which only loosely matches "
               f"known chemistry (similarity {low.get('similarity', 0):.2f}).")

    if edges:
        return ("Boundary exploration",
               f"Pushes {', '.join(map(str, edges[:3]))} to the edge of what's been tried.")

    if risk in ("Moderate", "High"):
        risky = [c.get("utility", 0.0) for c in pool if c.get("risk") in ("Moderate", "High")]
        if risky and utility >= max(risky) - 1e-9:
            return ("High-risk high-reward",
                   f"{risk} risk, but the strongest predicted outcome among the riskier options.")

    if applicability < 40.0:
        return ("Gap-filling experiment",
               "Far from existing training data — informative for expanding dataset "
               "coverage even though the individual prediction is less certain.")

    return "Worth exploring", "A reasonable alternative not captured by the other categories."


# ---- active learning ---------------------------------------------------------
def information_gain_score(candidate: dict) -> float:
    """0-100: how much would performing this experiment likely teach the
    model, from signals already computed for the candidate — novelty
    (distance from the whole training domain) and dissimilarity to its
    single closest known experiment.

    Deliberately NOT a fabricated "epistemic uncertainty" term: this
    optimizer's model (gradient-boosted trees via XGBoost) has no per-sample
    prediction variance the way a bagging ensemble (Random Forest / Extra
    Trees) does, so a genuine third uncertainty component isn't available —
    see screening.py's uncertainty() for the same limitation acknowledged
    there. Novelty and dissimilarity are the two honest signals on hand.
    """
    applicability = candidate.get("applicability_pct", 0.0)
    similarity = candidate.get("similarity_pct")
    novelty = 100.0 - applicability
    if similarity is not None:
        score = 0.6 * novelty + 0.4 * (100.0 - similarity)
    else:
        score = novelty
    return round(max(0.0, min(100.0, score)), 1)


# ---- feasibility --------------------------------------------------------------
def feasibility_score(recipe, edges=None) -> tuple:
    """Estimate lab feasibility from objectively countable recipe structure
    — how many optional processing steps are used and how many pyrolysis
    stages are active — never a guessed equipment list or cost. Returns
    ``(label, reasons)`` with label in {"Easy", "Moderate", "Difficult"}.
    """
    recipe_dict = dict(recipe)
    used_steps = count_steps(recipe_dict)
    n_stages = count_stages(recipe_dict)

    reasons = [f"{used_steps} optional processing step(s) used",
              f"{n_stages} pyrolysis stage(s)"]
    complexity = used_steps + max(0, n_stages - 1) * 2
    if edges:
        reasons.append(f"{len(edges)} knob(s) at the edge of tested conditions "
                       "(needs precise control)")
        complexity += 1

    if complexity <= 1:
        label = "Easy"
    elif complexity <= 3:
        label = "Moderate"
    else:
        label = "Difficult"
    return label, reasons


# ---- batch planner -------------------------------------------------------------
DEFAULT_PORTIONS = {"safe": 0.3, "moderate": 0.3, "exploratory": 0.2, "high_risk": 0.2}

_STRATEGY_BUCKET = {
    "Best candidate": "safe", "Safe improvement": "safe", "Validation experiment": "safe",
    "Boundary exploration": "moderate", "Novel chemistry": "moderate",
    "Gap-filling experiment": "exploratory", "Worth exploring": "exploratory",
    "High-risk high-reward": "high_risk",
}


def enrich_candidates(pool: list) -> list:
    """Attach strategy/info-gain/feasibility to a copy of every candidate in
    ``pool`` (Top-N or Pareto list), without mutating the originals."""
    enriched = []
    for c in pool:
        c2 = dict(c)
        c2["strategy"], c2["strategy_reason"] = experiment_strategy(c, pool)
        c2["info_gain"] = information_gain_score(c)
        c2["feasibility"], c2["feasibility_reasons"] = feasibility_score(
            c.get("recipe", []), c.get("edges"))
        enriched.append(c2)
    return enriched


def build_batch_plan(pool: list, batch_size: int = 10, portions: dict | None = None) -> list:
    """A balanced experiment portfolio from a ranked candidate pool: a mix
    of safe / moderate / exploratory / high-risk picks (see
    ``DEFAULT_PORTIONS``) instead of `batch_size` near-identical "best"
    recipes. Each returned candidate carries strategy/info-gain/feasibility
    plus ``batch_rank``.

    Falls back to backfilling remaining slots from the overall utility-
    ranked pool if a bucket runs short, so the plan is never smaller than
    ``min(batch_size, len(pool))`` just because one category was thin.
    """
    if not pool:
        return []
    portions = portions or DEFAULT_PORTIONS
    enriched = enrich_candidates(pool)

    buckets: dict[str, list] = {}
    for c in enriched:
        buckets.setdefault(_STRATEGY_BUCKET.get(c["strategy"], "moderate"), []).append(c)
    for members in buckets.values():
        members.sort(key=lambda c: c.get("utility", 0.0), reverse=True)

    targets = {b: max(1, round(batch_size * frac)) for b, frac in portions.items()}
    plan, used_ids = [], set()
    for b, target in targets.items():
        for c in buckets.get(b, [])[:target]:
            if id(c) not in used_ids:
                plan.append(c)
                used_ids.add(id(c))

    remaining = sorted((c for c in enriched if id(c) not in used_ids),
                       key=lambda c: c.get("utility", 0.0), reverse=True)
    for c in remaining:
        if len(plan) >= batch_size:
            break
        plan.append(c)
        used_ids.add(id(c))

    plan.sort(key=lambda c: c.get("utility", 0.0), reverse=True)
    plan = plan[:batch_size]
    for i, c in enumerate(plan, start=1):
        c["batch_rank"] = i
    return plan
