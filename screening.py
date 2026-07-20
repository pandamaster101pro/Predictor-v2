"""
screening.py  —  BioCarbon Screen decision-support engine.
==========================================================

Turns a trained ``MultiOutputRegressor(ExtraTreesRegressor)`` into a research
assistant for biomass-derived hard-carbon synthesis.  Everything here is pure
logic (no GUI) so it can be unit-tested head-lessly and reused by reports.

For a set of synthesis conditions and a chosen performance target it produces:

  * a point prediction PLUS an honest prediction interval + confidence level
    (epistemic spread across the forest's trees, combined with the model's
    cross-validated residual error);
  * an applicability-domain / out-of-distribution verdict (is this recipe
    inside the model's experience?);
  * the most similar real experiments from the training data;
  * where the prediction ranks against every experiment seen;
  * a human-language recommendation (Excellent / Strong / Worth testing /
    Low priority / Poor candidate) with the reasons behind it;
  * per-feature contributions (SHAP when available, perturbation otherwise),
    partial-dependence curves and a local sensitivity ranking.

It never overstates confidence and always separates *prediction* from
*measurement*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

RANDOM_STATE = 42

# Verdict tiers, best -> worst, with an RGBA colour for the GUI.
VERDICTS = {
    "Excellent candidate": (0.15, 0.80, 0.35, 1.0),
    "Strong candidate":    (0.45, 0.78, 0.30, 1.0),
    "Worth testing":       (0.90, 0.75, 0.20, 1.0),
    "Low priority":        (0.90, 0.55, 0.20, 1.0),
    "Poor candidate":      (0.88, 0.32, 0.28, 1.0),
}

# A DELIBERATELY skipped step ("we didn't do it") — a real choice, encoded as the
# explicit "None" category. Kept distinct from UNKNOWN ("we don't know").
NOT_DONE_TOKENS = {"--", "---", "----", "—", "–", "none", "nil", "not applied"}
# Truly unrecorded / unknown values.
UNKNOWN_TOKENS = {"", "-", "unknown", "unspecified", "nan", "na", "n/a",
                  "missing", "missing or unspecified", "tbd", "?", "input"}
# Any empty-ish token (either kind).
BLANK_TOKENS = UNKNOWN_TOKENS | NOT_DONE_TOKENS


# =============================================================================
# ENCODING HELPERS  (mirror the app's build_matrix so screening lines up 1:1)
# =============================================================================
def feature_to_original(feature_columns, numeric_schema, categorical_schema):
    """
    Map every ENCODED column back to the ORIGINAL synthesis variable it came
    from, so one-hot dummies can be summed back into a readable feature name.

    ``{"Pyrolysis_Temp_C": "Pyrolysis_Temp_C",
       "Biomass_Source_Rice husk": "Biomass_Source"}``
    """
    mapping = {}
    cats = sorted(categorical_schema.keys(), key=len, reverse=True)  # longest first
    for col in feature_columns:
        if col in numeric_schema:
            mapping[col] = col
            continue
        hit = None
        for c in cats:
            if col == c or col.startswith(c + "_"):
                hit = c
                break
        mapping[col] = hit if hit is not None else col
    return mapping


def reconstruct_raw(X_enc, numeric_schema, categorical_schema):
    """
    Rebuild a readable raw-feature frame from the one-hot-encoded training
    matrix (numeric columns are exact; a categorical row's value is whichever
    dummy is 1, or the dropped baseline category when every dummy is 0).
    """
    n = len(X_enc)
    out = {}
    for col in numeric_schema:
        out[col] = X_enc[col].values if col in X_enc.columns else np.full(n, numeric_schema[col])
    # Assign each dummy to its true owner (longest matching prefix) so a column
    # like 'Electrolyte' does not steal 'Electrolyte_Additive_*' dummies.
    orig = feature_to_original(list(X_enc.columns), numeric_schema, categorical_schema)
    for col, choices in categorical_schema.items():
        dummies = [c for c in X_enc.columns if orig.get(c) == col]
        baseline = choices[0] if choices else "Missing"
        vals = np.array([baseline] * n, dtype=object)
        for d in dummies:
            level = d[len(col) + 1:]
            mask = X_enc[d].values.astype(float) > 0.5
            vals[mask] = level
        out[col] = vals
    return pd.DataFrame(out, index=X_enc.index)


def _confidence_from_relwidth(rel, in_domain):
    """Map a relative interval half-width (sigma / target-spread) to a label."""
    if not in_domain:
        return "Low" if rel < 0.8 else "Very low"
    if rel < 0.30:
        return "High"
    if rel < 0.55:
        return "Moderate"
    if rel < 0.90:
        return "Low"
    return "Very low"


_CONF_SCORE = {"High": 1.0, "Moderate": 0.66, "Low": 0.33, "Very low": 0.12}


# =============================================================================
# THE SCREENER  —  everything about one trained model, cached for speed
# =============================================================================
class Screener:
    """Wraps a trained model + its training data to answer screening questions."""

    def __init__(self, model, feature_columns, numeric_schema, categorical_schema,
                 targets, X_train, y_train, cv_rmse=None, cv_r2=None,
                 display_labels=None):
        self.model = model
        self.feature_columns = list(feature_columns)
        self._col_index = {c: i for i, c in enumerate(self.feature_columns)}
        self.numeric_schema = dict(numeric_schema)
        self.categorical_schema = {k: list(v) for k, v in categorical_schema.items()}
        self.targets = list(targets)
        # Display names for internal feature columns (e.g. the app's parsed
        # messy-column parts: numeric_feature_A1 -> "Pretreat 1: number/percent").
        self.labels = {str(k): str(v) for k, v in (display_labels or {}).items()}

        self.X_train = X_train[self.feature_columns].astype(float)
        self.y_train = y_train.reset_index(drop=True)
        self.cv_rmse = dict(cv_rmse or {})
        self.cv_r2 = dict(cv_r2 or {})

        # Raw (human-readable) view of every training experiment.
        self.train_raw = reconstruct_raw(self.X_train, self.numeric_schema,
                                         self.categorical_schema).reset_index(drop=True)

        # Observed numeric ranges (for out-of-range warnings + PD sweeps).
        self.numeric_ranges = {}
        for c in self.numeric_schema:
            s = pd.to_numeric(self.train_raw[c], errors="coerce").dropna()
            if len(s):
                self.numeric_ranges[c] = (float(s.min()), float(s.max()),
                                          float(np.percentile(s, 1)),
                                          float(np.percentile(s, 99)))

        # Standardise the feature space so distances aren't dominated by scale.
        self.scaler = StandardScaler().fit(self.X_train.values)
        self.Xs = self.scaler.transform(self.X_train.values)

        # k-NN model + the training set's own neighbour-distance distribution,
        # which defines "typical" so we can flag isolated (novel) recipes.
        self.k = int(min(8, max(2, len(self.Xs) - 1)))
        self._nn = NearestNeighbors(n_neighbors=min(self.k + 1, len(self.Xs))).fit(self.Xs)
        d_self, _ = self._nn.kneighbors(self.Xs)
        self._train_meandist = d_self[:, 1:].mean(axis=1)   # drop self (col 0)
        self._d_ref = float(np.median(self._train_meandist)) or 1.0

        # Target spread (for confidence) and forests for uncertainty/SHAP.
        self._y_std = {t: float(self.y_train[t].std() or 1.0) for t in self.targets}
        self._forests = self._extract_forests()
        self._shap = {}   # lazily-built SHAP explainers, per target

        # Aggregated (original-feature) importances for the whole model.
        self.orig_map = feature_to_original(self.feature_columns,
                                            self.numeric_schema, self.categorical_schema)
        self.importance = self._global_importance()

    def pretty(self, name):
        """Human-friendly label: display name when one exists, else _pretty."""
        return _pretty(self.labels.get(str(name), name))

    # ---- internals ---------------------------------------------------------
    def _extract_forests(self):
        """Return {target: fitted forest} when the model exposes per-tree access."""
        forests = {}
        subs = getattr(self.model, "estimators_", None)
        if subs is not None and len(subs) == len(self.targets):
            for t, est in zip(self.targets, subs):
                forests[t] = est if hasattr(est, "estimators_") else None
        return forests

    def _global_importance(self):
        agg = {}
        subs = getattr(self.model, "estimators_", None)
        if subs is None:
            return []
        try:
            imp = np.mean([e.feature_importances_ for e in subs], axis=0)
        except Exception:
            return []
        for col, v in zip(self.feature_columns, imp):
            agg[self.orig_map[col]] = agg.get(self.orig_map[col], 0.0) + float(v)
        return sorted(agg.items(), key=lambda kv: kv[1], reverse=True)

    def encode(self, raw):
        """Turn a {feature: value} dict into a model-aligned 1-row float frame.

        Builds the encoded vector directly by column index rather than via
        get_dummies + reindex. This is collision-proof: overlapping column
        names (e.g. 'Electrolyte' vs 'Electrolyte_Additive') or a categorical
        value that happens to look like another column can't create duplicate
        labels, and an unknown/baseline category simply leaves its dummies at 0.
        """
        x = np.zeros(len(self.feature_columns), dtype=float)
        for c, med in self.numeric_schema.items():
            i = self._col_index.get(c)
            if i is None:
                continue
            try:
                x[i] = float(raw.get(c, med))
            except (TypeError, ValueError):
                x[i] = med
        for c in self.categorical_schema:
            v = raw.get(c, "Missing")
            v = "Missing" if v is None else str(v)
            i = self._col_index.get(f"{c}_{v}")
            if i is not None:      # baseline / unseen level -> all dummies stay 0
                x[i] = 1.0
        return pd.DataFrame([x], columns=self.feature_columns)

    # ---- 1. prediction + uncertainty --------------------------------------
    def uncertainty(self, X_enc):
        """
        Per target: point prediction plus a 95% interval that combines the
        forest's tree-to-tree spread (epistemic) with the model's cross-
        validated residual error (aleatoric).
        """
        out = {}
        point = self.model.predict(X_enc)[0]
        for i, t in enumerate(self.targets):
            mean = float(point[i])
            forest = self._forests.get(t)
            if forest is not None:
                tree_preds = np.array([tr.predict(X_enc.values)[0] for tr in forest.estimators_])
                std_ens = float(tree_preds.std())
            else:
                std_ens = 0.0
            resid = float(self.cv_rmse.get(t, std_ens))
            sigma = float(np.sqrt(std_ens ** 2 + resid ** 2)) or (0.05 * abs(mean) + 1e-6)
            lo, hi = mean - 1.96 * sigma, mean + 1.96 * sigma
            rel = sigma / (self._y_std[t] or 1.0)
            out[t] = {
                "mean": mean, "sigma": sigma, "std_ensemble": std_ens,
                "resid_rmse": resid, "lo": lo, "hi": hi,
                "rel_width": rel,
                "expected_error": resid,
                "conf_raw": _confidence_from_relwidth(rel, True),
            }
        return out

    # ---- 2. applicability domain / novelty --------------------------------
    def applicability(self, X_enc):
        """How isolated is this recipe versus the training cloud?"""
        xs = self.scaler.transform(X_enc.values)
        d, _ = self._nn.kneighbors(xs, n_neighbors=min(self.k, len(self.Xs)))
        mean_d = float(d.mean())
        pct = float((self._train_meandist < mean_d).mean() * 100.0)   # 0..100
        in_domain = pct <= 95.0
        if pct < 50:
            label = "well inside the training domain"
        elif pct < 80:
            label = "inside the training domain"
        elif pct <= 95:
            label = "at the edge of the training domain"
        else:
            label = "OUTSIDE the training domain"
        similarity_pct = 100.0 * float(np.exp(-mean_d / (self._d_ref * self.k ** 0.0)))
        return {"distance": mean_d, "percentile": pct, "in_domain": in_domain,
                "label": label, "novelty": pct, "nn_similarity": similarity_pct}

    def ood_flags(self, raw):
        """Concrete out-of-distribution warnings for individual inputs."""
        flags = []
        for c, (lo, hi, p1, p99) in self.numeric_ranges.items():
            try:
                v = float(raw.get(c, self.numeric_schema[c]))
            except (TypeError, ValueError):
                continue
            if v < lo or v > hi:
                flags.append(
                    f"{self.pretty(c)} = {v:g} is outside the observed range "
                    f"[{lo:g}, {hi:g}] — extrapolation.")
        for c, choices in self.categorical_schema.items():
            v = raw.get(c)
            if v is None:
                continue
            sv = str(v)
            if sv not in choices and sv.strip().lower() not in BLANK_TOKENS:
                flags.append(f"{self.pretty(c)} = '{sv}' was never seen in training "
                             f"— unknown category.")
        return flags

    def missing_inputs(self, raw):
        """Categorical inputs left UNKNOWN (not recorded).

        A deliberate 'None' / 'not done' is a real choice, so it is NOT counted
        as missing supporting data.
        """
        missing = []
        for c in self.categorical_schema:
            v = str(raw.get(c, "Missing")).strip().lower()
            if v in UNKNOWN_TOKENS:
                missing.append(self.pretty(c))
        return missing

    # ---- 3. similar experiments -------------------------------------------
    def similar(self, X_enc, k=6):
        """The k closest real experiments, with their measured performance."""
        xs = self.scaler.transform(X_enc.values)
        kk = min(k, len(self.Xs))
        d, idx = self._nn.kneighbors(xs, n_neighbors=kk)
        rows = []
        for dist, j in zip(d[0], idx[0]):
            sim = 100.0 * float(np.exp(-dist / self._d_ref))
            measured = {t: float(self.y_train.iloc[j][t]) for t in self.targets}
            conditions = {c: self.train_raw.iloc[j][c] for c in self.train_raw.columns}
            rows.append({"index": int(j), "distance": float(dist),
                         "similarity": max(0.0, min(100.0, sim)),
                         "measured": measured, "conditions": conditions})
        return rows

    # ---- 4. ranking vs the dataset ----------------------------------------
    def ranking(self, target, value):
        col = self.y_train[target].values
        pct = float((col < value).mean() * 100.0)
        return {"percentile": pct, "obs_min": float(col.min()),
                "obs_max": float(col.max()), "obs_mean": float(col.mean()),
                "better_than": pct}

    # ---- 5. explanations ---------------------------------------------------
    def contributions(self, X_enc, target, top=10):
        """
        Per-original-feature contribution to THIS prediction.  Uses SHAP tree
        values when available (exact for trees); otherwise a perturbation
        fallback (set each feature to its training baseline, measure the change).
        """
        i = self.targets.index(target)
        forest = self._forests.get(target)
        contribs = {}
        base = None
        if forest is not None:
            try:
                import shap
                if target not in self._shap:
                    self._shap[target] = shap.TreeExplainer(forest)
                expl = self._shap[target]
                sv = np.asarray(expl.shap_values(X_enc.values))
                sv = sv[0] if sv.ndim == 2 else sv
                base = float(getattr(expl, "expected_value", np.mean(sv)) or 0.0)
                for col, val in zip(self.feature_columns, sv):
                    o = self.orig_map[col]
                    contribs[o] = contribs.get(o, 0.0) + float(val)
            except Exception:
                contribs = {}
        if not contribs:                       # perturbation fallback
            full = float(self.model.predict(X_enc)[0][i])
            base = full
            for o in set(self.orig_map.values()):
                pert = X_enc.copy()
                if o in self.numeric_schema:
                    pert[o] = float(self.X_train[o].median())
                else:
                    for d in [c for c in self.feature_columns if self.orig_map[c] == o]:
                        pert[d] = 0.0
                delta = full - float(self.model.predict(pert)[0][i])
                contribs[o] = delta
        ranked = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return {"base": base, "contributions": ranked[:top]}

    def partial_dependence(self, raw, feature, target, n=25):
        """Sweep one feature across its range, holding the rest at ``raw``."""
        i = self.targets.index(target)
        xs_out, ys_out = [], []
        if feature in self.numeric_schema:
            lo, hi = self.numeric_ranges.get(feature, (0, 1, 0, 1))[2:4]
            grid = np.linspace(lo, hi, n)
            for g in grid:
                r = dict(raw); r[feature] = g
                ys_out.append(float(self.model.predict(self.encode(r))[0][i]))
                xs_out.append(float(g))
        else:
            for lvl in self.categorical_schema.get(feature, []):
                r = dict(raw); r[feature] = lvl
                ys_out.append(float(self.model.predict(self.encode(r))[0][i]))
                xs_out.append(lvl)
        return xs_out, ys_out

    def sensitivity(self, raw, target, top=8):
        """How much the prediction swings as each feature spans its range."""
        rows = []
        feats = list(self.numeric_schema.keys()) + list(self.categorical_schema.keys())
        for f in feats:
            try:
                _, ys = self.partial_dependence(raw, f, target, n=12)
                if ys:
                    rows.append((f, float(max(ys) - min(ys))))
            except Exception:
                continue
        rows.sort(key=lambda kv: kv[1], reverse=True)
        return rows[:top]

    def effect_summary(self, raw, target, top=6):
        """
        Human-readable what-if effects for the most sensitive variables.

        Numeric variables are swept from the 1st to 99th percentile of the
        training data; categoricals are compared across known training levels.
        """
        sensitive = self.sensitivity(raw, target, top=top)
        effects = []
        curves = {}
        for feature, swing in sensitive:
            xs, ys = self.partial_dependence(raw, feature, target, n=18)
            yvals = [float(y) for y in ys]
            curves[feature] = {"x": list(xs), "y": yvals}
            if len(yvals) < 2:
                continue
            if feature in self.numeric_schema:
                delta = yvals[-1] - yvals[0]
                tol = max(1e-9, 0.03 * max(abs(v) for v in yvals))
                if delta > tol:
                    direction = "increase"
                    text = (f"Increasing {self.pretty(feature)} tends to increase "
                            f"predicted {self.pretty(target)}.")
                elif delta < -tol:
                    direction = "decrease"
                    text = (f"Increasing {self.pretty(feature)} tends to decrease "
                            f"predicted {self.pretty(target)}.")
                else:
                    direction = "flat"
                    text = (f"{self.pretty(feature)} has little directional effect "
                            f"across the observed range.")
                effects.append({
                    "feature": feature, "kind": "numeric", "direction": direction,
                    "swing": float(swing), "delta": float(delta),
                    "from_value": xs[0], "to_value": xs[-1],
                    "from_prediction": yvals[0], "to_prediction": yvals[-1],
                    "summary": text,
                })
            else:
                best = int(np.argmax(yvals))
                worst = int(np.argmin(yvals))
                effects.append({
                    "feature": feature, "kind": "categorical",
                    "direction": "level_choice", "swing": float(swing),
                    "best_level": xs[best], "best_prediction": yvals[best],
                    "worst_level": xs[worst], "worst_prediction": yvals[worst],
                    "summary": (f"For {self.pretty(feature)}, '{xs[best]}' gives the "
                                f"highest predicted {self.pretty(target)} among known "
                                f"training levels; '{xs[worst]}' gives the lowest."),
                })
        return {"sensitivity": sensitive, "effects": effects,
                "partial_dependence": curves}

    # ---- 6. recommendation -------------------------------------------------
    def recommend(self, target, unc, ad, ood, similar, missing):
        """Turn every signal into a verdict + plain-language reasons."""
        u = unc[target]
        rank = self.ranking(target, u["mean"])
        perf = rank["percentile"] / 100.0
        conf_label = _confidence_from_relwidth(u["rel_width"], ad["in_domain"])
        reliability = _CONF_SCORE[conf_label]
        support = (1.0 if ad["in_domain"] else 0.2)
        if similar:
            support = 0.5 * support + 0.5 * (similar[0]["similarity"] / 100.0)

        score = 0.5 * perf + 0.3 * reliability + 0.2 * support
        severe_ood = len(ood)

        # Domain / data guardrails cap optimistic verdicts.
        capped = None
        if not ad["in_domain"] or severe_ood >= 2:
            capped = "Low priority"
        if severe_ood >= 3:
            capped = "Poor candidate"

        if score >= 0.75 and conf_label in ("High", "Moderate"):
            verdict = "Excellent candidate"
        elif score >= 0.60:
            verdict = "Strong candidate"
        elif score >= 0.45:
            verdict = "Worth testing"
        elif score >= 0.30:
            verdict = "Low priority"
        else:
            verdict = "Poor candidate"
        if capped and list(VERDICTS).index(capped) > list(VERDICTS).index(verdict):
            verdict = capped

        reasons = []
        if perf >= 0.75:
            reasons.append(f"High predicted {self.pretty(target)} — better than "
                           f"{rank['percentile']:.0f}% of experiments in the dataset.")
        elif perf >= 0.5:
            reasons.append(f"Above-median predicted {self.pretty(target)} "
                           f"({rank['percentile']:.0f}th percentile of the dataset).")
        else:
            reasons.append(f"Modest predicted {self.pretty(target)} "
                           f"({rank['percentile']:.0f}th percentile of the dataset).")

        if conf_label == "High":
            reasons.append("Low prediction uncertainty — the interval is tight.")
        elif conf_label == "Moderate":
            reasons.append("Moderate prediction uncertainty.")
        else:
            reasons.append("High prediction uncertainty — treat the value with caution.")

        if ad["in_domain"]:
            if ad["percentile"] >= 80:
                reasons.append("Novel synthesis pathway, but still supported by nearby "
                               "experiments — interesting to explore.")
            elif similar and similar[0]["similarity"] >= 75:
                reasons.append("Very similar to real experiments the model has seen.")
            else:
                reasons.append("Inside the model's training experience.")
        else:
            reasons.append("Outside the training domain — the prediction may be "
                           "unreliable; validate before trusting it.")

        for f in ood:
            reasons.append(f"Warning: {f}")
        if missing:
            reasons.append("Missing supporting data for: " + ", ".join(missing[:6]) +
                           (" …" if len(missing) > 6 else "") + ".")

        return {"verdict": verdict, "score": float(score),
                "confidence": conf_label, "reasons": reasons,
                "ranking": rank, "color": VERDICTS[verdict]}

    # ---- 7. the whole workflow in one call --------------------------------
    def screen(self, raw, target, k_similar=6):
        """Run the full primary workflow for one recipe + target."""
        if target not in self.targets:
            raise ValueError(f"'{target}' is not a trained target.")
        X_enc = self.encode(raw)
        unc = self.uncertainty(X_enc)
        ad = self.applicability(X_enc)
        ood = self.ood_flags(raw)
        missing = self.missing_inputs(raw)
        sims = self.similar(X_enc, k=k_similar)
        rec = self.recommend(target, unc, ad, ood, sims, missing)
        contrib = self.contributions(X_enc, target)
        effects = self.effect_summary(raw, target)
        return {
            "target": target, "raw": dict(raw),
            "prediction": unc[target], "all_predictions": unc,
            "applicability": ad, "ood": ood, "missing": missing,
            "similar": sims, "recommendation": rec, "contributions": contrib,
            "sensitivity": effects["sensitivity"],
            "effect_summary": effects["effects"],
            "partial_dependence": effects["partial_dependence"],
            "model_quality": {"cv_r2": self.cv_r2.get(target),
                              "cv_rmse": self.cv_rmse.get(target),
                              "n_train": len(self.X_train),
                              "n_features": len(self.feature_columns)},
        }


def _pretty(name):
    """Human-friendly column label (strip units suffixes, underscores -> spaces)."""
    return str(name).replace("_", " ").strip()


# =============================================================================
# EXPERIMENT PRIORITIZATION
# =============================================================================
def prioritize(screener: Screener, candidates, target, weights=None):
    """
    Score & rank a list of candidate synthesis routes (each a raw dict).

    Composite score blends predicted performance, prediction confidence,
    novelty, similarity to the best real experiments, and feasibility
    (fewer synthesis steps / lower pyrolysis temperature = more feasible).
    Returns a DataFrame sorted best-first.
    """
    w = {"performance": 0.4, "confidence": 0.2, "novelty": 0.15,
         "similarity": 0.15, "feasibility": 0.10}
    if weights:
        w.update(weights)

    rows = []
    for i, raw in enumerate(candidates):
        try:
            res = screener.screen(raw, target, k_similar=3)
        except Exception:
            continue
        pred = res["prediction"]["mean"]
        rank = res["recommendation"]["ranking"]["percentile"] / 100.0
        conf = _CONF_SCORE[res["recommendation"]["confidence"]]
        novelty = res["applicability"]["percentile"] / 100.0
        in_dom = res["applicability"]["in_domain"]
        sim = (res["similar"][0]["similarity"] / 100.0) if res["similar"] else 0.0
        feas = _feasibility(raw, screener)
        score = (w["performance"] * rank + w["confidence"] * conf +
                 w["novelty"] * novelty + w["similarity"] * sim +
                 w["feasibility"] * feas)
        rows.append({
            "candidate": i + 1,
            "predicted": round(pred, 1),
            "interval_low": round(res["prediction"]["lo"], 1),
            "interval_high": round(res["prediction"]["hi"], 1),
            "confidence": res["recommendation"]["confidence"],
            "in_domain": in_dom,
            "novelty_pct": round(novelty * 100, 0),
            "top_similarity_pct": round(sim * 100, 0),
            "feasibility": round(feas, 2),
            "verdict": res["recommendation"]["verdict"],
            "priority_score": round(score, 4),
            **{f"in::{k}": v for k, v in raw.items()},
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _feasibility(raw, screener):
    """0..1 feasibility heuristic: lower temperature & fewer steps score higher."""
    score = 1.0
    # Pyrolysis temperature: cheaper/easier when lower within the observed range.
    for c, (lo, hi, p1, p99) in screener.numeric_ranges.items():
        cl = c.lower()
        if "temp" in cl and hi > lo:
            try:
                v = float(raw.get(c, screener.numeric_schema[c]))
                score -= 0.35 * max(0.0, min(1.0, (v - lo) / (hi - lo)))
            except (TypeError, ValueError):
                pass
            break
    # Synthesis steps: activation / second pyrolysis / additives add effort.
    # A blank, 'not-done' or zero value means that optional step was skipped.
    steps = 0
    for c, v in raw.items():
        sv = str(v).strip().lower()
        cl = c.lower()
        if sv in BLANK_TOKENS or sv in ("0", "0.0", "no"):
            continue
        if any(key in cl for key in ("activation", "additive", "py.2", "py2",
                                     "post", "pretreat", "pre_treat")):
            steps += 1
    score -= 0.08 * steps
    return float(max(0.0, min(1.0, score)))
