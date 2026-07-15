"""
charts.py  —  Chart generators for the Predictor app.
=====================================================

Pure matplotlib (Agg) chart makers that each write a PNG and return its path.
No GUI imports here on purpose, so this module can be tested head-lessly:

    python charts.py        # renders every chart from synthetic data into ./charts

The app (app_imgui.py) imports these and feeds them real data from the trained
model. Every function is defensive: it never raises for empty/degenerate input,
it just draws an explanatory placeholder so one bad chart can't sink the batch.

Charts provided:
    correlation_heatmap · predicted_vs_actual · residual_plot · feature_importance
    shap_summary · shap_dependence · optimization_heatmap · pareto_front
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

# ---- palette (shared with make_graphs.py: validated default, light surface) --
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, AQUA, RED, GOOD = "#2a78d6", "#1baf7a", "#d03b3b", "#0ca30c"

# Diverging blue<->red map for correlation / SHAP-style surfaces.
DIVERGE = LinearSegmentedColormap.from_list("bwr2", [BLUE, "#f4f3ee", RED])
# Sequential map for value surfaces (optimization heatmap).
SEQ = LinearSegmentedColormap.from_list("seq2", ["#eef4fb", BLUE, "#12325f"])


def style_ax(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def short(name, n=22):
    name = str(name)
    return name if len(name) <= n else name[:n - 1] + "…"


def _save(fig, out):
    fig.savefig(out, dpi=130, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    return out


def _placeholder(out, title, message):
    """Draw a single-panel 'why this is empty' card so the tab stays informative."""
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURF)
    style_ax(ax)
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", color=INK,
            fontsize=13, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.40, message, ha="center", va="center", color=MUTED,
            fontsize=9, wrap=True, transform=ax.transAxes)
    return _save(fig, out)


def _grid_shape(n):
    """A pleasant rows x cols layout for n subplots."""
    cols = 1 if n <= 1 else (2 if n <= 4 else 3)
    rows = int(np.ceil(n / cols))
    return rows, cols


# =============================================================================
# 1. CORRELATION HEATMAP
# =============================================================================
def correlation_heatmap(df, out, columns=None, title="Correlation heatmap"):
    """Pearson correlation matrix over the numeric columns (features + targets)."""
    num = df.select_dtypes(include=[np.number])
    if columns:
        num = num[[c for c in columns if c in num.columns]]
    num = num.loc[:, num.nunique(dropna=True) > 1]     # drop constant columns
    if num.shape[1] < 2:
        return _placeholder(out, title, "Need at least two varying numeric columns.")

    corr = num.corr().fillna(0.0)
    n = corr.shape[1]
    size = max(5.0, min(0.55 * n + 2.5, 16))
    fig, ax = plt.subplots(figsize=(size, size), facecolor=SURF)
    style_ax(ax)
    im = ax.imshow(corr.values, cmap=DIVERGE, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([short(c, 16) for c in corr.columns], rotation=90, fontsize=7)
    ax.set_yticklabels([short(c, 16) for c in corr.index], fontsize=7)
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold", pad=10)
    # Annotate values when the matrix is small enough to read.
    if n <= 14:
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color=INK if abs(v) < 0.6 else SURF)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_edgecolor(GRID)
    return _save(fig, out)


# =============================================================================
# 2. PREDICTED vs ACTUAL   /   3. RESIDUAL PLOT   (one panel per target)
# =============================================================================
def _as2d(a):
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def predicted_vs_actual(y_true, y_pred, target_names, out, r2_by_target=None):
    y_true, y_pred = _as2d(y_true), _as2d(y_pred)
    k = y_true.shape[1]
    names = list(target_names) if target_names is not None else [f"target {i}" for i in range(k)]
    if k == 0 or len(y_true) == 0:
        return _placeholder(out, "Predicted vs actual", "No predictions available.")

    rows, cols = _grid_shape(k)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.0 * rows), facecolor=SURF)
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle("Predicted vs actual  (out-of-fold)", color=INK, fontsize=13,
                 fontweight="bold", x=0.02, ha="left")
    for i in range(k):
        ax = axes[i]; style_ax(ax)
        t, p = y_true[:, i], y_pred[:, i]
        ax.scatter(t, p, s=16, color=BLUE, alpha=0.5, edgecolors="none")
        lo = float(np.nanmin([t.min(), p.min()])); hi = float(np.nanmax([t.max(), p.max()]))
        if lo == hi:
            hi = lo + 1.0
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls="--")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        r2 = None if r2_by_target is None else r2_by_target.get(names[i])
        sub = f"  (R²={r2:.2f})" if isinstance(r2, (int, float)) else ""
        ax.set_title(short(names[i], 26) + sub, fontsize=10, loc="left", fontweight="bold")
        ax.set_xlabel("actual"); ax.set_ylabel("predicted")
        ax.xaxis.set_major_locator(MaxNLocator(5))
    for j in range(k, len(axes)):
        axes[j].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out)


def residual_plot(y_true, y_pred, target_names, out):
    y_true, y_pred = _as2d(y_true), _as2d(y_pred)
    k = y_true.shape[1]
    names = list(target_names) if target_names is not None else [f"target {i}" for i in range(k)]
    if k == 0 or len(y_true) == 0:
        return _placeholder(out, "Residual plot", "No predictions available.")

    rows, cols = _grid_shape(k)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.0 * rows), facecolor=SURF)
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle("Residuals vs predicted  (out-of-fold)", color=INK, fontsize=13,
                 fontweight="bold", x=0.02, ha="left")
    for i in range(k):
        ax = axes[i]; style_ax(ax)
        p = y_pred[:, i]; resid = y_true[:, i] - p
        ax.scatter(p, resid, s=16, color=BLUE, alpha=0.5, edgecolors="none")
        ax.axhline(0, color=RED, lw=1.4, ls="--")
        ax.set_title(short(names[i], 26), fontsize=10, loc="left", fontweight="bold")
        ax.set_xlabel("predicted"); ax.set_ylabel("residual (actual − pred)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
    for j in range(k, len(axes)):
        axes[j].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out)


# =============================================================================
# 4. FEATURE IMPORTANCE
# =============================================================================
def feature_importance(importances, out, top=15, measured=None,
                       title="Feature importance"):
    """`importances` is a pandas Series {feature: importance} or a dict."""
    s = pd.Series(dict(importances)) if not isinstance(importances, pd.Series) else importances
    s = s[s > 0].sort_values(ascending=False)
    if s.empty:
        return _placeholder(out, title, "The model reported no feature importances.")
    measured = measured or set()
    top_s = s.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.42 * len(top_s) + 1.2)), facecolor=SURF)
    style_ax(ax)
    colors = [AQUA if f in measured else BLUE for f in top_s.index]
    ax.barh(range(len(top_s)), top_s.values, color=colors, height=0.72)
    ax.set_yticks(range(len(top_s)))
    ax.set_yticklabels([short(f, 30) for f in top_s.index], fontsize=8)
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    ax.set_xlabel("importance")
    if measured:
        ax.text(0.98, 0.04, "aqua = measured after synthesis\nblue = controllable",
                transform=ax.transAxes, fontsize=7, color=MUTED, ha="right", va="bottom")
    return _save(fig, out)


# =============================================================================
# 5. SHAP SUMMARY   /   6. SHAP DEPENDENCE   (lazy import; optional dependency)
# =============================================================================
def _shap_values_for(estimator, X_df):
    """Return (shap_values 2d, shap.Explainer output) for a single-output tree model."""
    import shap
    # Cap sample size so TreeExplainer stays fast on big frames.
    Xs = X_df if len(X_df) <= 400 else X_df.sample(400, random_state=42)
    explainer = shap.TreeExplainer(estimator)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):        # some versions wrap single-output in a list
        sv = sv[0]
    return np.asarray(sv), Xs


def shap_summary(estimator, X_df, out, target_name="", max_display=15):
    try:
        import shap  # noqa: F401
    except ImportError:
        return _placeholder(out, "SHAP summary",
                            "Install the 'shap' package to enable SHAP charts.")
    try:
        sv, Xs = _shap_values_for(estimator, X_df)
        import shap
        fig = plt.figure(facecolor=SURF)
        shap.summary_plot(sv, Xs, max_display=max_display, show=False,
                          plot_size=(9, max(4, 0.4 * min(max_display, X_df.shape[1]) + 2)))
        fig = plt.gcf()
        fig.patch.set_facecolor(SURF)
        ttl = "SHAP summary" + (f"  ·  {short(target_name, 28)}" if target_name else "")
        fig.suptitle(ttl, color=INK, fontsize=12, fontweight="bold", x=0.02, ha="left")
        return _save(fig, out)
    except Exception as e:  # noqa: BLE001
        return _placeholder(out, "SHAP summary", f"Could not compute SHAP values:\n{e}")


def shap_dependence(estimator, X_df, out, target_name="", top_k=4):
    try:
        import shap  # noqa: F401
    except ImportError:
        return _placeholder(out, "SHAP dependence",
                            "Install the 'shap' package to enable SHAP charts.")
    try:
        sv, Xs = _shap_values_for(estimator, X_df)
        # Rank features by mean |SHAP| and show dependence for the strongest few.
        order = np.argsort(-np.abs(sv).mean(axis=0))
        feats = [X_df.columns[i] for i in order[:max(1, top_k)]]
        rows, cols = _grid_shape(len(feats))
        fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.8 * rows), facecolor=SURF)
        axes = np.atleast_1d(axes).ravel()
        ttl = "SHAP dependence" + (f"  ·  {short(target_name, 28)}" if target_name else "")
        fig.suptitle(ttl, color=INK, fontsize=13, fontweight="bold", x=0.02, ha="left")
        col_idx = {c: i for i, c in enumerate(Xs.columns)}
        for n, feat in enumerate(feats):
            ax = axes[n]; style_ax(ax)
            xi = col_idx[feat]
            xv = Xs.iloc[:, xi].values
            ax.scatter(xv, sv[:, xi], s=16, color=BLUE, alpha=0.55, edgecolors="none")
            ax.axhline(0, color=MUTED, lw=1, ls="--")
            ax.set_title(short(feat, 26), fontsize=10, loc="left", fontweight="bold")
            ax.set_xlabel(short(feat, 26)); ax.set_ylabel("SHAP value")
            ax.xaxis.set_major_locator(MaxNLocator(5))
        for j in range(len(feats), len(axes)):
            axes[j].axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return _save(fig, out)
    except Exception as e:  # noqa: BLE001
        return _placeholder(out, "SHAP dependence", f"Could not compute SHAP values:\n{e}")


# =============================================================================
# 7. OPTIMIZATION HEATMAP  (2-D partial dependence of the model over two knobs)
# =============================================================================
def optimization_heatmap(predict_fn, numeric_schema, categorical_schema,
                         feat_x, feat_y, out, target_index=0, target_name="",
                         x_range=None, y_range=None, resolution=40,
                         maximize=True):
    """
    Sweep two numeric knobs on a grid, hold every other feature at its default
    (median / first category), predict the target, and draw the response surface.

    `predict_fn(raw_df) -> 2-D array` handles encoding + model.predict.
    `numeric_schema` {col: median}, `categorical_schema` {col: [choices]}.
    """
    if feat_x is None or feat_y is None or feat_x == feat_y:
        return _placeholder(out, "Optimization heatmap",
                            "Need two distinct numeric features to sweep.")
    try:
        xr = x_range or (0.0, 1.0)
        yr = y_range or (0.0, 1.0)
        xs = np.linspace(xr[0], xr[1], resolution)
        ys = np.linspace(yr[0], yr[1], resolution)
        gx, gy = np.meshgrid(xs, ys)

        # One base row from the schema defaults, replicated across the grid.
        base = {c: v for c, v in numeric_schema.items()}
        for c, choices in categorical_schema.items():
            base[c] = choices[0] if choices else "Missing"
        grid = pd.DataFrame([base] * gx.size)
        grid[feat_x] = gx.ravel()
        grid[feat_y] = gy.ravel()

        preds = np.asarray(predict_fn(grid))
        if preds.ndim == 1:
            z = preds
        else:
            z = preds[:, min(target_index, preds.shape[1] - 1)]
        z = z.reshape(gx.shape)

        fig, ax = plt.subplots(figsize=(7.6, 6.2), facecolor=SURF)
        style_ax(ax)
        im = ax.pcolormesh(gx, gy, z, cmap=SEQ, shading="auto")
        cs = ax.contour(gx, gy, z, colors=INK2, linewidths=0.5, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
        # Mark the grid optimum.
        opt = np.unravel_index(np.argmax(z) if maximize else np.argmin(z), z.shape)
        ax.scatter([gx[opt]], [gy[opt]], s=90, marker="*", color=RED,
                   edgecolors=SURF, linewidths=0.8, zorder=5,
                   label=f"{'max' if maximize else 'min'} ≈ {z[opt]:.0f}")
        ax.legend(loc="upper right", fontsize=7, facecolor=SURF, edgecolor=GRID)
        ttl = "Optimization heatmap" + (f"  ·  {short(target_name, 26)}" if target_name else "")
        ax.set_title(ttl, fontsize=12, loc="left", fontweight="bold")
        ax.set_xlabel(short(feat_x, 26)); ax.set_ylabel(short(feat_y, 26))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=MUTED, labelsize=7)
        cb.outline.set_edgecolor(GRID)
        cb.set_label("predicted " + short(target_name, 22), color=INK2, fontsize=8)
        return _save(fig, out)
    except Exception as e:  # noqa: BLE001
        return _placeholder(out, "Optimization heatmap", f"Could not build surface:\n{e}")


# =============================================================================
# 8. PARETO FRONT  (trade-off between two targets; non-dominated set highlighted)
# =============================================================================
def _pareto_mask(pts, maximize):
    """Boolean mask of non-dominated rows. `pts` is (n,2); `maximize` is (bool,bool)."""
    s = np.array([1.0 if m else -1.0 for m in maximize])
    v = pts * s                              # flip so bigger is always better
    n = len(v)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # i is dominated if some j is >= in both and > in at least one.
        dominated = np.any(np.all(v >= v[i], axis=1) & np.any(v > v[i], axis=1))
        keep[i] = not dominated
    return keep


def pareto_front(points_df, obj_a, obj_b, out, maximize_a=True, maximize_b=True,
                 title="Pareto front"):
    if obj_a not in points_df.columns or obj_b not in points_df.columns or obj_a == obj_b:
        return _placeholder(out, title, "Need two distinct target columns.")
    d = points_df[[obj_a, obj_b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 2:
        return _placeholder(out, title, "Not enough complete rows for a trade-off.")

    pts = d.values
    mask = _pareto_mask(pts, (maximize_a, maximize_b))
    front = d[mask].sort_values(obj_a)

    fig, ax = plt.subplots(figsize=(7.6, 6.0), facecolor=SURF)
    style_ax(ax)
    ax.scatter(pts[~mask, 0], pts[~mask, 1], s=22, color=MUTED, alpha=0.45,
               edgecolors="none", label="dominated")
    ax.scatter(d[mask][obj_a], d[mask][obj_b], s=46, color=RED,
               edgecolors=SURF, linewidths=0.6, zorder=4, label="Pareto-optimal")
    ax.plot(front[obj_a], front[obj_b], color=RED, lw=1.4, alpha=0.7, zorder=3)
    arrow = f"({'↑' if maximize_a else '↓'} {short(obj_a,18)},  {'↑' if maximize_b else '↓'} {short(obj_b,18)})"
    ax.set_title(f"{title}   {arrow}", fontsize=12, loc="left", fontweight="bold")
    ax.set_xlabel(short(obj_a, 30)); ax.set_ylabel(short(obj_b, 30))
    ax.legend(loc="best", fontsize=8, facecolor=SURF, edgecolor=GRID)
    return _save(fig, out)


# =============================================================================
# LATENT-VARIABLE CHARTS  (used by the Latent Variables tab)
# =============================================================================
# A small qualitative palette for coloring categories (e.g. by Material).
_CAT_COLORS = [BLUE, AQUA, RED, "#f4a11a", "#8b5cf6", "#0ea5a5",
               "#e05299", "#6b8f2e", "#3b6fb0", "#b5642e"]


def explained_variance(evr, out, title="PCA explained variance"):
    """Bar of per-component variance ratio + cumulative line (twin axis)."""
    evr = np.asarray(evr, dtype=float)
    if evr.size == 0:
        return _placeholder(out, title, "No PCA components to show.")
    cum = np.cumsum(evr)
    x = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURF)
    style_ax(ax)
    ax.bar(x, evr * 100, color=BLUE, width=0.66, label="per component")
    ax.set_xlabel("principal component"); ax.set_ylabel("variance explained (%)")
    ax.set_xticks(x)
    ax2 = ax.twinx()
    ax2.plot(x, cum * 100, color=RED, marker="o", ms=4, lw=1.6, label="cumulative")
    ax2.set_ylabel("cumulative (%)", color=INK2)
    ax2.set_ylim(0, 105)
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    return _save(fig, out)


def pca_score_scatter(scores_df, color_series, out, pc_x="PC1", pc_y="PC2",
                      color_name="Material", title="PCA scores"):
    """Scatter of two PCs, colored by a categorical series (e.g. Material)."""
    if pc_x not in scores_df.columns or pc_y not in scores_df.columns:
        return _placeholder(out, title, "Need at least two principal components.")
    fig, ax = plt.subplots(figsize=(7.6, 6.2), facecolor=SURF)
    style_ax(ax)
    if color_series is not None:
        cats = pd.Series(color_series).astype(str).to_numpy()
        for i, cat in enumerate(pd.unique(cats)):
            m = cats == cat
            ax.scatter(scores_df.loc[m, pc_x], scores_df.loc[m, pc_y], s=26,
                       color=_CAT_COLORS[i % len(_CAT_COLORS)], alpha=0.75,
                       edgecolors="none", label=short(cat, 18))
        ax.legend(loc="best", fontsize=7, facecolor=SURF, edgecolor=GRID,
                  title=short(color_name, 18), title_fontsize=7)
    else:
        ax.scatter(scores_df[pc_x], scores_df[pc_y], s=26, color=BLUE, alpha=0.7,
                   edgecolors="none")
    ax.set_title(f"{title}  ·  colored by {short(color_name, 18)}",
                 fontsize=12, loc="left", fontweight="bold")
    ax.set_xlabel(pc_x); ax.set_ylabel(pc_y)
    return _save(fig, out)


def pca_loading_bar(loadings, out, pc="PC1", top=15, title="PCA loadings"):
    """Horizontal bar of the largest-magnitude loadings for one component."""
    if pc not in loadings.columns:
        return _placeholder(out, title, f"Component {pc} not found.")
    s = loadings[pc]
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.42 * len(s) + 1.2)), facecolor=SURF)
    style_ax(ax)
    colors = [BLUE if v >= 0 else RED for v in s.values]
    ax.barh(range(len(s)), s.values, color=colors, height=0.72)
    ax.axvline(0, color=GRID, lw=1)
    ax.set_yticks(range(len(s))); ax.set_yticklabels([short(f, 28) for f in s.index], fontsize=8)
    ax.set_title(f"{title}  ·  {pc}  (blue +   red −)", fontsize=12, loc="left", fontweight="bold")
    ax.set_xlabel("loading")
    return _save(fig, out)


def pipeline_comparison_bar(results, out, metric="r2", title=None):
    """
    Grouped bar of pipeline A/B/C performance per model, with std error bars.
    `results` is the dict from latent.compare_pipelines.
    `metric` in {"r2","rmse","mae"}.
    """
    variants = ["A", "B", "C"]
    labels = {"A": "A: original", "B": "B: latent", "C": "C: both"}
    models = results.get("_meta", {}).get("models") or ["ExtraTrees"]
    have = False
    fig, ax = plt.subplots(figsize=(8.4, 5.0), facecolor=SURF)
    style_ax(ax)
    width = 0.8 / max(1, len(models))
    x = np.arange(len(variants))
    for j, m in enumerate(models):
        means, errs = [], []
        for v in variants:
            sc = results.get(v, {}).get(m, {})
            means.append(sc.get(f"{metric}_mean", np.nan))
            errs.append(sc.get(f"{metric}_std", 0.0))
            have = have or (f"{metric}_mean" in sc)
        ax.bar(x + j * width, means, width=width * 0.95, yerr=errs, capsize=3,
               color=_CAT_COLORS[j % len(_CAT_COLORS)], label=m,
               error_kw=dict(ecolor=MUTED, lw=1))
    if not have:
        return _placeholder(out, "Pipeline comparison", "No results to plot.")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([labels[v] for v in variants], fontsize=9)
    ax.set_ylabel(f"CV {metric.upper()}  (mean ± std)")
    ax.set_title(title or f"Pipeline comparison · {metric.upper()}",
                 fontsize=12, loc="left", fontweight="bold")
    ax.legend(loc="best", fontsize=8, facecolor=SURF, edgecolor=GRID)
    return _save(fig, out)


# =============================================================================
# DATASET-INTELLIGENCE CHARTS
# =============================================================================
def ranked_bar(series, out, title="Ranking", xlabel="value", top=15,
               diverging=False):
    """Horizontal bar of a {label: value} Series, largest-magnitude first."""
    s = pd.Series(dict(series)) if not isinstance(series, pd.Series) else series
    s = s.dropna()
    if s.empty:
        return _placeholder(out, title, "Nothing to rank.")
    s = s.reindex(s.abs().sort_values(ascending=False).index).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.42 * len(s) + 1.2)), facecolor=SURF)
    style_ax(ax)
    colors = ([BLUE if v >= 0 else RED for v in s.values] if diverging else BLUE)
    ax.barh(range(len(s)), s.values, color=colors, height=0.72)
    if diverging:
        ax.axvline(0, color=GRID, lw=1)
    ax.set_yticks(range(len(s))); ax.set_yticklabels([short(f, 30) for f in s.index], fontsize=8)
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    return _save(fig, out)


def target_distribution(values, out, target_name="target", stats=None):
    """Histogram + box plot + QQ plot in one figure, annotated with shape stats."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return _placeholder(out, "Target analysis", "Too few target values.")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor=SURF)
    for ax in axes:
        style_ax(ax)
    # Histogram + median line.
    axes[0].hist(y, bins=24, color=BLUE, edgecolor=SURF, linewidth=0.6)
    axes[0].axvline(np.median(y), color=RED, lw=1.4, ls="--")
    axes[0].set_title("Histogram", fontsize=10, loc="left", fontweight="bold")
    axes[0].set_xlabel(short(target_name, 24)); axes[0].set_ylabel("count")
    # Box plot.
    bp = axes[1].boxplot(y, vert=True, widths=0.5, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set(facecolor=AQUA, alpha=0.5, edgecolor=INK2)
    for med in bp["medians"]:
        med.set(color=RED, linewidth=1.5)
    axes[1].set_title("Box plot", fontsize=10, loc="left", fontweight="bold")
    axes[1].set_xticks([])
    # QQ plot vs normal.
    try:
        from scipy import stats as _st
        _st.probplot(y, dist="norm", plot=axes[2])
        axes[2].get_lines()[0].set(marker="o", ms=4, color=BLUE, alpha=0.6, ls="")
        axes[2].get_lines()[1].set(color=RED, lw=1.4)
    except Exception:  # noqa: BLE001
        axes[2].scatter(np.sort(y), np.linspace(0, 1, len(y)), s=10, color=BLUE)
    axes[2].set_title("QQ plot (normal)", fontsize=10, loc="left", fontweight="bold")

    sub = ""
    if stats:
        sub = (f"skew={stats.get('skewness', float('nan')):.2f}  ·  "
               f"kurtosis={stats.get('kurtosis', float('nan')):.2f}  ·  "
               f"normal={'yes' if stats.get('is_normal') else 'no'}  ·  "
               f"outliers={stats.get('n_outliers', 0)}")
    fig.suptitle(f"Target distribution · {short(target_name, 28)}   {sub}",
                 color=INK, fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out)


def causal_diagram(structure_present, out, message="",
                   title="Causal pipeline"):
    """Boxes+arrows: Material/Pretreat/Pyrolysis -> Carbon Structure -> Capacity.

    The 'Carbon Structure' node is drawn dashed when structural descriptors are
    absent from the dataset.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURF)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, text, dashed=False, color=BLUE):
        style = "round,pad=0.1"
        p = FancyBboxPatch((x - 1.05, y - 0.42), 2.1, 0.84, boxstyle=style,
                           linewidth=1.6, edgecolor=color, facecolor=SURF,
                           linestyle="--" if dashed else "-")
        ax.add_patch(p)
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
                color=MUTED if dashed else INK, fontweight="bold")
        return (x, y)

    def arrow(a, b, dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                     color=MUTED, lw=1.3,
                                     linestyle="--" if dashed else "-",
                                     shrinkA=42, shrinkB=42))

    mat = box(1.5, 5.0, "Material")
    pre = box(1.5, 3.0, "Pretreatment")
    pyr = box(1.5, 1.0, "Pyrolysis")
    struct = box(5.0, 3.0, "Carbon Structure", dashed=not structure_present,
                 color=(AQUA if structure_present else MUTED))
    cap = box(8.5, 3.0, "Capacity", color=GOOD)
    for src in (mat, pre, pyr):
        arrow(src, struct, dashed=not structure_present)
    arrow(struct, cap, dashed=not structure_present)

    ax.set_title(title, fontsize=13, loc="left", fontweight="bold")
    if message:
        ax.text(5.0, 2.35, message, ha="center", va="center", fontsize=8,
                color=RED, style="italic")
    return _save(fig, out)


def radar_chart(labels, values, out, title="Latent contribution"):
    """Radar/spider chart of {label: value} (e.g. latent contributions)."""
    labels = list(labels); vals = list(values)
    if len(labels) < 3:
        return _placeholder(out, title, "Need at least three axes for a radar chart.")
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals = vals + vals[:1]; ang = ang + ang[:1]
    fig, ax = plt.subplots(figsize=(6.6, 6.6), subplot_kw=dict(polar=True), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.plot(ang, vals, color=BLUE, lw=1.8)
    ax.fill(ang, vals, color=BLUE, alpha=0.25)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([short(l, 18) for l in labels], fontsize=8, color=INK2)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=18)
    return _save(fig, out)


def biplot(scores_df, loadings, out, color_series=None, pc_x="PC1", pc_y="PC2",
           color_name="Material", top_loadings=8, title="PCA biplot"):
    """Score scatter overlaid with the strongest loading vectors."""
    if pc_x not in scores_df.columns or pc_y not in scores_df.columns:
        return _placeholder(out, title, "Need at least two principal components.")
    fig, ax = plt.subplots(figsize=(8, 6.8), facecolor=SURF)
    style_ax(ax)
    if color_series is not None:
        cats = pd.Series(color_series).astype(str).to_numpy()
        for i, cat in enumerate(pd.unique(cats)):
            m = cats == cat
            ax.scatter(scores_df.loc[m, pc_x], scores_df.loc[m, pc_y], s=20,
                       color=_CAT_COLORS[i % len(_CAT_COLORS)], alpha=0.6,
                       edgecolors="none", label=short(cat, 16))
        ax.legend(loc="best", fontsize=7, facecolor=SURF, edgecolor=GRID,
                  title=short(color_name, 16), title_fontsize=7)
    else:
        ax.scatter(scores_df[pc_x], scores_df[pc_y], s=20, color=BLUE, alpha=0.6,
                   edgecolors="none")
    # Scale loading arrows to the score cloud.
    if pc_x in loadings.columns and pc_y in loadings.columns:
        span = float(np.nanmax(np.abs(scores_df[[pc_x, pc_y]].to_numpy()))) or 1.0
        lmag = loadings[[pc_x, pc_y]].abs().sum(axis=1).sort_values(ascending=False)
        for feat in lmag.head(top_loadings).index:
            vx, vy = loadings.loc[feat, pc_x], loadings.loc[feat, pc_y]
            ax.annotate("", xy=(vx * span, vy * span), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
            ax.text(vx * span * 1.08, vy * span * 1.08, short(feat, 16),
                    color=RED, fontsize=7, ha="center", va="center")
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    ax.set_xlabel(pc_x); ax.set_ylabel(pc_y)
    return _save(fig, out)


# =============================================================================
# HEAD-LESS SELF TEST  ->  python charts.py
# =============================================================================
def _selftest(out_dir="charts"):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "temp": rng.normal(800, 120, n),
        "time": rng.uniform(1, 10, n),
        "ratio": rng.uniform(0, 1, n),
        "cat": rng.choice(["A", "B", "C"], n),
    })
    signal = 0.03 * X["temp"] + 12 * X["time"] - 40 * X["ratio"]
    y = pd.DataFrame({
        "capacity": signal + rng.normal(0, 8, n),
        "ice": 60 + 0.5 * X["time"] + rng.normal(0, 3, n),
    })
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import cross_val_predict, KFold
    Xe = pd.get_dummies(X, drop_first=True).astype(float)
    est = ExtraTreesRegressor(n_estimators=120, random_state=0, n_jobs=-1)
    oof = cross_val_predict(est, Xe, y, cv=KFold(5, shuffle=True, random_state=0))
    est.fit(Xe, y["capacity"])

    paths = []
    paths.append(correlation_heatmap(pd.concat([X, y], axis=1), f"{out_dir}/corr.png"))
    paths.append(predicted_vs_actual(y.values, oof, list(y.columns), f"{out_dir}/pva.png"))
    paths.append(residual_plot(y.values, oof, list(y.columns), f"{out_dir}/resid.png"))
    imp = pd.Series(est.feature_importances_, index=Xe.columns)
    paths.append(feature_importance(imp, f"{out_dir}/imp.png"))
    paths.append(shap_summary(est, Xe, f"{out_dir}/shap_sum.png", "capacity"))
    paths.append(shap_dependence(est, Xe, f"{out_dir}/shap_dep.png", "capacity"))

    def predict_fn(raw):
        enc = pd.get_dummies(raw, drop_first=True).reindex(columns=Xe.columns, fill_value=0)
        return est.predict(enc.astype(float))
    paths.append(optimization_heatmap(
        predict_fn, {"temp": 800.0, "time": 5.0, "ratio": 0.5},
        {"cat": ["A", "B", "C"]}, "temp", "time", f"{out_dir}/opt.png",
        target_name="capacity", x_range=(560, 1040), y_range=(1, 10)))
    paths.append(pareto_front(y, "capacity", "ice", f"{out_dir}/pareto.png"))

    print("Wrote:")
    for p in paths:
        print("  ", p, "OK" if os.path.exists(p) else "MISSING")


if __name__ == "__main__":
    _selftest()
