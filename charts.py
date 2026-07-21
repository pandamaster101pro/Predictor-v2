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

import json
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

# ---- publication theme -------------------------------------------------------
EXPORT_DPI = 300
CAPTION_BOTTOM = 0.18
SURF, INK, INK2, MUTED, GRID = "#ffffff", "#111827", "#374151", "#6b7280", "#e5e7eb"
BLUE, AQUA, RED, GOOD, ORANGE = "#2563eb", "#059669", "#dc2626", "#16a34a", "#d97706"
PURPLE, GOLD = "#7c3aed", "#ca8a04"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": EXPORT_DPI,
    "savefig.dpi": EXPORT_DPI,
    "axes.unicode_minus": False,
})

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
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)


def short(name, n=22):
    name = str(name)
    return name if len(name) <= n else name[:n - 1] + "…"


def label(name):
    """Readable scientific-ish label from a spreadsheet/model column name."""
    s = str(name).replace("_", " ").replace("  ", " ").strip()
    for a, b in {
        "R2": "R²", "r2": "R²", "mAh per g": "mAh g⁻¹",
        "C min": "°C min⁻¹",
    }.items():
        s = s.replace(a, b)
    return s


def _companion_paths(out):
    root, _ = os.path.splitext(out)
    return {
        "png": root + ".png",
        "svg": root + ".svg",
        "pdf": root + ".pdf",
        "txt": root + ".txt",
        "json": root + ".json",
    }


def companion_paths(out):
    """Public helper used by the GUI to locate SVG/PDF/interpretation sidecars."""
    return _companion_paths(out)


def _write_meta(out, title, interpretation, stats=None):
    paths = _companion_paths(out)
    with open(paths["txt"], "w", encoding="utf-8") as f:
        f.write(interpretation or "")
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump({
            "title": title,
            "interpretation": interpretation or "",
            "stats": stats or {},
            "exports": {k: paths[k] for k in ("png", "svg", "pdf")},
        }, f, ensure_ascii=False, indent=2)


def _caption_text(text, fig):
    width = max(float(fig.get_size_inches()[0]), 6.0)
    line_width = max(72, min(130, int(width * 12)))
    wrapped = textwrap.wrap(str(text), width=line_width)
    if len(wrapped) > 3:
        wrapped = wrapped[:3]
        wrapped[-1] = textwrap.shorten(wrapped[-1], width=max(20, line_width - 4),
                                       placeholder="...")
        if not wrapped[-1].endswith("..."):
            wrapped[-1] = wrapped[-1].rstrip(". ") + "..."
    return "\n".join(wrapped)


def _figure_note(fig, text):
    if text:
        fig.text(0.015, 0.035, _caption_text(text, fig), ha="left", va="bottom",
                 fontsize=6.5, color=INK2, linespacing=1.25,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffffcc",
                           edgecolor="none"))


def _save(fig, out, title="", interpretation="", stats=None):
    paths = _companion_paths(out)
    fig.savefig(paths["png"], dpi=EXPORT_DPI, facecolor=SURF, bbox_inches="tight")
    fig.savefig(paths["svg"], facecolor=SURF, bbox_inches="tight")
    fig.savefig(paths["pdf"], facecolor=SURF, bbox_inches="tight")
    _write_meta(out, title, interpretation, stats)
    plt.close(fig)
    return paths["png"]


def _placeholder(out, title, message):
    """Draw a single-panel 'why this is empty' card so the tab stays informative."""
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURF)
    style_ax(ax)
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", color=INK,
            fontsize=13, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.40, message, ha="center", va="center", color=MUTED,
            fontsize=9, wrap=True, transform=ax.transAxes)
    return _save(fig, out, title=title, interpretation=message)


def _grid_shape(n):
    """A pleasant rows x cols layout for n subplots."""
    cols = 1 if n <= 1 else (2 if n <= 4 else 3)
    rows = int(np.ceil(n / cols))
    return rows, cols


# =============================================================================
# 1. CORRELATION HEATMAP
# =============================================================================
def correlation_heatmap(df, out, columns=None, title="Correlation heatmap",
                        method="pearson", alpha=0.05):
    """Pearson/Spearman correlation matrix over numeric columns."""
    num = df.select_dtypes(include=[np.number])
    if columns:
        num = num[[c for c in columns if c in num.columns]]
    num = num.loc[:, num.nunique(dropna=True) > 1]     # drop constant columns
    if num.shape[1] < 2:
        return _placeholder(out, title, "Need at least two varying numeric columns.")

    method = method.lower()
    corr = num.corr(method=method if method in ("pearson", "spearman") else "pearson").fillna(0.0)
    n = corr.shape[1]
    size = max(5.0, min(0.55 * n + 2.5, 16))
    fig, ax = plt.subplots(figsize=(size, size), facecolor=SURF)
    style_ax(ax)
    im = ax.imshow(corr.values, cmap=DIVERGE, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([short(label(c), 16) for c in corr.columns], rotation=90, fontsize=7)
    ax.set_yticklabels([short(label(c), 16) for c in corr.index], fontsize=7)
    ax.set_title(f"{title} ({method.title()})", fontsize=12, loc="left", fontweight="bold", pad=10)

    pvals = None
    try:
        from scipy import stats as _st
        pvals = np.ones_like(corr.values, dtype=float)
        for i, a in enumerate(corr.columns):
            for j, b in enumerate(corr.columns):
                if i == j:
                    pvals[i, j] = 0.0
                    continue
                xy = num[[a, b]].dropna()
                # A constant column has no defined correlation p-value; leave it
                # at 1.0 (not significant) instead of warning + returning NaN.
                if len(xy) < 4 or xy[a].nunique() < 2 or xy[b].nunique() < 2:
                    continue
                if method == "spearman":
                    _, p = _st.spearmanr(xy[a], xy[b])
                else:
                    _, p = _st.pearsonr(xy[a], xy[b])
                pvals[i, j] = p
    except Exception:  # noqa: BLE001
        pvals = None

    # Annotate values when the matrix is small enough to read.
    if n <= 14:
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                sig = "*" if pvals is not None and pvals[i, j] < alpha and i != j else ""
                ax.text(j, i, f"{v:.2f}{sig}", ha="center", va="center", fontsize=6,
                        color=INK if abs(v) < 0.6 else SURF)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_edgecolor(GRID)
    vals = corr.where(~np.eye(n, dtype=bool)).stack()
    if len(vals):
        strongest = vals.abs().idxmax()
        v = corr.loc[strongest]
        interp = (f"This {method} heatmap shows pairwise linear/monotonic association. "
                  f"The strongest relationship is {label(strongest[0])} vs "
                  f"{label(strongest[1])} (r={v:.2f}). Asterisks mark relationships "
                  f"with p<{alpha:g} when enough complete observations are available.")
    else:
        interp = "The heatmap shows no off-diagonal relationships after filtering."
    _figure_note(fig, interp)
    fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 1))
    return _save(fig, out, title=title, interpretation=interp,
                 stats={"method": method, "n_features": int(n)})


# =============================================================================
# 2. PREDICTED vs ACTUAL   /   3. RESIDUAL PLOT   (one pane l per target)
# =============================================================================
def _as2d(a):
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def predicted_vs_actual(y_true, y_pred, target_names, out, r2_by_target=None,
                        validation_method="cross-validation"):
    y_true, y_pred = _as2d(y_true), _as2d(y_pred)
    k = y_true.shape[1]
    names = list(target_names) if target_names is not None else [f"target {i}" for i in range(k)]
    if k == 0 or len(y_true) == 0:
        return _placeholder(out, "Predicted vs actual", "No predictions available.")

    rows, cols = _grid_shape(k)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.0 * rows), facecolor=SURF)
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(f"Predicted vs Actual - {validation_method} (pooled OOF)", color=INK, fontsize=13,
                 fontweight="bold", x=0.02, ha="left")
    interpretations = []
    for i in range(k):
        ax = axes[i]; style_ax(ax)
        t, p = y_true[:, i], y_pred[:, i]
        mask = np.isfinite(t) & np.isfinite(p)
        t, p = t[mask], p[mask]
        ax.scatter(t, p, s=18, color=BLUE, alpha=0.58, edgecolors="white",
                   linewidths=0.2, label="cross-validated prediction")
        lo = float(np.nanmin([t.min(), p.min()])); hi = float(np.nanmax([t.max(), p.max()]))
        if lo == hi:
            hi = lo + 1.0
        resid = t - p
        band = 1.96 * float(np.nanstd(resid))
        ax.fill_between([lo, hi], [lo - band, hi - band], [lo + band, hi + band],
                        color=BLUE, alpha=0.08, label="approx. 95% error band")
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.2, ls="--", label="ideal 1:1")
        if len(t) >= 3 and np.nanstd(t) > 0:
            coef = np.polyfit(t, p, 1)
            ax.plot([lo, hi], np.polyval(coef, [lo, hi]), color=RED, lw=1.3,
                    label="trend")
        rmse = float(np.sqrt(np.nanmean((t - p) ** 2)))
        mae = float(np.nanmean(np.abs(t - p)))
        denom = np.abs(t)
        mape = float(np.nanmean(np.abs((t - p) / denom)) * 100) if np.all(denom > 1e-12) else np.nan
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        r2 = None if r2_by_target is None else r2_by_target.get(names[i])
        sub = f"  (CV R²={r2:.2f})" if isinstance(r2, (int, float)) else ""
        ax.set_title(short(label(names[i]), 26) + sub, fontsize=10, loc="left", fontweight="bold")
        ax.set_xlabel("Measured value"); ax.set_ylabel("Model prediction")
        box = f"RMSE={rmse:.2g}\nMAE={mae:.2g}"
        if np.isfinite(mape):
            box += f"\nMAPE={mape:.1f}%"
        ax.text(0.04, 0.96, box, transform=ax.transAxes, ha="left", va="top",
                fontsize=7, color=INK2, bbox=dict(boxstyle="round,pad=0.25",
                                                  facecolor=SURF, edgecolor=GRID))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        if i == 0:
            ax.legend(loc="lower right", fontsize=6, facecolor=SURF, edgecolor=GRID)
        quality = "strong" if isinstance(r2, (int, float)) and r2 >= 0.6 else (
            "weak" if isinstance(r2, (int, float)) and r2 < 0.2 else "moderate")
        interpretations.append(
            f"{label(names[i])}: {quality} predictive agreement; RMSE={rmse:.3g}, MAE={mae:.3g}."
        )
    for j in range(k, len(axes)):
        axes[j].axis("off")
    interp = ("Actual-vs-predicted plots compare experimental measurements with "
              "cross-validated predictions. Points near the dashed 1:1 line indicate "
              "good calibration; broad scatter or systematic trend-line deviation "
              "signals bias or limited learnability. " + " ".join(interpretations[:4]))
    _figure_note(fig, interp)
    fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 0.96))
    return _save(fig, out, title="Actual vs predicted", interpretation=interp)


def residual_plot(y_true, y_pred, target_names, out,
                  validation_method="cross-validation"):
    y_true, y_pred = _as2d(y_true), _as2d(y_pred)
    k = y_true.shape[1]
    names = list(target_names) if target_names is not None else [f"target {i}" for i in range(k)]
    if k == 0 or len(y_true) == 0:
        return _placeholder(out, "Residual plot", "No predictions available.")

    rows, cols = _grid_shape(k)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.0 * rows), facecolor=SURF)
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(f"Residual Diagnostics - {validation_method} (pooled OOF)", color=INK, fontsize=13,
                 fontweight="bold", x=0.02, ha="left")
    summaries = []
    for i in range(k):
        ax = axes[i]; style_ax(ax)
        p = y_pred[:, i]; resid = y_true[:, i] - p
        mask = np.isfinite(p) & np.isfinite(resid)
        p, resid = p[mask], resid[mask]
        sigma = float(np.nanstd(resid))
        ax.axhspan(-1.96 * sigma, 1.96 * sigma, color=BLUE, alpha=0.08,
                   label="approx. 95% residual range")
        ax.scatter(p, resid, s=18, color=BLUE, alpha=0.55, edgecolors="white",
                   linewidths=0.2)
        ax.axhline(0, color=RED, lw=1.4, ls="--")
        if len(p) >= 3 and np.nanstd(p) > 0:
            coef = np.polyfit(p, resid, 1)
            xs = np.linspace(float(np.nanmin(p)), float(np.nanmax(p)), 50)
            ax.plot(xs, np.polyval(coef, xs), color=INK2, lw=1.1, label="residual trend")
        bias = float(np.nanmean(resid))
        ax.set_title(short(label(names[i]), 26), fontsize=10, loc="left", fontweight="bold")
        ax.set_xlabel("Model prediction"); ax.set_ylabel("Residual (measured - predicted)")
        ax.text(0.04, 0.96, f"bias={bias:.2g}\nSD={sigma:.2g}", transform=ax.transAxes,
                ha="left", va="top", fontsize=7, color=INK2,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=SURF, edgecolor=GRID))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        if i == 0:
            ax.legend(loc="best", fontsize=6, facecolor=SURF, edgecolor=GRID)
        summaries.append(f"{label(names[i])}: mean residual {bias:.3g}, residual SD {sigma:.3g}.")
    for j in range(k, len(axes)):
        axes[j].axis("off")
    interp = ("Residual plots reveal bias, heteroscedasticity, and outliers. "
              "A reliable model should have residuals centered near zero with no "
              "obvious trend across the prediction range. " + " ".join(summaries[:4]))
    _figure_note(fig, interp)
    fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 0.96))
    return _save(fig, out, title="Residual diagnostics", interpretation=interp)


def residual_distribution(y_true, y_pred, target_names, out,
                          validation_method="cross-validation"):
    """Residual histograms/error distributions for one or more targets."""
    y_true, y_pred = _as2d(y_true), _as2d(y_pred)
    k = y_true.shape[1]
    names = list(target_names) if target_names is not None else [f"target {i}" for i in range(k)]
    if k == 0 or len(y_true) == 0:
        return _placeholder(out, "Residual distribution", "No predictions available.")
    rows, cols = _grid_shape(k)
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.8 * rows), facecolor=SURF)
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(f"Residual Distribution - {validation_method} (pooled OOF)", color=INK, fontsize=13,
                 fontweight="bold", x=0.02, ha="left")
    summaries = []
    for i in range(k):
        ax = axes[i]; style_ax(ax)
        resid = y_true[:, i] - y_pred[:, i]
        resid = resid[np.isfinite(resid)]
        if len(resid) == 0:
            ax.axis("off")
            continue
        ax.hist(resid, bins=min(28, max(8, int(np.sqrt(len(resid))))),
                color=BLUE, alpha=0.8, edgecolor=SURF)
        ax.axvline(0, color=INK2, lw=1.1, ls="--")
        ax.axvline(np.mean(resid), color=RED, lw=1.3, label="mean error")
        ax.set_title(short(label(names[i]), 26), loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("Residual (measured - predicted)"); ax.set_ylabel("Count")
        ax.legend(loc="best", fontsize=7, facecolor=SURF, edgecolor=GRID)
        summaries.append(f"{label(names[i])}: median error {np.median(resid):.3g}.")
    for j in range(k, len(axes)):
        axes[j].axis("off")
    interp = ("The residual histogram shows whether prediction errors are centered, "
              "symmetric, or dominated by outliers. Long tails indicate experiments "
              "where the model is less reliable. " + " ".join(summaries[:4]))
    _figure_note(fig, interp)
    fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 0.95))
    return _save(fig, out, title="Residual distribution", interpretation=interp)


def cv_metric_distribution(distributions, out, validation_method="cross-validation"):
    """Plot fold/repeat distributions for R2, RMSE, and MAE by target."""
    data = dict(distributions or {})
    if not data:
        return _placeholder(out, "CV metric distributions", "No fold metrics are available.")
    targets = list(data)
    metrics = [("r2", "R2"), ("rmse", "RMSE"), ("mae", "MAE")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor=SURF)
    for ax, (key, display) in zip(axes, metrics):
        style_ax(ax)
        values = [np.asarray(data[t].get(key, []), dtype=float) for t in targets]
        values = [v[np.isfinite(v)] for v in values]
        positions = np.arange(1, len(targets) + 1)
        if any(len(v) for v in values):
            bp = ax.boxplot(values, positions=positions, patch_artist=True,
                            showmeans=True, widths=0.55)
            for box in bp["boxes"]:
                box.set(facecolor=BLUE, alpha=0.42, edgecolor=INK2)
            for median in bp["medians"]:
                median.set(color=RED, linewidth=1.3)
            for i, vals in enumerate(values, start=1):
                if len(vals):
                    jitter = np.linspace(-0.12, 0.12, len(vals))
                    ax.scatter(i + jitter, vals, s=10, color=PURPLE, alpha=0.55)
        ax.set_xticks(positions)
        ax.set_xticklabels([short(label(t), 15) for t in targets], rotation=35, ha="right")
        ax.set_ylabel(display)
        ax.set_title(f"Fold/repeat {display}", loc="left", fontweight="bold")
        if key == "r2":
            ax.axhline(0, color=INK2, lw=0.8)
    title = f"Cross-validation uncertainty - {validation_method}"
    interp = ("Each point is one validation fold in one repeat. Boxes summarise "
              "cross-validation uncertainty; they are not prediction intervals for "
              "individual recipes.")
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", color=INK)
    _figure_note(fig, interp)
    fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 0.92))
    return _save(fig, out, title=title, interpretation=interp)


# =============================================================================
# 4. FEATURE IMPORTANCE
# =============================================================================
def feature_importance(importances, out, top=15, measured=None,
                       title="Feature importance", sort="importance"):
    """`importances` is a pandas Series {feature: importance} or a dict."""
    s = pd.Series(dict(importances)) if not isinstance(importances, pd.Series) else importances
    s = s[s > 0].sort_values(ascending=False)
    if s.empty:
        return _placeholder(out, title, "The model reported no feature importances.")
    measured = measured or set()
    total = float(s.sum()) or 1.0
    if top is None or int(top) <= 0 or int(top) >= len(s):
        top_s = s
    else:
        top_s = s.head(int(top))
    cumulative = top_s.cumsum() / total
    draw = top_s[::-1]
    cum_draw = cumulative[::-1]
    fig, ax = plt.subplots(figsize=(9.4, max(3.4, 0.38 * len(draw) + 1.5)), facecolor=SURF)
    style_ax(ax)
    dominant_cut = max(0.20 * total, (s.iloc[1] * 1.8 if len(s) > 1 else s.iloc[0] + 1))
    colors = []
    for f, v in draw.items():
        if v >= dominant_cut:
            colors.append(ORANGE)
        elif f in measured:
            colors.append(AQUA)
        else:
            colors.append(BLUE)
    ax.barh(range(len(draw)), draw.values, color=colors, height=0.72)
    ax.set_yticks(range(len(draw)))
    ax.set_yticklabels([short(label(f), 34) for f in draw.index], fontsize=8)
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    ax.set_xlabel("Raw model importance")
    xmax = float(draw.max()) or 1.0
    for y, (feat, val), cval in zip(range(len(draw)), draw.items(), cum_draw.values):
        ax.text(val + 0.012 * xmax, y, f"{100 * val / total:.1f}%  raw={val:.3g}  cum={100*cval:.0f}%",
                va="center", fontsize=7, color=INK2)
    ax.set_xlim(0, xmax * 1.36)
    ax.axvline(0.05 * total, color=GRID, lw=1, ls=":")
    if measured:
        ax.text(0.98, 0.04, "aqua = measured after synthesis\nblue = controllable",
                transform=ax.transAxes, fontsize=7, color=MUTED, ha="right", va="bottom")
    dominant = [label(f) for f, v in s.items() if v >= dominant_cut]
    interp = (f"The plot ranks {len(top_s)} of {len(s)} grouped model features by importance. "
              f"Bars show raw importance; labels show percent and cumulative contribution. ")
    if dominant:
        interp += ("Dominant variables are highlighted in orange: " +
                   ", ".join(short(f, 28) for f in dominant[:5]) + ".")
    else:
        interp += "No single variable dominates, so prediction is distributed across several inputs."
    _figure_note(fig, interp)
    fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 1))
    return _save(fig, out, title=title, interpretation=interp,
                 stats={"n_features": int(len(s)), "shown": int(len(top_s))})


def mutual_information_bar(X_df, y, out, target_name="target", top=20):
    """Mutual-information ranking for encoded/numeric features."""
    try:
        from sklearn.feature_selection import mutual_info_regression
        X = pd.DataFrame(X_df).select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        yy = pd.to_numeric(pd.Series(y), errors="coerce")
        mask = yy.notna()
        X, yy = X.loc[mask], yy.loc[mask]
        if X.shape[1] == 0 or len(yy) < 5:
            return _placeholder(out, "Mutual information", "Need more complete numeric data.")
        vals = mutual_info_regression(X.astype(float), yy.astype(float), random_state=42)
        s = pd.Series(vals, index=X.columns).sort_values(ascending=False)
        shown = s.head(top)[::-1]
        fig, ax = plt.subplots(figsize=(9.0, max(3.2, 0.38 * len(shown) + 1.3)), facecolor=SURF)
        style_ax(ax)
        ax.barh(range(len(shown)), shown.values, color=PURPLE, height=0.7)
        ax.set_yticks(range(len(shown)))
        ax.set_yticklabels([short(label(c), 34) for c in shown.index], fontsize=8)
        ax.set_xlabel("Mutual information with target")
        ax.set_title(f"Nonlinear Feature Signal vs {short(label(target_name), 28)}",
                     loc="left", fontweight="bold")
        for yloc, v in enumerate(shown.values):
            ax.text(v + 0.01 * (shown.max() or 1), yloc, f"{v:.3g}",
                    va="center", fontsize=7, color=INK2)
        interp = (f"Mutual information ranks variables by any detectable relationship "
                  f"with {label(target_name)}, including nonlinear effects. High values "
                  "indicate features that may help prediction, but they do not prove causality.")
        _figure_note(fig, interp)
        fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 1))
        return _save(fig, out, title="Mutual information ranking", interpretation=interp)
    except Exception as e:  # noqa: BLE001
        return _placeholder(out, "Mutual information", f"Could not compute MI:\n{e}")


def model_performance_summary(metrics, out, title="Model performance summary"):
    """Train and pooled-OOF R2/RMSE per target (rich or legacy metrics)."""
    rows = list(metrics or [])
    if not rows:
        return _placeholder(out, title, "Train a model to view performance metrics.")
    if isinstance(rows[0], dict):
        names = [r["target"] for r in rows]
        train_r2 = np.array([r["train_r2"] for r in rows], dtype=float)
        train_rmse = np.array([r["train_rmse"] for r in rows], dtype=float)
        cv_r2 = np.array([r["pooled_oof"]["r2"] for r in rows], dtype=float)
        cv_rmse = np.array([r["pooled_oof"]["rmse"] for r in rows], dtype=float)
        validation = rows[0].get("validation_method", "cross-validation")
    else:
        names = [r[0] for r in rows]
        train_r2 = np.array([r[1] for r in rows], dtype=float)
        train_rmse = np.array([r[2] for r in rows], dtype=float)
        cv_r2 = np.array([r[3] for r in rows], dtype=float)
        cv_rmse = np.array([r[4] for r in rows], dtype=float)
        validation = "cross-validation"
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4.2, 0.35 * len(names) + 2)),
                             facecolor=SURF)
    for ax in axes:
        style_ax(ax)
    w = 0.38
    axes[0].bar(x - w / 2, train_r2, width=w, color=AQUA, label="Train R²")
    axes[0].bar(x + w / 2, cv_r2, width=w, color=BLUE, label="CV R²")
    axes[0].axhline(0, color=INK2, lw=0.8)
    axes[0].set_ylabel("R²")
    axes[0].set_title("Explained variance", loc="left", fontweight="bold")
    axes[0].legend(facecolor=SURF, edgecolor=GRID)
    axes[1].bar(x - w / 2, train_rmse, width=w, color=AQUA, label="Train RMSE")
    axes[1].bar(x + w / 2, cv_rmse, width=w, color=BLUE, label="CV RMSE")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Prediction error", loc="left", fontweight="bold")
    axes[1].legend(facecolor=SURF, edgecolor=GRID)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([short(label(n), 18) for n in names], rotation=45, ha="right")
    gap = float(np.nanmean(train_r2 - cv_r2))
    best = names[int(np.nanargmax(cv_r2))] if len(cv_r2) else "target"
    interp = (f"This chart compares in-sample training fit with {validation}. "
              f"The best CV R² is for {label(best)}. The average Train-CV R² gap is "
              f"{gap:.2f}; large positive gaps indicate overfitting.")
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", color=INK)
    _figure_note(fig, interp)
    fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 0.94))
    return _save(fig, out, title=title, interpretation=interp)


def model_comparison_plot(results, out, title="Model comparison"):
    """Visual comparison of architecture benchmark rows from the Compare tab."""
    rows = list(results or [])
    if not rows:
        return _placeholder(out, title, "Run model comparison first.")
    if isinstance(rows[0], dict):
        rows = [r for r in rows if "r2" in r]
        if not rows:
            return _placeholder(out, title, "Every model evaluation failed.")
        names = [r["name"] for r in rows]
        cv_r2 = np.array([r["r2"]["mean"] for r in rows], dtype=float)
        cv_rmse = np.array([r["rmse"]["mean"] for r in rows], dtype=float)
        cv_mae = np.array([r["mae"]["mean"] for r in rows], dtype=float)
        train_r2 = np.array([r["train_r2"] for r in rows], dtype=float)
        pred_ms = np.array([r["predict_ms"] for r in rows], dtype=float)
        r2_low = np.array([r["r2"]["lower"] for r in rows], dtype=float)
        r2_high = np.array([r["r2"]["upper"] for r in rows], dtype=float)
    else:
        rows = [r for r in rows if len(r) >= 7]
        names = [r[0] for r in rows]
        cv_r2 = np.array([r[1] for r in rows], dtype=float)
        cv_rmse = np.array([r[3] for r in rows], dtype=float)
        cv_mae = np.array([r[4] for r in rows], dtype=float)
        train_r2 = np.array([r[5] for r in rows], dtype=float)
        pred_ms = np.array([r[6] for r in rows], dtype=float)
        r2_low = cv_r2.copy(); r2_high = cv_r2.copy()
    order = np.argsort(np.nan_to_num(cv_r2, nan=-1e9))[::-1]
    names = [names[i] for i in order]
    cv_r2, cv_rmse, cv_mae, train_r2, pred_ms, r2_low, r2_high = [
        a[order] for a in (cv_r2, cv_rmse, cv_mae, train_r2, pred_ms, r2_low, r2_high)]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, max(4.0, 0.45 * len(names) + 1.4)),
                             facecolor=SURF)
    for ax in axes:
        style_ax(ax)
    axes[0].barh(y, cv_r2, color=[GOOD if v >= 0.5 else (RED if v < 0 else BLUE) for v in cv_r2])
    axes[0].errorbar(cv_r2, y, xerr=np.maximum(
        np.vstack([cv_r2 - r2_low, r2_high - cv_r2]), 0.0),
                     fmt="none", ecolor=INK2, capsize=3, lw=1)
    axes[0].set_yticks(y); axes[0].set_yticklabels([short(n, 26) for n in names], fontsize=8)
    axes[0].invert_yaxis(); axes[0].set_xlabel("CV R²"); axes[0].set_title("Generalization", loc="left", fontweight="bold")
    axes[1].barh(y - 0.18, cv_rmse, height=0.34, color=BLUE, label="RMSE")
    axes[1].barh(y + 0.18, cv_mae, height=0.34, color=AQUA, label="MAE")
    axes[1].invert_yaxis(); axes[1].set_yticks([]); axes[1].set_xlabel("Error"); axes[1].set_title("Error magnitude", loc="left", fontweight="bold")
    axes[1].legend(facecolor=SURF, edgecolor=GRID)
    axes[2].barh(y - 0.18, np.maximum(train_r2 - cv_r2, 0), height=0.34, color=ORANGE, label="overfit gap")
    axes[2].barh(y + 0.18, pred_ms, height=0.34, color=PURPLE, label="ms/row")
    axes[2].invert_yaxis(); axes[2].set_yticks([]); axes[2].set_xlabel("Gap / latency"); axes[2].set_title("Risk and speed", loc="left", fontweight="bold")
    axes[2].legend(facecolor=SURF, edgecolor=GRID)
    best = names[0] if names else "model"
    interp = (f"Models are ranked by cross-validated R². {best} is currently strongest. "
              "Compare CV error with the Train-CV gap: high training fit plus weak CV "
              "performance signals overfitting rather than reliable prediction.")
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", color=INK)
    _figure_note(fig, interp)
    fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 0.94))
    return _save(fig, out, title=title, interpretation=interp)


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


def shap_summary(estimator, X_df, out, target_name="", max_display=15,
                 display_labels=None):
    try:
        import shap  # noqa: F401
    except ImportError:
        return _placeholder(out, "SHAP summary",
                            "Install the 'shap' package to enable SHAP charts.")
    try:
        sv, Xs = _shap_values_for(estimator, X_df)
        fig = plt.figure(facecolor=SURF)
        names = [dict(display_labels or {}).get(str(c), str(c)) for c in Xs.columns]
        shap.summary_plot(sv, Xs, feature_names=names, max_display=max_display, show=False,
                          plot_size=(9, max(4, 0.4 * min(max_display, X_df.shape[1]) + 2)))
        fig = plt.gcf()
        fig.patch.set_facecolor(SURF)
        ttl = "SHAP summary" + (f"  ·  {short(target_name, 28)}" if target_name else "")
        fig.suptitle(ttl, color=INK, fontsize=12, fontweight="bold", x=0.02, ha="left")
        return _save(fig, out)
    except Exception as e:  # noqa: BLE001
        return _placeholder(out, "SHAP summary", f"Could not compute SHAP values:\n{e}")


def shap_dependence(estimator, X_df, out, target_name="", top_k=4,
                    display_labels=None):
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
            display = dict(display_labels or {}).get(str(feat), str(feat))
            ax.set_title(short(display, 26), fontsize=10, loc="left", fontweight="bold")
            ax.set_xlabel(short(display, 26)); ax.set_ylabel("SHAP value")
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
    interp = (f"Loadings show which original variables define {pc}. Positive and negative "
              "directions separate samples along the component; large absolute loadings "
              "are the most influential variables for that axis.")
    _figure_note(fig, interp)
    fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 1))
    return _save(fig, out, title=title, interpretation=interp)


def pca_variable_contribution(loadings, out, pcs=("PC1", "PC2"), top=20,
                              title="PCA variable contribution"):
    """Contribution/cos2-style ranking from squared loadings across selected PCs."""
    cols = [c for c in pcs if c in loadings.columns]
    if not cols:
        return _placeholder(out, title, "No requested PCA loading columns found.")
    contrib = loadings[cols].pow(2).sum(axis=1)
    contrib = contrib.sort_values(ascending=False)
    shown = contrib.head(top)[::-1]
    total = float(contrib.sum()) or 1.0
    fig, ax = plt.subplots(figsize=(9.0, max(3.4, 0.38 * len(shown) + 1.3)), facecolor=SURF)
    style_ax(ax)
    ax.barh(range(len(shown)), shown.values / total * 100.0, color=PURPLE, height=0.72)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels([short(label(c), 34) for c in shown.index], fontsize=8)
    ax.set_xlabel(f"Contribution to {' + '.join(cols)} (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    for yloc, v in enumerate(shown.values / total * 100.0):
        ax.text(v + 0.01 * max(1, shown.max() / total * 100.0), yloc,
                f"{v:.1f}%", va="center", fontsize=7, color=INK2)
    interp = (f"This chart ranks variables by squared loading across {' and '.join(cols)}. "
              "Higher contribution means the variable strongly structures the PCA score plot. "
              "This is descriptive variance structure, not proof of causality.")
    _figure_note(fig, interp)
    fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 1))
    return _save(fig, out, title=title, interpretation=interp)


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
    """Publication target distribution: histogram, density, box, violin, QQ, stats."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return _placeholder(out, "Target analysis", "Too few target values.")
    mean = float(np.mean(y))
    median = float(np.median(y))
    std = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    outliers = int(((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)).sum())
    ci = (mean - 1.96 * std / np.sqrt(len(y)), mean + 1.96 * std / np.sqrt(len(y)))
    skew = kurt = np.nan
    pval = np.nan
    normal = False
    try:
        from scipy import stats as _st
        skew = float(_st.skew(y, bias=False))
        kurt = float(_st.kurtosis(y, bias=False))
        if len(y) <= 5000:
            _, pval = _st.shapiro(y)
        else:
            _, pval = _st.normaltest(y)
        normal = bool(pval > 0.05)
    except Exception:  # noqa: BLE001
        pass
    if stats:
        skew = stats.get("skewness", skew)
        kurt = stats.get("kurtosis", kurt)
        normal = stats.get("is_normal", normal)
        outliers = stats.get("n_outliers", outliers)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), facecolor=SURF)
    axes = axes.ravel()
    for ax in axes:
        style_ax(ax)

    # Histogram + density + CI.
    axes[0].hist(y, bins=min(32, max(10, int(np.sqrt(len(y))))), density=True,
                 color=BLUE, alpha=0.72, edgecolor=SURF, linewidth=0.6, label="histogram")
    try:
        from scipy import stats as _st
        xs = np.linspace(float(y.min()), float(y.max()), 240)
        kde = _st.gaussian_kde(y)
        axes[0].plot(xs, kde(xs), color=RED, lw=1.6, label="density")
    except Exception:  # noqa: BLE001
        pass
    axes[0].axvline(mean, color=INK2, lw=1.3, label="mean")
    axes[0].axvline(median, color=GOLD, lw=1.3, ls="--", label="median")
    axes[0].axvspan(ci[0], ci[1], color=AQUA, alpha=0.15, label="95% CI mean")
    axes[0].set_title("Histogram and density", fontsize=10, loc="left", fontweight="bold")
    axes[0].set_xlabel(label(target_name)); axes[0].set_ylabel("Density")
    axes[0].legend(loc="best", fontsize=6, facecolor=SURF, edgecolor=GRID)

    # Box plot.
    bp = axes[1].boxplot(y, vert=True, widths=0.5, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set(facecolor=AQUA, alpha=0.5, edgecolor=INK2)
    for med in bp["medians"]:
        med.set(color=RED, linewidth=1.5)
    axes[1].scatter(np.ones_like(y) + np.random.default_rng(0).normal(0, 0.025, len(y)),
                    y, s=8, color=BLUE, alpha=0.25, edgecolors="none")
    axes[1].set_title("Box plot with samples", fontsize=10, loc="left", fontweight="bold")
    axes[1].set_xticks([])

    # Violin plot.
    vp = axes[2].violinplot(y, showmeans=True, showmedians=True)
    for body in vp["bodies"]:
        body.set_facecolor(PURPLE); body.set_edgecolor(PURPLE); body.set_alpha(0.35)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in vp:
            vp[key].set_color(INK2)
    axes[2].set_title("Violin plot", fontsize=10, loc="left", fontweight="bold")
    axes[2].set_xticks([])

    # QQ plot vs normal.
    try:
        from scipy import stats as _st
        _st.probplot(y, dist="norm", plot=axes[3])
        axes[3].get_lines()[0].set(marker="o", ms=4, color=BLUE, alpha=0.6, ls="")
        axes[3].get_lines()[1].set(color=RED, lw=1.4)
    except Exception:  # noqa: BLE001
        axes[3].scatter(np.sort(y), np.linspace(0, 1, len(y)), s=10, color=BLUE)
    axes[3].set_title("QQ plot vs normal", fontsize=10, loc="left", fontweight="bold")

    # Cumulative distribution.
    xs = np.sort(y)
    axes[4].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=BLUE, lw=1.8)
    axes[4].set_title("Empirical CDF", fontsize=10, loc="left", fontweight="bold")
    axes[4].set_xlabel(label(target_name)); axes[4].set_ylabel("Cumulative probability")

    # Stats panel.
    axes[5].axis("off")
    stat_lines = [
        f"n = {len(y)}",
        f"mean = {mean:.4g}",
        f"median = {median:.4g}",
        f"standard deviation = {std:.4g}",
        f"95% CI(mean) = [{ci[0]:.4g}, {ci[1]:.4g}]",
        f"skewness = {skew:.3g}",
        f"kurtosis = {kurt:.3g}",
        f"outliers (1.5 IQR) = {outliers}",
        f"normality p = {pval:.3g}" if np.isfinite(pval) else "normality p = n/a",
    ]
    axes[5].text(0.02, 0.96, "\n".join(stat_lines), ha="left", va="top",
                 fontsize=9, color=INK2, transform=axes[5].transAxes,
                 bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor=GRID))

    shape = "approximately normal" if normal else "not well described by a normal distribution"
    interp = (f"The target distribution for {label(target_name)} is {shape}. "
              f"Mean={mean:.3g}, median={median:.3g}, SD={std:.3g}, skewness={skew:.2g}, "
              f"and {outliers} potential outliers were detected. Non-normality or strong "
              "outliers can make cross-validation less stable and may justify transforms or robust models.")
    fig.suptitle(f"Target Distribution · {short(label(target_name), 32)}",
                 color=INK, fontsize=12, fontweight="bold", x=0.02, ha="left")
    _figure_note(fig, interp)
    fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 0.94))
    return _save(fig, out, title="Target distribution", interpretation=interp,
                 stats={"n": int(len(y)), "mean": mean, "median": median,
                        "std": std, "skewness": float(skew), "kurtosis": float(kurt),
                        "outliers": int(outliers), "normality_p": float(pval) if np.isfinite(pval) else None})


def missingness_chart(df, out, title="Missing values"):
    """Per-column missingness chart for dataset intelligence."""
    if df is None or len(df.columns) == 0:
        return _placeholder(out, title, "No dataset columns available.")
    miss = df.isna().mean().sort_values(ascending=False) * 100.0
    miss = miss[miss > 0]
    if miss.empty:
        miss = pd.Series({"No missing values": 0.0})
    shown = miss.head(30)[::-1]
    fig, ax = plt.subplots(figsize=(9.0, max(3.3, 0.32 * len(shown) + 1.4)), facecolor=SURF)
    style_ax(ax)
    colors = [RED if v >= 50 else (ORANGE if v >= 20 else BLUE) for v in shown.values]
    ax.barh(range(len(shown)), shown.values, color=colors, height=0.72)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels([short(label(c), 34) for c in shown.index], fontsize=8)
    ax.set_xlabel("Missing values (%)")
    ax.set_xlim(0, max(100, float(shown.max()) * 1.12))
    ax.set_title(title, loc="left", fontweight="bold")
    for yloc, v in enumerate(shown.values):
        ax.text(v + 1, yloc, f"{v:.1f}%", va="center", fontsize=7, color=INK2)
    overall = float(df.isna().mean().mean() * 100.0)
    interp = (f"Overall missingness is {overall:.1f}%. Variables above 20% missingness "
              "may weaken learnability or require careful imputation; variables above "
              "50% should be treated as low-support evidence.")
    _figure_note(fig, interp)
    fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 1))
    return _save(fig, out, title=title, interpretation=interp,
                 stats={"overall_missing_pct": overall})


def dataset_difficulty_chart(summary, diff, learn, out, title="Dataset difficulty"):
    """Compact visual explanation of dataset/modeling difficulty."""
    n = float(summary.get("n_samples", 0))
    missing = float(summary.get("missing_pct_overall", 0))
    dup = float(summary.get("duplicate_conditions", summary.get("duplicate_rows", 0)))
    best_r2 = float(learn.get("best_r2", np.nan))
    est_low = float(diff.get("est_r2_low", np.nan))
    est_high = float(diff.get("est_r2_high", np.nan))
    metrics = pd.Series({
        "Sample support": min(1.0, n / 200.0),
        "Completeness": max(0.0, 1.0 - missing / 100.0),
        "Uniqueness": max(0.0, 1.0 - dup / max(n, 1.0)),
        "Best CV R²": np.clip((best_r2 + 1) / 2, 0, 1) if np.isfinite(best_r2) else 0,
        "Estimated learnability": np.clip(((est_low + est_high) / 2 + 1) / 2, 0, 1)
        if np.isfinite(est_low) and np.isfinite(est_high) else 0,
    })
    fig, ax = plt.subplots(figsize=(8.8, 4.8), facecolor=SURF)
    style_ax(ax)
    colors = [GOOD if v >= 0.7 else (ORANGE if v >= 0.4 else RED) for v in metrics.values]
    ax.bar(range(len(metrics)), metrics.values * 100, color=colors, width=0.62)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([short(k, 18) for k in metrics.index], rotation=25, ha="right")
    ax.set_ylabel("Favorable score (%)")
    ax.set_title(f"{title}: {diff.get('label', 'unknown')}", loc="left", fontweight="bold")
    for i, v in enumerate(metrics.values * 100):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=8, color=INK2)
    interp = (f"Difficulty is classified as {diff.get('label', 'unknown')}. "
              f"The dataset has {int(n)} samples, {missing:.1f}% missingness, and "
              f"best observed CV R² of {best_r2:.2f}. Low bars identify the constraints "
              "most likely limiting reliable prediction.")
    _figure_note(fig, interp)
    fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 1))
    return _save(fig, out, title=title, interpretation=interp)


def similarity_chart(similar, out, target_name="", applicability=None,
                     title="Similarity and applicability"):
    """Nearest-neighbor similarity chart for screening reports."""
    rows = list(similar or [])
    if not rows:
        return _placeholder(out, title, "No similar experiments available.")
    sims = np.array([r.get("similarity", np.nan) for r in rows], dtype=float)
    measured = []
    for r in rows:
        m = r.get("measured", {})
        measured.append(m.get(target_name, np.nan) if target_name else np.nan)
    measured = np.array(measured, dtype=float)
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8.8, max(3.4, 0.45 * len(rows) + 1.4)), facecolor=SURF)
    style_ax(ax)
    colors = [GOOD if s >= 75 else (ORANGE if s >= 45 else RED) for s in sims[::-1]]
    ax.barh(y, sims[::-1], color=colors, height=0.68)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Similarity to screened recipe (%)")
    labels = []
    for r in rows[::-1]:
        idx = r.get("index", "?")
        labels.append(f"Experiment {idx}")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, loc="left", fontweight="bold")
    for yloc, s, m in zip(y, sims[::-1], measured[::-1]):
        suffix = "" if not np.isfinite(m) else f"  measured={m:.3g}"
        ax.text(min(s + 1.5, 98), yloc, f"{s:.0f}%{suffix}",
                va="center", fontsize=7, color=INK2)
    ad = applicability or {}
    ad_text = ""
    if ad:
        ad_text = (f" Applicability domain: {ad.get('label', 'unknown')} "
                   f"(novelty percentile {ad.get('percentile', np.nan):.0f}).")
    interp = ("Nearest-neighbor similarity compares the candidate recipe against "
              "training experiments. High similarity means stronger empirical support; "
              "low similarity or outside-domain status means the prediction should be "
              "treated as exploratory." + ad_text)
    _figure_note(fig, interp)
    fig.tight_layout(rect=(0, CAPTION_BOTTOM, 1, 1))
    return _save(fig, out, title=title, interpretation=interp)


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
