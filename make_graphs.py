"""
make_graphs.py  —  Visualise the connections in the capacity data.
==================================================================

Produces one dashboard PNG that shows, for the reversible-capacity dataset:
  A. distribution of the target (capacity)
  B. which inputs drive capacity   (model feature importance)
  C. model fit                     (out-of-fold predicted vs actual)
  D. linear connections            (feature<->capacity correlation, signed)
  E-F. the two strongest single-feature relationships (scatter + trend)

Run:  python make_graphs.py      ->  writes capacity_connections.png
"""

import sys, subprocess, importlib.util

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def auto_bootstrap():
    deps = {"pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn",
            "xgboost": "xgboost", "matplotlib": "matplotlib", "openpyxl": "openpyxl"}
    miss = [p for m, p in deps.items() if importlib.util.find_spec(m) is None]
    if miss:
        print(f"[*] Installing {miss} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--only-binary=:all:", *miss])


auto_bootstrap()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

# ---- palette (validated default, light surface) ----------------------------
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, AQUA, RED, GOOD = "#2a78d6", "#1baf7a", "#d03b3b", "#0ca30c"

DATA = r"C:/Users/28jay/Downloads/lignin_hard_carbon_dataset_fixed.xlsx"
TARGET = "Reversible_Capacity_mAh_per_g"
# features that are MEASURED after synthesis (flagged, not excluded, in graphs)
MEASURED = {"Lignin_Purity_wt%", "Ash_Content_wt%", "Sulfur_Content_wt%",
            "d002_Angstrom", "La_nm", "Lc_nm", "ID_IG_Ratio", "BET_Surface_Area_m2_per_g",
            "Total_Pore_Volume_cm3_per_g", "Micropore_Fraction", "Closed_Pore_Fraction",
            "True_Density_g_per_cm3", "Carbon_Yield_wt%"}
OUTCOMES = {"Plateau_Capacity_mAh_per_g", "Slope_Capacity_mAh_per_g", "ICE_%",
            "Rate_Cap_Retention_%", "Cycle_Retention_100cyc_%", "Avg_Sodiation_Voltage_V"}


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
    return name if len(name) <= n else name[:n - 1] + "…"


def main():
    df = pd.read_excel(DATA).drop(columns=["Sample_ID"], errors="ignore")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    # The file has a junk first row ('INPUT') that forces numeric columns to be
    # text — recover their real numeric dtype so they're not mistaken for
    # categories. A column that is >=80% parseable as numbers becomes numeric.
    for c in df.columns:
        if df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() >= 0.8:
                df[c] = coerced

    y = df[TARGET].values
    feats = [c for c in df.columns if c not in OUTCOMES | {TARGET}]

    # encode + model (all inputs, to reveal the real drivers)
    X = df[feats].copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].astype(str).fillna("NA")
    Xe = pd.get_dummies(X, drop_first=False).astype(float)
    model = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=42, n_jobs=-1, verbosity=0)
    oof = cross_val_predict(model, Xe.values, y, cv=KFold(5, shuffle=True, random_state=42))
    r2 = r2_score(y, oof)
    model.fit(Xe.values, y)

    # feature importance -> map dummies back to their source column, sum
    imp = pd.Series(model.feature_importances_, index=Xe.columns)
    src = {}
    for col in Xe.columns:
        base = col
        for f in feats:
            if col == f or col.startswith(f + "_"):
                base = f; break
        src[col] = base
    imp_by_feat = imp.groupby(src).sum().sort_values(ascending=False)

    # numeric correlations with target (signed)
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    corr = df[num + [TARGET]].corr()[TARGET].drop(TARGET).dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)

    # ---- figure ----
    fig = plt.figure(figsize=(16, 9), facecolor=SURF)
    fig.suptitle(f"What drives reversible capacity  ·  {len(df)} experiments  ·  "
                 f"model out-of-fold R² = {r2:.2f}",
                 color=INK, fontsize=14, fontweight="bold", x=0.5, y=0.98)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                          left=0.11, right=0.98, top=0.9, bottom=0.08)

    # A. target distribution
    ax = fig.add_subplot(gs[0, 0]); style_ax(ax)
    ax.hist(y, bins=24, color=BLUE, edgecolor=SURF, linewidth=0.6)
    ax.axvline(np.median(y), color=RED, lw=1.5, ls="--")
    ax.text(np.median(y), ax.get_ylim()[1] * 0.92, f" median {np.median(y):.0f}",
            color=RED, fontsize=8, va="top")
    ax.set_title("A. Capacity distribution", fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("Reversible capacity (mAh/g)"); ax.set_ylabel("count")

    # B. feature importance (top 12), measured flagged in aqua
    ax = fig.add_subplot(gs[0, 1]); style_ax(ax)
    top = imp_by_feat.head(12)[::-1]
    colors = [AQUA if f in MEASURED else BLUE for f in top.index]
    ax.barh(range(len(top)), top.values, color=colors, height=0.72)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([short(f) for f in top.index], fontsize=7)
    ax.set_title("B. What drives it (model importance)", fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("importance")
    ax.text(0.98, 0.04, "aqua = measured after synthesis\nblue = controllable",
            transform=ax.transAxes, fontsize=7, color=MUTED, ha="right", va="bottom")

    # C. predicted vs actual (OOF)
    ax = fig.add_subplot(gs[0, 2]); style_ax(ax)
    ax.scatter(y, oof, s=16, color=BLUE, alpha=0.5, edgecolors="none")
    lim = [min(y.min(), oof.min()), max(y.max(), oof.max())]
    ax.plot(lim, lim, color=MUTED, lw=1.2, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"C. Predicted vs actual (5-fold OOF, R²={r2:.2f})", fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("actual mAh/g"); ax.set_ylabel("predicted mAh/g")

    # D. signed correlations (diverging blue/red)
    ax = fig.add_subplot(gs[1, 0]); style_ax(ax)
    cc = corr.head(12)[::-1]
    bar_colors = [BLUE if v >= 0 else RED for v in cc.values]
    ax.barh(range(len(cc)), cc.values, color=bar_colors, height=0.72)
    ax.axvline(0, color=GRID, lw=1)
    ax.set_yticks(range(len(cc))); ax.set_yticklabels([short(f, 16) for f in cc.index], fontsize=7)
    ax.set_title("D. Linear link to capacity (Pearson r)", fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("correlation  (blue +   red −)"); ax.set_xlim(-1, 1)

    # E, F. two strongest single-feature relationships
    strongest = corr.abs().sort_values(ascending=False).index[:2]
    for j, feat in enumerate(strongest):
        ax = fig.add_subplot(gs[1, 1 + j]); style_ax(ax)
        xv = pd.to_numeric(df[feat], errors="coerce")
        mask = xv.notna()
        ax.scatter(xv[mask], y[mask.values], s=16, color=BLUE, alpha=0.5, edgecolors="none")
        # linear trend line
        if mask.sum() > 2:
            b, a = np.polyfit(xv[mask], y[mask.values], 1)
            xs = np.linspace(xv[mask].min(), xv[mask].max(), 50)
            ax.plot(xs, a + b * xs, color=RED, lw=1.6)
        flag = "  (measured)" if feat in MEASURED else "  (controllable)"
        ax.set_title(f"{'E' if j == 0 else 'F'}. {short(feat,26)}{flag}",
                     fontsize=10, loc="left", fontweight="bold")
        ax.set_xlabel(feat); ax.set_ylabel("capacity (mAh/g)")
        ax.xaxis.set_major_locator(MaxNLocator(5))

    out = "capacity_connections.png"
    fig.savefig(out, dpi=130, facecolor=SURF)
    print(f"[OK] wrote {out}  ({len(df)} rows, R²={r2:.3f})")
    print(f"Top drivers: {', '.join(imp_by_feat.head(5).index)}")


if __name__ == "__main__":
    main()
