"""
slideshow.py  —  Turn a run's charts into a narrated slideshow PDF.
==================================================================

Collects the diagnostic charts already produced by the Charts / Latent /
Dataset-Intelligence tabs and lays them out one-per-slide in a landscape PDF:

  * a title slide (model, dataset, target(s), headline metric);
  * optional section-divider slides ("Model Diagnostics", ...);
  * one slide per chart — the figure plus its plain-language explanation
    (read from the companion .json each chart writes, or supplied directly);
  * a synthesised conclusion slide (overall verdict from the CV metrics, the
    most influential variables, caveats) that always keeps the promise the app
    makes elsewhere: these are MODEL PREDICTIONS, not laboratory measurements.

Pure logic + matplotlib only (headless Agg, via PdfPages) so it needs no extra
dependency, runs fully offline, and can be unit-tested without a GUI.
"""

from __future__ import annotations

import json
import os
import textwrap

# --- palette (self-contained; mirrors the charts look) ----------------------
SURF = "#ffffff"
BAND = "#1f6f3f"        # header/brand green
BAND_DK = "#144d2b"
INK = "#14211b"
INK2 = "#3a4a42"
MUTED = "#6b7a72"
ACCENT = "#c9d8e8"
RULE = "#d7dedb"

# 16:9 landscape "slide".
SLIDE_W, SLIDE_H = 12.8, 7.2


# =============================================================================
# Text helpers
# =============================================================================
def _wrap(text: str, width: int) -> list[str]:
    out = []
    for para in str(text).replace("\r", "").split("\n"):
        para = para.strip()
        if not para:
            out.append("")
            continue
        out.extend(textwrap.wrap(para, width=width) or [""])
    return out


def read_interpretation(png_path: str) -> str:
    """Explanation text a chart wrote next to its PNG (charts._write_meta)."""
    if not png_path:
        return ""
    root, _ = os.path.splitext(png_path)
    for path in (root + ".json", root + ".txt"):
        try:
            with open(path, encoding="utf-8") as fh:
                if path.endswith(".json"):
                    return str(json.load(fh).get("interpretation", "")).strip()
                return fh.read().strip()
        except (OSError, ValueError):
            continue
    return ""


# =============================================================================
# Conclusion synthesis
# =============================================================================
def _target_r2(metric: dict):
    """CV R2 for one target metric dict, tolerant of the app's shapes."""
    pooled = metric.get("pooled_oof") or {}
    if "r2" in pooled:
        return pooled["r2"]
    cv = (metric.get("cv") or {}).get("r2") or {}
    return cv.get("mean")


def _target_rmse(metric: dict):
    pooled = metric.get("pooled_oof") or {}
    if "rmse" in pooled:
        return pooled["rmse"]
    cv = (metric.get("cv") or {}).get("rmse") or {}
    return cv.get("mean")


def _verdict(mean_r2) -> str:
    if mean_r2 is None or mean_r2 != mean_r2:            # None / NaN
        return "could not be summarised from the available metrics"
    if mean_r2 >= 0.75:
        return "strong predictive performance"
    if mean_r2 >= 0.5:
        return "moderate performance — useful for ranking and prioritisation"
    if mean_r2 >= 0.3:
        return "weak but directional performance"
    return "limited predictive power — treat results as exploratory"


def conclusion_lines(metrics=None, top_features=None, targets=None,
                     n_charts=0) -> list[str]:
    """Build the conclusion narrative from CV metrics + top features."""
    metrics = list(metrics or [])
    lines: list[str] = []

    r2s = [_target_r2(m) for m in metrics]
    r2s = [r for r in r2s if isinstance(r, (int, float)) and r == r]
    mean_r2 = sum(r2s) / len(r2s) if r2s else None

    if metrics:
        head = f"The model shows {_verdict(mean_r2)}"
        if mean_r2 is not None:
            head += f" (mean cross-validated R2 = {mean_r2:.2f} across {len(metrics)} target(s))."
        else:
            head += "."
        lines.append(head)
        lines.append("")
        lines.append("Per-target cross-validated performance:")
        for m in metrics:
            r2, rmse = _target_r2(m), _target_rmse(m)
            tr = m.get("train_r2")
            bits = []
            if isinstance(r2, (int, float)) and r2 == r2:
                bits.append(f"CV R2 = {r2:.2f}")
            if isinstance(rmse, (int, float)) and rmse == rmse:
                bits.append(f"RMSE = {rmse:.3g}")
            if isinstance(tr, (int, float)) and tr == tr:
                bits.append(f"train R2 = {tr:.2f}")
            lines.append(f"  - {m.get('target', 'target')}: " + ", ".join(bits))
            # Over-fitting flag.
            if (isinstance(tr, (int, float)) and isinstance(r2, (int, float))
                    and tr == tr and r2 == r2 and tr - r2 > 0.25):
                lines.append("      (large train-vs-CV gap — some over-fitting; "
                             "trust the CV number.)")
        lines.append("")
    elif targets:
        lines.append(f"Model targets: {', '.join(map(str, targets))}.")
        lines.append("")

    if top_features:
        named = ", ".join(f"{name} ({val:.2f})" if isinstance(val, (int, float))
                          else str(name) for name, val in list(top_features)[:5])
        lines.append(f"Most influential variables (by model importance): {named}.")
        lines.append("These are the levers to prioritise when planning experiments.")
        lines.append("")

    if n_charts:
        lines.append(f"This deck summarises {n_charts} diagnostic chart(s): correlation "
                     "and mutual-information signal, out-of-fold accuracy and residual "
                     "behaviour, feature importance/SHAP attributions, and optimisation "
                     "trade-offs where available.")
        lines.append("")

    lines.append("Reminder: every value here is a MODEL PREDICTION meant to prioritise "
                 "experiments — not a laboratory measurement. Validate promising "
                 "candidates at the bench and retrain as new data arrives.")
    return lines


# =============================================================================
# Slide primitives
# =============================================================================
def _new_slide(plt):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), facecolor=SURF)
    fig.patch.set_facecolor(SURF)
    return fig


def _header_band(fig, title, kicker=""):
    """Top brand band with a slide title; returns the band's bottom (fig coords)."""
    import matplotlib.patches as mpatches
    band_h = 0.135
    fig.add_artist(mpatches.Rectangle((0, 1 - band_h), 1, band_h,
                                      transform=fig.transFigure, facecolor=BAND,
                                      edgecolor="none", zorder=0))
    if kicker:
        fig.text(0.045, 1 - band_h * 0.34, kicker.upper(), color=ACCENT,
                 fontsize=9, fontweight="bold", va="center")
    fig.text(0.045, 1 - band_h * 0.66, title, color="white", fontsize=17,
             fontweight="bold", va="center")
    return 1 - band_h


def _footer(fig, idx, total, note="BioCarbon Screen"):
    fig.text(0.045, 0.035, note, color=MUTED, fontsize=7.5, va="center")
    if total:
        fig.text(0.955, 0.035, f"{idx} / {total}", color=MUTED, fontsize=7.5,
                 va="center", ha="right")
    import matplotlib.lines as mlines
    fig.add_artist(mlines.Line2D([0.045, 0.955], [0.065, 0.065],
                                 transform=fig.transFigure, color=RULE, lw=0.8))


def _title_slide(pdf, plt, title, subtitle, meta_rows, headline):
    import matplotlib.patches as mpatches
    fig = _new_slide(plt)
    fig.add_artist(mpatches.Rectangle((0, 0.60), 1, 0.40, transform=fig.transFigure,
                                      facecolor=BAND, edgecolor="none"))
    fig.add_artist(mpatches.Rectangle((0, 0.575), 1, 0.025, transform=fig.transFigure,
                                      facecolor=BAND_DK, edgecolor="none"))
    fig.text(0.06, 0.80, title, color="white", fontsize=30, fontweight="bold", va="center")
    if subtitle:
        fig.text(0.06, 0.685, subtitle, color=ACCENT, fontsize=13, va="center")

    y = 0.46
    for label, value in (meta_rows or []):
        fig.text(0.06, y, f"{label}", color=MUTED, fontsize=10.5, va="center")
        fig.text(0.30, y, f"{value}", color=INK, fontsize=10.5, va="center",
                 fontweight="bold")
        y -= 0.052
    if headline:
        for i, line in enumerate(_wrap(headline, 96)[:3]):
            fig.text(0.06, 0.16 - i * 0.036, line, color=INK2, fontsize=11, va="center")
    _footer(fig, 1, 0, note="Generated summary — model predictions, not measurements")
    pdf.savefig(fig, facecolor=SURF)
    plt.close(fig)


def _section_slide(pdf, plt, name, subtitle=""):
    fig = _new_slide(plt)
    fig.text(0.06, 0.55, name, color=BAND, fontsize=26, fontweight="bold", va="center")
    import matplotlib.lines as mlines
    fig.add_artist(mlines.Line2D([0.06, 0.55], [0.47, 0.47], transform=fig.transFigure,
                                 color=BAND, lw=2.5))
    if subtitle:
        fig.text(0.06, 0.40, subtitle, color=INK2, fontsize=12, va="center")
    pdf.savefig(fig, facecolor=SURF)
    plt.close(fig)


def _chart_slide(pdf, plt, title, png_path, explanation, kicker, idx, total):
    fig = _new_slide(plt)
    top = _header_band(fig, title, kicker)

    have_text = bool(explanation)
    # Image occupies the upper area; explanation sits in a panel below it.
    img_bottom = 0.30 if have_text else 0.09
    img_ax = fig.add_axes([0.045, img_bottom, 0.91, top - img_bottom - 0.02])
    img_ax.axis("off")
    drawn = False
    if png_path and os.path.exists(png_path):
        try:
            img_ax.imshow(plt.imread(png_path))
            img_ax.set_aspect("equal")           # preserve chart aspect, centered
            drawn = True
        except Exception:                        # noqa: BLE001
            drawn = False
    if not drawn:
        img_ax.text(0.5, 0.5, "(chart image unavailable)", ha="center", va="center",
                    color=MUTED, fontsize=12, transform=img_ax.transAxes)

    if have_text:
        import matplotlib.patches as mpatches
        fig.add_artist(mpatches.Rectangle((0.045, 0.085), 0.91, 0.20,
                                          transform=fig.transFigure, facecolor="#f3f7f5",
                                          edgecolor=RULE, lw=0.8, zorder=0))
        fig.text(0.06, 0.255, "What this shows", color=BAND, fontsize=10,
                 fontweight="bold", va="center")
        lines = _wrap(explanation, 132)[:5]
        for i, line in enumerate(lines):
            fig.text(0.06, 0.215 - i * 0.033, line, color=INK2, fontsize=9.3, va="center")
    _footer(fig, idx, total)
    pdf.savefig(fig, facecolor=SURF)
    plt.close(fig)


def _conclusion_slides(pdf, plt, lines, idx, total):
    """One or more slides holding the conclusion narrative."""
    per_slide = 15
    chunks = [lines[i:i + per_slide] for i in range(0, len(lines), per_slide)] or [[]]
    for n, chunk in enumerate(chunks):
        fig = _new_slide(plt)
        title = "Conclusion" if len(chunks) == 1 else f"Conclusion ({n + 1}/{len(chunks)})"
        top = _header_band(fig, title, "Summary")
        y = top - 0.06
        for line in chunk:
            if not line:
                y -= 0.028
                continue
            wrapped = _wrap(line, 118)
            bold = not line.startswith(" ") and line.endswith(":")
            for w in wrapped:
                fig.text(0.05, y, w, color=INK if bold else INK2,
                         fontsize=11 if bold else 10.2, va="top",
                         fontweight="bold" if bold else "normal")
                y -= 0.040
            y -= 0.006
        _footer(fig, idx + n, total)
        pdf.savefig(fig, facecolor=SURF)
        plt.close(fig)


# =============================================================================
# Public API
# =============================================================================
def build_slideshow(path, sections, *, title="Model Analysis Summary", subtitle="",
                    meta_rows=None, interpretations=None, metrics=None,
                    top_features=None, targets=None, conclusion=None) -> str:
    """Render a narrated slideshow PDF and return its path.

    Parameters
    ----------
    path : str
        Output .pdf path.
    sections : list
        Either ``[(chart_title, png_path), ...]`` or, to group charts under
        headings, ``[(section_name, [(chart_title, png_path), ...]), ...]``.
    interpretations : dict, optional
        ``{png_path or chart_title: explanation}``. Missing entries fall back to
        the companion .json/.txt each chart wrote, then to a generic note.
    metrics, top_features, targets : optional
        Feed the auto-generated conclusion (see :func:`conclusion_lines`).
        ``top_features`` is ``[(display_name, importance), ...]``.
    conclusion : str or list[str], optional
        Explicit conclusion text; overrides the auto-generated narrative.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Normalise `sections` into [(name_or_None, [(title, png), ...]), ...].
    grouped: list[tuple] = []
    if sections and isinstance(sections[0], (list, tuple)) and len(sections[0]) == 2 \
            and isinstance(sections[0][1], (list, tuple)) \
            and (len(sections[0][1]) == 0
                 or isinstance(sections[0][1][0], (list, tuple))):
        grouped = [(name, list(items)) for name, items in sections]
    else:
        grouped = [(None, list(sections))]

    interpretations = dict(interpretations or {})
    flat = [(name, it) for name, items in grouped for it in items]
    n_charts = len(flat)

    def explain(chart_title, png):
        if png in interpretations:
            return interpretations[png]
        if chart_title in interpretations:
            return interpretations[chart_title]
        found = read_interpretation(png)
        return found or f"{chart_title}: see the figure above for the model diagnostics."

    # Total slide count for footers: title + sections + charts + conclusion.
    show_sections = [name for name, _ in grouped if name]
    if isinstance(conclusion, str):
        concl_lines = _wrap(conclusion, 118)
    elif isinstance(conclusion, (list, tuple)):
        concl_lines = list(conclusion)
    else:
        concl_lines = conclusion_lines(metrics=metrics, top_features=top_features,
                                       targets=targets, n_charts=n_charts)
    n_concl = max(1, (len(concl_lines) + 14) // 15)
    total = 1 + len(show_sections) + n_charts + n_concl

    with PdfPages(path) as pdf:
        headline = ""
        if metrics:
            r2s = [_target_r2(m) for m in metrics]
            r2s = [r for r in r2s if isinstance(r, (int, float)) and r == r]
            if r2s:
                headline = (f"Headline: mean cross-validated R2 = "
                            f"{sum(r2s) / len(r2s):.2f} across {len(metrics)} target(s). "
                            "All figures are model predictions, not measurements.")
        _title_slide(pdf, plt, title, subtitle, meta_rows, headline)

        idx = 2
        for name, items in grouped:
            if name:
                _section_slide(pdf, plt, name)
                idx += 1
            for chart_title, png in items:
                _chart_slide(pdf, plt, chart_title, png,
                             explain(chart_title, png), name or "", idx, total)
                idx += 1

        _conclusion_slides(pdf, plt, concl_lines, idx, total)
    return path
