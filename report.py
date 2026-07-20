"""
report.py  —  Professional screening reports for BioCarbon Screen.
==================================================================

Turns one screening result (from :class:`screening.Screener`) into a shareable
PDF and/or a multi-sheet Excel workbook.  Both make the same promise the app
makes on screen: these are MODEL PREDICTIONS to prioritise experiments, not
laboratory measurements.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd

import charts as C


# =============================================================================
# Small matplotlib helpers  (headless Agg backend so it never needs a window)
# =============================================================================
def _contrib_png(result, screener, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contribs = result["contributions"]["contributions"][:8][::-1]
    if not contribs:
        return None
    names = [screener.pretty(n) for n, _ in contribs]
    vals = [v for _, v in contribs]
    colors = ["#3aa856" if v >= 0 else "#cc4b3b" for v in vals]
    fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=130)
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel(f"Contribution to predicted {screener.pretty(result['target'])}", fontsize=8)
    ax.set_title("Why the model made this prediction", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _interval_png(result, screener, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = result["target"]
    pred = result["prediction"]
    fig, ax = plt.subplots(figsize=(6.6, 1.9), dpi=130)
    col = screener.y_train[t].values
    ax.hist(col, bins=25, color="#c9d8e8", edgecolor="#9bb")
    ax.axvspan(pred["lo"], pred["hi"], color="#3aa856", alpha=0.18,
               label="95% prediction interval")
    ax.axvline(pred["mean"], color="#1f6f3f", lw=2, label="prediction")
    ax.set_xlabel(f"{screener.pretty(t)}  (dataset distribution vs this prediction)", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# PDF report  (reportlab)
# =============================================================================
def build_prediction_pdf(path, result, screener, model_summary=""):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image, HRFlowable)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=18, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=8,
                         textColor=colors.HexColor("#666"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12,
                        textColor=colors.HexColor("#1f6f3f"), spaceBefore=10)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=13)

    tmpdir = tempfile.mkdtemp(prefix="bcs_report_")
    story = []
    rec = result["recommendation"]
    pred = result["prediction"]
    t = result["target"]
    mq = result["model_quality"]

    story.append(Paragraph("BioCarbon Screen — Screening Report", h1))
    story.append(Paragraph(
        "AI-assisted screening for biomass-derived hard-carbon synthesis. "
        "Values below are MODEL PREDICTIONS to help prioritise experiments — "
        "they are not laboratory measurements.", sub))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc"),
                            spaceBefore=6, spaceAfter=6))

    # --- Verdict banner ---
    vcol = colors.Color(*rec["color"][:3])
    banner = Table([[Paragraph(
        f"<b>Recommendation: {rec['verdict']}</b>  &nbsp; "
        f"(priority score {rec['score']:.2f} · confidence {rec['confidence']})",
        ParagraphStyle("ban", parent=body, textColor=colors.white, fontSize=11))]],
        colWidths=[170 * mm])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), vcol),
                                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                ("TOPPADDING", (0, 0), (-1, -1), 6),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(banner)
    story.append(Spacer(1, 6))

    # --- Prediction summary ---
    story.append(Paragraph("Prediction summary", h2))
    ci = mq["cv_r2"]
    pred_tbl = [
        ["Target", screener.pretty(t)],
        ["Estimated value", f"{pred['mean']:.1f}"],
        ["95% prediction interval", f"{pred['lo']:.1f}  to  {pred['hi']:.1f}"],
        ["Expected error (CV RMSE)", f"± {pred['expected_error']:.1f}"],
        ["Confidence level", rec["confidence"]],
        ["Ranking vs dataset", f"better than {rec['ranking']['percentile']:.0f}% "
                               f"of experiments"],
        ["Applicability domain", result["applicability"]["label"]],
        ["Model quality (CV R²)", f"{ci:.3f}" if ci is not None else "n/a"],
        ["Training experiments", f"{mq['n_train']} rows · {mq['n_features']} features"],
    ]
    tb = Table(pred_tbl, colWidths=[55 * mm, 115 * mm])
    tb.setStyle(_kv_style(colors))
    story.append(tb)

    # --- Interval plot ---
    try:
        img = _interval_png(result, screener, os.path.join(tmpdir, "iv.png"))
        if img:
            story.append(Spacer(1, 4))
            story.append(Image(img, width=165 * mm, height=47 * mm))
    except Exception:
        pass

    # --- Why (reasons + contributions) ---
    story.append(Paragraph("Why this recommendation", h2))
    for r in rec["reasons"]:
        story.append(Paragraph("• " + r, body))
    try:
        img = _contrib_png(result, screener, os.path.join(tmpdir, "contrib.png"))
        if img:
            story.append(Spacer(1, 4))
            story.append(Image(img, width=165 * mm, height=85 * mm))
    except Exception:
        pass

    if result.get("effect_summary"):
        story.append(Paragraph("What-if sensitivity", h2))
        for e in result["effect_summary"][:6]:
            story.append(Paragraph("- " + e["summary"], body))
        rows = [["Variable", "Prediction swing", "Direction"]]
        for e in result["effect_summary"][:6]:
            rows.append([screener.pretty(e["feature"]), f"{e['swing']:.1f}",
                         str(e["direction"])])
        st = Table(rows, colWidths=[75 * mm, 45 * mm, 50 * mm])
        st.setStyle(_grid_style(colors))
        story.append(st)

    # --- Warnings ---
    if result["ood"] or result["missing"]:
        story.append(Paragraph("Warnings", h2))
        for f in result["ood"]:
            story.append(Paragraph("⚠ " + f, body))
        if result["missing"]:
            story.append(Paragraph("⚠ Unspecified inputs: " +
                                   ", ".join(result["missing"]), body))

    # --- Input conditions ---
    story.append(Paragraph("Input synthesis conditions", h2))
    rows = [["Variable", "Value"]]
    for c, v in result["raw"].items():
        rows.append([screener.pretty(c), _fmt(v)])
    it = Table(rows, colWidths=[85 * mm, 85 * mm])
    it.setStyle(_grid_style(colors))
    story.append(it)

    # --- Similar experiments ---
    if result["similar"]:
        story.append(Paragraph("Most similar real experiments", h2))
        try:
            sim_img = C.similarity_chart(
                result["similar"], os.path.join(tmpdir, "similarity.png"),
                target_name=t, applicability=result["applicability"])
            story.append(Image(sim_img, width=165 * mm, height=75 * mm))
            story.append(Spacer(1, 4))
        except Exception:
            pass
        hdr = ["Similarity", "Measured " + screener.pretty(t)] + \
              [screener.pretty(c) for c in list(result["similar"][0]["conditions"])[:4]]
        rows = [hdr]
        for s in result["similar"][:6]:
            row = [f"{s['similarity']:.0f}%", f"{s['measured'][t]:.1f}"]
            for c in list(s["conditions"])[:4]:
                row.append(_fmt(s["conditions"][c]))
            rows.append(row)
        stt = Table(rows)
        stt.setStyle(_grid_style(colors))
        story.append(stt)

    # --- Suggested next experiments ---
    story.append(Paragraph("Suggested next steps", h2))
    story.append(Paragraph(_next_steps(result, screener), body))

    if model_summary:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Model: " + model_summary, sub))

    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=15 * mm,
                            bottomMargin=15 * mm, leftMargin=20 * mm, rightMargin=20 * mm)
    doc.build(story)
    return path


def _kv_style(colors):
    from reportlab.platypus import TableStyle
    return TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef4ef")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3)])


def _grid_style(colors):
    from reportlab.platypus import TableStyle
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f6f3f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f8f5")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2)])


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _next_steps(result, screener):
    rec = result["recommendation"]
    t = screener.pretty(result["target"])
    v = rec["verdict"]
    if v in ("Excellent candidate", "Strong candidate"):
        base = (f"This route is a priority for the bench: the model predicts a "
                f"strong {t} with acceptable confidence and it sits inside the "
                f"model's experience. Synthesise and characterise it, then feed the "
                f"measured result back to retrain and tighten the model.")
    elif v == "Worth testing":
        base = (f"A reasonable candidate. Predicted {t} is promising but the "
                f"uncertainty or novelty is non-trivial — worth an exploratory run "
                f"if capacity is limited only after the top candidates.")
    else:
        base = (f"Low priority. Either the predicted {t} is modest, the uncertainty "
                f"is high, or the recipe is outside the model's experience. Prefer "
                f"other candidates first, or gather more data around this region.")
    if not result["applicability"]["in_domain"]:
        base += (" Because it is outside the training domain, treat any single-run "
                 "result as data collection rather than confirmation.")
    return base


# =============================================================================
# Excel report  (pandas + openpyxl)
# =============================================================================
def export_prediction_excel(path, result, screener):
    t = result["target"]
    pred = result["prediction"]
    rec = result["recommendation"]
    mq = result["model_quality"]

    summary = pd.DataFrame({
        "Field": ["Target", "Estimated value", "Interval low", "Interval high",
                  "Expected error (CV RMSE)", "Confidence", "Verdict",
                  "Priority score", "Ranking percentile", "Applicability domain",
                  "In domain?", "Model CV R2", "Training rows", "Encoded features"],
        "Value": [screener.pretty(t), round(pred["mean"], 2), round(pred["lo"], 2),
                  round(pred["hi"], 2), round(pred["expected_error"], 2),
                  rec["confidence"], rec["verdict"], round(rec["score"], 3),
                  round(rec["ranking"]["percentile"], 1),
                  result["applicability"]["label"],
                  result["applicability"]["in_domain"],
                  mq["cv_r2"], mq["n_train"], mq["n_features"]],
    })
    inputs = pd.DataFrame({"Variable": list(result["raw"].keys()),
                           "Value": [_fmt(v) for v in result["raw"].values()]})
    reasons = pd.DataFrame({"Recommendation reasons": rec["reasons"]})
    warnings = pd.DataFrame({"Warnings": (result["ood"] +
                             (["Unspecified: " + ", ".join(result["missing"])]
                              if result["missing"] else [])) or ["None"]})
    contribs = pd.DataFrame(result["contributions"]["contributions"],
                            columns=["Feature", "Contribution"])
    sensitivity = pd.DataFrame(result.get("effect_summary", []))
    pd_rows = []
    for feature, curve in result.get("partial_dependence", {}).items():
        for x, y in zip(curve.get("x", []), curve.get("y", [])):
            pd_rows.append({"Feature": feature, "Value": x, "Prediction": y})
    partial = pd.DataFrame(pd_rows)
    sim_rows = []
    for s in result["similar"]:
        row = {"Similarity_%": round(s["similarity"], 1),
               f"Measured_{t}": round(s["measured"][t], 2)}
        row.update(s["conditions"])
        sim_rows.append(row)
    similar = pd.DataFrame(sim_rows)

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        inputs.to_excel(xl, sheet_name="Inputs", index=False)
        reasons.to_excel(xl, sheet_name="Recommendation", index=False)
        warnings.to_excel(xl, sheet_name="Warnings", index=False)
        contribs.to_excel(xl, sheet_name="FeatureContribution", index=False)
        if len(sensitivity):
            sensitivity.to_excel(xl, sheet_name="Sensitivity", index=False)
        if len(partial):
            partial.to_excel(xl, sheet_name="PartialDependence", index=False)
        if len(similar):
            similar.to_excel(xl, sheet_name="SimilarExperiments", index=False)
    return path
