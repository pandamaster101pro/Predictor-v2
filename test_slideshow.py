"""
test_slideshow.py  —  slideshow PDF builder + conclusion synthesis.

Run:  python -m pytest test_slideshow.py -q
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import slideshow as S


def _tiny_png(path, label="x"):
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1, 2], [0, 1, 0])
    ax.set_title(label)
    fig.savefig(path)
    plt.close(fig)


def _page_count(pdf_path):
    from pypdf import PdfReader
    return len(PdfReader(pdf_path).pages)


# ---------------------------------------------------------------------------
# Conclusion synthesis
# ---------------------------------------------------------------------------
def test_conclusion_uses_cv_metrics_and_verdict():
    metrics = [
        {"target": "LIB_1A", "train_r2": 0.95,
         "pooled_oof": {"r2": 0.82, "rmse": 12.3, "mae": 8.1}},
        {"target": "SIB_1A", "train_r2": 0.90,
         "pooled_oof": {"r2": 0.78, "rmse": 15.0, "mae": 9.0}},
    ]
    lines = S.conclusion_lines(metrics=metrics,
                               top_features=[("Pyrolysis temp", 0.31), ("Pretreat 1", 0.22)],
                               n_charts=8)
    text = "\n".join(lines)
    assert "strong predictive performance" in text
    assert "LIB_1A" in text and "CV R2 = 0.82" in text
    assert "Pyrolysis temp" in text
    assert "MODEL PREDICTION" in text.upper()


def test_conclusion_flags_overfitting_gap():
    metrics = [{"target": "T", "train_r2": 0.98, "pooled_oof": {"r2": 0.40, "rmse": 5}}]
    text = "\n".join(S.conclusion_lines(metrics=metrics))
    assert "over-fitting" in text.lower()


def test_conclusion_handles_missing_metrics():
    lines = S.conclusion_lines(metrics=[], targets=["A", "B"], n_charts=3)
    text = "\n".join(lines)
    assert "A, B" in text
    assert "model prediction" in text.lower()


def test_verdict_bands():
    assert S._verdict(0.9) == "strong predictive performance"
    assert "moderate" in S._verdict(0.6)
    assert "weak" in S._verdict(0.35)
    assert "limited" in S._verdict(0.1)
    assert "could not" in S._verdict(float("nan"))


# ---------------------------------------------------------------------------
# Interpretation lookup
# ---------------------------------------------------------------------------
def test_read_interpretation_from_json(tmp_path):
    png = tmp_path / "chart.png"
    _tiny_png(str(png))
    (tmp_path / "chart.json").write_text(
        json.dumps({"title": "t", "interpretation": "Residuals look unbiased."}),
        encoding="utf-8")
    assert S.read_interpretation(str(png)) == "Residuals look unbiased."


def test_read_interpretation_missing_returns_empty(tmp_path):
    png = tmp_path / "nope.png"
    _tiny_png(str(png))
    assert S.read_interpretation(str(png)) == ""


# ---------------------------------------------------------------------------
# Full slideshow build
# ---------------------------------------------------------------------------
def test_build_flat_slideshow(tmp_path):
    pngs = []
    for i in range(3):
        p = tmp_path / f"c{i}.png"
        _tiny_png(str(p), f"chart {i}")
        pngs.append((f"Chart {i}", str(p)))
    out = tmp_path / "deck.pdf"
    S.build_slideshow(str(out), pngs,
                      metrics=[{"target": "T", "train_r2": 0.8,
                                "pooled_oof": {"r2": 0.7, "rmse": 3.0}}],
                      top_features=[("A", 0.5)])
    assert os.path.exists(out)
    # title + 3 charts + 1 conclusion = 5 pages
    assert _page_count(str(out)) == 5


def test_build_sectioned_slideshow_counts_pages(tmp_path):
    def mk(n):
        items = []
        for i in range(n):
            p = tmp_path / f"s{n}_{i}.png"
            _tiny_png(str(p))
            items.append((f"c{i}", str(p)))
        return items

    sections = [("Model Diagnostics", mk(2)), ("Latent Variables", mk(1))]
    out = tmp_path / "deck2.pdf"
    S.build_slideshow(str(out), sections, title="Summary")
    # title + 2 section slides + 3 charts + 1 conclusion = 7
    assert _page_count(str(out)) == 7


def test_explicit_conclusion_override(tmp_path):
    p = tmp_path / "c.png"
    _tiny_png(str(p))
    out = tmp_path / "deck3.pdf"
    S.build_slideshow(str(out), [("Only chart", str(p))],
                      conclusion="Custom takeaway line.")
    assert _page_count(str(out)) == 3   # title + chart + conclusion


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
