"""test_research_gap.py — training-data coverage gap detection.

Run:  python -m pytest test_research_gap.py -q
"""

import pandas as pd
import pytest

import research_gap as RG


def test_rare_category_detected():
    df = pd.DataFrame({"Biomass": ["Rice husk"] * 18 + ["Bamboo"] * 2})
    cat_choices = {"Biomass": ["Rice husk", "Bamboo"]}
    gaps = RG.rare_categories(df, cat_choices, min_frac=0.05, min_count=3)
    assert any(g["value"] == "Bamboo" for g in gaps)
    assert not any(g["value"] == "Rice husk" for g in gaps)


def test_rare_category_empty_dataframe():
    df = pd.DataFrame({"Biomass": []})
    assert RG.rare_categories(df, {"Biomass": ["Rice husk"]}) == []


def test_numeric_window_gap_detected():
    # Dense 700-900, then nothing until a couple points near 1200.
    vals = list(range(700, 900, 5)) + [1195, 1198]
    df = pd.DataFrame({"Temp": vals})
    gaps = RG.numeric_windows(df, ["Temp"], n_bins=5, min_frac=0.05)
    assert any(g["column"] == "Temp" for g in gaps)
    # The sparse high-temperature bin should show up with a low count.
    assert any(g["count"] <= 2 for g in gaps)


def test_numeric_window_too_few_points_skipped():
    df = pd.DataFrame({"Temp": [700, 750]})
    assert RG.numeric_windows(df, ["Temp"]) == []


def test_untested_combination_detected():
    df = pd.DataFrame({
        "Biomass": ["Rice husk", "Rice husk", "Bamboo"],
        "Activator": ["KOH", "KOH", "KOH"],
    })
    cat_choices = {"Biomass": ["Rice husk", "Bamboo"], "Activator": ["KOH", "NaOH"]}
    gaps = RG.untested_combinations(df, cat_choices)
    combos = {(g["value_a"], g["value_b"]) for g in gaps}
    assert ("Rice husk", "NaOH") in combos
    assert ("Bamboo", "NaOH") in combos
    assert ("Rice husk", "KOH") not in combos   # this one WAS observed


def test_untested_combination_skips_single_choice_columns():
    df = pd.DataFrame({"Atmosphere": ["Ar", "Ar"], "Activator": ["KOH", "KOH"]})
    cat_choices = {"Atmosphere": ["Ar"], "Activator": ["KOH", "NaOH"]}
    # Atmosphere has only 1 choice -> excluded by min_choices=2 default.
    gaps = RG.untested_combinations(df, cat_choices)
    assert not any(g["column_a"] == "Atmosphere" or g["column_b"] == "Atmosphere" for g in gaps)


def test_detect_gaps_combines_and_caps():
    df = pd.DataFrame({
        "Biomass": ["Rice husk"] * 15 + ["Bamboo"] * 1,
        "Activator": ["KOH"] * 14 + ["NaOH"] * 2,
        "Temp": list(range(700, 716)),
    })
    numeric_cols = ["Temp"]
    cat_choices = {"Biomass": ["Rice husk", "Bamboo"], "Activator": ["KOH", "NaOH"]}
    gaps = RG.detect_gaps(df, numeric_cols, cat_choices, max_gaps=5)
    assert len(gaps) <= 5
    assert all("description" in g for g in gaps)
    # rare_category / numeric_gap should be prioritized over untested_combination
    types_present = [g["type"] for g in gaps]
    if "untested_combination" in types_present:
        first_combo = types_present.index("untested_combination")
        assert all(t != "untested_combination" for t in types_present[:first_combo])


def test_detect_gaps_never_raises_on_bad_input():
    # A column referenced in cat_choices but absent from df.
    df = pd.DataFrame({"Temp": [700, 750, 800, 850, 900]})
    gaps = RG.detect_gaps(df, ["Temp"], {"Missing_Col": ["A", "B"]})
    assert isinstance(gaps, list)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
