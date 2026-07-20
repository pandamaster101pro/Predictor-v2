"""
test_units.py  —  unit-standardization behavior for units.standardize_units.

Run:  python -m pytest test_units.py -q
"""

import numpy as np
import pandas as pd
import pytest

import units as U


def _col(df, name):
    return list(df[name])


# ---------------------------------------------------------------------------
# Per-cell explicit units
# ---------------------------------------------------------------------------
def test_time_mixed_cell_units_convert_to_minutes():
    df = pd.DataFrame({"Time": ["30 min", "2 h", "1.5 h", "90 min"]})
    notes = []
    out = U.standardize_units(df, notes=notes)
    assert _col(out, "Time") == [30.0, 120.0, 90.0, 90.0]
    assert notes and "Time" in notes[0]


def test_mass_mg_and_g_to_grams():
    df = pd.DataFrame({"Additive": ["500 mg", "1 g", "250 mg"]})
    out = U.standardize_units(df)
    assert _col(out, "Additive") == pytest.approx([0.5, 1.0, 0.25])


def test_bare_numbers_follow_modal_cell_unit():
    # Three explicit hours + one bare number -> bare assumed hours too.
    df = pd.DataFrame({"Dwell": ["2 h", "1 h", "3 h", "4"]})
    out = U.standardize_units(df)
    assert _col(out, "Dwell") == [120.0, 60.0, 180.0, 240.0]


# ---------------------------------------------------------------------------
# Header-declared units (bare numeric cells)
# ---------------------------------------------------------------------------
def test_header_kelvin_to_celsius():
    df = pd.DataFrame({"Pyrolysis temp (K)": [1173.15, 1273.15, 273.15]})
    out = U.standardize_units(df)
    assert _col(out, "Pyrolysis temp (K)") == pytest.approx([900.0, 1000.0, 0.0])


def test_header_hours_to_minutes():
    df = pd.DataFrame({"Hold time (h)": [1, 2, 0.5]})
    out = U.standardize_units(df)
    assert _col(out, "Hold time (h)") == pytest.approx([60.0, 120.0, 30.0])


def test_header_celsius_bare_numbers_is_noop():
    # Already canonical (°C) with bare numbers -> column left untouched.
    df = pd.DataFrame({"Temp (oC)": [700, 800, 900]})
    notes = []
    out = U.standardize_units(df, notes=notes)
    assert _col(out, "Temp (oC)") == [700, 800, 900]
    assert notes == []


# ---------------------------------------------------------------------------
# Safety / conservatism
# ---------------------------------------------------------------------------
def test_text_column_untouched():
    df = pd.DataFrame({"Atmosphere": ["Ar", "N2", "Ar", "N2"]})
    out = U.standardize_units(df)
    assert _col(out, "Atmosphere") == ["Ar", "N2", "Ar", "N2"]


def test_unitless_column_untouched():
    df = pd.DataFrame({"pH": [6.5, 7.0, 7.2]})
    out = U.standardize_units(df)
    assert _col(out, "pH") == [6.5, 7.0, 7.2]


def test_mixed_dimensions_are_left_alone():
    # Cells span temperature AND time -> ambiguous -> untouched.
    df = pd.DataFrame({"Weird": ["900 C", "900 C", "30 min", "2 h"]})
    out = U.standardize_units(df)
    assert _col(out, "Weird") == ["900 C", "900 C", "30 min", "2 h"]


def test_protected_column_skipped():
    df = pd.DataFrame({"Time (h)": [1, 2, 3]})
    out = U.standardize_units(df, protected=["Time (h)"])
    assert _col(out, "Time (h)") == [1, 2, 3]


def test_blank_and_not_done_tokens_preserved():
    df = pd.DataFrame({"Time": ["30 min", "--", "", "2 h", None]})
    out = U.standardize_units(df)
    got = _col(out, "Time")
    assert got[0] == 30.0 and got[3] == 120.0
    assert got[1] == "--" and got[2] == ""
    assert got[4] is None or (isinstance(got[4], float) and np.isnan(got[4]))


def test_canonical_override_linear_dimension():
    df = pd.DataFrame({"Time": ["60 min", "120 min", "30 min"]})
    out = U.standardize_units(df, canonical={"time": "h"})
    assert _col(out, "Time") == pytest.approx([1.0, 2.0, 0.5])


def test_heating_rate_per_hour_to_per_minute():
    df = pd.DataFrame({"Ramp": ["300 C/h", "600 C/h"]})
    out = U.standardize_units(df)
    assert _col(out, "Ramp") == pytest.approx([5.0, 10.0])


def test_micro_sign_variants_normalize():
    # micro sign (µ) and Greek mu (μ) both mean micrograms here.
    df = pd.DataFrame({"Dose": ["500 µg", "500 μg", "0.001 g"]})
    out = U.standardize_units(df)
    assert _col(out, "Dose") == pytest.approx([5e-4, 5e-4, 1e-3])


# ---------------------------------------------------------------------------
# Percent concentration -> molarity
# ---------------------------------------------------------------------------
def test_molarity_cells_standardize_to_float():
    df = pd.DataFrame({"Pretreat": ["1M KOH", "0.5 M KOH", "2MHNO3"]})
    notes = []
    out = U.standardize_units(df, notes=notes)
    assert _col(out, "Pretreat") == pytest.approx([1.0, 0.5, 2.0])
    assert notes and "concentration" in notes[0].lower()


def test_percent_vv_h2so4_to_molarity():
    df = pd.DataFrame({"Acid": ["1.25v% H2SO4", "2.5 v% H2SO4"]})
    out = U.standardize_units(df)
    h2so4 = U.CHEMICALS["h2so4"]
    expected = [
        10 * 1.25 * h2so4["pure_density_g_ml"] / h2so4["molar_mass_g_mol"],
        10 * 2.5 * h2so4["pure_density_g_ml"] / h2so4["molar_mass_g_mol"],
    ]
    assert _col(out, "Acid") == pytest.approx(expected)


def test_header_percent_v_bare_values_to_molarity():
    df = pd.DataFrame({"H2SO4 concentration (%V)": [1.25, 2.5]})
    out = U.standardize_units(df)
    h2so4 = U.CHEMICALS["h2so4"]
    expected = [
        10 * 1.25 * h2so4["pure_density_g_ml"] / h2so4["molar_mass_g_mol"],
        10 * 2.5 * h2so4["pure_density_g_ml"] / h2so4["molar_mass_g_mol"],
    ]
    assert _col(out, "H2SO4 concentration (%V)") == pytest.approx(expected)


def test_percent_wv_salt_to_molarity():
    df = pd.DataFrame({"Salt": ["0.2 w/v% NaHCO3", "2 w/v% NaHCO3"]})
    out = U.standardize_units(df)
    nahco3 = U.CHEMICALS["nahco3"]
    assert _col(out, "Salt") == pytest.approx([
        10 * 0.2 / nahco3["molar_mass_g_mol"],
        10 * 2.0 / nahco3["molar_mass_g_mol"],
    ])


def test_wt_percent_stock_solution_to_molarity_when_density_known():
    df = pd.DataFrame({"Oxidizer": ["98wt% H2SO4", "38wt% H2O2"]})
    out = U.standardize_units(df)
    h2so4 = U.CHEMICALS["h2so4"]
    h2o2 = U.CHEMICALS["h2o2"]
    assert _col(out, "Oxidizer") == pytest.approx([
        10 * 98 * h2so4["solution_density_g_ml"] / h2so4["molar_mass_g_mol"],
        10 * 38 * h2o2["solution_density_g_ml"] / h2o2["molar_mass_g_mol"],
    ])


def test_plain_doping_percent_without_basis_is_untouched():
    df = pd.DataFrame({"Dopant": ["15% Mn", "5% Zn", "30mole% citric acid"]})
    out = U.standardize_units(df)
    assert _col(out, "Dopant") == ["15% Mn", "5% Zn", "30mole% citric acid"]


# ---------------------------------------------------------------------------
# Parsed messy-column triples -> molarity, label columns removed
# ---------------------------------------------------------------------------
def test_parsed_triple_vv_to_molarity_and_labels_removed():
    # '3.2%V H2SO4' was parsed to (3.2, 'V', 'h2so4') by the app.
    df = pd.DataFrame({
        "numeric_feature_A1": [3.2, 1.6, 0.0],
        "group_label_B1": ["V", "V", "None"],
        "text_modifier_C1": ["h2so4", "h2so4", "None"],
    })
    notes = []
    out = U.standardize_parsed_mixed(df, notes=notes)
    h2so4 = U.CHEMICALS["h2so4"]
    f = 10 * h2so4["pure_density_g_ml"] / h2so4["molar_mass_g_mol"]
    assert set(out.columns) == {"numeric_feature_A1", "solute_label_D1"}
    assert _col(out, "numeric_feature_A1") == pytest.approx([3.2 * f, 1.6 * f, 0.0])
    assert _col(out, "solute_label_D1") == ["H2SO4", "H2SO4", "None"]
    assert notes and "removed its label columns" in notes[0]


def test_parsed_triple_molar_code_passthrough():
    # '1M KOH' -> (1.0, 'M', 'koh'): value is already molarity.
    df = pd.DataFrame({
        "numeric_feature_A": [1.0, 0.5, 0.0],
        "group_label_B": ["M", "M", "None"],
        "text_modifier_C": ["koh", "koh", "None"],
    })
    out = U.standardize_parsed_mixed(df)
    assert set(out.columns) == {"numeric_feature_A", "solute_label_D"}
    assert _col(out, "numeric_feature_A") == pytest.approx([1.0, 0.5, 0.0])
    assert _col(out, "solute_label_D") == ["KOH", "KOH", "None"]


def test_parsed_triple_wv_basis_split_across_labels():
    # '0.2 w/v% NaHCO3' parses to code 'W' + text '/v  nahco3' — reunited.
    df = pd.DataFrame({
        "numeric_feature_A": [0.2, 2.0],
        "group_label_B": ["W", "W"],
        "text_modifier_C": ["/v  nahco3", "/v  nahco3"],
    })
    out = U.standardize_parsed_mixed(df)
    nahco3 = U.CHEMICALS["nahco3"]
    assert set(out.columns) == {"numeric_feature_A", "solute_label_D"}
    assert _col(out, "numeric_feature_A") == pytest.approx(
        [10 * 0.2 / nahco3["molar_mass_g_mol"], 10 * 2.0 / nahco3["molar_mass_g_mol"]])
    assert _col(out, "solute_label_D") == ["NaHCO3", "NaHCO3"]


def test_parsed_triple_doping_keeps_labels():
    # '15% Mn' -> (15.0, 'MN', 'None'): no basis, no registry chemical.
    df = pd.DataFrame({
        "numeric_feature_A2": [15.0, 5.0],
        "group_label_B2": ["MN", "ZN"],
        "text_modifier_C2": ["None", "None"],
    })
    out = U.standardize_parsed_mixed(df)
    assert set(out.columns) == {"numeric_feature_A2", "group_label_B2",
                                "text_modifier_C2"}
    assert _col(out, "numeric_feature_A2") == [15.0, 5.0]


def test_parsed_triple_below_threshold_untouched():
    # Only 1 of 3 informative rows converts -> whole triple left alone.
    df = pd.DataFrame({
        "numeric_feature_A": [3.2, 10.0, 1.0],
        "group_label_B": ["V", "MN", "ZN"],
        "text_modifier_C": ["h2so4", "None", "None"],
    })
    out = U.standardize_parsed_mixed(df)
    assert set(out.columns) == {"numeric_feature_A", "group_label_B",
                                "text_modifier_C"}
    assert _col(out, "numeric_feature_A") == [3.2, 10.0, 1.0]


# ---------------------------------------------------------------------------
# Display composition ('1M NaOH' instead of split internal columns)
# ---------------------------------------------------------------------------
def test_compose_value_molarity_with_solute():
    assert U.compose_value(1.0, solute="NaOH") == "1M NaOH"
    assert U.compose_value(0.5972, solute="H2SO4") == "0.5972M H2SO4"
    assert U.compose_value(0.0, solute="None") == "--"
    assert U.compose_value(0.3, solute="Missing") == "0.3 M"


def test_compose_value_unconverted_triple():
    assert U.compose_value(15.0, code="MN") == "15% MN"
    assert U.compose_value(5.0, code="ZN", detail="znso4-7h2o") == "5% ZN (znso4-7h2o)"
    assert U.compose_value(0.0, detail="bio oil") == "bio oil"
    assert U.compose_value(0.0, code="None", detail="None") == "--"


def test_recipe_groups_and_compose_group():
    cols = ["numeric_feature_A1", "solute_label_D1",
            "numeric_feature_A2", "group_label_B2", "text_modifier_C2",
            "numeric_feature_A3", "Temp_C"]
    groups = U.recipe_groups(cols)
    assert set(groups) == {"1", "2"}          # A3 has no companions -> no group
    row = {"numeric_feature_A1": 1.0, "solute_label_D1": "NaOH",
           "numeric_feature_A2": 15.0, "group_label_B2": "MN",
           "text_modifier_C2": "None"}
    assert U.compose_group(groups["1"], row.get) == "1M NaOH"
    assert U.compose_group(groups["2"], row.get) == "15% MN"


def test_compose_grouped_passthrough_and_labels():
    raw = {"numeric_feature_A1": 1.0, "solute_label_D1": "NaOH", "Temp_C": 900}
    labels = {"numeric_feature_A1": "Pretreat 1: number/percent"}
    rows = dict(U.compose_grouped(raw, labels))
    assert rows == {"Pretreat 1": "1M NaOH", "Temp_C": 900}


def test_roundtrip_cell_to_parts_to_display():
    # '1M NaOH' -> parse -> molarity + solute -> compose -> '1M NaOH'
    mol, solute, basis = U.parse_concentration_to_molarity("1M NaOH")
    assert U.compose_value(mol, solute=solute) == "1M NaOH"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
