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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
