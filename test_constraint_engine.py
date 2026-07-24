"""test_constraint_engine.py — constraint parsing/expansion/application.

Run:  python -m pytest test_constraint_engine.py -q
"""

import numpy as np
import pandas as pd
import pytest

import constraint_engine as CE


# ---- count_steps / count_stages -----------------------------------------------
def test_count_steps_and_stages():
    recipe = {"Pretreat 1": "1M NaOH", "Pretreat 2": "None", "Py.1 temp. (oC)": 900,
              "Py.2 temp. (oC)": 950}
    assert CE.count_steps(recipe) == 1
    assert CE.count_stages(recipe) == 2


def test_count_stages_defaults_to_one():
    assert CE.count_stages({"Temperature": 900}) == 1


# ---- parse: original column syntax unchanged ------------------------------------
def test_parse_numeric_comparison():
    c = CE.parse("Temperature <= 900")
    assert c == [{"kind": "column", "column": "Temperature", "op": "<=", "value": "900"}]


def test_parse_in_and_not_in():
    c = CE.parse("Biomass IN [Rice husk, Bamboo]\nActivator NOT IN [KOH]")
    assert c[0] == {"kind": "column", "column": "Biomass", "op": "IN",
                    "values": ["Rice husk", "Bamboo"]}
    assert c[1]["op"] == "NOT IN" and c[1]["values"] == ["KOH"]


def test_parse_ignores_blank_and_comment_lines():
    assert CE.parse("\n# a comment\n  \n") == []


def test_parse_unparsable_line_skipped():
    assert CE.parse("this has no operator at all") == []


# ---- parse: chemical-class shortcut ---------------------------------------------
def test_parse_chemical_class_shortcut():
    c = CE.parse("NO STRONG ACID")
    assert c == [{"kind": "chemical_class", "class": "strong acid", "raw": "NO STRONG ACID"}]


def test_parse_chemical_class_case_insensitive():
    c = CE.parse("no oxidizer")
    assert c[0]["kind"] == "chemical_class"
    assert c[0]["class"] == "oxidizer"


# ---- parse: process shortcut -----------------------------------------------------
def test_parse_process_stage_limit():
    c = CE.parse("stages <= 2")
    assert c == [{"kind": "process", "column": "max_stages", "op": "<=",
                 "value": 2, "raw": "stages <= 2"}]


def test_parse_process_step_limit():
    c = CE.parse("steps <= 1")
    assert c[0] == {"kind": "process", "column": "max_steps", "op": "<=",
                    "value": 1, "raw": "steps <= 1"}


# ---- expand_chemical_constraints -------------------------------------------------
def _schema_with_roles(*roles):
    return {"columns": {
        role: {"prefix": role, "descriptor_columns": [f"{role}_Is_Strong_Acid",
                                                       f"{role}_Is_Oxidizer"]}
        for role in roles
    }}


def test_expand_chemical_class_across_all_matching_roles():
    schema = _schema_with_roles("Pretreat1", "Activator")
    parsed = CE.parse("NO STRONG ACID")
    expanded, notes = CE.expand_chemical_constraints(parsed, schema)
    cols = {e["column"] for e in expanded}
    assert cols == {"Pretreat1_Is_Strong_Acid", "Activator_Is_Strong_Acid"}
    assert all(e["op"] == "<=" and e["value"] == "0" for e in expanded)
    assert len(notes) == 1 and "Pretreat1" in notes[0] and "Activator" in notes[0]


def test_expand_chemical_class_no_matching_role_reports_no_effect():
    schema = {"columns": {}}
    parsed = CE.parse("NO STRONG BASE")
    expanded, notes = CE.expand_chemical_constraints(parsed, schema)
    assert expanded == []
    assert "no effect" in notes[0]


def test_expand_passes_through_column_constraints_unchanged():
    parsed = CE.parse("Temperature <= 900")
    expanded, notes = CE.expand_chemical_constraints(parsed, {"columns": {}})
    assert expanded == parsed
    assert notes == []


# ---- process_limits / satisfies_process ------------------------------------------
def test_process_limits_extracts_tightest():
    parsed = CE.parse("stages <= 2\nstages <= 1")
    limits = CE.process_limits(parsed)
    assert limits == {"max_stages": 1}


def test_satisfies_process_respects_limit():
    limits = {"max_stages": 1}
    ok_recipe = {"Py.1 temp. (oC)": 900}
    bad_recipe = {"Py.1 temp. (oC)": 900, "Py.2 temp. (oC)": 950}
    assert CE.satisfies_process(ok_recipe, limits)
    assert not CE.satisfies_process(bad_recipe, limits)


def test_satisfies_process_empty_limits_always_true():
    assert CE.satisfies_process({"anything": 1}, {})
    assert CE.satisfies_process({"anything": 1}, None)


# ---- apply: numeric / categorical (same contract as before) ---------------------
def test_apply_numeric_tightens_bounds():
    X_raw = pd.DataFrame({"Temperature": np.linspace(600, 1000, 50)})
    parsed = CE.parse("Temperature <= 800")
    overrides, notes = CE.apply(parsed, ["Temperature"], {}, X_raw, {})
    assert overrides["Temperature"][1] == 800.0
    assert notes


def test_apply_categorical_in_filters_choices():
    cat_choices = {"Biomass": ["Rice husk", "Bamboo", "Corn cob"]}
    parsed = CE.parse("Biomass IN [Rice husk, Bamboo]")
    X_raw = pd.DataFrame({"Biomass": ["Rice husk"]})
    overrides, notes = CE.apply(parsed, [], cat_choices, X_raw, {})
    assert cat_choices["Biomass"] == ["Rice husk", "Bamboo"]


def test_apply_unknown_column_reports_ignored():
    parsed = CE.parse("NotAKnob <= 5")
    X_raw = pd.DataFrame({"Temperature": [700, 800]})
    overrides, notes = CE.apply(parsed, ["Temperature"], {}, X_raw, {})
    assert overrides == {}
    assert "not a controllable knob" in notes[0]


def test_apply_skips_process_and_chemical_class_kinds():
    # apply() only consumes "column"-kind entries; process/chemical_class
    # entries must be handled by process_limits/expand_chemical_constraints
    # respectively, and are silently skipped here rather than mis-parsed.
    parsed = CE.parse("stages <= 2\nNO STRONG ACID\nTemperature <= 900")
    X_raw = pd.DataFrame({"Temperature": [700, 800, 900]})
    overrides, notes = CE.apply(parsed, ["Temperature"], {}, X_raw, {})
    assert "Temperature" in overrides
    assert len(notes) == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
