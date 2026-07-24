"""test_cost_model.py — user-editable cost/hazard database.

Run:  python -m pytest test_cost_model.py -q
"""

import os

import pytest

import cost_model as CM


def test_entry_defaults_to_unknown_not_a_guess():
    e = CM.ReagentCostEntry(name="KOH")
    assert e.cost_per_kg is None and e.cost_per_liter is None
    assert e.hazard_class == "Unknown"
    assert not e.has_cost


def test_recipe_cost_none_when_any_reagent_unpriced():
    db = CM.CostDatabase(path="unused.json")
    db.set("KOH", CM.ReagentCostEntry(name="KOH", cost_per_kg=5.0))
    result = db.recipe_cost({"KOH": 2.0, "NaOH": 1.0})   # NaOH never entered
    assert result["total_cost"] is None
    assert result["partial_cost"] == pytest.approx(10.0)
    assert result["unpriced"] == ["NaOH"]
    assert result["priced"] == ["KOH"]


def test_recipe_cost_complete_when_all_priced():
    db = CM.CostDatabase(path="unused.json")
    db.set("KOH", CM.ReagentCostEntry(name="KOH", cost_per_kg=5.0))
    db.set("NaOH", CM.ReagentCostEntry(name="NaOH", cost_per_liter=3.0))
    result = db.recipe_cost({"KOH": 2.0, "NaOH": 1.0})
    assert result["total_cost"] == pytest.approx(13.0)
    assert set(result["priced"]) == {"KOH", "NaOH"}


def test_recipe_hazard_tracks_max_and_unknown():
    db = CM.CostDatabase(path="unused.json")
    db.set("KOH", CM.ReagentCostEntry(name="KOH", hazard_class="High", corrosive=True))
    db.set("Cellulose", CM.ReagentCostEntry(name="Cellulose", hazard_class="None"))
    result = db.recipe_hazard(["KOH", "Cellulose", "Unlisted"])
    assert result["max_hazard"] == "High"
    assert result["corrosive"] is True
    assert result["unknown"] == ["Unlisted"]
    assert set(result["known"]) == {"KOH", "Cellulose"}


def test_coverage_fraction():
    db = CM.CostDatabase(path="unused.json")
    db.set("KOH", CM.ReagentCostEntry(name="KOH", cost_per_kg=1.0))
    assert db.coverage(["KOH", "NaOH"]) == pytest.approx(0.5)
    assert db.coverage([]) == 0.0


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "costs.json")
    db = CM.CostDatabase(path=path)
    db.set("KOH", CM.ReagentCostEntry(
        name="KOH", cost_per_kg=5.5, hazard_class="High",
        corrosive=True, notes="from supplier X"))
    db.save()
    assert os.path.exists(path)

    reloaded = CM.CostDatabase(path=path).load()
    e = reloaded.get("KOH")
    assert e is not None
    assert e.cost_per_kg == pytest.approx(5.5)
    assert e.hazard_class == "High"
    assert e.corrosive is True
    assert e.notes == "from supplier X"


def test_load_missing_file_is_empty_not_an_error(tmp_path):
    db = CM.CostDatabase(path=str(tmp_path / "nope.json")).load()
    assert db.entries == {}


def test_load_corrupt_file_is_empty_not_a_crash(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    db = CM.CostDatabase(path=str(path)).load()
    assert db.entries == {}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
