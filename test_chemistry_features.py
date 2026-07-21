"""Regression tests for chemistry-aware feature engineering."""

import math

import pandas as pd
import pytest
from sklearn.linear_model import Ridge

import chemistry_features as chemistry
import screening


def test_formula_parser_handles_parentheses_and_molecular_weight():
    assert chemistry.FormulaParser.parse("Fe(NO3)3") == {"Fe": 1, "N": 3, "O": 9}
    assert chemistry.FormulaParser.molecular_weight("HCl") == pytest.approx(36.46, abs=0.02)
    assert chemistry.FormulaParser.parse("not a formula") == {}


def test_lookup_hcl_generates_scientific_descriptors():
    descriptor = chemistry.DescriptorGenerator().describe("hydrochloric acid")
    values = descriptor.feature_values()
    assert descriptor.canonical_name == "HCl"
    assert descriptor.confidence == 1.0
    assert values["Is_Acid"] == 1.0
    assert values["Is_Strong_Acid"] == 1.0
    assert values["Is_Mineral_Acid"] == 1.0
    assert values["Is_Base"] == 0.0
    assert values["Contains_Chloride"] == 1.0
    assert values["MolecularWeight"] == pytest.approx(36.46, abs=0.02)
    assert values["Num_H"] == 1.0
    assert values["Num_Cl"] == 1.0
    assert descriptor.categorical["Halogen"] == "Chlorine"


def test_unlisted_formula_is_inferred_without_crashing():
    generator = chemistry.DescriptorGenerator()
    descriptor = generator.describe("HBr")
    similarities = generator.similarities("HBr", top=3)
    assert descriptor.source == "formula_heuristic"
    assert 0.0 < descriptor.confidence < 1.0
    assert descriptor.numeric["Is_Strong_Acid"] == 1.0
    assert descriptor.numeric["Num_Br"] == 1.0
    assert descriptor.categorical["ChemicalClass"] == "Strong acid"
    assert similarities[0].name == "HCl"
    assert 0.0 <= similarities[0].score <= 1.0


def test_feature_engineering_drops_label_but_retains_reporting_metadata():
    frame = pd.DataFrame({
        "Pretreat1_Chemical": ["HCl", "NaOH", "Unknown"],
        "Pretreat1_Molarity": [2.0, 1.5, 0.5],
        "PyrolysisTemperature": [700.0, 800.0, 750.0],
        "HoldingTime": [1.0, 2.0, 1.5],
    })
    expansion = chemistry.ChemistryFeatureEngineer().transform(frame)
    output = expansion.frame
    assert "Pretreat1_Chemical" not in output
    assert expansion.original_values["Pretreat1_Chemical"].tolist() == [
        "HCl", "NaOH", "Unknown"]
    assert output.loc[0, "Pretreat1_Is_Strong_Acid"] == 1.0
    assert output.loc[1, "Pretreat1_Is_Strong_Base"] == 1.0
    assert output.loc[0, "Pretreat1_StrongAcid_x_Molarity"] == pytest.approx(2.0)
    assert output.loc[1, "Pretreat1_StrongBase_x_Molarity"] == pytest.approx(1.5)
    assert "Pretreat1_Hydroxide_x_PyrolysisTemperature" in output
    assert {
        "feature": "Pretreat1_StrongAcid_x_Molarity",
        "left": "Pretreat1_Is_Strong_Acid",
        "right": "Pretreat1_Molarity",
    } in expansion.metadata["interaction_specs"]
    assert expansion.metadata["columns"]["Pretreat1_Chemical"]["observed_chemicals"] == [
        "HCl", "NaOH", "Unknown"]


def test_rdkit_unavailable_keeps_a_stable_fallback_schema():
    rdkit = chemistry.RDKitInterface()
    rdkit.available = False
    values = rdkit.describe("CCO")
    assert set(chemistry.RDKitInterface.DESCRIPTOR_DEFAULTS).issubset(values)
    assert all(f"MorganBit_{i:02d}" in values
               for i in range(chemistry.RDKitInterface.FINGERPRINT_BITS))
    assert math.isnan(values["TPSA"])


def test_pubchem_adapter_can_enrich_unknown_names_without_network_dependency():
    pubchem = chemistry.PubChemInterface(enabled=True)
    pubchem.lookup = lambda name: {
        "name": "Formic acid", "formula": "CH2O2", "smiles": "C(=O)O"}
    generator = chemistry.DescriptorGenerator(pubchem=pubchem)
    descriptor = generator.describe("formic acid")
    assert descriptor.source == "pubchem+formula_heuristic"
    assert descriptor.formula == "CH2O2"
    assert descriptor.numeric["MolecularWeight"] == pytest.approx(46.025, abs=0.02)
    assert descriptor.confidence == pytest.approx(0.84)


def test_manual_override_invalidates_cache_and_can_be_removed():
    generator = chemistry.DescriptorGenerator()
    assert generator.describe("HCl").numeric["pKa"] == pytest.approx(-6.3)
    generator.set_override("HCl", {"pKa": -5.9, "Corrosiveness": "Extreme"})
    edited = generator.describe("HCl")
    assert edited.numeric["pKa"] == pytest.approx(-5.9)
    assert edited.categorical["Corrosiveness"] == "Extreme"
    assert edited.confidence == 1.0
    generator.clear_override("HCl")
    assert generator.describe("HCl").numeric["pKa"] == pytest.approx(-6.3)


def test_nonchemical_categories_are_not_misclassified_as_reagents():
    frame = pd.DataFrame({
        "route": ["A", "B", "A", "B"],
        "material": ["Bamboo", "Wood", "Bamboo", "Wood"],
        "temperature": [1, 2, 3, 4],
    })
    assert chemistry.ChemistryFeatureEngineer().detect_columns(frame) == []
    assert chemistry.ChemicalLookup().formula_from_text("Bamboo") == ""


def test_screener_expands_reagent_only_candidate_and_flags_unseen_chemical():
    raw_train = pd.DataFrame({
        "Pretreat1_Chemical": ["HCl", "H2SO4", "NaOH", "KOH", "HCl", "NaOH"],
        "Pretreat1_Molarity": [1.0, 1.5, 1.0, 2.0, 2.5, 0.5],
        "PyrolysisTemperature": [650, 700, 750, 800, 850, 900],
    })
    expansion = chemistry.ChemistryFeatureEngineer().transform(raw_train)
    model_raw = expansion.frame.copy()
    numeric_schema, categorical_schema = {}, {}
    for column in model_raw:
        if pd.api.types.is_numeric_dtype(model_raw[column]):
            present = model_raw[column].dropna()
            median = present.median() if len(present) else math.nan
            numeric_schema[column] = 0.0 if pd.isna(median) else float(median)
            model_raw[column] = model_raw[column].fillna(numeric_schema[column])
        else:
            model_raw[column] = model_raw[column].fillna("Missing").astype(str)
            categorical_schema[column] = sorted(model_raw[column].unique().tolist())
    encoded = pd.get_dummies(model_raw, drop_first=True).astype(float)
    target = pd.DataFrame({"capacity": [100, 110, 90, 95, 125, 85]})
    model = Ridge().fit(encoded, target["capacity"])
    screener = screening.Screener(
        model, encoded.columns, numeric_schema, categorical_schema,
        ["capacity"], encoded, target, chemistry_schema=expansion.metadata,
        chemistry_originals=expansion.original_values,
    )

    candidate = {
        "Pretreat1_Chemical": "HBr",
        "Pretreat1_Molarity": 1.2,
        "PyrolysisTemperature": 725,
    }
    encoded_candidate = screener.encode(candidate)
    assert encoded_candidate.loc[0, "Pretreat1_Is_Strong_Acid"] == 1.0
    assert encoded_candidate.loc[0, "Pretreat1_Num_Br"] == 1.0
    evidence = screener.chemical_evidence(candidate)[0]
    assert evidence["original"] == "HBr"
    assert evidence["exactly_observed"] is False
    assert evidence["prediction_confidence_adjustment"] < 1.0
    nearest_conditions = screener.similar(encoded_candidate, k=1)[0]["conditions"]
    assert "Pretreat1_Chemical" in nearest_conditions
    assert "Pretreat1_MolecularWeight" not in nearest_conditions
    screener.contributions = lambda *args, **kwargs: {"base": 0.0, "contributions": []}
    screener.effect_summary = lambda *args, **kwargs: {
        "sensitivity": [], "effects": [], "partial_dependence": {}}
    result = screener.screen(candidate, "capacity", k_similar=1)
    assert any("exact chemical was not observed" in warning for warning in result["ood"])
    assert result["prediction"]["sigma"] > 0
    assert result["chemistry"][0]["summary"] == (
        "Known strong mineral acid. Descriptors inferred.")


def test_application_training_builder_uses_descriptors_and_keeps_originals(tmp_path):
    import app_imgui as app

    path = tmp_path / "chemistry_training.csv"
    pd.DataFrame({
        "Pretreat1_Chemical": ["HCl", "H2SO4", "NaOH", "KOH"] * 3,
        "Pretreat1_Molarity": [0.5, 1.0, 1.5, 2.0] * 3,
        "Temperature": list(range(650, 770, 10)),
        "capacity": list(range(100, 112)),
    }).to_csv(path, index=False)
    cfg = {
        "data_path": str(path), "ids": [], "mixed": [],
        "targets": ["capacity"], "all_target_columns": ["capacity"],
        "single_target_mode": True, "col_types": {}, "exclude": [],
        "sheet": None, "standardize_units": False,
        "validation": {
            "method": "random_kfold", "group_column": "", "n_splits": 3,
            "n_repeats": 1, "confidence_level": .95, "random_state": 42,
            "interval_method": "percentile",
        },
        "chemistry_enabled": True,
    }
    data = app.build_training_data(cfg)
    assert "Pretreat1_Chemical" not in data["X_raw"]
    assert "Pretreat1_Is_Strong_Acid" in data["X_raw"]
    assert not any("Chemical_HCl" in column for column in data["X_encoded"])
    assert data["chemistry_originals"]["Pretreat1_Chemical"].iloc[0] == "HCl"
    assert data["chemistry_schema"]["descriptor_feature_count"] > 20
    
