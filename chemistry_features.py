"""Chemistry-aware feature engineering for biomass activation workflows.

The engine is intentionally useful offline.  A curated, extensible lookup table
is supplemented by formula parsing and conservative chemical heuristics.  RDKit
is used lazily when installed, while a stable fallback schema keeps saved models
portable to machines without RDKit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import importlib.util
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote
from urllib.request import build_opener, HTTPErrorProcessor

import numpy as np
import pandas as pd


class _NoRaiseHTTPErrorProcessor(HTTPErrorProcessor):
    """Pass HTTP error responses (4xx/5xx) through instead of raising.

    A PubChem "compound not found" is a normal 404 response, not a real
    error — the default urllib opener raises HTTPError for it, which would
    otherwise trip a debugger's "raise exceptions" breakpoint on every
    single lookup miss. Genuine network failures (DNS, timeout, connection
    refused) still raise via URLError as usual.
    """

    def http_response(self, _request, response):
        return response

    https_response = http_response


_HTTP_OPENER = build_opener(_NoRaiseHTTPErrorProcessor)


UNKNOWN_TOKENS = {"", "unknown", "missing", "na", "n/a", "nan", "?", "unspecified"}
NONE_TOKENS = {"none", "nil", "not applied", "no additive", "no pretreatment", "--", "-"}

ATOMIC_WEIGHTS = {
    "H": 1.00794, "Li": 6.941, "B": 10.811, "C": 12.0107, "N": 14.0067,
    "O": 15.9994, "F": 18.9984, "Na": 22.9898, "Mg": 24.305, "Al": 26.9815,
    "Si": 28.0855, "P": 30.9738, "S": 32.065, "Cl": 35.453, "K": 39.0983,
    "Ar": 39.948, "Ca": 40.078, "Cr": 51.9961, "Mn": 54.938, "Fe": 55.845, "Co": 58.933,
    "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "Ag": 107.868,
    "I": 126.904, "Ba": 137.327,
}
METALS = {"Li", "Na", "K", "Mg", "Ca", "Ba", "Al", "Cr", "Mn", "Fe", "Co",
          "Ni", "Cu", "Zn", "Ag"}
TRANSITION_METALS = {"Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ag"}
ALKALI_METALS = {"Li", "Na", "K"}
ALKALINE_EARTH_METALS = {"Mg", "Ca", "Ba"}
HALOGENS = {"F": "Fluorine", "Cl": "Chlorine", "Br": "Bromine", "I": "Iodine"}


@dataclass(frozen=True)
class ChemicalRecord:
    """One easy-to-extend chemical knowledge-base entry."""

    name: str
    formula: str = ""
    aliases: tuple[str, ...] = ()
    smiles: str = ""
    pka: float | None = None
    pkb: float | None = None
    tags: frozenset[str] = frozenset()
    metal: str = "None"
    cation: str = "None"
    anion: str = "None"
    ionic_charge: float = 0.0
    oxidation_tendency: float = 0.0
    reduction_tendency: float = 0.0
    ionic_strength_class: str = "Low"
    corrosiveness: str = "Low"
    water_soluble: bool = True


@dataclass
class ChemicalDescriptor:
    """Stable descriptor result returned for every input, including unknowns."""

    original: str
    canonical_name: str
    formula: str
    source: str
    confidence: float
    numeric: dict[str, float] = field(default_factory=dict)
    categorical: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def feature_values(self) -> dict[str, Any]:
        return {**self.numeric, **self.categorical,
                "DescriptorConfidence": float(self.confidence)}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChemicalSimilarity:
    name: str
    score: float


@dataclass
class ChemistryExpansion:
    frame: pd.DataFrame
    metadata: dict[str, Any]
    original_values: pd.DataFrame


@dataclass(frozen=True)
class ChemicalColumnDetection:
    column: str
    confidence: float
    reason: str
    recognized_fraction: float
    unique_values: int


@dataclass
class ChemistryAutoConfig:
    mode: str = "automatic"
    detected_columns: list[str] = field(default_factory=list)
    selected_descriptor_keys: list[str] = field(default_factory=list)
    selected_categorical_keys: list[str] = field(default_factory=list)
    selected_interactions: list[str] = field(default_factory=list)
    retain_original_labels: bool = False
    collapse_rare_chemicals: bool = True
    rare_category_min_groups: int = 5
    drop_constant: bool = True
    near_constant_threshold: float = 0.99
    correlation_threshold: float = 0.95
    max_chemistry_features: int = 40
    rationale: list[str] = field(default_factory=list)


COMPACT_DESCRIPTOR_KEYS = (
    "Is_Acid", "Is_Base", "Is_Strong_Acid", "Is_Strong_Base",
    "Is_Oxidizer", "Is_Reducing_Agent", "Is_Transition_Metal_Salt",
    "Contains_Hydroxide", "Contains_Chloride", "Contains_Sulfate",
    "Contains_Nitrate", "Contains_Carbonate", "Organic", "Inorganic",
    "MolecularWeight", "pKa", "pKb", "EstimatedAcidity",
    "EstimatedBasicity", "EstimatedOxidationTendency",
    "EstimatedReductionTendency", "EstimatedIonicCharge",
    "DescriptorConfidence",
)

STANDARD_EXTRA_DESCRIPTOR_KEYS = (
    "Num_H", "Num_C", "Num_N", "Num_O", "Num_S", "Num_P",
    "HalogenCount", "Num_MetalAtoms", "WaterSoluble",
)

STANDARD_CATEGORICAL_KEYS = (
    "ChemicalClass", "AnionType", "CationType", "Metal", "Halogen",
    "IonicStrengthClass", "Corrosiveness",
)

DESCRIPTOR_FAMILIES: dict[str, tuple[str, ...]] = {
    "acid_base": (
        "Is_Acid", "Is_Base", "Is_Strong_Acid", "Is_Weak_Acid",
        "Is_Strong_Base", "Is_Weak_Base", "pKa", "pKb",
        "EstimatedAcidity", "EstimatedBasicity",
    ),
    "functional_groups": (
        "Contains_Hydroxide", "Contains_Chloride", "Contains_Sulfate",
        "Contains_Nitrate", "Contains_Carbonate", "Contains_Phosphate",
        "Contains_Fluoride", "Is_Carbonate", "Is_Bicarbonate",
        "Is_Chelating_Agent", "Is_Organic_Solvent",
    ),
    "elemental_composition": (
        "NumAtoms", "Num_H", "Num_C", "Num_N", "Num_O", "Num_S",
        "Num_P", "Num_F", "Num_Cl", "Num_Br", "Num_I",
        "Num_MetalAtoms", "HalogenCount", "Organic", "Inorganic",
    ),
    "metal_properties": (
        "Is_Transition_Metal_Salt", "Is_Alkali_Hydroxide",
        "Is_Alkaline_Earth_Hydroxide", "Is_Metal_Nitrate",
        "Is_Metal_Chloride", "Is_Metal_Sulfate", "EstimatedIonicCharge",
    ),
    "redox": (
        "Is_Oxidizer", "Is_Reducing_Agent", "EstimatedOxidationTendency",
        "EstimatedReductionTendency",
    ),
    "physical_properties": (
        "MolecularWeight", "WaterSoluble", "HBD", "HBA", "TPSA", "LogP",
    ),
    "categorical_identity": STANDARD_CATEGORICAL_KEYS,
    "rdkit_topology": (
        "RotatableBonds", "RingCount", "RDKitMolecularWeight",
        "HeavyAtomCount", "AromaticRingCount", "FractionCSP3", "BalabanJ",
    ),
    "morgan_fingerprint": tuple(f"MorganBit_{i:02d}" for i in range(16)),
    "interactions": (),
    "confidence": ("DescriptorConfidence",),
}

# Earlier entries win when two numerical descriptors are redundant.
DESCRIPTOR_PREFERENCE = (
    "Is_Strong_Acid", "Is_Strong_Base", "Is_Acid", "Is_Base",
    "Is_Oxidizer", "Is_Reducing_Agent", "Is_Transition_Metal_Salt",
    "Contains_Hydroxide", "Contains_Chloride", "Contains_Sulfate",
    "Contains_Nitrate", "Contains_Carbonate", "MolecularWeight", "pKa", "pKb",
    "EstimatedOxidationTendency", "EstimatedReductionTendency",
    "EstimatedIonicCharge", "DescriptorConfidence", "Organic", "Inorganic",
    "Num_C", "Num_O", "Num_N", "Num_H", "Num_S", "Num_P", "HalogenCount",
    "WaterSoluble", "EstimatedAcidity", "EstimatedBasicity",
)


def _norm(text: Any) -> str:
    value = str(text or "").strip().lower()
    value = value.replace("₂", "2").replace("₃", "3").replace("₄", "4")
    value = value.replace("·", "").replace("⋅", "")
    return re.sub(r"[^a-z0-9]", "", value)


def normalize_chemical_key(text: Any) -> str:
    return _norm(text)


def _tags(*values: str) -> frozenset[str]:
    return frozenset(values)


def _record(name: str, formula: str = "", aliases: Sequence[str] = (), **kwargs) -> ChemicalRecord:
    return ChemicalRecord(name=name, formula=formula, aliases=tuple(aliases), **kwargs)


# Scientific class labels are deliberately explicit.  Adding a new reagent is a
# single record rather than a change to the feature-engineering algorithm.
COMMON_CHEMICALS: tuple[ChemicalRecord, ...] = (
    _record("HCl", "HCl", ("hydrochloric acid",), pka=-6.3,
            tags=_tags("acid", "strong_acid", "mineral_acid", "chloride"),
            anion="Chloride", ionic_charge=1, ionic_strength_class="High", corrosiveness="High"),
    _record("H2SO4", "H2SO4", ("sulfuric acid", "sulphuric acid"), pka=-3.0,
            tags=_tags("acid", "strong_acid", "mineral_acid", "sulfate", "oxidizer"),
            anion="Sulfate", ionic_charge=2, oxidation_tendency=.55,
            ionic_strength_class="High", corrosiveness="Very high"),
    _record("HNO3", "HNO3", ("nitric acid",), pka=-1.4,
            tags=_tags("acid", "strong_acid", "mineral_acid", "nitrate", "oxidizer"),
            anion="Nitrate", ionic_charge=1, oxidation_tendency=.95,
            ionic_strength_class="High", corrosiveness="Very high"),
    _record("HF", "HF", ("hydrofluoric acid",), pka=3.17,
            tags=_tags("acid", "weak_acid", "mineral_acid", "fluoride"),
            anion="Fluoride", ionic_charge=1, ionic_strength_class="Medium", corrosiveness="Very high"),
    _record("H3PO4", "H3PO4", ("phosphoric acid",), pka=2.15,
            tags=_tags("acid", "weak_acid", "mineral_acid", "phosphate"),
            anion="Phosphate", ionic_charge=3, ionic_strength_class="Medium", corrosiveness="High"),
    _record("Acetic acid", "C2H4O2", ("ethanoic acid", "ch3cooh"), smiles="CC(=O)O", pka=4.76,
            tags=_tags("acid", "weak_acid", "organic_acid", "organic"), corrosiveness="Medium"),
    _record("Citric acid", "C6H8O7", ("citrate",), smiles="OC(=O)CC(O)(CC(O)=O)C(O)=O", pka=3.13,
            tags=_tags("acid", "weak_acid", "organic_acid", "organic", "chelating"), corrosiveness="Medium"),
    _record("Oxalic acid", "C2H2O4", ("ethanedioic acid",), smiles="OC(=O)C(O)=O", pka=1.25,
            tags=_tags("acid", "weak_acid", "organic_acid", "organic", "chelating", "reducing_agent"),
            reduction_tendency=.45, corrosiveness="High"),
    _record("NaOH", "NaOH", ("sodium hydroxide",), pkb=-1.7,
            tags=_tags("base", "strong_base", "hydroxide", "alkali_hydroxide"), metal="Sodium",
            cation="Sodium", anion="Hydroxide", ionic_charge=1, ionic_strength_class="High", corrosiveness="Very high"),
    _record("KOH", "KOH", ("potassium hydroxide",), pkb=-1.7,
            tags=_tags("base", "strong_base", "hydroxide", "alkali_hydroxide"), metal="Potassium",
            cation="Potassium", anion="Hydroxide", ionic_charge=1, ionic_strength_class="High", corrosiveness="Very high"),
    _record("LiOH", "LiOH", ("lithium hydroxide",), pkb=-.5,
            tags=_tags("base", "strong_base", "hydroxide", "alkali_hydroxide"), metal="Lithium",
            cation="Lithium", anion="Hydroxide", ionic_charge=1, ionic_strength_class="High", corrosiveness="High"),
    _record("NH4OH", "NH4OH", ("ammonium hydroxide", "ammonia solution"), pkb=4.75,
            tags=_tags("base", "weak_base", "hydroxide"), cation="Ammonium", anion="Hydroxide",
            ionic_charge=1, ionic_strength_class="Medium", corrosiveness="High"),
    _record("ZnCl2", "ZnCl2", ("zinc chloride",), tags=_tags("salt", "chloride", "metal_chloride", "transition_metal_salt"),
            metal="Zinc", cation="Zinc", anion="Chloride", ionic_charge=2, ionic_strength_class="High", corrosiveness="High"),
    _record("FeCl3", "FeCl3", ("ferric chloride", "iron iii chloride"), tags=_tags("salt", "chloride", "metal_chloride", "transition_metal_salt", "oxidizer"),
            metal="Iron", cation="Iron(III)", anion="Chloride", ionic_charge=3, oxidation_tendency=.65, ionic_strength_class="High", corrosiveness="High"),
    _record("Fe(NO3)3", "Fe(NO3)3", ("ferric nitrate", "iron iii nitrate"), tags=_tags("salt", "nitrate", "metal_nitrate", "transition_metal_salt", "oxidizer"),
            metal="Iron", cation="Iron(III)", anion="Nitrate", ionic_charge=3, oxidation_tendency=.8, ionic_strength_class="High", corrosiveness="High"),
    _record("MgCl2", "MgCl2", ("magnesium chloride",), tags=_tags("salt", "chloride", "metal_chloride"),
            metal="Magnesium", cation="Magnesium", anion="Chloride", ionic_charge=2, ionic_strength_class="High"),
    _record("CaCl2", "CaCl2", ("calcium chloride",), tags=_tags("salt", "chloride", "metal_chloride"),
            metal="Calcium", cation="Calcium", anion="Chloride", ionic_charge=2, ionic_strength_class="High"),
    _record("K2CO3", "K2CO3", ("potassium carbonate",), tags=_tags("salt", "carbonate", "base"),
            metal="Potassium", cation="Potassium", anion="Carbonate", pkb=3.7, ionic_charge=2, ionic_strength_class="High"),
    _record("Na2CO3", "Na2CO3", ("sodium carbonate",), tags=_tags("salt", "carbonate", "base"),
            metal="Sodium", cation="Sodium", anion="Carbonate", pkb=3.7, ionic_charge=2, ionic_strength_class="High"),
    _record("NaHCO3", "NaHCO3", ("sodium bicarbonate", "sodium hydrogen carbonate"), tags=_tags("salt", "bicarbonate", "weak_base"),
            metal="Sodium", cation="Sodium", anion="Bicarbonate", pkb=6.3, ionic_charge=1, ionic_strength_class="Medium"),
    _record("KHCO3", "KHCO3", ("potassium bicarbonate", "potassium hydrogen carbonate"), tags=_tags("salt", "bicarbonate", "weak_base"),
            metal="Potassium", cation="Potassium", anion="Bicarbonate", pkb=6.3, ionic_charge=1, ionic_strength_class="Medium"),
    _record("KCl", "KCl", ("potassium chloride",), tags=_tags("salt", "chloride"), metal="Potassium", cation="Potassium", anion="Chloride", ionic_charge=1, ionic_strength_class="High"),
    _record("NaCl", "NaCl", ("sodium chloride",), tags=_tags("salt", "chloride"), metal="Sodium", cation="Sodium", anion="Chloride", ionic_charge=1, ionic_strength_class="High"),
    _record("NH4Cl", "NH4Cl", ("ammonium chloride",), tags=_tags("salt", "chloride"), cation="Ammonium", anion="Chloride", ionic_charge=1, ionic_strength_class="High"),
    _record("Urea", "CH4N2O", ("carbamide",), smiles="NC(N)=O", tags=_tags("organic", "weak_base"), pkb=13.9),
    _record("Melamine", "C3H6N6", (), smiles="NC1=NC(N)=NC(N)=N1", tags=_tags("organic", "weak_base"), pkb=8.0, water_soluble=True),
    _record("Sucrose", "C12H22O11", ("table sugar",), tags=_tags("organic"), water_soluble=True),
    _record("Glucose", "C6H12O6", ("dextrose",), tags=_tags("organic", "reducing_agent"), reduction_tendency=.35),
    _record("Fructose", "C6H12O6", (), tags=_tags("organic", "reducing_agent"), reduction_tendency=.35),
    _record("PEG", "C2H4O", ("polyethylene glycol",), tags=_tags("organic", "polymer"), water_soluble=True),
    _record("PVP", "C6H9NO", ("polyvinylpyrrolidone",), tags=_tags("organic", "polymer"), water_soluble=True),
    _record("CTAB", "C19H42BrN", ("cetyltrimethylammonium bromide",), tags=_tags("organic", "surfactant", "bromide", "salt"), cation="Cetyltrimethylammonium", anion="Bromide", ionic_charge=1),
    _record("SDS", "C12H25NaO4S", ("sodium dodecyl sulfate",), tags=_tags("organic", "surfactant", "sulfate", "salt"), metal="Sodium", cation="Sodium", anion="Sulfate", ionic_charge=1),
    _record("Ethanol", "C2H6O", ("ethyl alcohol",), smiles="CCO", tags=_tags("organic", "organic_solvent", "reducing_agent"), reduction_tendency=.2),
    _record("Methanol", "CH4O", ("methyl alcohol",), smiles="CO", tags=_tags("organic", "organic_solvent", "reducing_agent"), reduction_tendency=.2),
    _record("IPA", "C3H8O", ("isopropanol", "isopropyl alcohol", "2-propanol"), smiles="CC(O)C", tags=_tags("organic", "organic_solvent", "reducing_agent"), reduction_tendency=.25),
    _record("Acetone", "C3H6O", ("propanone",), smiles="CC(=O)C", tags=_tags("organic", "organic_solvent"), water_soluble=True),
    _record("Water", "H2O", ("distilled water", "deionized water", "h2o"), smiles="O", tags=_tags("inorganic"), anion="None", water_soluble=True),
    _record("Steam", "H2O", ("water vapor", "water vapour"), tags=_tags("inorganic", "gas")),
    _record("CO2", "CO2", ("carbon dioxide",), tags=_tags("inorganic", "gas", "weak_acid"), water_soluble=True),
    _record("NH3", "NH3", ("ammonia",), pkb=4.75, tags=_tags("base", "weak_base", "inorganic", "gas"), water_soluble=True),
    _record("Argon", "Ar", ("ar",), tags=_tags("inorganic", "gas", "inert"), water_soluble=False),
    _record("Nitrogen", "N2", ("n2", "nitrogen gas"), tags=_tags("inorganic", "gas", "inert"), water_soluble=False),
    _record("Hydrogen", "H2", ("h2", "hydrogen gas"), tags=_tags("inorganic", "gas", "reducing_agent"), reduction_tendency=.8, water_soluble=False),
    _record("Air", "", (), tags=_tags("inorganic", "gas"), water_soluble=False),
    _record("Oxygen", "O2", ("o2", "oxygen gas"), tags=_tags("inorganic", "gas", "oxidizer"), oxidation_tendency=1.0, water_soluble=False),
    _record("Vacuum", "", (), tags=_tags("inorganic", "gas", "inert"), water_soluble=False),
    _record("None", "", tuple(NONE_TOKENS), tags=_tags("none"), water_soluble=False),
    _record("Unknown", "", tuple(UNKNOWN_TOKENS), tags=_tags("unknown"), water_soluble=False),
)


class FormulaParser:
    """Small cached molecular-formula parser with parenthesis support."""

    @staticmethod
    @lru_cache(maxsize=1024)
    def parse(formula: str) -> dict[str, int]:
        text = str(formula or "").strip().replace("[", "(").replace("]", ")")
        text = re.sub(r"(?:\^?[+-]\d*|\d*[+-])$", "", text)
        text = text.split("·", 1)[0].split(".", 1)[0]
        if not text:
            return {}

        def section(pos: int, end: str | None = None):
            counts: dict[str, int] = {}
            while pos < len(text):
                if end and text[pos] == end:
                    return counts, pos + 1
                if text[pos] == "(":
                    inner, pos = section(pos + 1, ")")
                    match = re.match(r"\d+", text[pos:])
                    mult = int(match.group()) if match else 1
                    pos += len(match.group()) if match else 0
                    for element, count in inner.items():
                        counts[element] = counts.get(element, 0) + count * mult
                    continue
                match = re.match(r"([A-Z][a-z]?)(\d*)", text[pos:])
                if not match:
                    return {}, len(text)
                element, raw_count = match.groups()
                if element not in ATOMIC_WEIGHTS:
                    return {}, len(text)
                counts[element] = counts.get(element, 0) + int(raw_count or 1)
                pos += len(match.group())
            return (counts, pos) if end is None else ({}, pos)

        parsed, position = section(0)
        return parsed if position == len(text) else {}

    @classmethod
    def molecular_weight(cls, formula: str) -> float:
        atoms = cls.parse(formula)
        return float(sum(ATOMIC_WEIGHTS[e] * n for e, n in atoms.items())) if atoms else math.nan


class ChemicalLookup:
    def __init__(self, records: Iterable[ChemicalRecord] = COMMON_CHEMICALS):
        self.records = tuple(records)
        self._aliases: dict[str, ChemicalRecord] = {}
        self._search_aliases: list[tuple[str, ChemicalRecord]] = []
        for record in self.records:
            for alias in (record.name, record.formula, *record.aliases):
                key = _norm(alias)
                if key:
                    self._aliases[key] = record
                    self._search_aliases.append((str(alias).strip(), record))

    def exact(self, value: Any) -> ChemicalRecord | None:
        return self._aliases.get(_norm(value))

    def find_in_text(self, value: Any) -> ChemicalRecord | None:
        text = str(value or "")
        hits = []
        for alias, record in self._search_aliases:
            if len(_norm(alias)) < 2:
                continue
            pattern = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
            if re.search(pattern, text, flags=re.I):
                hits.append((len(alias), record))
        return max(hits, key=lambda item: item[0])[1] if hits else None

    def formula_from_text(self, value: Any) -> str:
        text = str(value or "").strip()
        # Remove concentration prefixes, then prefer a standalone formula token.
        text = re.sub(r"^\s*[-+]?\d*\.?\d+\s*(?:mol\s*/\s*l|m|%\w*)\s*", "", text, flags=re.I)
        # Parse complete alphanumeric tokens only.  Substring matching would
        # incorrectly read words such as "Bamboo" as elemental Ba.
        candidates = re.findall(r"[A-Za-z][A-Za-z0-9()]*", text)
        candidates = sorted(candidates, key=len, reverse=True)
        return next((c for c in candidates if FormulaParser.parse(c)), "")


class RDKitInterface:
    """Optional molecular descriptor adapter; imports RDKit only when called."""

    FINGERPRINT_BITS = 16
    DESCRIPTOR_DEFAULTS = {
        "HBD": math.nan, "HBA": math.nan, "TPSA": math.nan, "LogP": math.nan,
        "RotatableBonds": math.nan, "RingCount": math.nan, "RDKitMolecularWeight": math.nan,
        "HeavyAtomCount": math.nan, "AromaticRingCount": math.nan,
        "FractionCSP3": math.nan, "BalabanJ": math.nan,
    }

    def __init__(self):
        # Check availability without importing — RDKit is an optional, heavy
        # dependency and this class must not raise just to detect its absence
        # (a raise-and-catch here fires debugger "on raised exception" breaks
        # every time the app starts on a machine without RDKit installed).
        self.available = importlib.util.find_spec("rdkit") is not None

    @lru_cache(maxsize=512)
    def describe(self, smiles: str) -> dict[str, float]:
        values = dict(self.DESCRIPTOR_DEFAULTS)
        values.update({f"MorganBit_{i:02d}": 0.0 for i in range(self.FINGERPRINT_BITS)})
        if not self.available or not smiles:
            return values
        try:
            from rdkit import Chem
            from rdkit.Chem import (Crippen, Descriptors, GraphDescriptors,
                                    Lipinski, rdMolDescriptors)
            from rdkit.Chem import AllChem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return values
            values.update({
                "HBD": float(Lipinski.NumHDonors(mol)),
                "HBA": float(Lipinski.NumHAcceptors(mol)),
                "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
                "LogP": float(Crippen.MolLogP(mol)),
                "RotatableBonds": float(Lipinski.NumRotatableBonds(mol)),
                "RingCount": float(Lipinski.RingCount(mol)),
                "RDKitMolecularWeight": float(Descriptors.MolWt(mol)),
                "HeavyAtomCount": float(Lipinski.HeavyAtomCount(mol)),
                "AromaticRingCount": float(Lipinski.NumAromaticRings(mol)),
                "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
                "BalabanJ": float(GraphDescriptors.BalabanJ(mol)),
            })
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2,
                                                       nBits=self.FINGERPRINT_BITS)
            for i, bit in enumerate(fp):
                values[f"MorganBit_{i:02d}"] = float(bit)
        except Exception:
            pass
        return values


class PubChemInterface:
    """Optional, cached PubChem name lookup for unrecognized chemicals.

    Network access is opt-in so dataset scans remain fast and private by
    default.  A failed request is indistinguishable from no match and never
    interrupts training or prediction.
    """

    def __init__(self, enabled: bool = False, timeout: float = 2.0):
        self.enabled = bool(enabled)
        self.timeout = float(timeout)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.lookup.cache_clear()

    @lru_cache(maxsize=256)
    def lookup(self, name: str) -> dict[str, str] | None:
        text = str(name or "").strip()
        if not self.enabled or len(text) < 2:
            return None
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{quote(text, safe='')}/property/Title,MolecularFormula,CanonicalSMILES/JSON"
        )
        try:
            # noqa: S310 - fixed PubChem host; _HTTP_OPENER never raises for a
            # plain 404 ("name not found" is a normal, expected response here).
            with _HTTP_OPENER.open(url, timeout=self.timeout) as response:
                if response.status != 200:
                    return None
                payload = json.loads(response.read().decode("utf-8"))
            item = payload["PropertyTable"]["Properties"][0]
            formula = str(item.get("MolecularFormula", "")).strip()
            if not FormulaParser.parse(formula):
                return None
            return {
                "name": str(item.get("Title") or text),
                "formula": formula,
                "smiles": str(item.get("ConnectivitySMILES")
                              or item.get("CanonicalSMILES") or ""),
            }
        except Exception:
            return None


class DescriptorGenerator:
    """Resolve names/formulas and convert them into stable feature dictionaries."""

    FLAG_TAGS = {
        "Is_Acid": "acid", "Is_Base": "base", "Is_Strong_Acid": "strong_acid",
        "Is_Weak_Acid": "weak_acid", "Is_Strong_Base": "strong_base",
        "Is_Weak_Base": "weak_base", "Is_Mineral_Acid": "mineral_acid",
        "Is_Organic_Acid": "organic_acid", "Is_Oxidizer": "oxidizer",
        "Is_Reducing_Agent": "reducing_agent", "Is_Chelating_Agent": "chelating",
        "Is_Transition_Metal_Salt": "transition_metal_salt",
        "Is_Alkali_Hydroxide": "alkali_hydroxide",
        "Is_Alkaline_Earth_Hydroxide": "alkaline_earth_hydroxide",
        "Is_Salt": "salt", "Is_Carbonate": "carbonate", "Is_Bicarbonate": "bicarbonate",
        "Is_Organic_Solvent": "organic_solvent", "Is_Metal_Nitrate": "metal_nitrate",
        "Is_Metal_Chloride": "metal_chloride", "Is_Metal_Sulfate": "metal_sulfate",
    }

    def __init__(self, lookup: ChemicalLookup | None = None,
                 rdkit: RDKitInterface | None = None,
                 pubchem: PubChemInterface | None = None):
        self.lookup = lookup or ChemicalLookup()
        self.rdkit = rdkit or RDKitInterface()
        self.pubchem = pubchem or PubChemInterface()
        self.overrides: dict[str, dict[str, Any]] = {}

    def set_pubchem_enabled(self, enabled: bool) -> None:
        self.pubchem.set_enabled(enabled)
        self.describe.cache_clear()

    def set_override(self, chemical: str, values: Mapping[str, Any]) -> None:
        self.overrides[_norm(chemical)] = dict(values)
        self.describe.cache_clear()

    def clear_overrides(self) -> None:
        self.overrides.clear()
        self.describe.cache_clear()

    def clear_override(self, chemical: str) -> None:
        self.overrides.pop(_norm(chemical), None)
        self.describe.cache_clear()

    def export_overrides(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self.overrides.items()}

    @lru_cache(maxsize=2048)
    def describe(self, value: str, allow_pubchem: bool = True) -> ChemicalDescriptor:
        raw = str(value if value is not None else "Unknown").strip()
        normalized = _norm(raw)
        formula_hint = ""
        if normalized in {_norm(x) for x in NONE_TOKENS}:
            record, source, confidence = self.lookup.exact("None"), "lookup", 1.0
        elif normalized in {_norm(x) for x in UNKNOWN_TOKENS}:
            record, source, confidence = self.lookup.exact("Unknown"), "unknown", 0.0
        else:
            record = self.lookup.exact(raw)
            formula_hint = "" if record else self.lookup.formula_from_text(raw)
            if not record and formula_hint:
                # A complete formula token is more reliable than substring
                # alias matching (Ca(OH)2 must not be mistaken for H2).
                record = self.lookup.exact(formula_hint)
            if not record and not formula_hint:
                record = self.lookup.find_in_text(raw)
            source, confidence = ("lookup", 1.0) if record else ("formula_heuristic", .68)
        if record is None:
            formula = formula_hint or self.lookup.formula_from_text(raw)
            remote = (self.pubchem.lookup(raw)
                     if (allow_pubchem and not formula) else None)
            if remote:
                record = replace(
                    self._infer_record(remote["name"], remote["formula"]),
                    name=remote["name"], smiles=remote.get("smiles", ""),
                )
                source, confidence = "pubchem+formula_heuristic", .84
            else:
                record = self._infer_record(raw, formula)
                if not formula:
                    source, confidence = "unknown_heuristic", .25
        descriptor = self._from_record(raw, record, source, confidence)
        override = self.overrides.get(normalized)
        if override:
            numeric, categorical = dict(descriptor.numeric), dict(descriptor.categorical)
            for key, value_override in override.items():
                if key in categorical or isinstance(value_override, str):
                    categorical[key] = str(value_override)
                else:
                    try:
                        numeric[key] = float(value_override)
                    except (TypeError, ValueError):
                        continue
            descriptor.numeric = numeric
            descriptor.categorical = categorical
            descriptor.confidence = 1.0
            descriptor.source += "+manual_override"
            descriptor.notes.append("One or more descriptors were manually overridden.")
        return descriptor

    def _infer_record(self, raw: str, formula: str) -> ChemicalRecord:
        atoms = FormulaParser.parse(formula)
        tags: set[str] = set()
        metal_symbols = [e for e in atoms if e in METALS]
        transition = [e for e in metal_symbols if e in TRANSITION_METALS]
        halogens = [e for e in atoms if e in HALOGENS]
        anion = "None"
        if "OH" in formula:
            tags.add("base"); tags.add("hydroxide")
            if metal_symbols:
                tags.add("strong_base")
            if any(e in ALKALI_METALS for e in metal_symbols):
                tags.add("alkali_hydroxide")
            if any(e in ALKALINE_EARTH_METALS for e in metal_symbols):
                tags.add("alkaline_earth_hydroxide")
            anion = "Hydroxide"
        if formula.startswith("H") and len(atoms) > 1:
            tags.update(("acid", "mineral_acid"))
            if formula in {"HCl", "HBr", "HI", "HNO3", "H2SO4"}:
                tags.add("strong_acid")
            else:
                tags.add("weak_acid")
        patterns = (("HCO3", "bicarbonate"), ("CO3", "carbonate"),
                    ("SO4", "sulfate"), ("NO3", "nitrate"),
                    ("PO4", "phosphate"), ("Cl", "chloride"),
                    ("Br", "bromide"), ("F", "fluoride"))
        for token, tag in patterns:
            if token in formula:
                tags.add(tag); anion = tag.title()
                break
        if metal_symbols and anion != "None" and "hydroxide" not in tags:
            tags.add("salt")
            if transition:
                tags.add("transition_metal_salt")
            if "chloride" in tags:
                tags.add("metal_chloride")
            if "nitrate" in tags:
                tags.add("metal_nitrate"); tags.add("oxidizer")
            if "sulfate" in tags:
                tags.add("metal_sulfate")
        organic = "C" in atoms and "H" in atoms
        tags.add("organic" if organic else "inorganic")
        metal = transition[0] if transition else (metal_symbols[0] if metal_symbols else "None")
        cation = metal
        pka = -6.0 if "strong_acid" in tags else (4.0 if "weak_acid" in tags else None)
        pkb = -1.0 if "strong_base" in tags else (5.0 if "weak_base" in tags else None)
        estimated_charge = 1.0 if tags & {"acid", "base", "salt"} else 0.0
        if metal_symbols:
            n_metals = max(1, sum(atoms.get(element, 0) for element in metal_symbols))
            if "hydroxide" in tags:
                charge_units = atoms.get("O", 0)
            elif tags & {"chloride", "bromide", "fluoride"}:
                charge_units = sum(atoms.get(element, 0) for element in HALOGENS)
            elif "nitrate" in tags:
                charge_units = atoms.get("N", 0)
            elif "sulfate" in tags:
                charge_units = 2 * atoms.get("S", 0)
            elif "carbonate" in tags:
                charge_units = 2 * atoms.get("C", 0)
            else:
                charge_units = n_metals
            estimated_charge = float(np.clip(round(charge_units / n_metals), 1, 4))
        return ChemicalRecord(
            name=formula or raw or "Unknown", formula=formula, pka=pka, pkb=pkb,
            tags=frozenset(tags), metal=metal, cation=cation, anion=anion,
            ionic_charge=estimated_charge,
            oxidation_tendency=.55 if "oxidizer" in tags else 0.0,
            ionic_strength_class="High" if tags & {"strong_acid", "strong_base", "salt"} else "Low",
            corrosiveness="High" if tags & {"strong_acid", "strong_base"} else "Low",
            water_soluble=bool(tags & {"acid", "base", "salt"}) or not organic,
        )

    def _from_record(self, raw: str, record: ChemicalRecord, source: str,
                     confidence: float) -> ChemicalDescriptor:
        atoms = FormulaParser.parse(record.formula)
        tags = set(record.tags)
        metal_atoms = sum(count for element, count in atoms.items() if element in METALS)
        halogen_count = sum(atoms.get(element, 0) for element in HALOGENS)
        halogen = next((name for element, name in HALOGENS.items() if atoms.get(element, 0)), "None")
        organic = "organic" in tags or (atoms.get("C", 0) > 0 and atoms.get("H", 0) > 0)
        absent = bool(tags & {"none", "unknown"})
        numeric = {name: float(tag in tags) for name, tag in self.FLAG_TAGS.items()}
        numeric.update({
            "Contains_Hydroxide": float("hydroxide" in tags or "OH" in record.formula),
            "Contains_Chloride": float("chloride" in tags or atoms.get("Cl", 0) > 0),
            "Contains_Sulfate": float("sulfate" in tags),
            "Contains_Nitrate": float("nitrate" in tags),
            "Contains_Carbonate": float("carbonate" in tags or "bicarbonate" in tags),
            "Contains_Phosphate": float("phosphate" in tags),
            "Contains_Fluoride": float("fluoride" in tags or atoms.get("F", 0) > 0),
            "pKa": float(record.pka) if record.pka is not None else math.nan,
            "pKb": float(record.pkb) if record.pkb is not None else math.nan,
            "MolecularWeight": FormulaParser.molecular_weight(record.formula),
            "NumAtoms": float(sum(atoms.values())),
            "Num_H": float(atoms.get("H", 0)), "Num_C": float(atoms.get("C", 0)),
            "Num_N": float(atoms.get("N", 0)), "Num_O": float(atoms.get("O", 0)),
            "Num_S": float(atoms.get("S", 0)), "Num_P": float(atoms.get("P", 0)),
            "Num_F": float(atoms.get("F", 0)), "Num_Cl": float(atoms.get("Cl", 0)),
            "Num_Br": float(atoms.get("Br", 0)), "Num_I": float(atoms.get("I", 0)),
            "Num_MetalAtoms": float(metal_atoms), "HalogenCount": float(halogen_count),
            "EstimatedIonicCharge": float(record.ionic_charge),
            "EstimatedAcidity": 1.0 if "strong_acid" in tags else .55 if "weak_acid" in tags else 0.0,
            "EstimatedBasicity": 1.0 if "strong_base" in tags else .55 if "weak_base" in tags else 0.0,
            "EstimatedOxidationTendency": float(record.oxidation_tendency),
            "EstimatedReductionTendency": float(record.reduction_tendency),
            "WaterSoluble": float(record.water_soluble),
            "Organic": float(organic and not absent),
            "Inorganic": float(not organic and not absent),
        })
        numeric.update(self.rdkit.describe(record.smiles))
        categorical = {
            "ChemicalClass": self._primary_class(tags),
            "Metal": record.metal or "None", "Halogen": halogen,
            "AnionType": record.anion or "None", "CationType": record.cation or "None",
            "IonicStrengthClass": record.ionic_strength_class,
            "Corrosiveness": record.corrosiveness,
        }
        notes = []
        if source.startswith("formula"):
            notes.append("Descriptors inferred from the molecular formula and chemical heuristics.")
        elif source.startswith("unknown"):
            notes.append("Chemical was not recognized; conservative fallback descriptors were used.")
        return ChemicalDescriptor(raw, record.name, record.formula, source,
                                  float(min(max(confidence, 0.0), 1.0)),
                                  numeric, categorical, notes)

    @staticmethod
    def _primary_class(tags: set[str]) -> str:
        for tag, label in (("strong_acid", "Strong acid"), ("weak_acid", "Weak acid"),
                           ("strong_base", "Strong base"), ("weak_base", "Weak base"),
                           ("transition_metal_salt", "Transition metal salt"),
                           ("organic_solvent", "Organic solvent"), ("salt", "Salt"),
                           ("gas", "Gas"), ("polymer", "Polymer"),
                           ("none", "None"), ("unknown", "Unknown")):
            if tag in tags:
                return label
        return "Organic compound" if "organic" in tags else "Inorganic compound"

    def similarities(self, value: str, top: int = 3,
                     allow_pubchem: bool = True) -> list[ChemicalSimilarity]:
        query = self.describe(str(value), allow_pubchem=allow_pubchem)
        q = self._similarity_vector(query)
        results = []
        for record in self.lookup.records:
            if record.name in {"None", "Unknown"}:
                continue
            other = self._from_record(record.name, record, "lookup", 1.0)
            v = self._similarity_vector(other)
            denom = np.linalg.norm(q) * np.linalg.norm(v)
            score = float(np.dot(q, v) / denom) if denom else 0.0
            if _norm(value) in {_norm(record.name), _norm(record.formula)}:
                score = 1.0
            results.append(ChemicalSimilarity(record.name, min(max(score, 0.0), 1.0)))
        return sorted(results, key=lambda item: item.score, reverse=True)[:top]

    @staticmethod
    def _similarity_vector(descriptor: ChemicalDescriptor) -> np.ndarray:
        keys = ["Is_Acid", "Is_Base", "Is_Strong_Acid", "Is_Strong_Base", "Is_Oxidizer",
                "Is_Reducing_Agent", "Is_Transition_Metal_Salt", "Contains_Hydroxide",
                "Contains_Chloride", "Contains_Sulfate", "Contains_Nitrate",
                "EstimatedAcidity", "EstimatedBasicity", "EstimatedOxidationTendency",
                "EstimatedReductionTendency", "Organic", "Inorganic"]
        values = [descriptor.numeric.get(k, 0.0) for k in keys]
        mw = descriptor.numeric.get("MolecularWeight", math.nan)
        pka = descriptor.numeric.get("pKa", math.nan)
        pkb = descriptor.numeric.get("pKb", math.nan)
        values.extend([
            0.0 if not np.isfinite(mw) else min(mw / 250.0, 2.0),
            0.0 if not np.isfinite(pka) else np.clip((pka + 10.0) / 25.0, 0.0, 1.0),
            0.0 if not np.isfinite(pkb) else np.clip((pkb + 5.0) / 25.0, 0.0, 1.0),
            descriptor.numeric.get("EstimatedIonicCharge", 0.0) / 3.0,
            min(descriptor.numeric.get("Num_O", 0.0) / 8.0, 1.0),
            min(descriptor.numeric.get("HalogenCount", 0.0) / 4.0, 1.0),
        ])
        return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)


class ChemistryFeatureEngineer:
    """Detect chemical columns, expand descriptors, and add targeted interactions."""

    COLUMN_HINTS = ("chemical", "reagent", "solute", "activator", "activation_agent",
                    "acid", "base", "electrolyte")

    # Internal parsed-part columns (numeric_feature_A*/group_label_B*/text_modifier_C*),
    # spreadsheet junk (Unnamed:*) and index/serial columns never hold a single
    # chemical identity — expanding them explodes the feature count for nothing.
    # Gas-atmosphere columns (Ar/N2/CO2 ...) are recognized chemicals but are a
    # process environment, not a dissolved solute — the acid/base/molarity
    # descriptors don't apply, so keep them as ordinary categoricals.
    _SKIP_COLUMN_RE = re.compile(
        r"(?i)^(?:numeric_feature_a|group_label_b|text_modifier_c|unnamed(?::|\b)"
        r"|index$|serial|atmosphere|gas\b)")

    # Cap on auto-detected chemical columns: a guardrail so a mis-recognition can
    # never expand dozens of columns (each adds ~86 descriptors).
    MAX_DETECTED_COLUMNS = 6

    def __init__(self, generator: DescriptorGenerator | None = None):
        self.generator = generator or DescriptorGenerator()

    def detect_column_details(self, frame: pd.DataFrame) -> list[ChemicalColumnDetection]:
        blankish = {_norm(t) for t in (*UNKNOWN_TOKENS, "none", "--", "---",
                                       "not applied", "nil", "n.a.")}
        detections: list[ChemicalColumnDetection] = []
        for column in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[column]):
                continue
            name = str(column)
            if self._SKIP_COLUMN_RE.match(name):
                continue
            lname = name.lower()
            # Only real (non-blank) values count toward recognition, so a column
            # full of '--'/'None' is not mistaken for a recognized chemical.
            raw_sample = frame[column].head(120)
            sample = [v for v in raw_sample.dropna().astype(str)
                      if _norm(v) and _norm(v) not in blankish]
            if not sample:
                continue
            descriptions = [self.generator.describe(v, allow_pubchem=False) for v in sample]
            recognized = float(np.mean([d.confidence >= 0.65 for d in descriptions]))
            formula_fraction = float(np.mean([
                bool(self.generator.lookup.formula_from_text(v)) for v in sample]))
            always = lname.startswith("solute_label_d")
            hinted = any(hint in lname for hint in self.COLUMN_HINTS)
            blank_fraction = 1.0 - len(sample) / max(len(raw_sample), 1)
            unique_values = int(pd.Series(sample).nunique())
            cardinality_penalty = .12 if unique_values > max(30, len(sample) * .7) else 0.0
            confidence = (1.0 if always else
                          .55 * recognized + .25 * float(hinted) +
                          .15 * formula_fraction + .05 -
                          .15 * blank_fraction - cardinality_penalty)
            confidence = float(np.clip(confidence, 0.0, 1.0))
            reasons = []
            if always:
                reasons.append("standardized parsed-solute column")
            if hinted:
                reasons.append("chemical column-name hint")
            reasons.append(f"{recognized:.0%} of sampled values recognized")
            if formula_fraction:
                reasons.append(f"{formula_fraction:.0%} formula-like")
            if blank_fraction > .4:
                reasons.append(f"{blank_fraction:.0%} blank/missing")
            detections.append(ChemicalColumnDetection(
                column=name, confidence=confidence, reason="; ".join(reasons),
                recognized_fraction=recognized, unique_values=unique_values))
        return sorted(detections, key=lambda item: item.confidence, reverse=True)

    def detect_columns(self, frame: pd.DataFrame) -> list[str]:
        return [item.column for item in self.detect_column_details(frame)
                if item.confidence >= .70][:self.MAX_DETECTED_COLUMNS]

    @staticmethod
    def _feature_budget(n_groups: int, nonchem_encoded_estimate: int = 0) -> int:
        if n_groups < 75:
            base = 15
        elif n_groups < 150:
            base = 25
        elif n_groups < 300:
            base = 40
        elif n_groups < 600:
            base = 60
        else:
            base = 100
        # Aim for at least two independent groups per total encoded predictor,
        # while reserving room for ordinary process variables.
        total_budget = max(base, n_groups // 2)
        available = max(5, total_budget - int(nonchem_encoded_estimate))
        return int(min(base, available))

    @staticmethod
    def _group_representative(frame: pd.DataFrame, groups: pd.Series | None) -> pd.DataFrame:
        data = pd.DataFrame(frame).reset_index(drop=True)
        if groups is None or len(groups) != len(data):
            return data
        marker = pd.Series(groups).reset_index(drop=True).astype(str)
        keep = ~marker.duplicated()
        return data.loc[keep].reset_index(drop=True)

    def _candidate_descriptor_frame(self, frame: pd.DataFrame,
                                    chemical_columns: Sequence[str]):
        columns: dict[str, list[Any]] = {}
        feature_to_key: dict[str, str] = {}
        for column in chemical_columns:
            prefix = self.prefix(column)
            descriptors = [self.generator.describe(str(value), allow_pubchem=False)
                           for value in frame[column]]
            if not descriptors:
                continue
            for key in descriptors[0].feature_values():
                feature = f"{prefix}_{key}"
                columns[feature] = [d.feature_values().get(key) for d in descriptors]
                feature_to_key[feature] = key
        return pd.DataFrame(columns, index=frame.index), feature_to_key

    @staticmethod
    def _nonchem_encoded_estimate(frame: pd.DataFrame,
                                  chemical_columns: Sequence[str]) -> int:
        total = 0
        for column in frame.columns:
            if column in chemical_columns:
                continue
            if pd.api.types.is_numeric_dtype(frame[column]):
                total += 1
            else:
                total += max(1, min(20, int(frame[column].dropna().astype(str).nunique()) - 1))
        return total

    def _quality_filter(self, candidates: pd.DataFrame, feature_to_key: Mapping[str, str],
                        groups: pd.Series | None, near_constant_threshold: float,
                        correlation_threshold: float):
        view = self._group_representative(candidates, groups)
        dropped = {"constant": [], "near_constant": [], "missing": [],
                   "duplicate": [], "correlated": [], "high_cardinality": []}
        allowed: list[str] = []
        signatures: dict[tuple, str] = {}
        max_categories = max(8, min(25, len(view) // 3 if len(view) else 8))
        preference = {key: i for i, key in enumerate(DESCRIPTOR_PREFERENCE)}
        ordered = sorted(candidates.columns,
                         key=lambda c: (preference.get(feature_to_key[c], 10_000), c))
        for feature in ordered:
            series = view[feature]
            if float(series.isna().mean()) > .60:
                dropped["missing"].append(feature); continue
            present = series.dropna()
            if present.nunique(dropna=True) <= 1:
                dropped["constant"].append(feature); continue
            top_fraction = float(present.astype(str).value_counts(normalize=True).iloc[0])
            if top_fraction > near_constant_threshold:
                dropped["near_constant"].append(feature); continue
            if not pd.api.types.is_numeric_dtype(series) and present.astype(str).nunique() > max_categories:
                dropped["high_cardinality"].append(feature); continue
            signature = tuple(pd.util.hash_pandas_object(
                present.reset_index(drop=True).astype(str), index=False).tolist())
            if signature in signatures:
                dropped["duplicate"].append(feature); continue
            signatures[signature] = feature
            allowed.append(feature)

        numeric_allowed = [c for c in allowed if pd.api.types.is_numeric_dtype(view[c])]
        kept_numeric: list[str] = []
        for feature in numeric_allowed:
            correlated_with = None
            a = pd.to_numeric(view[feature], errors="coerce")
            for previous in kept_numeric:
                b = pd.to_numeric(view[previous], errors="coerce")
                pair = pd.concat([a, b], axis=1).dropna()
                if (len(pair) < 3 or pair.iloc[:, 0].nunique() < 2
                        or pair.iloc[:, 1].nunique() < 2):
                    continue
                corr = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
                if pd.notna(corr) and abs(float(corr)) >= correlation_threshold:
                    correlated_with = previous
                    break
            if correlated_with:
                dropped["correlated"].append(feature)
            else:
                kept_numeric.append(feature)
        allowed = [c for c in allowed if c not in set(dropped["correlated"])]
        return allowed, dropped

    def _interaction_candidates(self, frame: pd.DataFrame,
                                chemical_columns: Sequence[str]):
        required = {"Is_Strong_Acid", "Is_Strong_Base", "Is_Oxidizer",
                    "Contains_Hydroxide", "Is_Acid", "Is_Transition_Metal_Salt"}
        descriptor_columns: dict[str, list[Any]] = {}
        metadata = {"columns": {}, "interactions": [], "interaction_specs": []}
        for column in chemical_columns:
            prefix = self.prefix(column)
            descriptors = [self.generator.describe(str(value), allow_pubchem=False)
                           for value in frame[column]]
            names = []
            for key in required:
                feature = f"{prefix}_{key}"
                descriptor_columns[feature] = [d.feature_values().get(key, 0.0)
                                               for d in descriptors]
                names.append(feature)
            metadata["columns"][column] = {"prefix": prefix, "descriptor_columns": names}
        result = pd.concat([frame, pd.DataFrame(descriptor_columns, index=frame.index)], axis=1)
        result = self._add_interactions(result, frame, chemical_columns, metadata)
        return result, metadata

    def _supported_interactions(self, frame: pd.DataFrame,
                                chemical_columns: Sequence[str], groups: pd.Series | None,
                                selected_keys: set[str], mode: str):
        candidate_frame, metadata = self._interaction_candidates(frame, chemical_columns)
        selected, omitted = [], []
        view = self._group_representative(candidate_frame, groups)
        limit = 3 if len(view) < 150 else 6 if len(view) < 300 else 12
        for spec in metadata["interaction_specs"]:
            left_key = spec["left"].split(metadata["columns"][next(
                c for c, info in metadata["columns"].items()
                if spec["left"].startswith(info["prefix"] + "_"))]["prefix"] + "_", 1)[-1]
            if mode != "full" and left_key not in selected_keys:
                omitted.append((spec["feature"], "required descriptor was not selected")); continue
            values = pd.to_numeric(view[spec["feature"]], errors="coerce")
            left = pd.to_numeric(view[spec["left"]], errors="coerce")
            right = pd.to_numeric(view[spec["right"]], errors="coerce")
            nonzero_groups = int(((values.notna()) & (values.abs() > 1e-12)).sum())
            supported = (left.nunique(dropna=True) >= 2 and right.nunique(dropna=True) >= 2
                         and values.nunique(dropna=True) >= 2 and nonzero_groups >= 10)
            if mode == "full" or supported:
                if len(selected) < limit or mode == "full":
                    selected.append(spec["feature"])
                else:
                    omitted.append((spec["feature"], f"group-count limit of {limit}"))
            else:
                omitted.append((spec["feature"],
                                f"only {nonzero_groups} independent groups had nonzero support"))
        return selected, omitted

    def auto_configure(self, frame: pd.DataFrame, chemical_columns: Sequence[str] | None = None,
                       groups: pd.Series | None = None, target: pd.Series | None = None,
                       requested_mode: str = "automatic") -> ChemistryAutoConfig:
        del target  # Deliberately unused: global target-informed selection would leak.
        source = pd.DataFrame(frame).copy()
        detections = self.detect_column_details(source)
        columns = list(chemical_columns or [d.column for d in detections
                                            if d.confidence >= .70][:self.MAX_DETECTED_COLUMNS])
        columns = [c for c in columns if c in source]
        mode = str(requested_mode or "automatic").lower().split(" ", 1)[0]
        if mode not in {"off", "automatic", "compact", "standard", "full", "custom"}:
            mode = "automatic"
        n_rows = len(source)
        n_groups = int(pd.Series(groups).nunique()) if groups is not None else n_rows
        budget = self._feature_budget(
            n_groups, self._nonchem_encoded_estimate(source, columns))
        rationale = [f"Configuration used {n_groups} independent groups ({n_rows} rows)."]
        if mode == "off" or not columns:
            rationale.append("Chemistry expansion is off." if mode == "off"
                             else "No chemical columns met the 0.70 detection threshold.")
            config = ChemistryAutoConfig(mode=mode, detected_columns=columns,
                                         retain_original_labels=True,
                                         max_chemistry_features=0, rationale=rationale)
            config._detections = detections
            config._diagnostics = {"candidate_count": 0, "selected_count": 0,
                                   "dropped_count": 0}
            return config

        candidates, feature_to_key = self._candidate_descriptor_frame(source, columns)
        allowed, dropped = self._quality_filter(
            candidates, feature_to_key, groups, .99, .95)
        all_numeric_keys = list(dict.fromkeys(
            key for feature, key in feature_to_key.items()
            if pd.api.types.is_numeric_dtype(candidates[feature])))
        all_categorical_keys = list(dict.fromkeys(
            key for feature, key in feature_to_key.items()
            if not pd.api.types.is_numeric_dtype(candidates[feature])))
        allowed_by_key = {key: [f for f in allowed if feature_to_key[f] == key]
                          for key in set(feature_to_key.values())}

        if mode == "full":
            selected_numeric, selected_categorical = all_numeric_keys, all_categorical_keys
            selected_feature_set = set(candidates.columns)
            drop_constant = False
            rationale.append("Full mode retained the legacy descriptor set.")
        else:
            if mode == "compact":
                requested = list(COMPACT_DESCRIPTOR_KEYS)
                categorical_requested: list[str] = []
            elif mode == "standard":
                requested = list(dict.fromkeys(
                    (*COMPACT_DESCRIPTOR_KEYS, *STANDARD_EXTRA_DESCRIPTOR_KEYS)))
                categorical_requested = list(STANDARD_CATEGORICAL_KEYS)
            elif mode == "custom":
                requested = list(COMPACT_DESCRIPTOR_KEYS)
                categorical_requested = []
            else:
                excluded_small = set(DESCRIPTOR_FAMILIES["morgan_fingerprint"])
                if n_groups < 300:
                    excluded_small.update(DESCRIPTOR_FAMILIES["rdkit_topology"])
                    excluded_small.update(("Num_F", "Num_Cl", "Num_Br", "Num_I"))
                    rationale.append(
                        "Morgan fingerprints and broad RDKit topology were disabled because "
                        "fewer than 300 independent groups are available.")
                requested = [key for key in dict.fromkeys(
                    (*DESCRIPTOR_PREFERENCE, *COMPACT_DESCRIPTOR_KEYS,
                     *STANDARD_EXTRA_DESCRIPTOR_KEYS, *all_numeric_keys))
                             if key not in excluded_small]
                categorical_requested = (list(STANDARD_CATEGORICAL_KEYS)
                                         if n_groups >= 300 else [])

            selected_numeric, selected_categorical = [], []
            selected_feature_set: set[str] = set()
            reserve = 3 if n_groups < 150 else 6 if n_groups < 300 else 10
            descriptor_budget = budget if mode in {"compact", "standard", "custom"} \
                else max(1, budget - reserve)
            for key in requested:
                additions = [f for f in allowed_by_key.get(key, [])
                             if f not in selected_feature_set]
                if not additions:
                    continue
                if mode == "automatic" and len(selected_feature_set) + len(additions) > descriptor_budget:
                    continue
                selected_numeric.append(key)
                selected_feature_set.update(additions)
            for key in categorical_requested:
                additions = [f for f in allowed_by_key.get(key, [])
                             if f not in selected_feature_set]
                if additions and (mode != "automatic" or
                                  len(selected_feature_set) + len(additions) <= descriptor_budget):
                    selected_categorical.append(key)
                    selected_feature_set.update(additions)
            drop_constant = True

        selected_interactions, omitted_interactions = self._supported_interactions(
            source, columns, groups, set(selected_numeric), mode)
        if mode == "automatic":
            available_for_interactions = max(0, budget - len(selected_feature_set))
            if len(selected_interactions) > available_for_interactions:
                omitted_interactions.extend(
                    (name, "automatic chemistry feature budget")
                    for name in selected_interactions[available_for_interactions:])
            selected_interactions = selected_interactions[:available_for_interactions]

        semantic = {_norm(v) for v in (*UNKNOWN_TOKENS, *NONE_TOKENS)}
        rare_by_column: dict[str, list[str]] = {}
        total_common = 0
        group_values = (pd.Series(groups).reset_index(drop=True).astype(str)
                        if groups is not None and len(groups) == len(source)
                        else pd.Series(np.arange(len(source))).astype(str))
        for column in columns:
            counts = (pd.DataFrame({"chemical": source[column].astype(str).reset_index(drop=True),
                                    "group": group_values})
                      .drop_duplicates().groupby("chemical")["group"].nunique())
            rare = [str(name) for name, count in counts.items()
                    if count < 5 and _norm(name) not in semantic]
            rare_by_column[column] = rare
            total_common += int(sum(count >= 5 and _norm(name) not in semantic
                                    for name, count in counts.items()))
        retain_labels = bool(total_common and n_groups >= 10 * total_common)
        if mode in {"automatic", "compact"} and not retain_labels:
            rationale.append("Rare chemical identities were replaced by transferable descriptors.")
        if mode == "full":
            retain_labels = False  # Preserve the previous Full transform behavior.
        if mode == "automatic":
            rationale.insert(0,
                f"Compact chemistry representation selected because only {n_groups} "
                "independent synthesis conditions are available.")
        rationale.append(
            f"{len(selected_interactions)} chemically meaningful interaction(s) passed "
            "variation and independent-group support checks.")
        if omitted_interactions:
            rationale.append(f"{len(omitted_interactions)} interaction candidate(s) were omitted.")

        config = ChemistryAutoConfig(
            mode=mode, detected_columns=columns,
            selected_descriptor_keys=list(dict.fromkeys(selected_numeric)),
            selected_categorical_keys=list(dict.fromkeys(selected_categorical)),
            selected_interactions=selected_interactions,
            retain_original_labels=retain_labels, collapse_rare_chemicals=retain_labels,
            rare_category_min_groups=5, drop_constant=drop_constant,
            near_constant_threshold=.99, correlation_threshold=.95,
            max_chemistry_features=budget, rationale=rationale)
        config._allowed_descriptor_features = selected_feature_set
        config._rare_categories_by_column = rare_by_column
        config._detections = detections
        config._diagnostics = {
            "candidate_count": int(candidates.shape[1]),
            "selected_count": int(len(selected_feature_set) + len(selected_interactions)),
            "dropped_count": int(candidates.shape[1] - len(selected_feature_set)),
            "dropped_constant": dropped["constant"],
            "dropped_near_constant": dropped["near_constant"],
            "dropped_missing": dropped["missing"],
            "dropped_duplicate": dropped["duplicate"],
            "dropped_correlated": dropped["correlated"],
            "dropped_high_cardinality": dropped["high_cardinality"],
            "omitted_interactions": omitted_interactions,
            "n_rows": n_rows, "n_groups": n_groups,
        }
        return config

    @staticmethod
    def prefix(column: str) -> str:
        name = re.sub(r"(?i)(?:[_\s-]*(?:chemical|reagent|solute))$", "", str(column)).strip(" _-")
        return re.sub(r"[^A-Za-z0-9_]+", "_", name) or "Chemical"

    def transform(self, frame: pd.DataFrame, chemical_columns: Sequence[str] | None = None,
                  include_original: bool | None = None,
                  add_interactions: bool | None = None,
                  config: ChemistryAutoConfig | Mapping | None = None) -> ChemistryExpansion:
        source = pd.DataFrame(frame).copy()
        legacy_full = config is None
        if isinstance(config, Mapping):
            fields = ChemistryAutoConfig.__dataclass_fields__
            config = ChemistryAutoConfig(**{k: v for k, v in dict(config).items() if k in fields})
        if config is None:
            detected = list(chemical_columns or self.detect_columns(source))
            example = self.generator.describe("HCl")
            config = ChemistryAutoConfig(
                mode="full", detected_columns=detected,
                selected_descriptor_keys=list(example.numeric) + ["DescriptorConfidence"],
                selected_categorical_keys=list(example.categorical),
                retain_original_labels=False, drop_constant=False,
                max_chemistry_features=10_000,
                rationale=["Legacy Full transform requested."],
            )
        chemical_columns = list(chemical_columns or config.detected_columns)
        chemical_columns = [c for c in chemical_columns if c in source.columns]
        originals = source[chemical_columns].copy() if chemical_columns else pd.DataFrame(index=source.index)
        if config.mode == "off" or not chemical_columns:
            metadata = {
                "mode": config.mode, "columns": {}, "detected_columns": chemical_columns,
                "rdkit_available": self.generator.rdkit.available,
                "interactions": [], "interaction_specs": [],
                "candidate_descriptor_count": 0, "selected_descriptor_count": 0,
                "descriptor_feature_count": 0, "feature_budget": config.max_chemistry_features,
                "rationale": list(config.rationale), "retain_original_labels": True,
                "chemistry_config": asdict(config),
            }
            return ChemistryExpansion(source.copy(), metadata, originals)

        include_original = (config.retain_original_labels if include_original is None
                            else bool(include_original))
        add_interactions = ((True if legacy_full else bool(config.selected_interactions))
                            if add_interactions is None else bool(add_interactions))
        selected_keys = list(dict.fromkeys(
            (*config.selected_descriptor_keys, *config.selected_categorical_keys)))
        allowed_features = getattr(config, "_allowed_descriptor_features", None)
        rare_by_column = getattr(config, "_rare_categories_by_column", {})
        diagnostics = dict(getattr(config, "_diagnostics", {}))
        metadata: dict[str, Any] = {
            "mode": config.mode, "detected_columns": chemical_columns,
            "columns": {}, "rdkit_available": self.generator.rdkit.available,
            "interactions": [], "interaction_specs": [],
            "selected_interactions": list(config.selected_interactions),
            "descriptor_feature_count": 0,
            "candidate_descriptor_count": diagnostics.get(
                "candidate_count", len(selected_keys) * len(chemical_columns)),
            "selected_descriptor_count": 0,
            "dropped_constant": diagnostics.get("dropped_constant", []),
            "dropped_near_constant": diagnostics.get("dropped_near_constant", []),
            "dropped_correlated": diagnostics.get("dropped_correlated", []),
            "dropped_missing": diagnostics.get("dropped_missing", []),
            "dropped_duplicate": diagnostics.get("dropped_duplicate", []),
            "omitted_interactions": diagnostics.get("omitted_interactions", []),
            "feature_budget": config.max_chemistry_features,
            "rationale": list(config.rationale),
            "retain_original_labels": include_original,
            "chemistry_config": asdict(config),
        }
        # Accumulate every descriptor column and add them all at once — assigning
        # them one by one fragments the DataFrame (pandas PerformanceWarning).
        descriptor_columns: dict[str, list] = {}
        drop_columns: list[str] = []
        for column in chemical_columns:
            prefix = self.prefix(column)
            descriptors = [self.generator.describe(str(value), allow_pubchem=False)
                           for value in source[column]]
            feature_names: list[str] = []
            descriptor_keys: list[str] = []
            if descriptors:
                values0 = descriptors[0].feature_values()
                for key in selected_keys:
                    if key not in values0:
                        continue
                    feature = f"{prefix}_{key}"
                    if allowed_features is not None and feature not in allowed_features:
                        continue
                    values = pd.Series([d.feature_values().get(key) for d in descriptors],
                                       index=source.index)
                    if config.drop_constant and allowed_features is None:
                        present = values.dropna()
                        if present.nunique(dropna=True) <= 1:
                            metadata["dropped_constant"].append(feature); continue
                        if (not present.empty and
                                present.astype(str).value_counts(normalize=True).iloc[0]
                                > config.near_constant_threshold):
                            metadata["dropped_near_constant"].append(feature); continue
                        if float(values.isna().mean()) > .60:
                            metadata["dropped_missing"].append(feature); continue
                    descriptor_columns[feature] = values.tolist()
                    feature_names.append(feature)
                    descriptor_keys.append(key)
            metadata["columns"][column] = {
                "prefix": prefix,
                "descriptor_columns": feature_names,
                "descriptor_keys": descriptor_keys,
                "observed_chemicals": sorted(set(map(str, source[column].dropna().unique()))),
                "mean_confidence": float(np.mean([d.confidence for d in descriptors])) if descriptors else 0.0,
                "rare_categories": list(rare_by_column.get(column, [])),
            }
            if not include_original:
                drop_columns.append(column)
            elif config.collapse_rare_chemicals and rare_by_column.get(column):
                source[column] = source[column].where(
                    ~source[column].astype(str).isin(rare_by_column[column]), "Other")
            metadata["descriptor_feature_count"] += len(feature_names)

        base = source.drop(columns=drop_columns) if drop_columns else source
        if descriptor_columns:
            # Overwrite (not duplicate) any pre-existing same-named columns, e.g.
            # when re-expanding a row that already carries descriptor columns.
            overlap = [c for c in base.columns if c in descriptor_columns]
            if overlap:
                base = base.drop(columns=overlap)
            result = pd.concat(
                [base, pd.DataFrame(descriptor_columns, index=source.index)], axis=1)
        else:
            result = base.copy()

        if add_interactions and chemical_columns:
            allowed_interactions = (None if legacy_full else set(config.selected_interactions))
            result = self._add_interactions(
                result, source, chemical_columns, metadata,
                allowed_interactions=allowed_interactions)
        metadata["selected_descriptor_count"] = metadata["descriptor_feature_count"]
        metadata["selected_interactions"] = list(metadata["interactions"])
        metadata["chemistry_feature_diagnostics"] = {
            "candidate_count": metadata["candidate_descriptor_count"],
            "selected_count": metadata["descriptor_feature_count"] + len(metadata["interactions"]),
            "dropped_count": max(0, metadata["candidate_descriptor_count"] -
                                 metadata["descriptor_feature_count"]),
            "groups_per_chemistry_feature": (
                diagnostics.get("n_groups", len(source)) /
                max(1, metadata["descriptor_feature_count"] + len(metadata["interactions"]))),
            "rationale": list(config.rationale),
        }
        return ChemistryExpansion(result, metadata, originals)

    def _add_interactions(self, result: pd.DataFrame, source: pd.DataFrame,
                          chemical_columns: Sequence[str], metadata: dict[str, Any],
                          allowed_interactions: set[str] | None = None) -> pd.DataFrame:
        numeric = [c for c in source.columns if pd.api.types.is_numeric_dtype(source[c])]
        temperatures = [c for c in numeric if re.search(r"temp|pyro", str(c), re.I)]
        times = [c for c in numeric if re.search(r"holding|hold.*time|time", str(c), re.I)]
        # Collect interaction columns and add them in one concat (avoids the
        # DataFrame-fragmentation PerformanceWarning from per-column inserts).
        additions: dict[str, pd.Series] = {}
        for column in chemical_columns:
            info = metadata["columns"][column]
            prefix = info["prefix"]
            suffix = re.search(r"solute_label_D(\d*)$", column)
            molarity_candidates = []
            if suffix:
                candidate = f"numeric_feature_A{suffix.group(1)}"
                if candidate in source:
                    molarity_candidates.append(candidate)
            root = _norm(prefix)
            molarity_candidates += [c for c in numeric if
                                     ("molar" in str(c).lower() or "concentration" in str(c).lower())
                                     and (root in _norm(c) or not root)]
            molarity = molarity_candidates[0] if molarity_candidates else None

            def interaction(flag: str, companion: str | None, label: str):
                feature = f"{prefix}_{label}"
                flag_column = f"{prefix}_{flag}"
                if allowed_interactions is not None and feature not in allowed_interactions:
                    return
                if companion and flag_column in result and companion in source:
                    values = pd.to_numeric(source[companion], errors="coerce")
                    additions[feature] = pd.to_numeric(result[flag_column], errors="coerce") * values
                    metadata["interactions"].append(feature)
                    metadata["interaction_specs"].append({
                        "feature": feature, "left": flag_column, "right": companion,
                    })

            interaction("Is_Strong_Acid", molarity, "StrongAcid_x_Molarity")
            interaction("Is_Strong_Base", molarity, "StrongBase_x_Molarity")
            interaction("Is_Oxidizer", molarity, "Oxidizer_x_Molarity")
            interaction("Contains_Hydroxide", temperatures[0] if temperatures else None,
                        "Hydroxide_x_PyrolysisTemperature")
            interaction("Is_Acid", times[0] if times else None, "Acid_x_HoldingTime")
            interaction("Is_Transition_Metal_Salt", temperatures[0] if temperatures else None,
                        "TransitionMetal_x_Temperature")

        if additions:
            overlap = [c for c in result.columns if c in additions]
            if overlap:
                result = result.drop(columns=overlap)
            return pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1)
        return result

    def transform_with_schema(self, frame: pd.DataFrame,
                              chemistry_schema: Mapping[str, Any]) -> pd.DataFrame:
        """Apply exactly a trained chemistry schema; never auto-configure here."""
        schema = dict(chemistry_schema or {})
        source = pd.DataFrame(frame).copy()
        descriptor_columns: dict[str, list[Any]] = {}
        drop_columns: list[str] = []
        retain = bool(schema.get("retain_original_labels", False))
        for column, info in schema.get("columns", {}).items():
            if column not in source:
                source[column] = "Unknown"
            prefix = info.get("prefix", self.prefix(column))
            keys = list(info.get("descriptor_keys", []))
            if not keys:
                marker = prefix + "_"
                keys = [name[len(marker):] for name in info.get("descriptor_columns", [])
                        if str(name).startswith(marker)]
            descriptors = [self.generator.describe(str(value), allow_pubchem=False)
                           for value in source[column]]
            expected = set(info.get("descriptor_columns", []))
            for key in keys:
                feature = f"{prefix}_{key}"
                if expected and feature not in expected:
                    continue
                descriptor_columns[feature] = [d.feature_values().get(key) for d in descriptors]
            if retain:
                rare = set(map(str, info.get("rare_categories", [])))
                if rare:
                    source[column] = source[column].where(
                        ~source[column].astype(str).isin(rare), "Other")
            else:
                drop_columns.append(column)
        base = source.drop(columns=drop_columns, errors="ignore")
        overlap = [c for c in base if c in descriptor_columns]
        if overlap:
            base = base.drop(columns=overlap)
        result = (pd.concat([base, pd.DataFrame(descriptor_columns, index=source.index)], axis=1)
                  if descriptor_columns else base.copy())
        additions = {}
        for spec in schema.get("interaction_specs", []):
            left = spec.get("left")
            right = spec.get("right")
            if left in result and right in source:
                additions[spec["feature"]] = (
                    pd.to_numeric(result[left], errors="coerce") *
                    pd.to_numeric(source[right], errors="coerce"))
        if additions:
            result = result.drop(columns=[c for c in additions if c in result], errors="ignore")
            result = pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1)
        return result

    def expand_row(self, raw: Mapping[str, Any], chemistry_schema: Mapping[str, Any]) -> dict[str, Any]:
        expanded = self.transform_with_schema(pd.DataFrame([dict(raw)]), chemistry_schema)
        return expanded.iloc[0].to_dict()

    def estimate_feature_counts(self, frame: pd.DataFrame, config: ChemistryAutoConfig,
                                groups: pd.Series | None = None) -> dict[str, Any]:
        source = pd.DataFrame(frame)
        expansion = self.transform(source, config=config)
        chemistry_features = {
            c for info in expansion.metadata.get("columns", {}).values()
            for c in info.get("descriptor_columns", [])}
        chemistry_features.update(expansion.metadata.get("interactions", []))
        numerical_chemistry = sum(
            c in expansion.frame and pd.api.types.is_numeric_dtype(expansion.frame[c])
            for c in chemistry_features)
        categorical_dummy = 0
        total_encoded = 0
        for column in expansion.frame:
            if pd.api.types.is_numeric_dtype(expansion.frame[column]):
                total_encoded += 1
            else:
                levels = int(expansion.frame[column].dropna().astype(str).nunique())
                dummies = max(1, levels - 1)
                total_encoded += dummies
                if column in chemistry_features:
                    categorical_dummy += dummies
        n_groups = int(pd.Series(groups).nunique()) if groups is not None else len(source)
        ratio = n_groups / max(total_encoded, 1)
        risk = "Good" if ratio >= 5 else "Caution" if ratio >= 1 else "High risk"
        return {
            "original_predictors": int(source.shape[1]),
            "estimated_numeric_descriptors": int(numerical_chemistry),
            "estimated_categorical_dummy_columns": int(categorical_dummy),
            "estimated_total_encoded_predictors": int(total_encoded),
            "independent_groups": n_groups,
            "groups_per_encoded_predictor": float(ratio),
            "risk": risk,
            "chemistry_features": int(len(chemistry_features)),
            "expansion": expansion,
        }

    def knowledge_for_values(self, values: Iterable[Any],
                             allow_pubchem: bool = False) -> list[dict[str, Any]]:
        """Describe every distinct value (bulk scan).

        ``allow_pubchem`` defaults to False here specifically: this method
        iterates every observed chemical in a dataset, potentially dozens at
        once, and runs synchronously on the caller's thread (including the
        GUI's render thread). Allowing PubChem here would mean one sequential
        blocking network round-trip per unrecognized name, freezing the whole
        app for the total time. A single deliberate lookup still goes through
        ``DescriptorGenerator.describe()`` directly, which defaults to True.
        """
        rows = []
        for value in sorted(set(map(str, values))):
            descriptor = self.generator.describe(value, allow_pubchem=allow_pubchem)
            rows.append({
                "original": value, "descriptor": descriptor,
                "similarities": self.generator.similarities(
                    value, top=3, allow_pubchem=allow_pubchem),
            })
        return rows

    def nearest_known_profile(self, values: Mapping[str, Any], prefix: str,
                              top: int = 3) -> list[ChemicalSimilarity]:
        """Map an optimized descriptor profile back to feasible known chemicals."""
        numeric = {}
        categorical = {}
        marker = prefix + "_"
        for key, value in values.items():
            if not str(key).startswith(marker):
                continue
            descriptor_key = str(key)[len(marker):]
            if isinstance(value, str):
                categorical[descriptor_key] = value
            else:
                try:
                    numeric[descriptor_key] = float(value)
                except (TypeError, ValueError):
                    pass
        query = ChemicalDescriptor("optimized profile", "optimized profile", "",
                                   "optimizer", .5, numeric, categorical)
        q = self.generator._similarity_vector(query)
        ranked = []
        for record in self.generator.lookup.records:
            if record.name in {"None", "Unknown"}:
                continue
            descriptor = self.generator._from_record(record.name, record, "lookup", 1.0)
            v = self.generator._similarity_vector(descriptor)
            denom = np.linalg.norm(q) * np.linalg.norm(v)
            score = float(np.dot(q, v) / denom) if denom else 0.0
            ranked.append(ChemicalSimilarity(record.name, min(max(score, 0.0), 1.0)))
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:top]


def descriptor_display_name(feature: str) -> str:
    """Readable chemistry labels for importance, SHAP, charts, and reports."""
    text = str(feature)
    replacements = {
        "Is_Strong_Acid": "Strong acid", "Is_Weak_Acid": "Weak acid",
        "Is_Strong_Base": "Strong base", "Is_Weak_Base": "Weak base",
        "Is_Acid": "Acid", "Is_Base": "Base", "MolecularWeight": "Molecular weight",
        "Contains_Chloride": "Contains chloride", "Contains_Hydroxide": "Hydroxide",
        "Contains_Sulfate": "Contains sulfate", "Contains_Nitrate": "Contains nitrate",
        "Is_Transition_Metal_Salt": "Transition metal salt",
        "DescriptorConfidence": "Descriptor confidence",
        "StrongAcid_x_Molarity": "Strong acid x molarity",
        "StrongBase_x_Molarity": "Strong base x molarity",
        "Oxidizer_x_Molarity": "Oxidizer x molarity",
        "Hydroxide_x_PyrolysisTemperature": "Hydroxide x pyrolysis temperature",
        "Acid_x_HoldingTime": "Acid x holding time",
        "TransitionMetal_x_Temperature": "Transition metal x temperature",
    }
    for token, label in replacements.items():
        if text.endswith("_" + token) or text == token:
            prefix = text[:-(len(token) + 1)].replace("_", " ").strip()
            return f"{prefix}: {label}" if prefix else label
    text = text.replace("_", " ").strip()
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)


def descriptor_is_discrete(feature: str) -> bool:
    """Whether an optimized chemistry descriptor should stay integer-valued."""
    token = str(feature).rsplit("_", 1)[-1]
    full = str(feature)
    return (
        "_Is_" in full or "_Contains_" in full or "_MorganBit_" in full
        or "_Num_" in full or full.endswith(("_WaterSoluble", "_Organic", "_Inorganic",
                                              "_HalogenCount", "_NumAtoms",
                                              "_EstimatedIonicCharge"))
    )


# Shared process-local engine. Descriptor generation and RDKit calls are cached.
ENGINE = ChemistryFeatureEngineer()


def auto_configure_chemistry(
    frame: pd.DataFrame,
    chemical_columns: Sequence[str],
    groups: pd.Series | None = None,
    target: pd.Series | None = None,
    requested_mode: str = "automatic",
) -> ChemistryAutoConfig:
    """Public group-aware configurator using the shared cached engine."""
    return ENGINE.auto_configure(
        frame, chemical_columns=chemical_columns, groups=groups, target=target,
        requested_mode=requested_mode)


def chemistry_config_as_dict(config: ChemistryAutoConfig | Mapping | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, ChemistryAutoConfig):
        return asdict(config)
    return dict(config)


def apply_custom_families(config: ChemistryAutoConfig,
                          enabled_families: Iterable[str]) -> ChemistryAutoConfig:
    """Replace a Custom configuration's descriptor keys with named families."""
    families = [name for name in enabled_families if name in DESCRIPTOR_FAMILIES]
    keys = list(dict.fromkeys(
        key for family in families if family not in {"categorical_identity", "interactions"}
        for key in DESCRIPTOR_FAMILIES[family]))
    categoricals = (list(STANDARD_CATEGORICAL_KEYS)
                    if "categorical_identity" in families else [])
    config.mode = "custom"
    config.selected_descriptor_keys = keys
    config.selected_categorical_keys = categoricals
    if "interactions" not in families:
        config.selected_interactions = []
    config.rationale.append("Custom descriptor families: " +
                            (", ".join(families) if families else "none"))
    if hasattr(config, "_allowed_descriptor_features"):
        delattr(config, "_allowed_descriptor_features")
    return config
