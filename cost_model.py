"""
cost_model.py — User-editable reagent cost/hazard database.

No invented chemistry economics: every price and hazard rating here comes
from data the USER enters (e.g. from a supplier price sheet or an MSDS),
never from a guess. A chemical absent from the database has UNKNOWN cost and
hazard — reported as such, never defaulted to zero or "safe". A recipe's
total cost is None (not a partial number that looks complete) if any reagent
it uses is unpriced, so a partial total is never mistaken for a real one.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

DEFAULT_PATH = "chemical_costs.json"

# Ordered worst-to-best is wrong here; ordered none-to-worst so max() by index
# gives the most severe hazard across a recipe's reagents.
HAZARD_LEVELS = ["Unknown", "None", "Low", "Moderate", "High", "Severe"]


@dataclass
class ReagentCostEntry:
    """One user-entered reagent's cost and hazard data. Every numeric/hazard
    field defaults to "not entered" (None / "Unknown"), never a guessed value.
    """
    name: str
    cost_per_kg: Optional[float] = None
    cost_per_liter: Optional[float] = None
    hazard_class: str = "Unknown"
    corrosive: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReagentCostEntry":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in dict(d).items() if k in known})

    @property
    def has_cost(self) -> bool:
        return self.cost_per_kg is not None or self.cost_per_liter is not None


@dataclass
class CostDatabase:
    """A small, persistent, user-editable {chemical_name: ReagentCostEntry} store."""
    path: str = DEFAULT_PATH
    entries: dict = field(default_factory=dict)

    def load(self) -> "CostDatabase":
        self.entries = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
                for name, d in data.items():
                    self.entries[name] = ReagentCostEntry.from_dict(d)
            except (OSError, ValueError):
                pass    # corrupt/missing file -> start empty, never crash the app
        return self

    def save(self) -> None:
        data = {name: e.to_dict() for name, e in self.entries.items()}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get(self, name: str) -> Optional[ReagentCostEntry]:
        return self.entries.get(name)

    def set(self, name: str, entry: ReagentCostEntry) -> None:
        self.entries[name] = entry

    def remove(self, name: str) -> None:
        self.entries.pop(name, None)

    def recipe_cost(self, reagent_amounts: dict) -> dict:
        """``reagent_amounts``: {chemical_name: amount_in_kg_or_L}.

        Returns ``{"total_cost", "partial_cost", "priced", "unpriced"}`` —
        ``total_cost`` is None whenever ANY reagent used lacks price data (so
        a partial sum, in ``partial_cost``, is never mistaken for a complete
        total); ``unpriced`` lists exactly which reagents need entering.
        """
        total = 0.0
        priced, unpriced = [], []
        for name, amount in reagent_amounts.items():
            entry = self.get(name)
            price = None
            if entry is not None and amount is not None:
                if entry.cost_per_kg is not None:
                    price = entry.cost_per_kg * amount
                elif entry.cost_per_liter is not None:
                    price = entry.cost_per_liter * amount
            if price is None:
                unpriced.append(name)
            else:
                total += price
                priced.append(name)
        return {
            "total_cost": total if not unpriced else None,
            "partial_cost": total,
            "priced": priced,
            "unpriced": unpriced,
        }

    def recipe_hazard(self, chemicals) -> dict:
        """Returns ``{"max_hazard", "corrosive", "known", "unknown"}`` across
        every named chemical. A chemical never entered is reported in
        ``unknown``, not silently treated as "None" hazard.
        """
        max_level = "Unknown"
        corrosive = False
        known, unknown = [], []
        for name in chemicals:
            entry = self.get(name)
            if entry is None:
                unknown.append(name)
                continue
            known.append(name)
            if HAZARD_LEVELS.index(entry.hazard_class) > HAZARD_LEVELS.index(max_level):
                max_level = entry.hazard_class
            corrosive = corrosive or entry.corrosive
        return {"max_hazard": max_level, "corrosive": corrosive,
                "known": known, "unknown": unknown}

    def coverage(self, chemicals) -> float:
        """Fraction (0-1) of the given chemicals that have SOME cost or
        hazard data entered — lets a caller decide whether "cost" is even a
        meaningful objective yet for this dataset's reagents."""
        chemicals = list(chemicals)
        if not chemicals:
            return 0.0
        covered = sum(1 for c in chemicals if self.get(c) is not None)
        return covered / len(chemicals)


ENGINE = CostDatabase().load()
