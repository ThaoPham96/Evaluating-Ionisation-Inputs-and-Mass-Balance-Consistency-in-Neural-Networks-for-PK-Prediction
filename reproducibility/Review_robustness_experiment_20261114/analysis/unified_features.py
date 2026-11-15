"""Derive the archived single-pH categories from the frozen pKa values."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal


PH_BY_ARM = {"PH2": "2.0", "PH5": "5.0", "PH65": "6.5", "P1": "7.4", "PH8": "8.0"}
EXTRA_ARMS = tuple(arm for arm in PH_BY_ARM if arm != "P1")
CLASS_ORDER = ("anionic", "cationic", "neutral", "zwitterionic")
CODE = {(True, False): 0, (False, True): 1, (False, False): 2, (True, True): 3}
EXPECTED = {
    "PH2": ((0, 280, 37, 10), 206), "PH5": ((31, 144, 91, 61), 85),
    "PH65": ((49, 112, 108, 58), 38), "P1": ((65, 90, 119, 53), 0),
    "PH8": ((82, 75, 128, 42), 32),
}


def category_codes(arm, records):
    ph = Decimal(PH_BY_ARM[arm])
    codes = []
    for row in records:
        acid, base = (Decimal(str(row[field])) for field in ("pka_acid", "pka_base"))
        if not acid.is_finite() or not base.is_finite():
            raise ValueError("Single-pH categories require finite pKa values")
        codes.append(CODE[acid < ph, base > ph])
    return codes


def verify_categories(records):
    reference = [CLASS_ORDER.index(row["ionisation_class"]) for row in records]
    inventory = {}
    for arm in PH_BY_ARM:
        codes = category_codes(arm, records)
        counts = Counter(codes)
        switches = sum(a != b for a, b in zip(codes, reference, strict=True))
        ordered = tuple(counts[k] for k in range(4))
        if (ordered, switches) != EXPECTED[arm]:
            raise ValueError(f"The frozen category inventory differs for {arm}")
        inventory[arm] = {"ph": PH_BY_ARM[arm], "counts": list(ordered), "switches_vs_74": switches}
    return inventory


def descriptors(arm, records):
    import torch

    codes = torch.tensor(category_codes(arm, records), dtype=torch.long)
    values = torch.nn.functional.one_hot(codes, num_classes=len(CLASS_ORDER)).float()
    recipe = dict(arm=arm, input_dim=len(CLASS_ORDER), method="one_hot", ph=PH_BY_ARM[arm],
                  class_order=list(CLASS_ORDER), acidic_rule="pka_acid < ph", basic_rule="pka_base > ph")
    return values, recipe
