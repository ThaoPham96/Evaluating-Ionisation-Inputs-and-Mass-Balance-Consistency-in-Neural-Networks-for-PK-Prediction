"""SUP03: fixed-lambda and fixed-alpha trade-offs on the same three seeds (101, 202, 303).

Every point is paired against the three-seed baseline; intervals are pointwise conditional
group-bootstrap intervals (not simultaneous bands). Selected policies are re-scored on the
same three seeds so they can be placed on the curves; their nine-seed results are in SUP02.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mbsupp as mb  # noqa: E402

OUT = mb.ROOT / "supplementary_completion" / "tables"
SEEDS = mb.SEEDS3
SELECTED = ("selected_positive", "zero_eligible", "selected_alpha_orthogonal", "selected_alpha_half_life",
            "exact_orthogonal", "exact_half_life")


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = mb.Source()
    boot = mb.Bootstrap(source.cohort)
    seed_rows, contrast_rows = [], []
    for backbone in mb.PRIMARY:
        raw_base = source.primary(backbone, "baseline", SEEDS)
        base = mb.units(raw_base, SEEDS)
        arms = []
        for value in mb.LAMBDAS:
            pred = mb.units(source.primary(backbone, f"fixed_lambda_{value:g}", SEEDS), SEEDS)
            arms.append(("fixed_lambda", "training_loss", value, f"fixed_lambda_{value:g}", pred))
        for method in ("orthogonal", "half_life"):
            for alpha in mb.ALPHA_GRID:
                corrected = {}
                for seed in SEEDS:
                    values = raw_base[seed].copy()
                    values[:, :3] = mb.partial_correction(values[:, :3], alpha, method)
                    corrected[seed] = values
                arms.append(("fixed_alpha", f"correction_{method}", alpha, f"fixed_alpha_{method}_{alpha:g}",
                             mb.units(corrected, SEEDS)))
        for policy in SELECTED:
            arms.append(("selected_same_three_seeds", policy, np.nan, policy,
                         mb.units(source.primary(backbone, policy, SEEDS), SEEDS)))
        for family, method, strength, policy, pred in arms:
            key = {"backbone": backbone, "family": family, "method": method, "strength": strength, "policy": policy}
            seed_rows += [{**key, **row} for row in mb.seed_metric_rows(source.cohort, pred, SEEDS)]
            contrast_rows += [{**key, "arm_b": "baseline", "interval": "pointwise 95% paired group bootstrap",
                               "arms_identical": bool(np.array_equal(pred, base)), **row}
                              for row in mb.contrast_suite(boot, source.cohort, pred, base)]
    write(OUT / "sup03_curve_seed_metrics.csv", seed_rows)
    write(OUT / "sup03_curve_contrasts_vs_baseline.csv", contrast_rows)
    print(f"SUP03: {len(seed_rows)} seed-metric rows, {len(contrast_rows)} contrast rows")


if __name__ == "__main__":
    main()
