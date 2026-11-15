"""Loss effect split at 1.5-fold observed triad disagreement (nine seeds, primary models).

Descriptive re-scoring of existing held-out predictions. Strata are defined on the observed labels
(674 same-row triads: within 1.5-fold versus above), so they do not identify causes.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mbsupp as mb  # noqa: E402

OUT = mb.ROOT / "supplementary_completion" / "tables" / "sup07_compatibility_split_1_5.csv"
POLICIES = ("selected_positive", "exact_orthogonal")


def main():
    source = mb.Source()
    cohort = source.cohort
    boot = mb.Bootstrap(cohort)
    eligible = cohort.triad_eligible
    observed = np.full(len(cohort.index), np.nan)
    observed[eligible] = 10 ** mb.abs_residual(cohort.triad_truth[eligible])
    strata = {"within 1.5-fold": eligible & (observed <= 1.5), "above 1.5-fold": eligible & (observed > 1.5)}
    assert {k: int(v.sum()) for k, v in strata.items()} == {"within 1.5-fold": 451, "above 1.5-fold": 223}
    rows = []
    for backbone in mb.PRIMARY:
        base = mb.units(source.primary(backbone, "baseline"), mb.SEEDS)
        for policy in POLICIES:
            pred = mb.units(source.primary(backbone, policy), mb.SEEDS)
            quantities = {"triad_mae_log10": (np.abs(pred[:, :, :3] - cohort.triad_truth).mean(axis=2),
                                              np.abs(base[:, :, :3] - cohort.triad_truth).mean(axis=2))}
            for column, name in enumerate(mb.NAMES[:3]):
                quantities[f"{name}_same_row_abs_error_log10"] = (np.abs(pred[:, :, column] - cohort.triad_truth[:, column]),
                                                                  np.abs(base[:, :, column] - cohort.triad_truth[:, column]))
            quantities["predictions_within_1_25_fold_percent"] = (100.0 * (mb.abs_residual(pred) <= math.log10(1.25)),
                                                                  100.0 * (mb.abs_residual(base) <= math.log10(1.25)))
            for metric, (a, b) in quantities.items():
                parts = {}
                for label, mask in strata.items():
                    pa, da, n = boot.mean(a, mask)
                    pb, db, _ = boot.mean(b, mask)
                    parts[label] = (pa - pb, da - db)
                    row = {"backbone": backbone, "policy": policy, "stratum": label, "n_triads": int(n.min()),
                           "metric": metric, "baseline_mean": float(pb.mean()), "policy_mean": float(pa.mean()),
                           **boot.interval(pa - pb, da - db)}
                    with np.errstate(invalid="ignore", divide="ignore"):
                        relative = 100 * (da - db).mean(axis=1) / db.mean(axis=1)
                    lower, upper = np.quantile(relative[np.isfinite(relative)], [.025, .975])
                    row.update({"relative_percent": float(100 * (pa - pb).mean() / pb.mean()),
                                "relative_ci95_lower": float(lower), "relative_ci95_upper": float(upper)})
                    rows.append(row)
                point = parts["above 1.5-fold"][0] - parts["within 1.5-fold"][0]
                draws = parts["above 1.5-fold"][1] - parts["within 1.5-fold"][1]
                rows.append({"backbone": backbone, "policy": policy, "stratum": "difference: above minus within",
                             "n_triads": 674, "metric": metric, "baseline_mean": "", "policy_mean": "",
                             **boot.interval(point, draws), "relative_percent": "", "relative_ci95_lower": "",
                             "relative_ci95_upper": ""})
    with open(OUT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("rows:", len(rows))


if __name__ == "__main__":
    main()
