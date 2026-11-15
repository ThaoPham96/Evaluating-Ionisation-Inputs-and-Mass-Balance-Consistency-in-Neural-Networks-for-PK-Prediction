"""SUP02: training loss versus post-hoc correction controls (nine seeds, primary models)."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mbsupp as mb  # noqa: E402

OUT = mb.ROOT / "supplementary_completion" / "tables"
POLICIES = mb.POLICIES      # baseline, selected_positive, zero_eligible, two selected-alpha, two exact
CONTRASTS = ([(policy, "baseline") for policy in POLICIES[1:]]
             + [("selected_positive", other) for other in
                ("zero_eligible", "selected_alpha_orthogonal", "selected_alpha_half_life",
                 "exact_orthogonal", "exact_half_life")])


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = mb.Source()
    boot = mb.Bootstrap(source.cohort)
    seed_rows, contrast_rows, selection_rows = [], [], []
    for backbone in mb.PRIMARY:
        arrays = {policy: mb.units(source.primary(backbone, policy), mb.SEEDS) for policy in POLICIES}
        for policy, pred in arrays.items():
            seed_rows += [{"backbone": backbone, "policy": policy, **row}
                          for row in mb.seed_metric_rows(source.cohort, pred, mb.SEEDS)]
        for a, b in CONTRASTS:
            identical = bool(np.array_equal(arrays[a], arrays[b]))
            contrast_rows += [{"backbone": backbone, "arm_a": a, "arm_b": b, "arms_identical": identical, **row}
                              for row in mb.contrast_suite(boot, source.cohort, arrays[a], arrays[b])]
        for fold in mb.FOLDS:
            lam = source.selection(backbone, fold, "lambda")
            base = source.selection(backbone, fold, "baseline")
            scores = {float(k): v["macro_log10_mae"] for k, v in lam["lambda_scores"].items()}
            row = {"backbone": backbone, "outer_fold": fold, "baseline_candidate": base["candidate_id"],
                   "selected_positive_lambda": lam["selected_lambda"],
                   "zero_eligible_lambda": lam["selected_zero_eligible_lambda"],
                   "zero_eligible_selects_baseline": lam["selected_zero_eligible_lambda"] == 0,
                   "all_positive_lambdas_worse_than_zero": lam["all_positives_worse_than_zero"],
                   "selected_alpha_orthogonal": base["alpha_controls"]["orthogonal"]["selected_alpha"],
                   "selected_alpha_half_life": base["alpha_controls"]["half_life"]["selected_alpha"]}
            row.update({f"inner_triad_mae_lambda_{value:g}": scores[value] for value in mb.LAMBDAS})
            for method in ("orthogonal", "half_life"):
                alpha_scores = {float(k): v["macro_log10_mae"] for k, v in base["alpha_controls"][method]["scores"].items()}
                row.update({f"inner_triad_mae_alpha_{method}_{value:g}": alpha_scores[value] for value in mb.ALPHA_GRID})
            selection_rows.append(row)
    write(OUT / "sup02_policy_seed_metrics.csv", seed_rows)
    write(OUT / "sup02_paired_contrasts.csv", contrast_rows)
    write(OUT / "sup02_selection_by_fold.csv", selection_rows)
    print(f"SUP02: {len(seed_rows)} seed-metric rows, {len(contrast_rows)} contrast rows, {len(selection_rows)} selection rows")


if __name__ == "__main__":
    main()
