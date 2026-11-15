"""SUP04-SUP06: alternative encoders, GraphConv depth, task weighting and single-task models.

All comparisons use seeds 101, 202, 303, paired on outer folds and seeds. The encoder, depth and
task-weight fits use the GraphConv fold's TRANSFERRED configuration and selected lambda (with their
own inner epoch calibration); they are not independently optimised. Single-task models were tuned
independently (six configurations per endpoint and outer fold).
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mbsupp as mb  # noqa: E402

OUT = mb.ROOT / "supplementary_completion" / "tables"
SEEDS = mb.SEEDS3


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = mb.Source()
    cohort = source.cohort
    boot = mb.Bootstrap(cohort)
    reference = {"baseline": mb.units(source.primary("GraphConv", "baseline", SEEDS), SEEDS),
                 "loss": mb.units(source.primary("GraphConv", "selected_positive", SEEDS), SEEDS)}

    # ---- SUP04 encoders and depth -------------------------------------
    cells = [("primary_final", "GraphConv", 2, "inner-selected for GraphConv (reference)")]
    cells += [("alternative_encoders", b, d, "transferred GraphConv configuration and lambda") for b, d in mb.SECONDARY_BLOCKS["alternative_encoders"]]
    cells += [("depth_1_to_5", b, d, "transferred GraphConv configuration and lambda") for b, d in mb.SECONDARY_BLOCKS["depth_1_to_5"]]
    seed_rows, contrast_rows = [], []
    for block, backbone, depth, settings in cells:
        arms = reference if block == "primary_final" else {
            arm: mb.units(source.secondary(block, backbone, depth, arm, SEEDS), SEEDS) for arm in ("baseline", "loss")}
        key = {"study": "depth" if backbone == "GraphConv" else "encoder", "block": block, "backbone": backbone,
               "depth": depth, "settings": settings}
        for arm, pred in arms.items():
            seed_rows += [{**key, "arm": arm, **row} for row in mb.seed_metric_rows(cohort, pred, SEEDS)]
        contrast_rows += [{**key, "contrast": "loss - baseline (same encoder/depth)", **row}
                          for row in mb.contrast_suite(boot, cohort, arms["loss"], arms["baseline"])]
        if block != "primary_final":
            for arm in ("baseline", "loss"):
                contrast_rows += [{**key, "contrast": f"{arm}: this cell - GraphConv depth 2", **row}
                                  for row in mb.contrast_suite(boot, cohort, arms[arm], reference[arm])]
    write(OUT / "sup04_encoder_depth_seed_metrics.csv", seed_rows)
    write(OUT / "sup04_encoder_depth_contrasts.csv", contrast_rows)

    # ---- SUP05 task weights (2 x 2) ------------------------------------
    weighted = {arm: mb.units(source.secondary("task_weight_sensitivity", "GraphConv", 2, arm, SEEDS), SEEDS)
                for arm in ("baseline", "loss")}
    arms = {("equal", "baseline"): reference["baseline"], ("equal", "loss"): reference["loss"],
            ("Vd_CL_x5", "baseline"): weighted["baseline"], ("Vd_CL_x5", "loss"): weighted["loss"]}
    seed_rows = [{"weights": w, "arm": arm, **row} for (w, arm), pred in arms.items()
                 for row in mb.seed_metric_rows(cohort, pred, SEEDS)]
    pairs = [("loss effect, equal weights", ("equal", "loss"), ("equal", "baseline")),
             ("loss effect, Vd/CL weights 5", ("Vd_CL_x5", "loss"), ("Vd_CL_x5", "baseline")),
             ("weight effect, baseline", ("Vd_CL_x5", "baseline"), ("equal", "baseline")),
             ("weight effect, with loss", ("Vd_CL_x5", "loss"), ("equal", "loss"))]
    contrast_rows = [{"contrast": label, "arm_a": "/".join(a), "arm_b": "/".join(b), **row}
                     for label, a, b in pairs for row in mb.contrast_suite(boot, cohort, arms[a], arms[b])]
    stats = [mb.arm_stats(boot, cohort, arms[k]) for k in
             (("Vd_CL_x5", "loss"), ("Vd_CL_x5", "baseline"), ("equal", "loss"), ("equal", "baseline"))]
    contrast_rows += [{"contrast": "interaction: loss effect (weights 5) - loss effect (equal)",
                       "arm_a": "Vd_CL_x5/loss - Vd_CL_x5/baseline", "arm_b": "equal/loss - equal/baseline", **row}
                      for row in mb.linear_contrast(boot, stats, (1, -1, -1, 1))]
    write(OUT / "sup05_task_weight_seed_metrics.csv", seed_rows)
    write(OUT / "sup05_task_weight_contrasts.csv", contrast_rows)

    # ---- SUP06 single-task ---------------------------------------------
    single = mb.units(source.single_task(SEEDS), SEEDS)
    arms = {"single_task": single, "multitask_baseline": reference["baseline"], "multitask_loss": reference["loss"]}
    seed_rows = [{"model": name, **row} for name, pred in arms.items() for row in mb.seed_metric_rows(cohort, pred, SEEDS)]
    contrast_rows = [{"contrast": f"single_task - {other}", **row} for other in ("multitask_baseline", "multitask_loss")
                     for row in mb.contrast_suite(boot, cohort, single, arms[other])]
    write(OUT / "sup06_single_task_seed_metrics.csv", seed_rows)
    write(OUT / "sup06_single_task_contrasts.csv", contrast_rows)
    budget = []
    for fold in mb.FOLDS:
        multitask = source.selection("GraphConv", fold, "lambda")
        mt_id = multitask["aliases"]["baseline"][str(SEEDS[0])]
        row = {"outer_fold": fold, "multitask_candidate": multitask["candidate_id"],
               "multitask_parameters": int(source.all_index[mt_id]["parameter_count"]),
               "multitask_configurations_searched": 20, "single_task_configurations_searched_per_endpoint": 6}
        total = 0
        for endpoint in mb.NAMES:
            sel = source.single_selection(endpoint, fold)
            st_id = source.find_final("single_task_final", "GraphConv", fold, SEEDS[0], 0.0, 2, endpoint)
            count = int(source.all_index[st_id]["parameter_count"])
            total += count
            row[f"single_task_{endpoint}_candidate"] = sel["candidate"]
            row[f"single_task_{endpoint}_parameters"] = count
        row["single_task_total_parameters_seven_models"] = total
        budget.append(row)
    write(OUT / "sup06_single_task_budget.csv", budget)
    write(OUT / "inventory_supplementary_final_fits.csv", source.records)
    print("SUP04-06 written;", len(source.records), "supplementary fits verified")


if __name__ == "__main__":
    main()
