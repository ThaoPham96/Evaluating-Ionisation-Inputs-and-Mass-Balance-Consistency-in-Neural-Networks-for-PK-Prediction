"""SUP07: prespecified source-definition sensitivity and observed-compatibility strata (nine seeds).

Both analyses RE-SCORE existing held-out predictions. They are scoring sensitivities and descriptive
strata; they say nothing about how models retrained without these labels would behave, and strata
defined on the observed labels do not identify causes.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mbsupp as mb  # noqa: E402

OUT = mb.ROOT / "supplementary_completion" / "tables"
POLICIES = ("selected_positive", "exact_orthogonal", "exact_half_life")
STRATA = (("within 1.25-fold", 1.0, 1.25), ("1.25- to 2-fold", 1.25, 2.0), ("2- to 5-fold", 2.0, 5.0),
          ("above 5-fold", 5.0, np.inf))


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = mb.Source()
    cohort = source.cohort
    boot = mb.Bootstrap(cohort)
    data = mb.ROOT / "data"
    ids = list(cohort.index)
    names = json.loads((data / "summary.json").read_text(encoding="utf-8"))["source_sensitivity_excluded_names"]
    records = [row for row in mb.read_csv(data / "reviewed_source_records.csv") if row["Name"] in names]
    if sorted(row["Name"] for row in records) != sorted(names):
        raise ValueError("Prespecified sensitivity names do not map one-to-one onto reviewed source records")
    mapping = {row["row_idx"]: row for row in mb.read_csv(data / "source_row_mapping.csv")}
    triads = {row["structure_id"]: row for row in mb.read_csv(data / "same_row_complete_triads.csv")}
    excluded = np.zeros(len(ids), bool)
    affected = []
    for record in records:
        sid = mapping[record["ORIG_INDEX"]]["structure_id"]
        i = cohort.index[sid]
        excluded[i] = True
        labelled = [name for column, name in enumerate(mb.NAMES) if np.isfinite(cohort.endpoint_truth[i, column])]
        triad_flag = triads.get(sid, {}).get("source_definition_sensitivity_exclude", "")
        affected.append({"name": record["Name"], "source_row": record["ORIG_INDEX"], "structure_id": sid,
                         "volume_definition": record["volume_definition"],
                         "study_compatibility": record["study_compatibility"],
                         "in_674_same_row_triads": sid in triads, "triad_file_flag": triad_flag,
                         "endpoint_labels_removed": "|".join(labelled), "outer_fold": int(cohort.folds[i])})
    flagged_in_file = {sid for sid, row in triads.items() if row["source_definition_sensitivity_exclude"] == "True"}
    if flagged_in_file != {row["structure_id"] for row in affected if row["in_674_same_row_triads"]}:
        raise ValueError("Triad-file exclusion flags disagree with the prespecified name list")
    write(OUT / "sup07_source_sensitivity_affected_structures.csv", affected)

    counts = [{"quantity": "structures (consistency denominator)", "n_all": len(ids), "n_after_exclusion": int((~excluded).sum())},
              {"quantity": "same-row complete triads", "n_all": int(cohort.triad_eligible.sum()),
               "n_after_exclusion": int((cohort.triad_eligible & ~excluded).sum())}]
    for column, name in enumerate(mb.NAMES):
        labelled = np.isfinite(cohort.endpoint_truth[:, column])
        counts.append({"quantity": f"{name} endpoint labels", "n_all": int(labelled.sum()),
                       "n_after_exclusion": int((labelled & ~excluded).sum())})
    write(OUT / "sup07_source_sensitivity_counts.csv", counts)

    observed_fold = np.full(len(ids), np.nan)
    observed_fold[cohort.triad_eligible] = 10 ** mb.abs_residual(cohort.triad_truth[cohort.triad_eligible])
    sensitivity, strata = [], []
    for backbone in mb.PRIMARY:
        base = mb.units(source.primary(backbone, "baseline"), mb.SEEDS)
        for policy in POLICIES:
            pred = mb.units(source.primary(backbone, policy), mb.SEEDS)
            for label, include in (("all structures", None), ("prespecified exclusions removed", ~excluded)):
                sensitivity += [{"backbone": backbone, "arm_a": policy, "arm_b": "baseline", "scoring_set": label, **row}
                                for row in mb.contrast_suite(boot, cohort, pred, base, include)]
            for label, low, high in STRATA:
                include = cohort.triad_eligible & (observed_fold > low) & (observed_fold <= high)
                rows = [row for row in mb.contrast_suite(boot, cohort, pred, base, include)
                        if row["metric"].startswith(("triad_mae", "within_", "median_fold"))]
                strata += [{"backbone": backbone, "arm_a": policy, "arm_b": "baseline", "observed_stratum": label,
                            "n_triads_in_stratum": int(include.sum()), **row} for row in rows]
    write(OUT / "sup07_source_sensitivity_contrasts.csv", sensitivity)
    write(OUT / "sup07_compatibility_strata_contrasts.csv", strata)
    print("SUP07:", len(affected), "structures excluded;", counts[1])


if __name__ == "__main__":
    main()
