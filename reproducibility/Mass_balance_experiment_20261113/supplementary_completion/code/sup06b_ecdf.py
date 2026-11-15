"""Export mean-of-seed ECDFs of predicted triad disagreement: single-task versus multitask GraphConv."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mbsupp as mb  # noqa: E402


def main():
    source = mb.Source()
    arms = {"single_task": source.single_task(mb.SEEDS3),
            "multitask_baseline": source.primary("GraphConv", "baseline", mb.SEEDS3),
            "multitask_loss": source.primary("GraphConv", "selected_positive", mb.SEEDS3)}
    grid = np.geomspace(1.0, 10.0, 601)
    rows = []
    for name, arrays in arms.items():
        fold = 10 ** mb.abs_residual(mb.units(arrays, mb.SEEDS3))
        curve = np.mean([(np.sort(f)[None, :] <= grid[:, None]).mean(axis=1) * 100 for f in fold], axis=0)
        rows += [{"model": name, "fold_disagreement": float(x), "mean_cumulative_percent": float(y), "n_seeds": 3,
                  "n_structures": fold.shape[1]} for x, y in zip(grid, curve)]
    path = mb.ROOT / "supplementary_completion" / "tables" / "sup06_predicted_disagreement_ecdf.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("ECDF rows:", len(rows))


if __name__ == "__main__":
    main()
