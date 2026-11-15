"""SUP08: which outputs move when the consistency loss is added (nine seeds, primary models).

Descriptive. For each structure and seed, q = (log10 t_half, log10 Vd, log10 CL), a = (1, -1, 1),
residual r = a.q - log10(ln 2). With shift d = q_policy - q_baseline, the residual change is
r_policy - r_baseline = a.d, so each endpoint's contribution a_e * d_e is additive.

absorption_share_e = -Cov(a_e d_e, r_baseline) / Var(r_baseline): the fraction of the baseline residual
removed through endpoint e (orthogonal correction gives alpha/3 for each endpoint; half-life-only
correction gives alpha for half-life and 0 for the others). The three shares sum to the total
fraction removed. Statistics are recomputed within each bootstrap draw from per-group sums.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mbsupp as mb  # noqa: E402

OUT = mb.ROOT / "supplementary_completion" / "tables"
POLICIES = ("selected_positive", "exact_orthogonal", "exact_half_life")
SIGN = (1.0, -1.0, 1.0)


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def moments(boot, x, y):
    """Point and draw covariance(x, y) and variance(y) per seed from group sums."""
    ones = np.ones_like(x)
    out = []
    for values in (ones, x, y, x * y, y * y, x * x):
        out.append(boot.sums(values))
    (n, nd), (sx, sxd), (sy, syd), (sxy, sxyd), (syy, syyd), (sxx, sxxd) = out
    point = (sxy / n - sx / n * sy / n, syy / n - (sy / n) ** 2, sxx / n - (sx / n) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        draw = (sxyd / nd - sxd / nd * syd / nd, syyd / nd - (syd / nd) ** 2, sxxd / nd - (sxd / nd) ** 2)
    return point, draw


def main():
    source = mb.Source()
    cohort = source.cohort
    boot = mb.Bootstrap(cohort)
    everyone = np.ones(len(cohort.index), bool)
    rows = []

    def push(key, statistic, unit, point, draws):
        rows.append({**key, "statistic": statistic, "unit": unit, **boot.interval(point, draws)})

    for backbone in mb.PRIMARY:
        base = mb.units(source.primary(backbone, "baseline"), mb.SEEDS)
        r_base = base[:, :, 0] - base[:, :, 1] + base[:, :, 2] - mb.LOG10_LN2
        for policy in POLICIES:
            pred = mb.units(source.primary(backbone, policy), mb.SEEDS)
            r_policy = pred[:, :, 0] - pred[:, :, 1] + pred[:, :, 2] - mb.LOG10_LN2
            shares_point, shares_draw = 0.0, 0.0
            for column, name in enumerate(mb.NAMES):
                key = {"backbone": backbone, "policy": policy, "endpoint": name}
                unit = "percentage points" if column in mb.PERCENT_COLUMNS else "log10"
                shift = pred[:, :, column] - base[:, :, column]
                point, draws, _ = boot.mean(shift, everyone)
                push(key, "mean_signed_shift", unit, point, draws)
                point, draws, _ = boot.mean(np.abs(shift), everyone)
                push(key, "mean_absolute_shift", unit, point, draws)
                (cov_p, var_b_p, var_l_p), (cov_d, var_b_d, var_l_d) = moments(boot, pred[:, :, column], base[:, :, column])
                with np.errstate(invalid="ignore", divide="ignore"):
                    push(key, "spread_ratio_policy_over_baseline", "ratio of across-structure SD",
                         np.sqrt(var_l_p / var_b_p), np.sqrt(var_l_d / var_b_d))
                    push(key, "baseline_across_structure_sd", unit, np.sqrt(var_b_p), np.sqrt(var_b_d))
                    push(key, "policy_baseline_prediction_correlation", "Pearson r",
                         cov_p / np.sqrt(var_l_p * var_b_p), cov_d / np.sqrt(var_l_d * var_b_d))
                for label, arr in (("baseline", base), ("policy", pred)):
                    seed_sd = arr[:, :, column].std(axis=0, ddof=1)[None, :]
                    point, draws, _ = boot.mean(seed_sd, everyone)
                    push(key, f"mean_across_seed_sd_{label}", unit, point, draws)
                if column < 3:
                    (cov_p, var_p, _), (cov_d, var_d, _) = moments(boot, SIGN[column] * shift, r_base)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        share_p, share_d = -cov_p / var_p, -cov_d / var_d
                    push(key, "absorption_share_of_baseline_residual", "fraction", share_p, share_d)
                    shares_point, shares_draw = shares_point + share_p, shares_draw + share_d
                    truth = cohort.triad_truth[:, column]
                    pa, da, _ = boot.mean(np.abs(pred[:, :, column] - truth), cohort.triad_eligible)
                    pb, db, _ = boot.mean(np.abs(base[:, :, column] - truth), cohort.triad_eligible)
                    push(key, "same_row_triad_component_mae_change", "log10", pa - pb, da - db)
            key = {"backbone": backbone, "policy": policy, "endpoint": "triad"}
            push(key, "total_fraction_of_baseline_residual_removed", "fraction", shares_point, shares_draw)
            point, draws, _ = boot.mean(r_base, everyone)
            push(key, "mean_signed_residual_baseline", "log10", point, draws)
            point, draws, _ = boot.mean(r_policy, everyone)
            push(key, "mean_signed_residual_policy", "log10", point, draws)
            (_, var_b_p, var_l_p), (_, var_b_d, var_l_d) = moments(boot, r_policy, r_base)
            with np.errstate(invalid="ignore", divide="ignore"):
                push(key, "residual_sd_ratio_policy_over_baseline", "ratio", np.sqrt(var_l_p / var_b_p),
                     np.sqrt(var_l_d / var_b_d))
    write(OUT / "sup08_endpoint_shifts_and_spread.csv", rows)
    print("SUP08:", len(rows), "rows")


if __name__ == "__main__":
    main()
