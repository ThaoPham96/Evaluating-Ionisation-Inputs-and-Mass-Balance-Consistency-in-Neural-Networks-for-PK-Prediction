"""SUP09: complete endpoint metric suite for the primary models (nine seeds, seven policies).

Scales. Half-life, Vd, CL, Cmax and Tmax are scored in log10 units: MAE, RMSE, Pearson r, Spearman
rho and R-squared all use log10 values. F and PPB are scored in percentage points on the 0-100 scale.
R-squared = 1 - SSE/SST with SST about the mean observed value of the scored structures.
Fold metrics (five positive endpoints): fold error = 10**|log10 prediction - log10 observation|.
  gmfe                    geometric mean fold error = 10**mean(|log10 error|)
  mean_fold_error         arithmetic mean of the fold ratios (always >= gmfe)
  median_fold_error       median fold ratio
  within_2_fold_percent / within_3_fold_percent
For F and PPB the same ratio metrics are reported only on strictly positive observations
(PPB: 781 of 805; the 24 zero labels stay in MAE, RMSE, r and R-squared). Percent and log errors
are never averaged together. Intervals are conditional paired group-bootstrap intervals; Spearman
rho is reported per seed without an interval.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mbsupp as mb  # noqa: E402

OUT = mb.ROOT / "supplementary_completion" / "tables"


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bundle(boot, pred, truth, mask, log_scale):
    """{metric: (point (seed,), draws (draw, seed))} from per-group sums; pred (seed, N), truth (N,)."""
    truth = np.where(mask, truth, 0.0)
    pred = np.where(mask, pred, 0.0)
    error = pred - truth
    t = np.broadcast_to(truth, pred.shape)
    S = lambda values: boot.sums(values, mask)  # noqa: E731,N806
    (n, nd) = S(np.ones_like(pred))
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, values in (("mae", np.abs(error)), ("mse", error ** 2)):
            p, d = S(values)
            out[name] = (p / n, d / nd)
        out["rmse"] = (np.sqrt(out["mse"][0]), np.sqrt(out["mse"][1]))
        (sp, spd), (st, std), (spt, sptd), (spp, sppd), (stt, sttd) = S(pred), S(t), S(pred * t), S(pred ** 2), S(t ** 2)
        def corr(n, sp, st, spt, spp, stt):
            cov = spt / n - sp / n * st / n
            return cov / np.sqrt((spp / n - (sp / n) ** 2) * (stt / n - (st / n) ** 2))
        out["pearson_r"] = (corr(n, sp, st, spt, spp, stt), corr(nd, spd, std, sptd, sppd, sttd))
        out["r_squared"] = (1 - out["mse"][0] / (stt / n - (st / n) ** 2), 1 - out["mse"][1] / (sttd / nd - (std / nd) ** 2))
        if log_scale:
            absolute = np.abs(error)
            out["gmfe"] = (10 ** out["mae"][0], 10 ** out["mae"][1])
            p, d = S(10 ** absolute)
            out["mean_fold_error"] = (p / n, d / nd)
            for limit in (2, 3):
                p, d = S(100.0 * (absolute <= np.log10(limit)))
                out[f"within_{limit}_fold_percent"] = (p / n, d / nd)
            p, d = boot.median(absolute, mask)
            out["median_fold_error"] = (10 ** p, 10 ** d)
    del out["mse"]
    return out, n.astype(int)


def spearman(pred, truth, mask):
    return np.array([np.corrcoef(rankdata(row[mask]), rankdata(truth[mask]))[0, 1] for row in pred])


def main():
    source = mb.Source()
    cohort = source.cohort
    boot = mb.Bootstrap(cohort)
    absolute_rows, seed_rows, delta_rows = [], [], []
    for backbone in mb.PRIMARY:
        stats = {}
        for policy in mb.POLICIES:
            pred = mb.units(source.primary(backbone, policy), mb.SEEDS)
            for column, endpoint in enumerate(mb.NAMES):
                truth = cohort.endpoint_truth[:, column]
                labelled = np.isfinite(truth)
                percent = column in mb.PERCENT_COLUMNS
                sets = [("percentage points" if percent else "log10", pred[:, :, column], truth, labelled, not percent)]
                if percent:   # ratio metrics on strictly positive observations only
                    positive = labelled & (truth > 0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        sets.append(("log10 ratio, positive observations", np.log10(pred[:, :, column]),
                                     np.log10(np.where(positive, truth, 1.0)), positive, True))
                for scale, values, target, mask, log_scale in sets:
                    result, n = bundle(boot, values, target, mask, log_scale)
                    if percent and log_scale:   # keep only the ratio metrics from the positive subset
                        result = {k: v for k, v in result.items() if k in
                                  ("gmfe", "mean_fold_error", "median_fold_error", "within_2_fold_percent", "within_3_fold_percent")}
                    else:
                        rho = spearman(values, target, mask)
                        seed_rows += [{"backbone": backbone, "policy": policy, "endpoint": endpoint, "scale": scale,
                                       "metric": "spearman_rho", "seed": seed, "value": float(v), "n_scored": int(n.min())}
                                      for seed, v in zip(mb.SEEDS, rho)]
                        absolute_rows.append({"backbone": backbone, "policy": policy, "endpoint": endpoint, "scale": scale,
                                              "metric": "spearman_rho", "estimate": float(rho.mean()), "ci95_lower": "",
                                              "ci95_upper": "", "seed_sd": float(rho.std(ddof=1)), "n_seeds": len(mb.SEEDS),
                                              "n_scored": int(n.min()), "n_defined_draws": ""})
                    for metric, (point, draws) in result.items():
                        key = {"backbone": backbone, "policy": policy, "endpoint": endpoint, "scale": scale, "metric": metric}
                        summary = boot.interval(point, draws)
                        absolute_rows.append({**key, "estimate": summary["estimate"], "ci95_lower": summary["ci95_lower"],
                                              "ci95_upper": summary["ci95_upper"], "seed_sd": summary["seed_sd"],
                                              "n_seeds": summary["n_seeds"], "n_scored": int(n.min()),
                                              "n_defined_draws": summary["n_defined_draws"]})
                        seed_rows += [{**key, "seed": seed, "value": float(v), "n_scored": int(n.min())}
                                      for seed, v in zip(mb.SEEDS, point)]
                        stats[(policy, endpoint, scale, metric)] = (point, draws, int(n.min()))
        for (policy, endpoint, scale, metric), (point, draws, n) in stats.items():
            if policy == "baseline":
                continue
            base_point, base_draws, _ = stats[("baseline", endpoint, scale, metric)]
            delta_rows.append({"backbone": backbone, "arm_a": policy, "arm_b": "baseline", "endpoint": endpoint,
                               "scale": scale, "metric": metric, "mean_a": float(point.mean()),
                               "mean_b": float(base_point.mean()), **boot.interval(point - base_point, draws - base_draws),
                               "n_scored": n})
    write(OUT / "sup09_endpoint_metrics_summary.csv", absolute_rows)
    write(OUT / "sup09_endpoint_metrics_by_seed.csv", seed_rows)
    write(OUT / "sup09_endpoint_metric_deltas_vs_baseline.csv", delta_rows)
    print("SUP09:", len(absolute_rows), "summary rows,", len(seed_rows), "seed rows,", len(delta_rows), "delta rows")


if __name__ == "__main__":
    main()
