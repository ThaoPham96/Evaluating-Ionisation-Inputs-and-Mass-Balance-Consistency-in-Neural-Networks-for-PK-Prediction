"""Shared evaluator for the supplementary mass-balance analyses.

Extends the package's primary validator (block == primary_final only) to the
supplementary outer-final blocks with equivalent identity, split, coverage and
provenance checks. Nothing here trains models or alters supplied files.

Conventions preserved from METHODS.txt / START_HERE.txt:
  * join on structure_id (arrays are in frozen model_dataset.csv order);
  * pool the five outer folds within a seed, score, then average seed metrics;
  * paired group bootstrap: 10,000 draws of 552 parent/scaffold groups (PCG64
    seed 20261112), the same draws for every arm and seed.
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis"))

from mass_balance_campaign import (  # noqa: E402
    ENDPOINTS, LAMBDAS, PRED_COLUMNS, PRIMARY, SEEDS, digest, load_predictions,
    min_with_tolerance, read_csv, sha256,
)
from mass_balance_controls import ALPHA_GRID, partial_correction, residual_log10  # noqa: E402,F401
from summarize_mass_balance_campaign import (  # noqa: E402,F401
    CURVES, FOLDS, LOG_COLUMNS, PERCENT_COLUMNS, POLICIES, Cohort, Predictions, scoring_units,
)
from bootstrap_mass_balance_overview import (  # noqa: E402
    BOOTSTRAP_SEED, N_RESAMPLES, group_membership, resample_group_counts,
)

NAMES = tuple(ENDPOINTS.values())            # t_half, Vd, CL, F, PPB, Cmax, Tmax
SEEDS3 = SEEDS[:3]
LOG10_LN2 = math.log10(math.log(2))
WEIGHTED = [1, 5, 5, 1, 1, 1, 1]
SECONDARY_BLOCKS = {
    "alternative_encoders": (("GIN", 2), ("GAT", 2), ("AttentiveFP", 2)),
    "depth_1_to_5": (("GraphConv", 1), ("GraphConv", 3), ("GraphConv", 4), ("GraphConv", 5)),
    "task_weight_sensitivity": (("GraphConv", 2),),
}
ARM_SUFFIX = {"baseline": "0", "loss": "GCN_inner_selected_positive"}


class Source(Predictions):
    """Primary validator plus verified access to supplementary outer-final fits."""

    def __init__(self, campaign=ROOT / "campaign", cohort=None, index=ROOT / "figure_data" / "all_fits_index.csv"):
        cohort = cohort or Cohort(ROOT / "data")
        super().__init__(campaign, cohort)
        self.final_rows = [row for row in read_csv(index) if row["inner_fold"] == ""]
        self.all_index = {row["logical_id"]: row for row in read_csv(index)}
        self._cache = {}
        self.records = []

    # ---- lookups -------------------------------------------------------
    def find_final(self, block, backbone, fold, seed, lambda_value, depth=None, endpoint=""):
        hits = [row for row in self.final_rows
                if row["block"] == block and row["backbone"] == backbone and int(row["outer_fold"]) == fold
                and int(row["seed"]) == seed and float(row["lambda_value"]) == float(lambda_value)
                and row["endpoint"] == endpoint
                and (depth is None or (row["depth"] != "" and int(float(row["depth"])) == depth))]
        if len(hits) != 1:
            raise ValueError(f"Expected one outer-final fit, found {len(hits)}: "
                             f"{block}/{backbone}/fold{fold}/seed{seed}/lambda{lambda_value}/depth{depth}/{endpoint}")
        return hits[0]["logical_id"]

    def secondary_selection(self, block, backbone, fold, depth, arm):
        name = f"{block}-{backbone}-outer{fold}-depth{depth}-{ARM_SUFFIX[arm]}"
        selection = self.json(self.path / "selections" / name / "selection.json")
        parent = self.selection("GraphConv", fold, "lambda")
        value = 0.0 if arm == "baseline" else parent["selected_lambda"]
        inner = selection["input_fits"]
        if (selection["campaign_id"] != self.campaign_id or selection["name"] != name
                or selection["kind"] != "secondary"
                or selection["parent_selection"] != f"GraphConv-outer{fold}-lambda"
                or selection["selected_lambda"] != parent["selected_lambda"]
                or selection["candidate_id"] != parent["candidate_id"]
                or set(selection["epochs"]) != {str(float(value))}
                or len(inner) != 5 or sorted(v["inner_fold"] for v in inner.values()) != [1, 2, 3, 4, 5]
                or any(self.all_index[k]["block"] != block or self.all_index[k]["backbone"] != backbone
                       or int(self.all_index[k]["outer_fold"]) != fold
                       or float(self.all_index[k]["lambda_value"]) != float(value) for k in inner)):
            raise ValueError(f"Secondary selection mismatch: {name}")
        epochs = selection["epochs"][str(float(value))]
        best = sorted(v["best_epoch"] for v in inner.values())
        return {"name": name, "lambda": float(value), "candidate": selection["candidate_id"], "epochs": epochs,
                "inner_best_epochs": best}

    def single_selection(self, endpoint, fold):
        name = f"ST-{endpoint}-outer{fold}"
        selection = self.json(self.path / "selections" / name / "selection.json")
        scores = selection["candidate_scores"]
        if (selection["campaign_id"] != self.campaign_id or selection["name"] != name
                or selection["kind"] != "single" or selection["parent_selection"] is not None
                or len(scores) != 6 or any(not math.isfinite(v) or v < 0 for v in scores.values())
                or selection["candidate_id"] != min_with_tolerance(scores)
                or set(selection["epochs"]) != {"0.0"}):
            raise ValueError(f"Single-task selection mismatch: {name}")
        return {"name": name, "lambda": 0.0, "candidate": selection["candidate_id"],
                "epochs": selection["epochs"]["0.0"], "candidate_scores": scores}

    # ---- verified fit --------------------------------------------------
    def verified_fit(self, logical_id, *, block, backbone, fold, seed, lambda_value, candidate, epochs,
                     depth=None, endpoint=None, task_weights=None):
        key = logical_id
        if key in self._cache:
            return self._cache[key]
        directory = self.path / "results" / logical_id
        published = self.json(directory / "published.json")
        complete = self.json(directory / "complete.json")
        wrapper = self.json(directory / "request.json")
        request = wrapper["request"]
        fit_id = f"{logical_id}-{digest({k: v for k, v in request.items() if k != 'fit_id'})[:20]}"
        rel = lambda name: (directory / name).relative_to(self.path).as_posix()  # noqa: E731
        if (published["campaign_id"] != self.campaign_id or request["campaign_id"] != self.campaign_id
                or published["trainer_marker_sha256"] != self.source_hashes[rel("complete.json")]
                or not complete["resolved_hash"] or complete["resolved_hash"] != wrapper["resolved_hash"]
                or published["fit_id"] != fit_id or complete["fit_id"] != fit_id or request["fit_id"] != fit_id
                or complete["files"]["request.json"] != self.source_hashes[rel("request.json")]
                or wrapper["request_hash"] != digest(request)):
            raise ValueError(f"Published fit identity mismatch: {logical_id}")
        fold_ids = self.cohort.ids_by_fold[fold]
        train, validation, predict = (request[name] for name in ("train_ids", "val_ids", "predict_ids"))
        expected_train = self.cohort.index.keys() - fold_ids
        if endpoint is not None:   # single-task fits train only on structures labelled for that endpoint
            column = NAMES.index(endpoint)
            labelled = {sid for sid, i in self.cohort.index.items() if np.isfinite(self.cohort.endpoint_truth[i, column])}
            expected_train = expected_train & labelled
        if (request["logical_id"] != logical_id or request["backbone"] != backbone or request["outer_fold"] != fold
                or request["block"] != block or request["endpoint"] != endpoint or request["inner_fold"] is not None
                or request["seed"] != seed or request["lambda_value"] != lambda_value
                or request["candidate_id"] != candidate or request["fixed_epochs"] != epochs
                or not isinstance(epochs, int) or not 1 <= epochs <= 200 or validation
                or (depth is not None and request["config"].get("depth") != depth)
                or request.get("task_weights") != task_weights
                or len(predict) != len(fold_ids) or set(predict) != fold_ids
                or len(train) != len(expected_train) or set(train) != expected_train
                or wrapper["effective_ids"] != {"train_ids": train, "val_ids": validation, "predict_ids": predict}):
            raise ValueError(f"Outer fit assignment differs from its block/frozen fold: {logical_id}")
        prediction_file = directory / "predictions.csv"
        checksum = sha256(prediction_file)
        if (checksum != complete["files"]["predictions.csv"]
                or prediction_file.stat().st_size != published["sizes"]["predictions.csv"]):
            raise ValueError(f"Prediction checksum or size mismatch: {logical_id}")
        self.remember(prediction_file, checksum)
        fit = load_predictions(directory, predict)
        self.records.append({"logical_id": logical_id, "block": block, "backbone": backbone, "outer_fold": fold,
                             "seed": seed, "lambda_value": lambda_value, "candidate_id": candidate,
                             "fixed_epochs": epochs, "depth": depth, "endpoint": endpoint or "",
                             "task_weights": "" if task_weights is None else ",".join(map(str, task_weights)),
                             "n_train": len(train), "n_predicted": len(fit),
                             "parameter_count": self.all_index[logical_id]["parameter_count"],
                             "predictions_sha256": checksum})
        self._cache[key] = fit
        return fit

    # ---- pooled arrays -------------------------------------------------
    def _pool(self, fits_by_fold):
        pooled = np.full((len(self.cohort.index), len(PRED_COLUMNS)), np.nan)
        seen = 0
        for fit in fits_by_fold:
            ids = sorted(fit)
            pooled[[self.cohort.index[sid] for sid in ids]] = np.array([fit[sid] for sid in ids])
            seen += len(ids)
        if seen != len(self.cohort.index):
            raise ValueError("Pooled outer folds do not cover the 952 structures exactly once")
        return pooled

    def secondary(self, block, backbone, depth, arm, seeds=SEEDS3):
        """Raw trainer-unit pooled predictions {seed: (952, 7)} for one supplementary arm."""
        if (backbone, depth) not in SECONDARY_BLOCKS[block]:
            raise ValueError(f"Unknown supplementary cell {block}/{backbone}/depth{depth}")
        weights = WEIGHTED if block == "task_weight_sensitivity" else None
        arrays = {}
        for seed in seeds:
            fits = []
            for fold in FOLDS:
                sel = self.secondary_selection(block, backbone, fold, depth, arm)
                logical_id = self.find_final(block, backbone, fold, seed, sel["lambda"], depth)
                fits.append(self.verified_fit(logical_id, block=block, backbone=backbone, fold=fold, seed=seed,
                                              lambda_value=sel["lambda"], candidate=sel["candidate"],
                                              epochs=sel["epochs"], depth=depth, task_weights=weights))
            arrays[seed] = self._pool(fits)
        return arrays

    def single_task(self, seeds=SEEDS3):
        """{seed: (952, 7)} where column j comes from the single-task model for endpoint j."""
        arrays = {}
        for seed in seeds:
            combined = np.full((len(self.cohort.index), len(PRED_COLUMNS)), np.nan)
            for column, endpoint in enumerate(NAMES):
                fits = []
                for fold in FOLDS:
                    sel = self.single_selection(endpoint, fold)
                    logical_id = self.find_final("single_task_final", "GraphConv", fold, seed, 0.0, 2, endpoint)
                    fits.append(self.verified_fit(logical_id, block="single_task_final", backbone="GraphConv",
                                                  fold=fold, seed=seed, lambda_value=0.0, candidate=sel["candidate"],
                                                  epochs=sel["epochs"], depth=2, endpoint=endpoint))
                pooled = self._pool(fits)
                others = [c for c in range(len(NAMES)) if c != column]
                if np.isfinite(pooled[:, others]).any() or not np.isfinite(pooled[:, column]).all():
                    raise ValueError(f"Single-task predictions for {endpoint} have unexpected columns")
                combined[:, column] = pooled[:, column]
            arrays[seed] = combined
        return arrays

    def primary(self, backbone, policy, seeds=SEEDS):
        arrays, missing = self.pooled(backbone, policy, seeds)
        if missing:
            raise ValueError(f"Incomplete {backbone}/{policy}: {missing}")
        return arrays


def units(arrays, seeds):
    """Stack {seed: raw} into (seed, structure, endpoint) scoring units (log10; F/PPB in percent)."""
    return np.stack([scoring_units(arrays[seed]) for seed in seeds])


def abs_residual(pred):
    """|log10 t_half - log10 Vd + log10 CL - log10 ln2| for (..., >=3) arrays in scoring units."""
    return np.abs(pred[..., 0] - pred[..., 1] + pred[..., 2] - LOG10_LN2)


class Bootstrap:
    """Saved paired group bootstrap; every statistic is rebuilt from per-group sums within each draw."""

    def __init__(self, cohort, saved=ROOT / "results" / "bootstrap" / "group_draw_counts.npz"):
        self.group_ids, self.membership = group_membership(cohort)
        counts = resample_group_counts(len(self.group_ids), N_RESAMPLES, BOOTSTRAP_SEED)
        stored = np.load(saved)
        if not (np.array_equal(stored["group_ids"], self.group_ids) and np.array_equal(stored["counts"], counts)):
            raise ValueError("Regenerated draws or group order differ from the saved bootstrap specification")
        self.draws = counts.astype(np.float64)
        self.n_groups = len(self.group_ids)

    def sums(self, values, mask=None):
        """values: (seed, structure). Returns point sums (seed,) and draw sums (draw, seed)."""
        values = np.asarray(values, dtype=np.float64)
        if mask is not None:
            values = np.where(mask, values, 0.0)
        by_group = np.zeros((self.n_groups, values.shape[0]))
        np.add.at(by_group, self.membership, values.T)
        return by_group.sum(axis=0), self.draws @ by_group

    def mean(self, values, mask):
        """Masked mean per seed: point (seed,) and draws (draw, seed)."""
        mask = np.broadcast_to(mask, np.shape(values))
        total, total_draw = self.sums(values, mask)
        n, n_draw = self.sums(mask.astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            return total / n, total_draw / n_draw, n.astype(int)

    @staticmethod
    def interval(point_by_seed, draw_by_seed):
        """Seed-averaged point estimate, percentile 95% interval, seed SD and defined-draw count."""
        point_by_seed = np.asarray(point_by_seed, dtype=float)
        draws = np.asarray(draw_by_seed, dtype=float).mean(axis=1)
        finite = draws[np.isfinite(draws)]
        lower, upper = (np.quantile(finite, [0.025, 0.975], method="linear") if finite.size else (np.nan, np.nan))
        return {"estimate": float(point_by_seed.mean()), "ci95_lower": float(lower), "ci95_upper": float(upper),
                "seed_sd": float(point_by_seed.std(ddof=1)) if point_by_seed.size > 1 else float("nan"),
                "seed_min": float(point_by_seed.min()), "seed_max": float(point_by_seed.max()),
                "n_seeds": int(point_by_seed.size), "n_bootstrap_draws": int(draws.size),
                "n_defined_draws": int(finite.size)}

    def paired_mean_delta(self, a, b, mask):
        """Seed-averaged mean(a) - mean(b) on one common mask, with paired percentile interval."""
        pa, da, n = self.mean(a, mask)
        pb, db, _ = self.mean(b, mask)
        out = self.interval(pa - pb, da - db)
        out.update({"mean_a": float(pa.mean()), "mean_b": float(pb.mean()),
                    "min_n_scored": int(n.min()), "max_n_scored": int(n.max())})
        return out


# ---------------------------------------------------------------------------
# Metric and contrast suites shared by SUP02-SUP09
# ---------------------------------------------------------------------------
THRESHOLDS = (1.25, 1.5, 2.0, 5.0)        # 1.25 is the frozen prespecified threshold


def _tag(threshold):
    return f"{threshold:g}".replace(".", "_")


def _quantile_draws(self, values, mask=None, q=0.5):
    """Median per seed on the cohort (np.median) and within every draw (exact median of the resample)."""
    if q != 0.5:
        raise NotImplementedError("Only the median is implemented")
    values = np.asarray(values, dtype=np.float64)
    point = np.empty(values.shape[0])
    draws = np.empty((self.draws.shape[0], values.shape[0]))
    for s in range(values.shape[0]):
        keep = np.ones(values.shape[1], bool) if mask is None else np.broadcast_to(mask, values.shape)[s]
        if not keep.any():
            point[s], draws[:, s] = np.nan, np.nan
            continue
        v, groups = values[s, keep], self.membership[keep]
        order = np.argsort(v, kind="stable")
        v, cumulative = v[order], np.cumsum(self.draws[:, groups[order]], axis=1)
        total = cumulative[:, -1]
        lower, upper = np.floor((total + 1) / 2), np.ceil((total + 1) / 2)
        i1 = (cumulative >= lower[:, None]).argmax(axis=1)
        i2 = (cumulative >= upper[:, None]).argmax(axis=1)
        draws[:, s] = np.where(total > 0, (v[i1] + v[i2]) / 2, np.nan)
        point[s] = np.median(v)
    return point, draws


Bootstrap.median = _quantile_draws


def seed_metric_rows(cohort, pred, seeds):
    """Absolute per-seed metrics for one arm. pred: (seed, 952, 7) in scoring units, all finite."""
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite predictions; use an explicit common mask")
    rows = []
    residual = abs_residual(pred)
    triad = np.abs(pred[:, :, :3] - cohort.triad_truth).mean(axis=2)
    for s, seed in enumerate(seeds):
        add = lambda metric, value, n: rows.append({"seed": seed, "metric": metric, "value": float(value), "n": int(n)})  # noqa: E731
        add("triad_mae_log10", triad[s, cohort.triad_eligible].mean(), cohort.triad_eligible.sum())
        for threshold in THRESHOLDS:
            add(f"within_{_tag(threshold)}_fold_percent", 100 * (residual[s] <= math.log10(threshold)).mean(), residual.shape[1])
        add("median_fold_disagreement", 10 ** np.median(residual[s]), residual.shape[1])
        add("median_abs_log10_residual", np.median(residual[s]), residual.shape[1])
        for column, name in enumerate(NAMES):
            truth = cohort.endpoint_truth[:, column]
            mask = np.isfinite(truth)
            unit = "percentage_point" if column in PERCENT_COLUMNS else "log10"
            add(f"{name}_mae_{unit}", np.abs(pred[s, mask, column] - truth[mask]).mean(), mask.sum())
    return rows


def contrast_suite(boot, cohort, a, b, include=None):
    """Paired contrasts a - b (seed-averaged) with conditional group-bootstrap intervals.

    Each statistic is recomputed within every draw from per-group sums (means, ratios of means)
    or from the resampled values (medians), then the paired seed differences are averaged.
    include: optional boolean structure mask restricting every metric (sensitivity and strata analyses).
    """
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or a.shape != b.shape:
        raise ValueError("Arms must be finite and have identical (seed, structure, endpoint) shape")
    rows = []

    def push(metric, unit, pa, pb, da, db, n, relative=True):
        row = {"metric": metric, "unit": unit, "mean_a": float(pa.mean()), "mean_b": float(pb.mean()),
               **boot.interval(pa - pb, da - db), "n_scored": int(n)}
        rows.append(row)
        if relative:   # 100 * mean_seed(a - b) / mean_seed(b), the main-figure panel D normalisation
            with np.errstate(invalid="ignore", divide="ignore"):
                draw = 100 * (da - db).mean(axis=1) / db.mean(axis=1)
            finite = draw[np.isfinite(draw)]
            lower, upper = np.quantile(finite, [.025, .975], method="linear")
            seedwise = 100 * (pa - pb) / pb.mean()
            rows.append({"metric": metric.replace("_mae_", "_mae_relative_"), "unit": "percent of mean arm-b MAE",
                         "mean_a": float(pa.mean()), "mean_b": float(pb.mean()),
                         "estimate": float(seedwise.mean()), "ci95_lower": float(lower), "ci95_upper": float(upper),
                         "seed_sd": float(seedwise.std(ddof=1)), "seed_min": float(seedwise.min()),
                         "seed_max": float(seedwise.max()), "n_seeds": int(pa.size),
                         "n_bootstrap_draws": int(draw.size), "n_defined_draws": int(finite.size), "n_scored": int(n)})

    include = np.ones(a.shape[1], bool) if include is None else np.asarray(include, bool)
    eligible = cohort.triad_eligible & include
    ta = np.abs(a[:, :, :3] - cohort.triad_truth).mean(axis=2)
    tb = np.abs(b[:, :, :3] - cohort.triad_truth).mean(axis=2)
    pa, da, n = boot.mean(ta, eligible)
    pb, db, _ = boot.mean(tb, eligible)
    push("triad_mae_log10", "log10", pa, pb, da, db, n.min())
    ra, rb = abs_residual(a), abs_residual(b)
    everyone = include
    for threshold in THRESHOLDS:
        pa, da, n = boot.mean(100.0 * (ra <= math.log10(threshold)), everyone)
        pb, db, _ = boot.mean(100.0 * (rb <= math.log10(threshold)), everyone)
        push(f"within_{_tag(threshold)}_fold_percent", "percentage points", pa, pb, da, db, n.min(), relative=False)
    pa, da = boot.median(ra, include)
    pb, db = boot.median(rb, include)
    push("median_fold_disagreement", "fold", 10 ** pa, 10 ** pb, 10 ** da, 10 ** db, int(include.sum()), relative=False)
    for column, name in enumerate(NAMES):
        truth = cohort.endpoint_truth[:, column]
        mask = np.isfinite(truth) & include
        unit = "percentage_point" if column in PERCENT_COLUMNS else "log10"
        pa, da, n = boot.mean(np.abs(a[:, :, column] - truth), mask)
        pb, db, _ = boot.mean(np.abs(b[:, :, column] - truth), mask)
        push(f"{name}_mae_{unit}", unit, pa, pb, da, db, n.min())
    return rows


def arm_stats(boot, cohort, pred):
    """Per-arm statistics as {metric: (unit, point (seed,), draws (draw, seed), n)} for linear combinations."""
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite predictions; use an explicit common mask")
    out = {}
    triad = np.abs(pred[:, :, :3] - cohort.triad_truth).mean(axis=2)
    point, draws, n = boot.mean(triad, cohort.triad_eligible)
    out["triad_mae_log10"] = ("log10", point, draws, int(n.min()))
    residual = abs_residual(pred)
    everyone = np.ones(pred.shape[1], bool)
    for threshold in THRESHOLDS:
        point, draws, n = boot.mean(100.0 * (residual <= math.log10(threshold)), everyone)
        out[f"within_{_tag(threshold)}_fold_percent"] = ("percentage points", point, draws, int(n.min()))
    point, draws = boot.median(residual)
    out["median_fold_disagreement"] = ("fold", 10 ** point, 10 ** draws, pred.shape[1])
    for column, name in enumerate(NAMES):
        truth = cohort.endpoint_truth[:, column]
        unit = "percentage_point" if column in PERCENT_COLUMNS else "log10"
        point, draws, n = boot.mean(np.abs(pred[:, :, column] - truth), np.isfinite(truth))
        out[f"{name}_mae_{unit}"] = (unit, point, draws, int(n.min()))
    return out


def linear_contrast(boot, stats, coefficients):
    """Seed-averaged linear combination of arm statistics, e.g. an interaction (+1, -1, -1, +1)."""
    rows = []
    for metric in stats[0]:
        unit, _, _, n = stats[0][metric]
        point = sum(c * s[metric][1] for c, s in zip(coefficients, stats))
        draws = sum(c * s[metric][2] for c, s in zip(coefficients, stats))
        rows.append({"metric": metric, "unit": unit, **boot.interval(point, draws), "n_scored": n})
    return rows
