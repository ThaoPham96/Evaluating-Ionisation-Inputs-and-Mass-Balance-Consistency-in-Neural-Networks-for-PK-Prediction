"""Summarize verified outer predictions from the frozen v6 mass-balance campaign."""

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np

from mass_balance_campaign import (
    ENDPOINTS, FINGERPRINT, LAMBDAS, PRED_COLUMNS, PRIMARY, PROTOCOL, SEEDS, TOLERANCE,
    checked_files, digest, load_predictions, min_with_tolerance, read_csv, read_json, sha256, write_csv, write_json,
)
from mass_balance_controls import (
    ALPHA_GRID, coherent_label_error_floor, partial_correction, residual_log10,
)


FOLDS = (1, 2, 3, 4, 5)
METHODS = ("orthogonal", "half_life")
POLICIES = ("baseline", "selected_positive", "zero_eligible",
            "selected_alpha_orthogonal", "selected_alpha_half_life",
            "exact_orthogonal", "exact_half_life")
CURVES = tuple(f"fixed_lambda_{value:g}" for value in LAMBDAS)
RELEASE_NAMES = ("model_dataset.csv", "outer.csv", "same_row_complete_triads.csv")
PERCENT_COLUMNS = (3, 4)
LOG_COLUMNS = (0, 1, 2, 5, 6)
SOURCE_ERRORS = (OSError, ValueError, KeyError, TypeError)


def saved_scores(receipts):
    scores = {float(key): receipt["macro_log10_mae"] for key, receipt in receipts.items()}
    if any(not math.isfinite(value) or value < 0 for value in scores.values()):
        raise ValueError("Invalid inner selection scores")
    return scores


class Cohort:
    def __init__(self, path):
        self.path = Path(path).resolve()
        manifest = read_json(self.path / "manifest.json")
        if manifest["fingerprint"] != FINGERPRINT:
            raise ValueError("Expected the frozen v6 cohort")
        self.hashes = checked_files(self.path, RELEASE_NAMES, manifest["files"])
        self.hashes["manifest.json"] = sha256(self.path / "manifest.json")
        dataset = read_csv(self.path / "model_dataset.csv")
        ids = [row["structure_id"] for row in dataset]
        self.index = {sid: position for position, sid in enumerate(ids)}
        outer_rows = read_csv(self.path / "outer.csv")
        outer = {row["structure_id"]: int(row["outer_fold"]) for row in outer_rows}
        if len(ids) != 952 or len(self.index) != len(ids) or len(outer_rows) != len(ids) or set(outer) != set(ids):
            raise ValueError("Outer folds must cover the frozen 952 structures exactly once")
        self.folds = np.array([outer[sid] for sid in ids])
        if set(self.folds) != set(FOLDS):
            raise ValueError("Expected all five frozen outer folds")
        self.ids_by_fold = {fold: {sid for sid in ids if outer[sid] == fold} for fold in FOLDS}
        raw = np.array([[float(row[column]) if row[column] else np.nan for column in ENDPOINTS]
                        for row in dataset])
        self.endpoint_truth = raw.copy()
        self.endpoint_truth[:, LOG_COLUMNS] = np.log10(raw[:, LOG_COLUMNS])
        self.triad_truth = np.full((len(ids), 3), np.nan)
        triad_rows = read_csv(self.path / "same_row_complete_triads.csv")
        for row in triad_rows:
            index = self.index[row["structure_id"]]
            values = np.array([float(row[column]) for column in tuple(ENDPOINTS)[:3]])
            if (np.isfinite(self.triad_truth[index]).any() or self.folds[index] != int(row["outer_fold"])
                    or not np.isfinite(values).all() or (values <= 0).any()):
                raise ValueError("Invalid complete same-row observed triad")
            self.triad_truth[index] = np.log10(values)
        self.triad_eligible = np.isfinite(self.triad_truth).all(axis=1)
        if len(triad_rows) != 674 or self.triad_eligible.sum() != 674:
            raise ValueError("Expected 674 complete same-row observed triads")
        self.ppb_zero_count = int((raw[:, 4] == 0).sum())
        if np.isfinite(raw[:, 4]).sum() != 805 or self.ppb_zero_count != 24:
            raise ValueError("Expected 805 frozen PPB labels, including 24 zeros")


class Predictions:
    def __init__(self, path, cohort):
        self.path = Path(path).resolve()
        self.cohort = cohort
        self.source_hashes = {}
        campaign = self.json(self.path / "campaign.json")
        identity = campaign["identity"]
        if (campaign["protocol"] != PROTOCOL or campaign["logical_fit_count"] != 4240
                or campaign["campaign_id"] != digest(identity) or identity["dataset_fingerprint"] != FINGERPRINT
                or any(identity["release"].get(name) != cohort.hashes[name] for name in RELEASE_NAMES)):
            raise ValueError("Campaign identity does not match the frozen v6 cohort")
        self.campaign_id = campaign["campaign_id"]

    def remember(self, path, checksum):
        self.source_hashes[path.relative_to(self.path).as_posix()] = checksum

    def json(self, path):
        content = path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        value = json.loads(content)
        self.remember(path, checksum)
        return value

    @lru_cache(maxsize=None)
    def selection(self, backbone, fold, stage):
        name = f"{backbone}-outer{fold}-{stage}"
        directory = self.path / "selections" / name
        selection = self.json(directory / "selection.json")
        if (selection["campaign_id"] != self.campaign_id or selection["name"] != name
                or selection["kind"] != stage):
            raise ValueError(f"Selection identity mismatch: {name}")
        if stage == "lambda":
            scores = saved_scores(selection["lambda_scores"])
            if set(scores) != set(LAMBDAS):
                raise ValueError(f"Invalid inner lambda selection: {name}")
            positive = min_with_tolerance({value: scores[value] for value in LAMBDAS[1:]})
            zero_eligible = min_with_tolerance({0.0: scores[0.0], positive: scores[positive]})
            all_worse = all(scores[value] > scores[0.0] + TOLERANCE for value in LAMBDAS[1:])
            if (selection["selected_lambda"] != positive or selection["selected_zero_eligible_lambda"] != zero_eligible
                    or selection["all_positives_worse_than_zero"] != all_worse
                    or selection["baseline_selection"] != f"{backbone}-outer{fold}-baseline"):
                raise ValueError(f"Selected lambda disagrees with inner scores: {name}")
        else:
            candidate_scores = selection["candidate_scores"]
            if (len(candidate_scores) != 20 or any(not math.isfinite(value) or value < 0
                    for value in candidate_scores.values())
                    or selection["candidate_id"] != min_with_tolerance(candidate_scores)):
                raise ValueError(f"Baseline candidate disagrees with inner scores: {name}")
            for method in METHODS:
                control = selection["alpha_controls"][method]
                scores = saved_scores(control["scores"])
                if set(scores) != set(ALPHA_GRID) or control["selected_alpha"] != min_with_tolerance(scores):
                    raise ValueError(f"Selected alpha disagrees with inner scores: {name}/{method}")
        return selection

    @lru_cache(maxsize=None)
    def fit(self, logical_id, backbone, fold, seed, lambda_value, candidate, epochs):
        if not isinstance(logical_id, str) or not re.fullmatch(r"MB5-[0-9]{5}", logical_id):
            raise ValueError(f"Invalid fit alias: {logical_id}")
        directory = self.path / "results" / logical_id
        published = self.json(directory / "published.json")
        complete = self.json(directory / "complete.json")
        wrapper = self.json(directory / "request.json")
        request = wrapper["request"]
        fit_id = f"{logical_id}-{digest({key: value for key, value in request.items() if key != 'fit_id'})[:20]}"
        complete_hash = self.source_hashes[(directory / "complete.json").relative_to(self.path).as_posix()]
        request_hash = self.source_hashes[(directory / "request.json").relative_to(self.path).as_posix()]
        if (published["campaign_id"] != self.campaign_id or request["campaign_id"] != self.campaign_id
                or published["trainer_marker_sha256"] != complete_hash
                or not complete["resolved_hash"] or complete["resolved_hash"] != wrapper["resolved_hash"]
                or published["fit_id"] != fit_id or complete["fit_id"] != fit_id or request["fit_id"] != fit_id
                or complete["files"]["request.json"] != request_hash or wrapper["request_hash"] != digest(request)):
            raise ValueError(f"Published fit identity mismatch: {logical_id}")
        expected_ids = self.cohort.ids_by_fold[fold]
        train, validation, predict = (request[name] for name in ("train_ids", "val_ids", "predict_ids"))
        if (request["logical_id"] != logical_id or request["backbone"] != backbone or request["outer_fold"] != fold
                or request["block"] != "primary_final" or request["endpoint"] is not None
                or request["inner_fold"] is not None or request["seed"] != seed or request["lambda_value"] != lambda_value
                or request["candidate_id"] != candidate or request["fixed_epochs"] != epochs
                or not isinstance(epochs, int) or not 1 <= epochs <= 200 or validation
                or len(predict) != len(expected_ids) or set(predict) != expected_ids
                or len(train) != len(self.cohort.index) - len(expected_ids)
                or set(train) != self.cohort.index.keys() - expected_ids
                or wrapper["effective_ids"] != {"train_ids": train, "val_ids": validation, "predict_ids": predict}):
            raise ValueError(f"Outer fit assignment differs from its policy/frozen fold: {logical_id}")
        prediction_file = directory / "predictions.csv"
        checksum = sha256(prediction_file)
        if (checksum != complete["files"]["predictions.csv"]
                or prediction_file.stat().st_size != published["sizes"]["predictions.csv"]):
            raise ValueError(f"Prediction checksum or size mismatch: {logical_id}")
        self.remember(prediction_file, checksum)
        return load_predictions(directory, predict)

    def fold_predictions(self, backbone, fold, seed, policy):
        selection = self.selection(backbone, fold, "lambda")
        aliases = selection["aliases"]
        value = selection["selected_lambda"]
        if policy in CURVES:
            value = LAMBDAS[CURVES.index(policy)]
            alias = aliases["fixed_lambda_curve"][str(value)]
        elif policy == "selected_positive" or (policy == "zero_eligible" and selection["selected_zero_eligible_lambda"] > 0):
            alias = aliases["selected_positive"]
        else:
            alias, value = aliases["baseline"], 0.0
        logical_id = alias.get(str(seed))
        if logical_id is None:
            raise ValueError(f"No published-fit alias for seed {seed}")
        fit = self.fit(logical_id, backbone, fold, seed, value, selection["candidate_id"],
                       selection["epochs"][str(float(value))])
        ids = sorted(fit)
        values = np.array([fit[sid] for sid in ids])
        for method in METHODS:
            if policy == f"selected_alpha_{method}":
                baseline = self.selection(backbone, fold, "baseline")
                if baseline["candidate_id"] != selection["candidate_id"]:
                    raise ValueError("Baseline and lambda selections have different candidates")
                alpha = baseline["alpha_controls"][method]["selected_alpha"]
            elif policy == f"exact_{method}":
                alpha = 1
            else:
                continue
            values[:, :3] = partial_correction(values[:, :3], alpha, method)
        return ids, values

    def pooled(self, backbone, policy, seeds):
        arrays, missing = {}, []
        for seed in seeds:
            pooled = np.full((len(self.cohort.index), len(PRED_COLUMNS)), np.nan)
            for fold in FOLDS:
                try:
                    ids, values = self.fold_predictions(backbone, fold, seed, policy)
                    pooled[[self.cohort.index[sid] for sid in ids]] = values
                except SOURCE_ERRORS as exc:
                    missing.append({"outer_fold": fold, "seed": seed, "reason": str(exc)})
            arrays[seed] = pooled
        return ({} if missing else arrays), missing

    def failure_records(self):
        records = {}
        for kind in ("failures", "selection_failures"):
            directory = self.path / kind
            records[kind] = ([{"id": path.stem, **self.json(path)} for path in sorted(directory.glob("*.json"))]
                             if directory.is_dir() else None)
        return records


def scoring_units(predictions):
    values = predictions.copy()
    percent_values = predictions[:, PERCENT_COLUMNS]
    with np.errstate(invalid="ignore", over="ignore"):
        logits = percent_values * math.log(10)
        physical = 100 * np.exp(-np.logaddexp(0, -logits))
    values[:, PERCENT_COLUMNS] = np.where(np.isfinite(percent_values), physical, np.nan)
    return values


def median_or_none(values):
    return float(np.median(values)) if len(values) else None


def error_metrics(predicted, baseline, observed, eligible):
    """Score absolute errors and the paired contrast on one identical mask."""
    valid = np.isfinite(predicted) & eligible
    valid_baseline = np.isfinite(baseline) & eligible
    common = valid & valid_baseline
    values = predicted[common] - observed[common]
    baseline_values = baseline[common] - observed[common]
    mae = float(np.abs(values).mean()) if values.size else None
    baseline_mae = float(np.abs(baseline_values).mean()) if values.size else None
    return {"mae": mae, "baseline_common_mae": baseline_mae,
            "delta_mae": mae - baseline_mae if mae is not None else None,
            "n_eligible": int(eligible.sum()), "n_scored": int(common.sum()),
            "n_invalid": int((eligible & ~valid).sum()),
            "n_baseline_invalid": int((eligible & ~valid_baseline).sum())}


def triad_metrics(predicted, baseline, observed, eligible):
    valid = np.isfinite(predicted).all(axis=1) & eligible
    valid_baseline = np.isfinite(baseline).all(axis=1) & eligible
    common = valid & valid_baseline
    values = np.abs(predicted[common] - observed[common])
    baseline_values = np.abs(baseline[common] - observed[common])
    return {"mae": float(values.mean()) if len(values) else None,
            "baseline_common_mae": float(baseline_values.mean()) if len(values) else None,
            "delta_mae": float((values - baseline_values).mean()) if len(values) else None,
            "n_eligible": int(eligible.sum()), "n_scored": int(common.sum()),
            "n_invalid": int((eligible & ~valid).sum()),
            "n_baseline_invalid": int((eligible & ~valid_baseline).sum())}


def coherence_metrics(predictions):
    valid = np.isfinite(predictions).all(axis=1)
    with np.errstate(invalid="ignore", over="ignore"):
        residuals = np.abs(residual_log10(predictions[valid]))
    residuals = residuals[np.isfinite(residuals)]
    n = len(predictions)
    n_within_125 = int((residuals <= math.log10(1.25)).sum())
    n_within_2 = int((residuals <= math.log10(2)).sum())
    with np.errstate(over="ignore"):
        median_fold = median_or_none(np.power(10.0, residuals))
    return {"median_abs_log10_residual": median_or_none(residuals),
            "median_disagreement_fold": median_fold if median_fold is None or math.isfinite(median_fold) else None,
            "within_1_25": n_within_125 / n, "within_2": n_within_2 / n,
            "n_eligible": n, "n_valid": len(residuals), "n_invalid": n - len(residuals),
            "n_within_1_25": n_within_125, "n_within_2": n_within_2}


def seed_metrics(cohort, predictions, baseline):
    pred = scoring_units(predictions)
    base = scoring_units(baseline)
    output = {
        "same_row_triad_log10": triad_metrics(pred[:, :3], base[:, :3], cohort.triad_truth, cohort.triad_eligible),
        "coherence_all_structures": coherence_metrics(pred[:, :3]),
    }
    for i, endpoint in enumerate(ENDPOINTS.values()):
        unit = "percentage_point" if i in PERCENT_COLUMNS else "log10"
        truth = cohort.endpoint_truth[:, i]
        output[f"{endpoint}_{unit}"] = error_metrics(pred[:, i], base[:, i], truth, np.isfinite(truth))
    return output


def summarize_seed_rows(rows, expected_seeds):
    groups = {}
    for row in rows:
        key = row["backbone"], row["policy"], row["domain"], row["metric"]
        groups.setdefault(key, {})[row["seed"]] = row["value"]
    aggregates = []
    for (backbone, policy, domain, metric), per_seed in groups.items():
        expected = expected_seeds[policy]
        if set(per_seed) != set(expected):
            raise ValueError(f"Incomplete seed rows for {backbone}/{policy}/{domain}/{metric}")
        values = [per_seed[seed] for seed in expected if per_seed[seed] is not None]
        complete = len(values) == len(expected)
        aggregates.append({"backbone": backbone, "policy": policy, "domain": domain, "metric": metric,
                           "n_seeds_expected": len(expected), "n_seeds_defined": len(values),
                           "mean": statistics.mean(values) if complete else None,
                           "sd": statistics.stdev(values) if complete and len(values) > 1 else None})
    return aggregates


def selection_summary(source):
    rows = []
    for backbone in PRIMARY:
        for fold in FOLDS:
            row = {"backbone": backbone, "outer_fold": fold}
            for stage in ("baseline", "lambda"):
                try:
                    selection = source.selection(backbone, fold, stage)
                    if stage == "baseline":
                        row["baseline_candidate"] = selection["candidate_id"]
                        for method in METHODS:
                            row[f"selected_alpha_{method}"] = selection["alpha_controls"][method]["selected_alpha"]
                    else:
                        for key in ("selected_lambda", "selected_zero_eligible_lambda", "all_positives_worse_than_zero"):
                            row[key] = selection[key]
                except SOURCE_ERRORS as exc:
                    row[f"{stage}_error"] = str(exc)
            rows.append(row)
    return rows


def summarize(source, cohort, include_curves=True):
    expected_seeds = {policy: SEEDS for policy in POLICIES}
    if include_curves:
        expected_seeds.update({policy: SEEDS[:3] for policy in CURVES})
    selection_rows = selection_summary(source)
    seed_rows, coverage = [], []
    for backbone in PRIMARY:
        baseline, baseline_missing = source.pooled(backbone, "baseline", SEEDS)
        for policy, seeds in expected_seeds.items():
            if policy == "baseline":
                arrays, missing = baseline, baseline_missing
            else:
                arrays, missing = source.pooled(backbone, policy, seeds)
            pairing_missing = [row for row in baseline_missing if row["seed"] in seeds] if policy != "baseline" else []
            coverage.append({"backbone": backbone, "policy": policy, "expected_fits": len(seeds) * len(FOLDS),
                             "verified_fits": len(seeds) * len(FOLDS) - len(missing),
                             "available": not (missing or pairing_missing), "missing": missing,
                             "missing_baseline": pairing_missing})
            if missing or pairing_missing:
                continue
            # The first-three-seed curve baseline can be complete even if another seed is missing.
            if not baseline:
                paired, missing_for_curve = source.pooled(backbone, "baseline", seeds)
                if missing_for_curve:
                    raise ValueError("The baseline coverage changed while building the summary")
            else:
                paired = baseline
            for seed in seeds:
                metrics = seed_metrics(cohort, arrays[seed], paired[seed])
                seed_rows.extend({"backbone": backbone, "policy": policy, "seed": seed,
                                  "domain": domain, "metric": metric, "value": value}
                                 for domain, measures in metrics.items() for metric, value in measures.items())

    frequencies = {}
    for backbone in PRIMARY:
        folds = [row for row in selection_rows if row["backbone"] == backbone]
        keys = ("selected_lambda", "selected_zero_eligible_lambda", "selected_alpha_orthogonal",
                "selected_alpha_half_life", "all_positives_worse_than_zero")
        frequencies[backbone] = {key: dict(Counter(str(row[key]) for row in folds if key in row)) for key in keys}
    observed = cohort.triad_truth[cohort.triad_eligible]
    return {"campaign_id": source.campaign_id, "metrics": summarize_seed_rows(seed_rows, expected_seeds),
            "coverage": coverage, "selections": selection_rows, "selection_frequencies": frequencies,
            "failures": source.failure_records(), "seed_metrics": seed_rows,
            "observed": {"n_structures": len(cohort.index), "n_same_row_triads": len(observed),
                         "ppb_zero_count": cohort.ppb_zero_count,
                         "coherent_log10_mae_floor": coherent_label_error_floor(observed),
                         **coherence_metrics(observed)}}


def write_table(path, columns, rows):
    write_csv(path, columns, ([row.get(column) for column in columns] for row in rows))


def write_report(output, report, source, cohort):
    output.mkdir(parents=True, exist_ok=True)
    metrics = ("backbone", "policy", "domain", "metric", "n_seeds_expected", "n_seeds_defined", "mean", "sd")
    write_table(output / "summary.csv", metrics, report["metrics"])
    write_table(output / "seed_metrics.csv", ("backbone", "policy", "seed", "domain", "metric", "value"),
                report["seed_metrics"])
    write_table(output / "selections.csv", ("backbone", "outer_fold", "baseline_candidate", "selected_lambda",
                "selected_zero_eligible_lambda", "selected_alpha_orthogonal", "selected_alpha_half_life",
                "all_positives_worse_than_zero", "baseline_error", "lambda_error"), report["selections"])
    write_json(output / "summary.json", {key: value for key, value in report.items() if key != "seed_metrics"})
    write_json(output / "manifest.json", {
        "campaign_id": source.campaign_id, "campaign_dir": str(source.path), "release_dir": str(cohort.path),
        "campaign_artifacts_read": source.source_hashes, "release_artifacts_read": cohort.hashes,
        "code": {name: sha256(Path(__file__).with_name(name)) for name in
                 (Path(__file__).name, "mass_balance_campaign.py", "mass_balance_controls.py")},
        "outputs": {name: sha256(output / name) for name in
                    ("summary.csv", "seed_metrics.csv", "selections.csv", "summary.json")},
        "verification_scope": "Published completion markers, requests, and scored prediction CSVs; checkpoints excluded",
        "aggregation": "Pool five outer test folds within each seed; mean and sample SD across seed metrics",
        "missingness": "Accuracy on common valid baseline/policy rows; coherence threshold denominator includes invalids",
    })


def compact_summary(report):
    metrics = {(row["backbone"], row["policy"], row["domain"], row["metric"]): row["mean"]
               for row in report["metrics"]}
    for row in report["coverage"]:
        backbone, policy = row["backbone"], row["policy"]
        if policy not in POLICIES:
            continue
        if not row["available"]:
            print(f"{backbone:15} {policy:28} unavailable ({len(row['missing'])} missing/invalid fits; "
                  f"{len(row['missing_baseline'])} missing/invalid baseline fits)")
            continue
        mae = metrics[(backbone, policy, "same_row_triad_log10", "mae")]
        within = metrics[(backbone, policy, "coherence_all_structures", "within_1_25")]
        mae_display = f"{mae:.5f}" if mae is not None else "undefined"
        print(f"{backbone:15} {policy:28} triad log10 MAE {mae_display:>9} | within 1.25-fold {within:.2%}")
    for kind, failures in report["failures"].items():
        if failures is None:
            print(f"{kind}: directory unavailable in the local snapshot")
        else:
            print(f"{kind}: {len(failures)} current failure markers")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-fixed-curves", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if any(output.is_relative_to(path.resolve()) for path in (args.campaign_dir, args.release_dir)):
        parser.error("Use an output directory outside the campaign and frozen release")
    cohort = Cohort(args.release_dir)
    source = Predictions(args.campaign_dir, cohort)
    report = summarize(source, cohort, include_curves=not args.skip_fixed_curves)
    write_report(output, report, source, cohort)
    compact_summary(report)


if __name__ == "__main__":
    main()
