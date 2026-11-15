"""Resolve and execute the frozen v6 mass-balance fit manifest on a shared filesystem."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import statistics
import sys
from tempfile import mkdtemp
import threading
import time
import traceback
from uuid import uuid4

import numpy as np

from mass_balance_controls import ALPHA_GRID, partial_correction


PROTOCOL = "PIPK_Net_mass_balance_experimental_protocol_v6_20261112"
FINGERPRINT = "5413771691c6981d22b6c9ca8e21584dc93cd01244c5e3b143c489dfdc9c7942"
ENDPOINTS = {
    "t1/2(hour)": "t_half", "VD(liter)": "Vd", "Cl(liter/hour)": "CL",
    "F(percentage)": "F", "PPB(percentage)": "PPB", "Cmax_uM": "Cmax", "Tmax": "Tmax",
}
PRED_COLUMNS = tuple(f"pred_{name}" for name in ENDPOINTS.values())
LAMBDAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909)
PRIMARY = ("GraphConv", "Chemprop", "ChemBERTa_head")
BLOCK_COUNTS = {
    "primary_baseline_search": 1500, "positive_lambda_calibration": 450,
    "primary_final": 495, "alternative_encoders": 240, "depth_1_to_5": 320,
    "single_task_search": 1050, "single_task_final": 105, "task_weight_sensitivity": 80,
}
SEARCH = {"primary_baseline_search", "single_task_search"}
SECONDARY = {"alternative_encoders", "depth_1_to_5", "task_weight_sensitivity"}
FIT_FILES = {"summary.json", "predictions.csv", "checkpoint.pt", "scaler.json",
             "request.json", "learning_curve.csv"}
RELEASE_FILES = ("model_dataset.csv", "same_row_complete_triads.csv", "outer.csv", "inner.csv")
TOLERANCE = 1e-8
PUBLISH_DELAYS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6) if os.name == "nt" else (0.0,)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    temp = path.with_name(f".{path.name}.{uuid4().hex}")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_move(temp, path, replace=True)
    finally:
        temp.unlink(missing_ok=True)


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_move(staged, target, *, replace):
    """Retry Windows sharing violations while preserving one atomic move per publication."""
    for attempt, delay in enumerate(PUBLISH_DELAYS, start=1):
        if delay:
            time.sleep(delay)
        try:
            if replace:
                os.replace(staged, target)
            else:
                staged.rename(target)
            return True
        except OSError as exc:
            if (not replace and target.is_dir() and
                    exc.errno in {errno.EEXIST, errno.ENOTEMPTY, errno.EACCES, errno.EPERM}):
                return False
            if not isinstance(exc, PermissionError) or attempt == len(PUBLISH_DELAYS):
                raise


def publish_directory(staged, target):
    """Atomically publish a populated directory, allowing a competing publisher to win."""
    published = atomic_move(staged, target, replace=False)
    if published:
        sync_directory(target.parent)
    return published


def sync_directory(directory):
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
            raise
    finally:
        os.close(descriptor)


def checked_files(directory, names, manifest):
    checksums = {name: sha256(directory / name) for name in names}
    if any(checksum != manifest.get(name) for name, checksum in checksums.items()):
        raise ValueError(f"Checksum mismatch in {directory}")
    return checksums


def min_with_tolerance(scores):
    best = min(scores.values())
    return min(key for key, value in scores.items() if value <= best + TOLERANCE)


def stopping_epochs(summaries):
    return max(1, min(200, round(statistics.median(item["best_epoch"] for item in summaries))))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Fit:
    logical_id: str
    block: str
    backbone: str
    outer: int
    inner: int | None
    seed: int
    policy: str
    candidate: str
    endpoint: str
    depth: int | None

    @classmethod
    def from_row(cls, row):
        return cls(row["logical_id"], row["block"], row["backbone"], int(row["outer_fold"]),
                   int(float(row["inner_fold"])) if row["inner_fold"] else None,
                   int(row["seed_id"]), row["lambda_policy"], row["candidate_id"],
                   row["endpoint"], int(float(row["depth"])) if row["depth"] else None)


@dataclass(frozen=True)
class Selection:
    name: str
    kind: str
    inputs: tuple[Fit, ...]
    outputs: tuple[Fit, ...]
    parent: str | None = None


class Plan:
    def __init__(self, release_dir, feature_dir, plan_dir):
        self.release_dir = Path(release_dir).resolve()
        self.feature_dir = Path(feature_dir).resolve()
        self.plan_dir = Path(plan_dir).resolve()
        frozen = read_json(self.release_dir / "manifest.json")
        protocol = read_json(self.plan_dir / f"{PROTOCOL}_manifest.json")
        if frozen["fingerprint"] != FINGERPRINT or protocol["dataset_fingerprint"] != FINGERPRINT:
            raise ValueError("Expected the frozen v6 mass-balance cohort")
        feature_manifest = read_json(self.feature_dir / "manifest.json")
        feature_data = feature_manifest["data"]
        if (feature_data["fingerprint"] != FINGERPRINT
                or feature_data["manifest.json"] != sha256(self.release_dir / "manifest.json")
                or feature_data["model_dataset.csv"] != frozen["files"]["model_dataset.csv"]):
            raise ValueError("Features do not match the frozen release")
        model_sources = ("pipknet/models.py", "pipknet/featurizers.py")
        model_hashes = checked_files(Path(__file__).parent, model_sources, feature_manifest["source_hashes"])
        configuration_name = f"{PROTOCOL}_candidate_configurations.csv"
        logical_name = f"{PROTOCOL}_logical_fits.csv"
        self.identity = {
            "dataset_fingerprint": FINGERPRINT,
            "release": checked_files(self.release_dir, RELEASE_FILES, frozen["files"]),
            "protocol": checked_files(self.plan_dir, (configuration_name, logical_name), protocol["files"]),
            "features": sha256(self.feature_dir / "manifest.json"),
            "code": {**model_hashes, **{name: sha256(Path(__file__).parent / name)
                     for name in ("mass_balance_campaign.py", "mass_balance_controls.py", "mass_balance_train.py")}},
        }
        self.campaign_id = digest(self.identity)
        rows = read_csv(self.plan_dir / logical_name)
        configurations = read_csv(self.plan_dir / configuration_name)
        if any(row["dataset_fingerprint"] != FINGERPRINT for row in rows + configurations):
            raise ValueError("Plan rows reference a different frozen cohort")
        self.fits = {row["logical_id"]: Fit.from_row(row) for row in rows}
        if len(self.fits) != len(rows) or Counter(fit.block for fit in self.fits.values()) != BLOCK_COUNTS:
            raise ValueError("The protocol requires 4,240 unique logical fit slots with the expected block counts")
        self.configs = {
            row["candidate_id"]: {
                "hidden_dim": int(row["hidden_dim"]),
                "depth": int(float(row["depth"])) if row["depth"] else None,
                "ffn_layers": int(row["ffn_layers"]), "dropout": float(row["dropout"]),
                "lr": float(row["lr"]), "weight_decay": float(row["weight_decay"]),
            } for row in configurations
        }
        self._load_folds()
        self.stages = self._build_stages()

    def _load_folds(self):
        outer = read_csv(self.release_dir / "outer.csv")
        dataset = read_csv(self.release_dir / "model_dataset.csv")
        if read_json(self.feature_dir / "structures.json") != [row["structure_id"] for row in dataset]:
            raise ValueError("Feature row order differs from the frozen modelling dataset")
        self.ids = tuple(sorted(row["structure_id"] for row in dataset))
        self.outer = {row["structure_id"]: int(row["outer_fold"]) for row in outer}
        if len(set(self.ids)) != 952 or len(self.ids) != 952 or set(self.outer) != set(self.ids):
            raise ValueError("Frozen outer folds do not match the 952-structure cohort")
        self.inner = defaultdict(dict)
        for row in read_csv(self.release_dir / "inner.csv"):
            self.inner[int(row["outer_fold"])][row["structure_id"]] = int(row["inner_fold"])
        for fold in range(1, 6):
            if set(self.inner[fold]) != {sid for sid in self.ids if self.outer[sid] != fold}:
                raise ValueError(f"Invalid inner fold membership for outer fold {fold}")
        self.labelled = {endpoint: {row["structure_id"] for row in dataset if row[column] != ""}
                         for column, endpoint in ENDPOINTS.items()}
        self.same_row_triads = {}
        for row in read_csv(self.release_dir / "same_row_complete_triads.csv"):
            sid = row["structure_id"]
            values = np.array([float(row[column]) for column in tuple(ENDPOINTS)[:3]])
            if (sid in self.same_row_triads or self.outer[sid] != int(row["outer_fold"])
                    or not np.isfinite(values).all() or (values <= 0).any()):
                raise ValueError(f"Invalid frozen same-row triad for {sid}")
            self.same_row_triads[sid] = np.log10(values)
        if len(self.same_row_triads) != 674:
            raise ValueError("Expected 674 complete same-row triads")

    def _build_stages(self):
        primary = defaultdict(list)
        single = defaultdict(list)
        secondary = defaultdict(list)
        for fit in self.fits.values():
            if fit.block in SECONDARY:
                secondary[(fit.outer, fit.block, fit.backbone, fit.depth, fit.policy)].append(fit)
            elif fit.endpoint:
                single[(fit.outer, fit.endpoint)].append(fit)
            else:
                primary[(fit.outer, fit.backbone)].append(fit)

        stages = []
        secondary_inners = defaultdict(list)
        secondary_stages = []
        for (outer, block, backbone, depth, policy), fits in secondary.items():
            name = f"{block}-{backbone}-outer{outer}-depth{depth}-{policy}"
            inners = tuple(fit for fit in fits if fit.inner is not None)
            finals = tuple(fit for fit in fits if fit.inner is None)
            secondary_inners[outer].extend(inners)
            secondary_stages.append(Selection(name, "secondary", inners, finals,
                                              f"GraphConv-outer{outer}-lambda"))
        for (outer, backbone), fits in primary.items():
            prefix = f"{backbone}-outer{outer}"
            baseline = tuple(fit for fit in fits if fit.block == "primary_baseline_search")
            lambdas = tuple(fit for fit in fits if fit.block == "positive_lambda_calibration")
            finals_zero = tuple(fit for fit in fits if fit.block == "primary_final" and fit.policy == "0")
            finals_positive = tuple(fit for fit in fits if fit.block == "primary_final" and fit.policy != "0")
            transferred = tuple(secondary_inners[outer]) if backbone == "GraphConv" else ()
            stages.append(Selection(f"{prefix}-baseline", "baseline", baseline, lambdas + finals_zero))
            stages.append(Selection(f"{prefix}-lambda", "lambda", lambdas,
                                    finals_positive + transferred, f"{prefix}-baseline"))
        for (outer, endpoint), fits in single.items():
            inners = tuple(fit for fit in fits if fit.inner is not None)
            finals = tuple(fit for fit in fits if fit.inner is None)
            stages.append(Selection(f"ST-{ENDPOINTS[endpoint]}-outer{outer}", "single", inners, finals))
        stages.extend(secondary_stages)

        produced = [fit.logical_id for stage in stages for fit in stage.outputs]
        produced.extend(fit.logical_id for fit in self.fits.values() if fit.block in SEARCH)
        if len(produced) != len(self.fits) or set(produced) != self.fits.keys():
            raise ValueError("Each authoritative fit must have exactly one assignment source")
        return stages

    def request(self, fit, selection=None):
        endpoint = ENDPOINTS[fit.endpoint] if fit.endpoint else None
        cohort = self.labelled[endpoint] if endpoint else set(self.ids)
        inner = self.inner[fit.outer]
        val_ids = [sid for sid in sorted(inner) if fit.inner == inner[sid] and sid in cohort]
        train_ids = [sid for sid in sorted(inner) if fit.inner != inner[sid] and sid in cohort]
        predict_ids = val_ids if fit.inner is not None else [sid for sid in self.ids if self.outer[sid] == fit.outer]
        candidate = fit.candidate if selection is None else selection["candidate_id"]
        config = self.configs[candidate].copy()
        if fit.depth is not None:
            config["depth"] = fit.depth
        if fit.policy == "0":
            lambda_value = 0.0
        elif fit.policy in {"inner_selected_positive", "GCN_inner_selected_positive"}:
            lambda_value = selection["selected_lambda"]
        elif fit.policy.startswith("other_positive_slot_"):
            index = int(fit.policy.rsplit("_", 1)[-1]) - 1
            lambda_value = [value for value in LAMBDAS[1:] if value != selection["selected_lambda"]][index]
        else:
            lambda_value = float(fit.policy)
        fixed_epochs = None
        if fit.inner is None:
            fixed_epochs = selection["epochs"][str(lambda_value)]
        request = {
            "logical_id": fit.logical_id, "block": fit.block,
            "outer_fold": fit.outer, "inner_fold": fit.inner,
            "backbone": "GraphConv" if endpoint else fit.backbone, "endpoint": endpoint,
            "candidate_id": candidate, "config": config, "seed": fit.seed, "lambda_value": lambda_value,
            "train_ids": train_ids, "val_ids": val_ids, "predict_ids": predict_ids,
            "fixed_epochs": fixed_epochs, "max_epochs": 200, "campaign_id": self.campaign_id,
        }
        if fit.block == "task_weight_sensitivity":
            request["task_weights"] = [1, 5, 5, 1, 1, 1, 1]
        request["fit_id"] = f"{fit.logical_id}-{digest(request)[:20]}"
        return request


def validate_trainer_output(directory, request, full=False):
    marker = read_json(directory / "complete.json")
    summary = read_json(directory / "summary.json")
    if (marker["fit_id"] != request["fit_id"] or not marker["resolved_hash"]
            or marker["resolved_hash"] != summary["resolved_hash"]):
        raise ValueError(f"Unexpected trainer result identity at {directory}")
    if not FIT_FILES.issubset(marker["files"]):
        raise ValueError(f"Incomplete trainer file manifest at {directory}")
    if any(Path(name).name != name for name in marker["files"]):
        raise ValueError(f"Unsafe trainer manifest filename at {directory}")
    files = marker["files"] if full else ("summary.json", "predictions.csv", "request.json")
    checked_files(directory, files, marker["files"])
    trainer_request = read_json(directory / "request.json")
    if (trainer_request["request"] != request or trainer_request["request_hash"] != digest(request)
            or trainer_request["resolved_hash"] != marker["resolved_hash"]
            or trainer_request["effective_ids"] != {key: request[key] for key in
                                                     ("train_ids", "val_ids", "predict_ids")}):
        raise ValueError(f"Trainer request differs from assignment at {directory}")
    if request["fixed_epochs"] is None:
        if (not math.isfinite(summary["val_equal_mse"]) or summary["val_equal_mse"] < 0
                or not 1 <= summary["best_epoch"] <= summary["actual_epochs"] <= request["max_epochs"]):
            raise ValueError(f"Invalid inner validation score or epoch at {directory}")
    elif summary["actual_epochs"] != request["fixed_epochs"]:
        raise ValueError(f"Final fit did not run its assigned epochs at {directory}")
    return summary


def load_predictions(directory, ids):
    values = {}
    for row in read_csv(directory / "predictions.csv"):
        sid = row["structure_id"]
        if sid in values:
            raise ValueError(f"Duplicate prediction ID {sid} at {directory}")
        values[sid] = [float(row[column]) if row[column] else float("nan") for column in PRED_COLUMNS]
    if set(values) != set(ids):
        raise ValueError(f"Prediction IDs differ from assignment at {directory}")
    return values


def prediction_csv_rows(pairs):
    for fit, predictions in pairs:
        for sid, values in sorted(predictions.items()):
            yield fit.logical_id, fit.inner, sid, *values


def collect_inner_oof(pairs, expected_ids):
    merged = {}
    for fit, predictions in pairs:
        for sid, values in predictions.items():
            if sid in merged:
                raise ValueError(f"OOF prediction repeats structure {sid} in inner fold {fit.inner}")
            merged[sid] = values
    if set(merged) != expected_ids:
        raise ValueError("OOF predictions must cover exactly the outer-training partition")
    return merged


def triad_arrays(plan, outer, oof):
    eligible = sorted(sid for sid in plan.inner[outer] if sid in plan.same_row_triads)
    truth = np.stack([plan.same_row_triads[sid] for sid in eligible])
    predictions = np.array([oof[sid][:3] for sid in eligible])
    return eligible, truth, predictions, np.isfinite(predictions).all(axis=1)


def triad_score(predictions, truth, mask):
    if not mask.any():
        raise ValueError("No valid predictions on the inner complete same-row triads")
    endpoint_mae = np.abs(predictions[mask] - truth[mask]).mean(axis=0)
    return {"macro_log10_mae": float(endpoint_mae.mean()),
            "endpoint_log10_mae": dict(zip(tuple(ENDPOINTS.values())[:3], endpoint_mae.tolist())),
            "n_scored": int(mask.sum())}


def alpha_selection(predictions, truth, valid):
    controls = {}
    for method in ("orthogonal", "half_life"):
        scores = {alpha: triad_score(partial_correction(predictions, alpha, method), truth, valid)
                  for alpha in ALPHA_GRID}
        selected = min_with_tolerance({alpha: score["macro_log10_mae"] for alpha, score in scores.items()})
        controls[method] = {"selected_alpha": selected, "scores": {str(a): s for a, s in scores.items()}}
    return controls


class Campaign:
    def __init__(self, plan, output_dir):
        self.plan = plan
        self.output = Path(output_dir).resolve()
        if any(self.output.is_relative_to(directory) for directory in (plan.release_dir, plan.feature_dir)):
            raise ValueError("Campaign outputs must be outside the frozen data and feature bundles")
        self.assignments = self.output / "assignments"
        self.selections = self.output / "selections"
        self.selection_failures = self.output / "selection_failures"
        self.claims = self.output / "claims"
        self.results = self.output / "results"
        self.attempts = self.output / "attempts"
        self.failures = self.output / "failures"

    def init(self):
        feature_manifest = read_json(self.plan.feature_dir / "manifest.json")
        checked_files(self.plan.feature_dir, ("structures.json", "graphs.pt", "chemberta.npy"),
                      feature_manifest["files"])
        self.output.mkdir(parents=True, exist_ok=True)
        for directory in (self.assignments, self.selections, self.selection_failures, self.claims,
                          self.results, self.attempts, self.failures):
            directory.mkdir(exist_ok=True)
        metadata = {"campaign_id": self.plan.campaign_id, "identity": self.plan.identity,
                    "logical_fit_count": len(self.plan.fits), "protocol": PROTOCOL}
        path = self.output / "campaign.json"
        if not path.exists():
            # A nonempty directory rename establishes the immutable campaign identity.
            state = self.output / "identity"
            staged = Path(mkdtemp(prefix="identity-", dir=self.attempts))
            try:
                write_json(staged / "campaign.json", metadata)
                publish_directory(staged, state)
                if read_json(state / "campaign.json") != metadata:
                    raise ValueError(f"A different campaign already owns {self.output}")
                write_json(path, metadata)
            finally:
                if staged.exists():
                    shutil.rmtree(staged)
        self.verify_identity()
        for fit in self.plan.fits.values():
            if fit.block in SEARCH and not self.assignment_path(fit.logical_id).is_file():
                self.assign(fit)

    def verify_identity(self):
        if read_json(self.output / "campaign.json")["campaign_id"] != self.plan.campaign_id:
            raise ValueError(f"Code, plan, release or features changed for {self.output}")

    def assignment_path(self, logical_id):
        return self.assignments / f"{logical_id}.json"

    def assign(self, fit, selection=None):
        path = self.assignment_path(fit.logical_id)
        request = self.plan.request(fit, selection)
        if path.exists():
            if read_json(path) != request:
                raise ValueError(f"Conflicting assignment for {fit.logical_id}")
        else:
            write_json(path, request)

    def request(self, logical_id):
        request = read_json(self.assignment_path(logical_id))
        fit_id = request["fit_id"]
        base = {name: value for name, value in request.items() if name != "fit_id"}
        if (request["logical_id"] != logical_id or request["campaign_id"] != self.plan.campaign_id
                or fit_id != f"{logical_id}-{digest(base)[:20]}"):
            raise ValueError(f"Invalid assignment identity for {logical_id}")
        return request

    def result_marker(self, logical_id):
        directory = self.results / logical_id
        marker = read_json(directory / "published.json")
        request = self.request(logical_id)
        if (marker["fit_id"] != request["fit_id"] or marker["campaign_id"] != self.plan.campaign_id
                or marker["trainer_marker_sha256"] != sha256(directory / "complete.json")
                or marker["environment_sha256"] != sha256(directory / "environment.json")):
            raise ValueError(f"Published result identity is inconsistent for {logical_id}")
        for name, size in marker["sizes"].items():
            if Path(name).name != name or (directory / name).stat().st_size != size:
                raise ValueError(f"Incomplete or changed artifact {name} for {logical_id}")
        return marker, request

    def inner_result(self, fit):
        marker, request = self.result_marker(fit.logical_id)
        path = self.results / fit.logical_id
        summary = validate_trainer_output(path, request)
        return {"fit_id": request["fit_id"], "inner_fold": fit.inner,
                "val_equal_mse": summary["val_equal_mse"], "best_epoch": summary["best_epoch"],
                "val_observed_counts": summary["val_observed_counts"],
                "trainer_marker_sha256": marker["trainer_marker_sha256"]}

    def predictions(self, fits):
        return [(fit, load_predictions(self.results / fit.logical_id, self.request(fit.logical_id)["predict_ids"]))
                for fit in fits]

    def select_candidate(self, stage, directory, summaries):
        candidates = defaultdict(list)
        for fit in stage.inputs:
            candidates[fit.candidate].append(summaries[fit.logical_id])
        for candidate, results in candidates.items():
            if sorted(row["inner_fold"] for row in results) != [1, 2, 3, 4, 5]:
                raise ValueError(f"Missing inner folds for candidate {candidate}")
        scores = {candidate: statistics.mean(item["val_equal_mse"] for item in results)
                  for candidate, results in candidates.items()}
        chosen = min_with_tolerance(scores)
        selected = {"candidate_id": chosen, "epochs": {"0.0": stopping_epochs(candidates[chosen])},
                    "candidate_scores": scores}
        if stage.kind == "single":
            return selected
        pairs = self.predictions(fit for fit in stage.inputs if fit.candidate == chosen)
        outer = stage.inputs[0].outer
        oof = collect_inner_oof(pairs, set(self.plan.inner[outer]))
        eligible, truth, predictions, valid = triad_arrays(self.plan, outer, oof)
        selected["alpha_controls"] = alpha_selection(predictions, truth, valid)
        selected["n_eligible"] = len(eligible)
        selected["n_invalid"] = int((~valid).sum())
        write_csv(directory / "baseline_oof.csv", ("logical_id", "inner_fold", "structure_id", *PRED_COLUMNS),
                  prediction_csv_rows(pairs))
        write_csv(directory / "alpha_selection_mask.csv",
                  ("structure_id", "inner_fold", "same_row_complete_triad", "valid_baseline_triad", "used"),
                  ((sid, self.plan.inner[outer][sid], int(sid in self.plan.same_row_triads),
                    int(np.isfinite(oof[sid][:3]).all()), int(used))
                   for sid, used in self.mask_rows(oof, eligible, valid)))
        return selected

    @staticmethod
    def mask_rows(oof, eligible, scored):
        by_id = dict(zip(eligible, scored.tolist()))
        return [(sid, by_id.get(sid, False)) for sid in sorted(oof)]

    def select_lambda(self, stage, directory, summaries, parent):
        baseline_stage = next(item for item in self.plan.stages if item.name == stage.parent)
        baseline_fits = [fit for fit in baseline_stage.inputs if fit.candidate == parent["candidate_id"]]
        for fit in baseline_fits:
            if self.inner_result(fit) != parent["input_fits"][fit.logical_id]:
                raise ValueError(f"Baseline completion changed after selection: {fit.logical_id}")
        by_lambda = {0.0: baseline_fits}
        for value in LAMBDAS[1:]:
            by_lambda[value] = [fit for fit in stage.inputs if float(fit.policy) == value]
        outer = stage.inputs[0].outer
        pairs = {value: self.predictions(fits) for value, fits in by_lambda.items()}
        oof = {value: collect_inner_oof(predictions, set(self.plan.inner[outer]))
               for value, predictions in pairs.items()}
        arrays = {value: triad_arrays(self.plan, outer, predictions) for value, predictions in oof.items()}
        eligible, truth, _, _ = arrays[0.0]
        common = np.logical_and.reduce([array[3] for array in arrays.values()])
        scores = {}
        for value, (_, _, predictions, valid) in arrays.items():
            scores[value] = {**triad_score(predictions, truth, common), "n_invalid": int((~valid).sum())}
        mae = {value: score["macro_log10_mae"] for value, score in scores.items()}
        chosen = min_with_tolerance({value: error for value, error in mae.items() if value > 0})
        chosen_or_zero = min_with_tolerance({0.0: mae[0.0], chosen: mae[chosen]})
        epochs = parent["epochs"].copy()
        for value in LAMBDAS[1:]:
            epochs[str(value)] = stopping_epochs([summaries[fit.logical_id] for fit in by_lambda[value]])
        selected = {"candidate_id": parent["candidate_id"], "selected_lambda": chosen,
                    "selected_zero_eligible_lambda": chosen_or_zero,
                    "all_positives_worse_than_zero": all(mae[value] > mae[0.0] + TOLERANCE for value in LAMBDAS[1:]),
                    "n_eligible": len(eligible), "epochs": epochs,
                    "lambda_scores": {str(value): score for value, score in scores.items()},
                    "baseline_selection": stage.parent}
        selected["aliases"] = self.curve_aliases(stage, baseline_stage, selected)
        write_csv(directory / "lambda_oof.csv", ("lambda", "logical_id", "inner_fold", "structure_id", *PRED_COLUMNS),
                  ((value, *row) for value, predictions in pairs.items() for row in prediction_csv_rows(predictions)))
        write_csv(directory / "lambda_selection_mask.csv",
                  ("structure_id", "inner_fold", "same_row_complete_triad", "used", *[f"valid_{v}" for v in LAMBDAS]),
                  ((sid, self.plan.inner[outer][sid], int(sid in self.plan.same_row_triads), int(used),
                    *[int(np.isfinite(oof[value][sid][:3]).all()) for value in LAMBDAS])
                   for sid, used in self.mask_rows(oof[0.0], eligible, common)))
        return selected

    def curve_aliases(self, stage, baseline_stage, selection):
        final_fits = [fit for fit in baseline_stage.outputs + stage.outputs if fit.block == "primary_final"]
        policies = {"baseline": {}, "selected_positive": {}, "fixed_lambda_curve": {str(v): {} for v in LAMBDAS}}
        physical = set()
        for fit in final_fits:
            request = self.plan.request(fit, selection)
            value = request["lambda_value"]
            seed = str(fit.seed)
            if (value, fit.seed) in physical:
                raise ValueError(f"Duplicate physical final fit for {stage.name}, lambda {value}, seed {seed}")
            physical.add((value, fit.seed))
            if value == 0:
                policies["baseline"][seed] = fit.logical_id
            elif value == selection["selected_lambda"]:
                policies["selected_positive"][seed] = fit.logical_id
            if fit.seed in SEEDS[:3]:
                policies["fixed_lambda_curve"][str(value)][seed] = fit.logical_id
        if (len(physical) != 33 or set(policies["baseline"]) != set(map(str, SEEDS))
                or set(policies["selected_positive"]) != set(map(str, SEEDS))
                or any(set(group) != set(map(str, SEEDS[:3])) for group in policies["fixed_lambda_curve"].values())):
            raise ValueError(f"Primary policies for {stage.name} do not cover the specified seeds")
        return policies

    def create_selection(self, stage):
        target = self.selections / stage.name
        directory = Path(mkdtemp(prefix=f"{stage.name}-", dir=self.attempts))
        try:
            summaries = {fit.logical_id: self.inner_result(fit) for fit in stage.inputs}
            parent = self.load_selection(stage.parent) if stage.parent else None
            if stage.kind in {"baseline", "single"}:
                selection = self.select_candidate(stage, directory, summaries)
            elif stage.kind == "lambda":
                selection = self.select_lambda(stage, directory, summaries, parent)
            else:
                lambda_value = 0.0 if stage.inputs[0].policy == "0" else parent["selected_lambda"]
                selection = {"candidate_id": parent["candidate_id"], "selected_lambda": parent["selected_lambda"],
                             "epochs": {str(lambda_value): stopping_epochs(list(summaries.values()))}}
            selection.update({"name": stage.name, "campaign_id": self.plan.campaign_id,
                              "kind": stage.kind, "input_fits": summaries,
                              "parent_selection": stage.parent,
                              "artifact_sha256": {file.name: sha256(file) for file in directory.iterdir()}})
            write_json(directory / "selection.json", selection)
            publish_directory(directory, target)
        finally:
            if directory.exists():
                shutil.rmtree(directory)
        return self.load_selection(stage.name)

    def load_selection(self, name):
        selection = read_json(self.selections / name / "selection.json")
        if selection["campaign_id"] != self.plan.campaign_id or selection["name"] != name:
            raise ValueError(f"Invalid selection identity for {name}")
        return selection

    def refresh(self):
        """Any worker can publish ready assignments. Concurrent refreshes are idempotent."""
        existing = {path.stem for path in self.assignments.glob("*.json")}
        results = {path.name for path in self.results.iterdir()}
        for stage in self.plan.stages:
            missing = [fit for fit in stage.outputs if fit.logical_id not in existing]
            if not missing:
                continue
            error_path = self.selection_failures / f"{stage.name}.json"
            if error_path.is_file():
                try:
                    if not read_json(error_path)["retryable"]:
                        continue
                except FileNotFoundError:
                    pass
            try:
                if (self.selections / stage.name).is_dir():
                    selection = self.load_selection(stage.name)
                elif all(fit.logical_id in results for fit in stage.inputs):
                    selection = self.create_selection(stage)
                else:
                    continue
                for fit in missing:
                    self.assign(fit, selection)
                error_path.unlink(missing_ok=True)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                write_json(error_path, {
                    "name": stage.name, "time": utc_now(), "error": str(exc),
                    "traceback": traceback.format_exc(), "retryable": isinstance(exc, OSError),
                })

    def claim(self, logical_id):
        target = self.claims / logical_id
        try:
            # An empty directory can only come from an interruption during release.
            target.rmdir()
        except OSError as exc:
            if exc.errno not in {errno.ENOENT, errno.ENOTEMPTY, errno.EEXIST}:
                raise
        if target.exists() or (self.results / logical_id).exists() or self.failure_path(logical_id).exists():
            return None
        token = uuid4().hex
        attempt = self.attempts / logical_id / token
        staged = attempt / "claim"
        staged.mkdir(parents=True)
        owner = {
            "token": token, "logical_id": logical_id, "hostname": socket.gethostname(),
            "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "claimed_at": utc_now(),
        }
        write_json(staged / f"{token}.json", owner)
        write_json(attempt / "heartbeat.json", {"time": utc_now()})
        if not publish_directory(staged, target):
            shutil.rmtree(attempt)
            return None
        if (self.results / logical_id).exists() or self.failure_path(logical_id).exists():
            self.release_claim(logical_id, token)
            return None
        return owner, attempt

    def owns_claim(self, logical_id, token):
        return (self.claims / logical_id / f"{token}.json").is_file()

    def release_claim(self, logical_id, token):
        path = self.claims / logical_id
        try:
            (path / f"{token}.json").unlink()
        except FileNotFoundError:
            return
        try:
            path.rmdir()
        except OSError as exc:
            if exc.errno not in {errno.ENOENT, errno.ENOTEMPTY, errno.EEXIST}:
                raise

    def failure_path(self, logical_id):
        return self.failures / f"{logical_id}.json"

    def execute(self, owner, attempt, device, trainer=None):
        logical_id, token = owner["logical_id"], owner["token"]
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(30):
                if not self.owns_claim(logical_id, token):
                    return
                write_json(attempt / "heartbeat.json", {"time": utc_now()})

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        started = time.monotonic()
        result = attempt / "fit"
        try:
            request = self.request(logical_id)
            if trainer is None:
                from mass_balance_train import run_fit as trainer
            trainer(request, self.plan.release_dir, self.plan.feature_dir, result, device=device)
            validate_trainer_output(result, request, full=True)
            load_predictions(result, request["predict_ids"])
            if not self.owns_claim(logical_id, token):
                raise RuntimeError(f"Claim for {logical_id} was released during training")
            write_json(result / "environment.json", {**owner, "platform": platform.platform(),
                       "python": sys.version, "requested_device": device, "finished_at": utc_now(),
                       "worker_seconds": time.monotonic() - started})
            published = {
                "fit_id": request["fit_id"], "campaign_id": self.plan.campaign_id,
                "trainer_marker_sha256": sha256(result / "complete.json"),
                "environment_sha256": sha256(result / "environment.json"),
                "sizes": {name: (result / name).stat().st_size for name in FIT_FILES},
            }
            for name in (*FIT_FILES, "complete.json"):
                with (result / name).open("r+b") as handle:
                    os.fsync(handle.fileno())
            write_json(result / "published.json", published)
            sync_directory(result)
            if not publish_directory(result, self.results / logical_id):
                raise RuntimeError(f"A result already exists for {logical_id}")
            return True
        except (Exception, KeyboardInterrupt) as exc:
            failure = {**owner, "finished_at": utc_now(), "error_type": type(exc).__name__,
                       "error": str(exc), "traceback": traceback.format_exc()}
            write_json(attempt / "failure.json", failure)
            if self.owns_claim(logical_id, token) and not (self.results / logical_id).is_dir():
                write_json(self.failure_path(logical_id), failure)
            if isinstance(exc, KeyboardInterrupt):
                raise
            print(f"{logical_id} failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return False
        finally:
            stop.set()
            thread.join()
            self.release_claim(logical_id, token)

    def claim_metadata(self, logical_id):
        files = list((self.claims / logical_id).glob("*.json"))
        if not files:
            return None
        if len(files) != 1:
            raise ValueError(f"Expected one claim owner for {logical_id}")
        try:
            owner = read_json(files[0])
            attempt = self.attempts / logical_id / owner["token"]
            heartbeat = read_json(attempt / "heartbeat.json")["time"]
        except FileNotFoundError:
            return None
        return {**owner, "heartbeat": heartbeat}

    def retry(self, logical_id, token=None, stale_seconds=900):
        if logical_id not in self.plan.fits:
            raise ValueError(f"Unknown logical ID {logical_id}")
        if (self.results / logical_id).exists():
            raise ValueError(f"Published result {logical_id} already exists")
        owner = self.claim_metadata(logical_id)
        if owner is not None:
            if owner["token"] != token:
                raise ValueError("Recovery requires the exact abandoned claim token from status")
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(owner["heartbeat"])).total_seconds()
            if age < stale_seconds:
                raise ValueError(f"The heartbeat for {logical_id} is only {age:.1f} seconds old")
            if os.name != "nt" and owner["hostname"] == socket.gethostname():
                try:
                    os.kill(owner["pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError(f"The owning process for {logical_id} is still running")
            attempt = self.attempts / logical_id / token
            write_json(attempt / "recovered.json", {**owner, "recovered_at": utc_now()})
            self.release_claim(logical_id, token)
        failed = self.failure_path(logical_id)
        if failed.exists():
            failed.unlink()
        elif owner is None:
            raise ValueError(f"No failed or abandoned attempt to retry for {logical_id}")

    def status(self, verify=False):
        assigned = {path.stem for path in self.assignments.glob("*.json")}
        result_ids = {path.name for path in self.results.iterdir()}
        failure_ids = {path.stem for path in self.failures.glob("*.json")}
        owners = [owner for path in self.claims.iterdir() if path.name not in result_ids | failure_ids
                  and (owner := self.claim_metadata(path.name)) is not None]
        active_ids = {owner["logical_id"] for owner in owners}
        invalid = {}
        if verify:
            for logical_id in sorted(result_ids):
                try:
                    _, request = self.result_marker(logical_id)
                    validate_trainer_output(self.results / logical_id, request, full=True)
                    load_predictions(self.results / logical_id, request["predict_ids"])
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    invalid[logical_id] = str(exc)
            for directory in self.selections.iterdir():
                try:
                    selection = self.load_selection(directory.name)
                    artifacts = selection["artifact_sha256"]
                    checked_files(directory, artifacts, artifacts)
                    for logical_id, value in selection["input_fits"].items():
                        if sha256(self.results / logical_id / "complete.json") != value["trainer_marker_sha256"]:
                            raise ValueError(f"Input completion changed for {logical_id}")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    invalid[directory.name] = str(exc)
        by_block = defaultdict(Counter)
        for logical_id, fit in self.plan.fits.items():
            if logical_id in invalid:
                state = "invalid"
            elif logical_id in result_ids:
                state = "completed"
            elif logical_id in failure_ids:
                state = "failed"
            elif logical_id in active_ids:
                state = "claimed"
            elif logical_id in assigned:
                state = "ready"
            else:
                state = "waiting"
            by_block[fit.block][state] += 1
        totals = sum(by_block.values(), Counter())
        return {"campaign_id": self.plan.campaign_id, "logical_fit_count": len(self.plan.fits),
                "totals": totals, "by_block": by_block, "claims": owners, "failed_ids": sorted(failure_ids),
                "selection_failures": sorted(path.stem for path in self.selection_failures.glob("*.json")),
                "invalid_artifacts": invalid, "verified": verify}

    def work(self, fits, device, max_fits=None, max_seconds=None, idle_timeout=300, poll_seconds=5):
        started = last_work = time.monotonic()
        next_refresh = 0.0
        attempted = succeeded = 0
        pool = {fit.logical_id for fit in fits}
        verified = set()
        invalid = {}
        while max_fits is None or attempted < max_fits:
            now = time.monotonic()
            if max_seconds is not None and now - started >= max_seconds:
                break
            if now >= next_refresh:
                self.refresh()
                next_refresh = now + 10 + os.getpid() % 11
            completed = {path.name for path in self.results.iterdir()}
            for logical_id in sorted((completed & pool) - verified - invalid.keys()):
                try:
                    self.result_marker(logical_id)
                    verified.add(logical_id)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    invalid[logical_id] = str(exc)
                    print(f"Invalid published result {logical_id}: {exc}", file=sys.stderr, flush=True)
            failed = {path.stem for path in self.failures.glob("*.json")}
            pending = [fit for fit in fits if fit.logical_id not in completed | failed]
            if not pending:
                break
            ready = {path.stem for path in self.assignments.glob("*.json")}
            for fit in pending:
                if fit.logical_id not in ready:
                    continue
                claim = self.claim(fit.logical_id)
                if claim is None:
                    continue
                owner, attempt = claim
                print(f"{utc_now()} started {fit.logical_id} {fit.block} {fit.backbone} {device}", flush=True)
                successful = self.execute(owner, attempt, device)
                attempted += 1
                succeeded += successful
                if successful:
                    verified.add(fit.logical_id)
                print(f"{utc_now()} {'completed' if successful else 'failed'} {fit.logical_id}", flush=True)
                last_work = time.monotonic()
                break
            else:
                if now - last_work >= idle_timeout:
                    break
                time.sleep(poll_seconds)
        self.refresh()
        return {"attempted": attempted, "succeeded": succeeded, "failed": attempted - succeeded,
                "invalid_artifacts": invalid}


def csv_option(value):
    return set(value.split(","))


def nonnegative(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("expected a finite nonnegative number")
    return result


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--release-dir", required=True, type=Path)
    common.add_argument("--feature-dir", required=True, type=Path)
    common.add_argument("--plan-dir", required=True, type=Path)
    common.add_argument("--output-dir", required=True, type=Path)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", parents=[common], help="verify bundles and publish root assignments")
    status = commands.add_parser("status", parents=[common], help="report campaign and claim status")
    status.add_argument("--verify", action="store_true", help="rehash every completed output and selection artifact")
    worker = commands.add_parser("worker", parents=[common], help="train one fit at a time")
    worker.add_argument("--device", default="cuda")
    worker.add_argument("--blocks", type=csv_option, help="comma-separated fit blocks to claim")
    worker.add_argument("--backbones", type=csv_option, help="comma-separated manifest backbones to claim")
    worker.add_argument("--logical-ids", type=csv_option, help="comma-separated logical IDs to claim")
    worker.add_argument("--max-fits", type=int, help="stop after this many attempted fits")
    worker.add_argument("--max-seconds", type=nonnegative, help="stop claiming after this runtime")
    worker.add_argument("--idle-timeout", default=300.0, type=nonnegative)
    worker.add_argument("--poll-seconds", default=5.0, type=nonnegative)
    retry = commands.add_parser("retry", parents=[common], help="retry a failed fit or a confirmed abandoned claim")
    retry.add_argument("--logical-id", required=True)
    retry.add_argument("--claim-token", help="required for a confirmed abandoned claim")
    retry.add_argument("--stale-seconds", default=900.0, type=nonnegative)
    retry_selection = commands.add_parser("retry-selection", parents=[common], help="recheck a failed selection")
    retry_selection.add_argument("--selection", required=True)
    args = parser.parse_args(argv)
    plan = Plan(args.release_dir, args.feature_dir, args.plan_dir)
    campaign = Campaign(plan, args.output_dir)
    if args.command == "init":
        campaign.init()
        report = campaign.status()
    else:
        campaign.verify_identity()
        if args.command == "status":
            report = campaign.status(verify=args.verify)
        elif args.command == "retry":
            campaign.retry(args.logical_id, args.claim_token, args.stale_seconds)
            report = {"retry_enabled": args.logical_id}
        elif args.command == "retry-selection":
            if args.selection not in {stage.name for stage in plan.stages}:
                parser.error(f"Unknown selection {args.selection}")
            path = campaign.selection_failures / f"{args.selection}.json"
            archived = campaign.attempts / f"selection-error-{args.selection}-{uuid4().hex}.json"
            path.rename(archived)
            campaign.refresh()
            report = campaign.status()
        else:
            if args.poll_seconds <= 0 or (args.max_fits is not None and args.max_fits <= 0):
                parser.error("--poll-seconds and --max-fits must be positive")
            for option, available in (("blocks", set(BLOCK_COUNTS)),
                                      ("backbones", {fit.backbone for fit in plan.fits.values()}),
                                      ("logical_ids", set(plan.fits))):
                requested = getattr(args, option)
                if requested and not requested <= available:
                    parser.error(f"Unknown {option}: {', '.join(sorted(requested - available))}")
            fits = [fit for fit in plan.fits.values()
                    if (args.blocks is None or fit.block in args.blocks)
                    and (args.backbones is None or fit.backbone in args.backbones)
                    and (args.logical_ids is None or fit.logical_id in args.logical_ids)]
            if not fits:
                parser.error("No logical fits match these worker filters")
            report = campaign.work(fits, args.device, args.max_fits, args.max_seconds,
                                   args.idle_timeout, args.poll_seconds)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report.get("invalid_artifacts") or report.get("failed") or report.get("selection_failures") else 0


if __name__ == "__main__":
    sys.exit(main())
