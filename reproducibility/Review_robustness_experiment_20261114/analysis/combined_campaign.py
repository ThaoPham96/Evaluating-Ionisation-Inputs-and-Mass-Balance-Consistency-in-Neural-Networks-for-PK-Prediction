"""Portable manifest, work queue, and prediction exports for the combined study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import threading
import time
import traceback
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "COMBINED_RUN_PLAN_v1.txt"
PRIOR_PLAN_SHA = "a9366c92a4d94fd9950faae10cb089bb45ad44358175a34098761b60342771d5"
FINGERPRINT = "99e83f5ad3b3802ab2374777eff38f8d9d66e2178237c0e1ea5f23d143948cb8"
TRIADS = "same_row_complete_triads.csv"
RECIPES = "p0_recipes.json"
BACKBONES = ("Chemprop", "GraphConv", "ChemBERTa_head")
FOLDS = tuple(range(1, 6))
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909)
LAMBDAS = (0., .03, .1, .3, 1., 3., 10.)
ARMS = ("P0", "P1")
CELLS = ("00", "10", "01", "11")
ENDPOINTS = {
    "t_half": "t1/2(hour)", "Vd": "VD(liter)", "CL": "Cl(liter/hour)",
    "Cmax": "Cmax_uM", "Tmax": "Tmax", "F": "F(percentage)", "PPB": "PPB(percentage)",
}
TRIAD_ENDPOINTS = tuple(ENDPOINTS)[:3]
BOUNDED = {"F", "PPB"}
N_LABELS = (316, 259, 246, 190, 249, 172, 292)
ARTIFACTS = {"summary.json", "predictions.csv", "checkpoint.pt", "scaler.json",
             "request.json", "learning_curve.csv"}
TOL = 1e-8


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(temp)
        replace_path(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def replace_path(source, destination, *, replace=True):
    operation = os.replace if replace else os.rename
    for _ in range(50):
        try:
            operation(source, destination)
            return
        except PermissionError:
            if os.name != "nt":
                raise
            time.sleep(.02)
    operation(source, destination)


def write_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write(path, lambda temp: temp.write_text(payload, encoding="utf-8", newline="\n"))


def write_csv(path, columns, records):
    def write(temp):
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)

    atomic_write(path, write)


def by_id(records):
    result = {record["structure_id"]: record for record in records}
    if len(result) != len(records) or "" in result:
        raise ValueError("Missing or duplicate structure IDs")
    return result


def code_hashes():
    analysis = ROOT / "analysis"
    paths = [analysis / name for name in ("combined_campaign.py", "combined_features.py", "combined_train.py")]
    vendor = analysis / "vendor_ionisation"
    paths.extend(vendor.rglob("*.py"))
    paths.extend(vendor.rglob("SOURCE_MANIFEST.json"))
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}


def selection_id(backbone, outer):
    return f"{backbone}-outer{outer}"


def bundles():
    result = {}
    for backbone in BACKBONES:
        for outer in FOLDS:
            group = selection_id(backbone, outer)
            for inner in FOLDS:
                key = f"inner-{group}-inner{inner}"
                result[key] = {"stage": "inner", "backbone": backbone, "outer_fold": outer,
                               "inner_fold": inner, "seed": 42, "selection_id": group}
            for seed in SEEDS:
                key = f"final-{group}-seed{seed}"
                result[key] = {"stage": "final", "backbone": backbone, "outer_fold": outer,
                               "inner_fold": None, "seed": seed, "selection_id": group}
    return result


def read_population(release_dir):
    dataset = by_id(rows(release_dir / "model_dataset.csv"))
    outer = by_id(rows(release_dir / "outer.csv"))
    inner_rows = rows(release_dir / "inner.csv")
    if len(dataset) != 327 or set(dataset) != set(outer):
        raise ValueError("Unexpected numerical cohort or outer fold coverage")
    groups = {}
    outer_map = {}
    for sid, row in outer.items():
        group, fold = row["group_id"], int(row["outer_fold"])
        if (row["group_id"], row["outer_fold"]) != (dataset[sid]["group_id"], dataset[sid]["outer_fold"]):
            raise ValueError(f"Outer fold does not match release metadata for {sid}")
        if group in groups and groups[group] != fold:
            raise ValueError(f"Group spans outer folds: {group}")
        groups[group] = fold
        outer_map[sid] = fold
    if len(groups) != 238 or set(groups.values()) != set(FOLDS):
        raise ValueError("Unexpected outer groups/folds")
    inner_map = {str(fold): {} for fold in FOLDS}
    for row in inner_rows:
        sid = row["structure_id"]
        outer_fold = row["outer_fold"]
        part = inner_map[outer_fold]
        if sid in part or row["group_id"] != dataset[sid]["group_id"]:
            raise ValueError(f"Invalid inner fold assignment for {sid}")
        part[sid] = int(row["inner_fold"])
    for fold, part in inner_map.items():
        if set(part) != {sid for sid in dataset if outer_map[sid] != int(fold)}:
            raise ValueError(f"Inner coverage/leakage for outer fold {fold}")
        if set(part.values()) != set(FOLDS):
            raise ValueError(f"Inner folds missing for outer fold {fold}")
        by_group = defaultdict(set)
        for sid, inner_fold in part.items():
            by_group[dataset[sid]["group_id"]].add(inner_fold)
        if any(len(folds) != 1 for folds in by_group.values()):
            raise ValueError(f"Inner group leakage for outer fold {fold}")
    for name, n_labels in zip(ENDPOINTS, N_LABELS):
        values = [float(row[ENDPOINTS[name]]) for row in dataset.values() if row[ENDPOINTS[name]]]
        if len(values) != n_labels or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Unexpected {name} label count or nonfinite labels")
        if name in BOUNDED and any(not 0 <= value <= 100 for value in values):
            raise ValueError(f"Invalid bounded labels for {name}")
        if name not in BOUNDED and any(value <= 0 for value in values):
            raise ValueError(f"Nonpositive labels for {name}")
    return dataset, {"outer": outer_map, "inner": inner_map}


def validate_triads(triads, dataset, splits):
    counts = Counter()
    excluded = 0
    for sid, row in triads.items():
        data = dataset[sid]
        for field in ("group_id", "outer_fold", "source_definition_sensitivity_exclude"):
            if row[field] != data[field]:
                raise ValueError(f"Same-row triad metadata mismatch for {sid}: {field}")
        for endpoint in TRIAD_ENDPOINTS:
            field = ENDPOINTS[endpoint]
            if float(row[field]) != float(data[field]):
                raise ValueError(f"Same-row triad value mismatch for {sid}: {field}")
        counts[int(row["outer_fold"])] += 1
        excluded += row["source_definition_sensitivity_exclude"] == "True"
    if (len(triads) != 242 or len({row["group_id"] for row in triads.values()}) != 181
            or counts != {1: 48, 2: 48, 3: 57, 4: 47, 5: 42} or excluded != 4):
        raise ValueError("Unexpected same-row triad population")
    inner_counts = [sum(sid in triads and fold == inner for sid, fold in part.items())
                    for part in splits["inner"].values() for inner in FOLDS]
    if (min(inner_counts), max(inner_counts)) != (27, 52):
        raise ValueError("Unexpected same-row triad coverage in inner folds")


def input_identity(release_dir, feature_dir):
    from combined_features import verify_inputs

    inputs = verify_inputs(release_dir, feature_dir)
    if (inputs["fingerprint"], inputs["structures"], inputs["groups"]) != (FINGERPRINT, 327, 238):
        raise ValueError("Frozen numerical release identity mismatch")
    return inputs


def build_plan(plan_path, release_dir, feature_dir, prior_plan, source_triads):
    if sha(prior_plan) != PRIOR_PLAN_SHA:
        raise ValueError("Prior plan differs from the frozen pH plan")
    source = read_json(prior_plan)
    anchor = source["identity"]["anchor_recipes"]
    keys = {selection_id(model, outer) for model in BACKBONES for outer in FOLDS}
    if set(anchor) != keys:
        raise ValueError("Unexpected P0 recipe coverage")
    recipes = {key: {field: anchor[key][field] for field in
                     ("config", "candidate_index", "prior_unit_id", "prior_selection_sha256",
                      "prior_complete_unit_sha256")} for key in sorted(keys)}
    inputs = input_identity(release_dir, feature_dir)
    if {key: inputs[key] for key in ("release", "features", "fingerprint", "structures")} != (
            source["identity"]["datasets"]["numerical"]):
        raise ValueError("The archived P0 selection used different frozen inputs")
    dataset, splits = read_population(release_dir)
    expected_sha = read_json(release_dir / "manifest.json")["inputs"]["mass_balance"][TRIADS]
    if sha(source_triads) != expected_sha:
        raise ValueError("Source triads differ from the version bound to the numerical release")
    selected_triads = [row for row in rows(source_triads) if row["structure_id"] in dataset]
    validate_triads(by_id(selected_triads), dataset, splits)

    triad_path = plan_path.with_name(TRIADS)
    recipe_path = plan_path.with_name(RECIPES)
    write_csv(triad_path, list(selected_triads[0]), selected_triads)
    write_json(recipe_path, recipes)
    identity = {
        "schema_version": 1, "protocol_sha256": sha(ROOT / PROTOCOL), "inputs": inputs,
        "code": code_hashes(), "prior_plan_sha256": PRIOR_PLAN_SHA, "prior_campaign_id": source["campaign_id"],
        "triads_sha256": sha(triad_path), "source_triads_sha256": expected_sha,
        "recipes_sha256": sha(recipe_path), "recipes": recipes, "splits": splits,
        "backbones": list(BACKBONES), "folds": list(FOLDS), "seeds": list(SEEDS), "search_seed": 42,
        "lambdas": list(LAMBDAS), "feature_arms": list(ARMS), "max_epochs": 200,
        "patience": 30, "min_delta": 1e-4, "task_weights": [1.] * 7,
        "counts": {"inner_bundles": 75, "selections": 15, "final_bundles": 135,
                   "inner_fits": 1050, "final_fits": 540, "total_fits": 1590, "final_rows": 35316},
    }
    plan = {"campaign_id": digest(identity), "identity": identity, "bundles": bundles()}
    write_json(plan_path, plan)
    return plan


def verify_plan(plan_path, release_dir, feature_dir):
    plan = read_json(plan_path)
    identity = plan["identity"]
    if plan["campaign_id"] != digest(identity) or plan["bundles"] != bundles():
        raise ValueError("Campaign manifest identity or bundle assignments changed")
    if identity["code"] != code_hashes() or identity["protocol_sha256"] != sha(ROOT / PROTOCOL):
        raise ValueError("Campaign source code or protocol changed")
    if identity["inputs"] != input_identity(release_dir, feature_dir):
        raise ValueError("Campaign inputs changed")
    triad_path, recipe_path = plan_path.with_name(TRIADS), plan_path.with_name(RECIPES)
    if (sha(triad_path) != identity["triads_sha256"] or sha(recipe_path) != identity["recipes_sha256"]
            or read_json(recipe_path) != identity["recipes"]):
        raise ValueError("Campaign triads or P0 recipes changed")
    dataset, splits = read_population(release_dir)
    if splits != identity["splits"]:
        raise ValueError("Campaign split assignments changed")
    triads = by_id(rows(triad_path))
    validate_triads(triads, dataset, splits)
    return plan, dataset, triads


def fit_request(plan, bundle, arm, lambda_value, fixed_epochs=None):
    outer, inner = bundle["outer_fold"], bundle["inner_fold"]
    splits = plan["identity"]["splits"]
    outer_train = splits["inner"][str(outer)]
    validation = sorted(sid for sid, fold in outer_train.items() if fold == inner)
    predict = validation if inner else sorted(sid for sid, fold in splits["outer"].items() if fold == outer)
    cell = f"{int(arm == 'P1')}{int(lambda_value > 0)}"
    request = {
        "campaign_id": plan["campaign_id"], "cohort": "numerical", "backbone": bundle["backbone"],
        "feature_arm": arm, "cell": cell, "outer_fold": outer, "inner_fold": inner,
        "config": plan["identity"]["recipes"][bundle["selection_id"]]["config"],
        "seed": bundle["seed"], "max_epochs": 200, "fixed_epochs": fixed_epochs,
        "lambda_value": lambda_value, "task_weights": [1.] * 7,
        "train_ids": sorted(sid for sid, fold in outer_train.items() if fold != inner),
        "val_ids": validation, "predict_ids": predict,
    }
    request["fit_id"] = "COM-" + digest(request)[:24]
    return request


def inner_requests(plan, bundle_id):
    bundle = plan["bundles"][bundle_id]
    if bundle["stage"] != "inner":
        raise ValueError(f"Expected an inner bundle: {bundle_id}")
    return [fit_request(plan, bundle, arm, value) for value in LAMBDAS for arm in ARMS]


def final_requests(plan, bundle_id, selection):
    bundle = plan["bundles"][bundle_id]
    if (bundle["stage"] != "final" or bundle["selection_id"] != selection["selection_id"]
            or plan["campaign_id"] != selection["campaign_id"]):
        raise ValueError(f"Selection does not match final bundle {bundle_id}")
    grid = {entry["lambda_value"]: entry for entry in selection["grid"]}
    return [fit_request(plan, bundle, arm, value, grid[value]["arms"][arm]["median_epoch"])
            for value in (0., selection["selected_lambda"]) for arm in ARMS]


def validate_fit(path, request):
    marker = read_json(path / "complete.json")
    if marker["fit_id"] != request["fit_id"] or set(marker["files"]) != ARTIFACTS:
        raise ValueError(f"Invalid fit completion marker: {path}")
    for filename, expected in marker["files"].items():
        if sha(path / filename) != expected:
            raise ValueError(f"Changed fit artifact: {path / filename}")
    saved = read_json(path / "request.json")
    if saved["request"] != request:
        raise ValueError(f"Fit request mismatch: {path}")
    ids = {key: request[key] for key in ("train_ids", "val_ids", "predict_ids")}
    if saved["effective_ids"] != ids or saved["resolved_hash"] != marker["resolved_hash"]:
        raise ValueError(f"Resolved fit identity mismatch: {path}")
    summary = read_json(path / "summary.json")
    for key in ("fit_id", "campaign_id", "backbone", "feature_arm", "cell", "seed", "outer_fold", "inner_fold",
                "lambda_value", "config", "task_weights"):
        if summary[key] != request[key]:
            raise ValueError(f"Fit summary {key} mismatch: {path}")
    if (summary["resolved_hash"] != marker["resolved_hash"]
            or summary["checkpoint_reload_verified"] is not True):
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    best, actual = summary["best_epoch"], summary["actual_epochs"]
    if not 1 <= best <= actual <= request["max_epochs"]:
        raise ValueError(f"Invalid fit epoch count: {path}")
    if request["fixed_epochs"] is not None and (best != request["fixed_epochs"] or actual != best):
        raise ValueError(f"Fixed fit did not train for requested epochs: {path}")
    if request["fixed_epochs"] is None and not math.isfinite(summary["val_equal_mse"]):
        raise ValueError(f"Invalid inner validation loss: {path}")
    predictions = rows(path / "predictions.csv")
    if [row["structure_id"] for row in predictions] != request["predict_ids"]:
        raise ValueError(f"Fit prediction IDs or order mismatch: {path}")
    for row in predictions:
        for endpoint in ENDPOINTS:
            transformed = float(row[f"pred_{endpoint}"])
            physical = float(row[f"physical_{endpoint}"])
            if not math.isfinite(transformed) or not math.isfinite(physical):
                raise ValueError(f"Nonfinite prediction: {path}")
            if endpoint in BOUNDED and not 0 <= physical <= 100:
                raise ValueError(f"Out-of-range {endpoint} prediction: {path}")
            if endpoint not in BOUNDED and physical <= 0:
                raise ValueError(f"Nonpositive {endpoint} prediction: {path}")
            if endpoint not in BOUNDED and not math.isclose(math.log10(physical), transformed, abs_tol=1e-12):
                raise ValueError(f"Log10 and physical {endpoint} predictions disagree: {path}")
    return summary, predictions


def validate_pairing(path, requests, summaries):
    if len({summary["environment_hash"] for summary in summaries}) != 1:
        raise ValueError(f"Matched fits have different runtime environments: {path}")
    if len({summary["common_state_hash"] for summary in summaries}) != 1:
        raise ValueError(f"Matched fits have different shared initialization: {path}")
    for arm in ARMS:
        if len({summary["initial_state_hash"] for summary in summaries if summary["feature_arm"] == arm}) != 1:
            raise ValueError(f"{arm} fits have different initial parameters: {path}")
    epoch_streams = {}
    for request, summary in zip(requests, summaries, strict=True):
        history = rows(path / request["fit_id"] / "learning_curve.csv")
        if [int(row["epoch"]) for row in history] != list(range(1, summary["actual_epochs"] + 1)):
            raise ValueError(f"Wrong learning-curve epoch sequence: {request['fit_id']}")
        for row in history:
            epoch = row["epoch"]
            stream = tuple(row[key] for key in ("batch_order_seed", "dropout_seed", "batch_order_sha256"))
            if epoch_streams.setdefault(epoch, stream) != stream:
                raise ValueError(f"Matched fits have different random streams at epoch {epoch}: {path}")


def checked_bundle(path, plan, bundle_id, requests):
    record = read_json(path / "bundle.json")
    if record["campaign_id"] != plan["campaign_id"] or record["bundle_id"] != bundle_id:
        raise ValueError(f"Wrong completed bundle identity: {path}")
    fits = record["fit_markers"]
    if set(fits) != {request["fit_id"] for request in requests}:
        raise ValueError(f"Wrong fit set for completed bundle: {path}")
    result = []
    for request in requests:
        fit_dir = path / request["fit_id"]
        if sha(fit_dir / "complete.json") != fits[request["fit_id"]]:
            raise ValueError(f"Changed fit completion marker: {fit_dir}")
        result.append(validate_fit(fit_dir, request))
    validate_pairing(path, requests, [summary for summary, _ in result])
    return record, result


def choose_strength(grid):
    best = min(entry["mean_score"] for entry in grid if entry["lambda_value"] > 0)
    selected = min(entry["lambda_value"] for entry in grid
                   if entry["lambda_value"] > 0 and entry["mean_score"] <= best + TOL)
    scores = {entry["lambda_value"]: entry["mean_score"] for entry in grid}
    zero_eligible = 0. if scores[0.] <= scores[selected] + TOL else selected
    return selected, zero_eligible


def selection_record(plan, group, completed, triads):
    units, sources, calibration = [], {}, []
    for fold in FOLDS:
        key = f"inner-{group}-inner{fold}"
        unit = plan["bundles"][key]
        requests = inner_requests(plan, key)
        path = completed / key
        _, fits = checked_bundle(path, plan, key, requests)
        sources[key] = sha(path / "bundle.json")
        units.append(unit)
        calibration.extend(zip(requests, fits, strict=True))
    backbone, outer = units[0]["backbone"], units[0]["outer_fold"]
    expected_ids = set(plan["identity"]["splits"]["inner"][str(outer)])
    return select_from_calibration(plan["campaign_id"], group, backbone, outer, expected_ids,
                                   triads, calibration, sources)


def select_from_calibration(campaign_id, group, backbone, outer, expected_ids, triads, calibration, sources):
    pool = defaultdict(dict)
    epochs = defaultdict(list)
    for request, (summary, predictions) in calibration:
        fit_key = (request["lambda_value"], request["feature_arm"])
        epochs[fit_key].append({"inner_fold": request["inner_fold"], "fit_id": request["fit_id"],
                                "best_epoch": summary["best_epoch"]})
        ids = by_id(predictions)
        if ids.keys() & pool[fit_key].keys():
            raise ValueError(f"Repeated inner prediction ID for {group}")
        pool[fit_key].update(ids)
    eligible = sorted(expected_ids & triads.keys())
    grid = []
    for value in LAMBDAS:
        arms = {}
        for arm in ARMS:
            key = (value, arm)
            if pool[key].keys() != expected_ids:
                raise ValueError(f"Incomplete pooled validation predictions for {group}: {key}")
            errors = {
                name: statistics.fmean(abs(float(pool[key][sid][f"pred_{name}"])
                                           - math.log10(float(triads[sid][ENDPOINTS[name]])))
                                       for sid in eligible)
                for name in TRIAD_ENDPOINTS
            }
            arms[arm] = {"score": statistics.fmean(errors.values()), "endpoint_log10_mae": errors,
                         "epochs": epochs[key], "median_epoch": int(statistics.median(
                             item["best_epoch"] for item in epochs[key]))}
        grid.append({"lambda_value": value, "mean_score": statistics.fmean(
            arms[arm]["score"] for arm in ARMS), "arms": arms})
    positive, zero_eligible = choose_strength(grid)
    return {"campaign_id": campaign_id, "selection_id": group, "backbone": backbone,
            "outer_fold": outer, "selected_lambda": positive, "zero_eligible_lambda": zero_eligible,
            "n_validation": len(expected_ids), "n_observed_triads": len(eligible), "grid": grid,
            "source_bundles": sources}


class Campaign:
    def __init__(self, plan, triads, output_dir):
        self.plan = plan
        self.triads = triads
        self.output_dir = output_dir
        self.claims = output_dir / "claims"
        self.attempts = output_dir / "attempts"
        self.completed = output_dir / "completed"
        self.selections = output_dir / "selections"
        self.groups = {bundle["selection_id"] for bundle in plan["bundles"].values()}

    def bind_output(self, *, create=False):
        # Atomic directory creation gives one output initializer. An immutable plan
        # record is visible only after its JSON is complete.
        path = self.output_dir / "campaign.json"
        owner = self.output_dir / ".initializing"
        if create and not path.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            try:
                owner.mkdir()
            except FileExistsError:
                for _ in range(100):
                    if path.is_file():
                        break
                    time.sleep(.1)
                else:
                    raise RuntimeError(f"Output initialization incomplete: {owner}")
            else:
                try:
                    if path.is_file():
                        return self.bind_output()
                    if set(self.output_dir.iterdir()) != {owner}:
                        raise ValueError(f"Unbound output directory contains existing files: {self.output_dir}")
                    for directory in (self.claims, self.attempts, self.completed, self.selections):
                        directory.mkdir()
                    write_json(path, {"campaign_id": self.plan["campaign_id"]})
                finally:
                    owner.rmdir()
        elif owner.is_dir() and not path.is_file():
            for _ in range(100):
                if path.is_file():
                    break
                time.sleep(.1)
        if path.exists():
            if read_json(path) != {"campaign_id": self.plan["campaign_id"]}:
                raise ValueError(f"Output belongs to a different campaign: {self.output_dir}")
        elif self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise ValueError(f"Output directory has no campaign identity: {self.output_dir}")

    def selection_path(self, group):
        return self.selections / f"{group}.json"

    def inner_paths(self, group):
        return {f"inner-{group}-inner{fold}": self.completed / f"inner-{group}-inner{fold}"
                for fold in FOLDS}

    def load_selection(self, group):
        path = self.selection_path(group)
        envelope = read_json(path)
        record = envelope["selection"]
        if (envelope["sha256"] != digest(record) or record["campaign_id"] != self.plan["campaign_id"]
                or record["selection_id"] != group):
            raise ValueError(f"Selection identity mismatch: {path}")
        sources = {key: sha(directory / "bundle.json") for key, directory in self.inner_paths(group).items()}
        if record["source_bundles"] != sources:
            raise ValueError(f"Selection source hashes changed: {path}")
        return record

    def publish_selections(self):
        for group in sorted(self.groups):
            path = self.selection_path(group)
            if not path.exists() and all(p.is_dir() for p in self.inner_paths(group).values()):
                # All inputs are immutable after bundle publication. Concurrent
                # selectors compute the same bytes and replace the same record.
                record = selection_record(self.plan, group, self.completed, self.triads)
                write_json(path, {"selection": record, "sha256": digest(record)})

    def requests(self, bundle_id):
        bundle = self.plan["bundles"][bundle_id]
        if bundle["stage"] == "inner":
            return inner_requests(self.plan, bundle_id)
        return final_requests(self.plan, bundle_id, self.load_selection(bundle["selection_id"]))

    def ready(self, bundle_id):
        bundle = self.plan["bundles"][bundle_id]
        return (not (self.completed / bundle_id).exists() and not (self.claims / bundle_id).exists()
                and not self.unresolved(bundle_id)
                and (bundle["stage"] == "inner" or self.selection_path(bundle["selection_id"]).is_file()))

    def unresolved(self, bundle_id):
        return [str(failure.relative_to(self.output_dir)) for failure in
                (self.attempts / bundle_id).glob("*/failure.json")
                if not failure.with_name("retry.json").is_file()]

    def pending(self):
        done = {path.name for path in self.completed.iterdir()}
        ready = [key for key in self.plan["bundles"] if self.ready(key)]

        def priority(key):
            bundle = self.plan["bundles"][key]
            group = bundle["selection_id"]
            # Ready finals and the expected long-running backbone first. Prefer
            # finishing an inner group so that subsequent seeds become runnable.
            return (bundle["stage"] != "final", BACKBONES.index(bundle["backbone"]),
                    -sum(inner in done for inner in self.inner_paths(group)), bundle["outer_fold"],
                    bundle["seed"], bundle["inner_fold"] or 0)

        return sorted(ready, key=priority)

    @contextmanager
    def claim(self, bundle_id, heartbeat_seconds):
        lock = self.claims / bundle_id
        try:
            lock.mkdir()
        except FileExistsError:
            yield None
            return
        attempt_id = uuid.uuid4().hex
        owner = {"bundle_id": bundle_id, "attempt_id": attempt_id, "hostname": socket.gethostname(),
                 "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                 "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "started_at": time.time()}
        stop = threading.Event()
        heartbeat = None

        def pulse():
            while not stop.wait(heartbeat_seconds):
                write_json(lock / "heartbeat.json", {"attempt_id": attempt_id, "at": time.time()})

        attempt = self.attempts / bundle_id / attempt_id
        try:
            if (self.completed / bundle_id).exists() or self.unresolved(bundle_id):
                yield None
                return
            # Publish the owner first so any interrupted start has an identifiable lease.
            write_json(lock / "owner.json", owner)
            attempt.mkdir(parents=True)
            write_json(attempt / "owner.json", owner)
            heartbeat = threading.Thread(target=pulse, daemon=True)
            heartbeat.start()
            yield attempt
        finally:
            stop.set()
            if heartbeat is not None:
                heartbeat.join()
            if (lock / "owner.json").exists():
                if read_json(lock / "owner.json")["attempt_id"] == attempt_id:
                    replace_path(lock, attempt / "claim", replace=False)
            elif lock.is_dir():
                lock.rmdir()

    def process(self, bundle_id, release_dir, feature_dir, device, heartbeat_seconds):
        from combined_train import run_fit

        with self.claim(bundle_id, heartbeat_seconds) as attempt:
            if attempt is None:
                return False
            owner = read_json(attempt / "owner.json")
            attempt_start = time.monotonic()
            work = attempt / "work"
            work.mkdir()
            try:
                requests = self.requests(bundle_id)
                fits = {}
                hardware = set()
                summaries = []
                for request in requests:
                    fit_path = work / request["fit_id"]
                    print(json.dumps({"event": "fit_start", "bundle_id": bundle_id,
                                      "fit_id": request["fit_id"]}), flush=True)
                    run_fit(request, release_dir, feature_dir, fit_path, device=device)
                    summary, _ = validate_fit(fit_path, request)
                    summaries.append(summary)
                    hardware.add(summary["device"])
                    fits[request["fit_id"]] = sha(fit_path / "complete.json")
                if len(hardware) != 1:
                    raise ValueError("Matched fits reported different hardware")
                validate_pairing(work, requests, summaries)
                lock_owner = read_json(self.claims / bundle_id / "owner.json")
                if lock_owner["attempt_id"] != owner["attempt_id"]:
                    raise RuntimeError("Bundle lease changed while processing")
                write_json(work / "bundle.json", {
                    **owner, "campaign_id": self.plan["campaign_id"], "fit_markers": fits,
                    "device": hardware.pop(), "elapsed_seconds": time.monotonic() - attempt_start,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                })
                replace_path(work, self.completed / bundle_id, replace=False)
            except BaseException as error:
                write_json(attempt / "failure.json", {
                    **owner, "time": time.time(), "error": repr(error), "traceback": traceback.format_exc(),
                })
                raise
            print(json.dumps({"event": "bundle_complete", "bundle_id": bundle_id,
                              "n_fits": len(requests)}), flush=True)
            return True

    def work(self, args):
        self.bind_output(create=True)
        started = time.monotonic()
        while True:
            if (args.max_seconds is not None and time.monotonic() - started >= args.max_seconds
                    or args.stop_file is not None and args.stop_file.exists()):
                break
            self.publish_selections()
            ready = [args.bundle_id] if args.bundle_id and self.ready(args.bundle_id) else self.pending()
            if args.bundle_id:
                ready = [key for key in ready if key == args.bundle_id]
            processed = next((True for key in ready
                              if self.process(key, args.release_dir, args.feature_dir,
                                              args.device, args.heartbeat_seconds)), False)
            if processed:
                self.publish_selections()
            if args.once or args.bundle_id:
                if not processed and args.bundle_id and not (self.completed / args.bundle_id).is_dir():
                    raise RuntimeError(f"Requested bundle is unavailable: {args.bundle_id}")
                break
            if not processed:
                state = self.status()
                if state["complete"]:
                    break
                if state["unresolved_failures"]:
                    raise RuntimeError("Work is blocked by failed attempts; inspect status and run recover")
                time.sleep(args.poll_seconds)

    def status(self):
        self.bind_output()
        counts = Counter()
        for key, bundle in self.plan["bundles"].items():
            if (self.completed / key / "bundle.json").is_file():
                counts[bundle["stage"]] += 1
        active = []
        if self.claims.is_dir():
            for path in sorted(self.claims.iterdir()):
                try:
                    owner = read_json(path / "owner.json")
                    heartbeat_path = path / "heartbeat.json"
                    heartbeat = read_json(heartbeat_path) if heartbeat_path.is_file() else None
                    active.append({**owner, "last_heartbeat": heartbeat["at"] if heartbeat else owner["started_at"]})
                except FileNotFoundError:
                    try:
                        modified = path.stat().st_mtime
                    except FileNotFoundError:
                        continue
                    active.append({"bundle_id": path.name, "last_heartbeat": modified})
        selection_count = sum(self.selection_path(group).is_file() for group in self.groups)
        inner, final = counts["inner"], counts["final"]
        failures = [failure for key in self.plan["bundles"] for failure in self.unresolved(key)]
        return {
            "campaign_id": self.plan["campaign_id"], "inner_bundles": inner,
            "final_bundles": final, "selections": selection_count,
            "inner_fits": 14 * inner, "final_fits": 4 * final, "total_fits": 14 * inner + 4 * final,
            "expected_total_fits": 1590, "active_claims": active, "unresolved_failures": failures,
            "complete": (inner, final, selection_count) == (75, 135, 15) and not failures,
        }

    def verify_results(self, complete=False):
        state = self.status()
        if complete and not state["complete"]:
            raise ValueError(f"Campaign incomplete: {state}")
        for group in sorted(self.groups):
            if self.selection_path(group).exists():
                saved = self.load_selection(group)
                expected = selection_record(self.plan, group, self.completed, self.triads)
                if saved != expected:
                    raise ValueError(f"Selection could not be independently regenerated: {group}")
        for key in self.plan["bundles"]:
            path = self.completed / key
            if path.exists():
                checked_bundle(path, self.plan, key, self.requests(key))
        return state

    def recover(self, bundle_id, *, owner_stopped, stale_seconds, reason):
        self.bind_output()
        lock = self.claims / bundle_id
        if lock.is_dir():
            if not owner_stopped:
                raise ValueError("Confirm that the recorded owner stopped with --confirm-owner-stopped")
            owner_path = lock / "owner.json"
            heartbeat_path = lock / "heartbeat.json"
            owner = read_json(owner_path) if owner_path.is_file() else None
            timestamp = lock.stat().st_mtime
            if owner:
                timestamp = (read_json(heartbeat_path)["at"] if heartbeat_path.is_file()
                             else owner["started_at"])
                if owner["hostname"] == socket.gethostname() and process_alive(owner["pid"]):
                    raise ValueError("Claim owner is still running on this host")
            if time.time() - timestamp < stale_seconds:
                raise ValueError("Claim heartbeat is too recent for recovery")
            attempt_id = owner["attempt_id"] if owner else f"incomplete-{uuid.uuid4().hex}"
            attempt = self.attempts / bundle_id / attempt_id
            attempt.mkdir(parents=True, exist_ok=True)
            if not (self.completed / bundle_id).is_dir() and not (attempt / "failure.json").exists():
                write_json(attempt / "failure.json", {"bundle_id": bundle_id, "reason": reason,
                                                       "owner": owner, "time": time.time()})
            replace_path(lock, attempt / "recovered_claim", replace=False)
        for failure in self.unresolved(bundle_id):
            path = self.output_dir / failure
            write_json(path.with_name("retry.json"), {"time": time.time(), "reason": reason})


def process_alive(pid):
    if os.name == "nt":
        # On Windows os.kill(pid, 0) would invoke TerminateProcess.
        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87
        try:
            return kernel.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 258
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


EXPORT_FIELDS = [
    "structure_id", "group_id", "backbone", "outer_fold", "inner_fold", "seed", "cell",
    "feature_arm", "lambda_value", "fit_id", "ionisation_class", "same_row_triad_eligible",
    "source_definition_sensitivity_exclude",
] + [f"{prefix}_{name}" for prefix in ("q", "true", "observed") for name in ENDPOINTS]


def export_row(request, prediction, dataset, triads):
    sid = prediction["structure_id"]
    entry = dataset[sid]
    row = {key: request[key] for key in
           ("backbone", "outer_fold", "inner_fold", "seed", "cell", "feature_arm", "lambda_value", "fit_id")}
    row.update(structure_id=sid, group_id=entry["group_id"], ionisation_class=entry["ionisation_class"],
               same_row_triad_eligible=int(sid in triads),
               source_definition_sensitivity_exclude=int(entry["source_definition_sensitivity_exclude"] == "True"))
    for name, column in ENDPOINTS.items():
        prefix = "physical" if name in BOUNDED else "pred"
        row[f"q_{name}"] = float(prediction[f"{prefix}_{name}"])
        observed = entry[column] != ""
        row[f"observed_{name}"] = int(observed)
        truth = float(entry[column]) if observed else None
        row[f"true_{name}"] = math.log10(truth) if observed and name not in BOUNDED else truth
    return row


def export(campaign, plan_path, dataset, destination):
    state = campaign.status()
    if not state["complete"]:
        raise ValueError(f"Campaign incomplete: {state}")
    plan = campaign.plan
    selections = {}
    for group in sorted(campaign.groups):
        record = campaign.load_selection(group)
        if record != selection_record(plan, group, campaign.completed, campaign.triads):
            raise ValueError(f"Selection failed recomputation: {group}")
        selections[group] = record
    records = {"inner": [], "final": []}
    for key, bundle in sorted(plan["bundles"].items()):
        stage = bundle["stage"]
        requests = (inner_requests(plan, key) if stage == "inner" else
                    final_requests(plan, key, selections[bundle["selection_id"]]))
        _, fits = checked_bundle(campaign.completed / key, plan, key, requests)
        for request, (_, predictions) in zip(requests, fits, strict=True):
            records[stage].extend(export_row(request, prediction, dataset, campaign.triads)
                                  for prediction in predictions)
    final = records["final"]
    expected_final = {(sid, model, seed, cell) for sid in dataset for model in BACKBONES
                      for seed in SEEDS for cell in CELLS}
    actual_final = {(row["structure_id"], row["backbone"], row["seed"], row["cell"]) for row in final}
    if len(final) != 35316 or actual_final != expected_final:
        raise ValueError("Unexpected final prediction coverage")
    p1_zero = [row for row in records["inner"] if row["feature_arm"] == "P1" and row["lambda_value"] == 0]
    if (len(records["inner"]), len(p1_zero)) != (54936, 3924):
        raise ValueError("Unexpected inner prediction coverage")
    write_csv(destination / "final_predictions.csv", EXPORT_FIELDS, final)
    write_csv(destination / "inner_grid_predictions.csv", EXPORT_FIELDS, records["inner"])
    write_csv(destination / "inner_p1_zero_predictions.csv", EXPORT_FIELDS, p1_zero)
    write_json(destination / "selections.json", list(selections.values()))
    files = {name: sha(destination / name) for name in
             ("final_predictions.csv", "inner_grid_predictions.csv", "inner_p1_zero_predictions.csv", "selections.json")}
    manifest = {"campaign_id": plan["campaign_id"], "plan_sha256": sha(plan_path),
                "final_rows": len(final), "inner_p1_zero_rows": len(p1_zero), "inner_grid_rows": len(records["inner"]),
                "final_fits": state["final_fits"], "inner_fits": state["inner_fits"], "files": files}
    write_json(destination / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "verify", "worker", "status", "export", "recover"))
    parser.add_argument("--plan", type=Path, default=ROOT / "runtime" / "plan.json")
    parser.add_argument("--release-dir", type=Path, default=ROOT / "data" / "release" / "numerical")
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "features" / "numerical")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runtime" / "run")
    parser.add_argument("--prior-plan", type=Path)
    parser.add_argument("--source-triads", type=Path)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--bundle-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.)
    parser.add_argument("--poll-seconds", type=float, default=5.)
    parser.add_argument("--stale-seconds", type=float, default=90.)
    parser.add_argument("--confirm-owner-stopped", action="store_true")
    parser.add_argument("--reason")
    args = parser.parse_args(argv)
    if args.command == "init":
        if args.prior_plan is None or args.source_triads is None:
            parser.error("init requires --prior-plan and --source-triads")
        if (args.output_dir / "campaign.json").is_file():
            parser.error("output is already bound to a frozen plan")
        plan = build_plan(args.plan, args.release_dir, args.feature_dir, args.prior_plan, args.source_triads)
        print(json.dumps({"campaign_id": plan["campaign_id"], **plan["identity"]["counts"]}, indent=2))
        return
    plan, dataset, triads = verify_plan(args.plan, args.release_dir, args.feature_dir)
    if args.bundle_id is not None and args.bundle_id not in plan["bundles"]:
        parser.error(f"unknown bundle ID: {args.bundle_id}")
    campaign = Campaign(plan, triads, args.output_dir)
    if args.command == "worker":
        if args.heartbeat_seconds <= 0 or args.poll_seconds <= 0 or (
                args.max_seconds is not None and args.max_seconds <= 0):
            parser.error("worker timing values must be positive")
        campaign.work(args)
        result = campaign.status()
    elif args.command == "recover":
        if args.bundle_id is None or not args.reason or args.stale_seconds < 0:
            parser.error("recover requires --bundle-id, --reason and a nonnegative --stale-seconds")
        campaign.recover(args.bundle_id, owner_stopped=args.confirm_owner_stopped,
                         stale_seconds=args.stale_seconds, reason=args.reason)
        result = campaign.status()
    elif args.command == "export":
        result = export(campaign, args.plan, dataset, args.export_dir or args.output_dir / "export")
    elif args.command == "verify":
        result = campaign.verify_results(complete=args.complete)
    else:
        result = campaign.status()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
