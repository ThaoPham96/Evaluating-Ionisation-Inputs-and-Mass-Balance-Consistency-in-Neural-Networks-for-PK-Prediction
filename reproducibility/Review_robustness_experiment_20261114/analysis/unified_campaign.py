"""Fit and export the unified numerical-pKa campaign in 15 independent units."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import combined_campaign as archived
from unified_features import EXTRA_ARMS, PH_BY_ARM, verify_categories


ROOT = Path(__file__).resolve().parents[1]
BACKBONES, FOLDS, SEEDS = archived.BACKBONES, archived.FOLDS, archived.SEEDS
ZERO_ARMS = ("P0", *PH_BY_ARM)
INNER_SETTINGS = tuple((arm, 0.) for arm in ZERO_ARMS) + tuple(
    (arm, strength) for strength in archived.LAMBDAS[1:] for arm in archived.ARMS
)
UNITS = {archived.selection_id(backbone, fold): (backbone, fold) for backbone in BACKBONES for fold in FOLDS}
INPUT_HASHES = {
    archived.RECIPES: "84adb53b8a99a222674c36f5030972bc26b7bc4fb9d4668c25c8eba3c6f81574",
    archived.TRIADS: "db51103863ca3abe0ff966a838a0715dc06e1657a599e9ce1d43af95e5d642bd",
}
INNER_COUNT = len(FOLDS) * len(INNER_SETTINGS)
FINAL_COUNT = len(SEEDS) * (len(ZERO_ARMS) + len(archived.ARMS))
PH_ALIASES = {"P0": "REF0", "P1": "REF74"}
ION_FIELDS = ["structure_id", *[f"{prefix}_{name}" for prefix in ("pred", "physical")
                                for name in archived.ENDPOINTS],
              "cohort", "backbone", "outer_fold", "procedure", "arm", "seed", "source_campaign_id", "source_fit_id"]


@dataclass(frozen=True)
class Campaign:
    plan: dict
    dataset: dict
    triads: dict
    release: Path
    features: Path


def load_campaign(root):
    release, features = root / "data/release/numerical", root / "features/numerical"
    inputs = archived.input_identity(release, features)
    for name, expected in INPUT_HASHES.items():
        if archived.sha(root / "runtime" / name) != expected:
            raise ValueError(f"Frozen selected recipes or same-row triads changed: {name}")
    recipes = archived.read_json(root / "runtime" / archived.RECIPES)
    if recipes.keys() != UNITS.keys():
        raise ValueError("Frozen selected recipes do not cover the planned units")
    dataset, splits = archived.read_population(release)
    triads = archived.by_id(archived.rows(root / "runtime" / archived.TRIADS))
    archived.validate_triads(triads, dataset, splits)
    inventory = verify_categories(list(dataset.values()))
    code = archived.code_hashes()
    code.update({f"analysis/{name}": archived.sha(ROOT / "analysis" / name)
                 for name in ("unified_campaign.py", "unified_features.py")})
    identity = dict(schema_version=1, inputs=inputs, files=INPUT_HASHES, code=code, recipes=recipes,
                    splits=splits, ph_inventory=inventory, inner_settings=[list(x) for x in INNER_SETTINGS],
                    zero_arms=list(ZERO_ARMS), seeds=list(SEEDS), search_seed=42, max_epochs=200,
                    patience=30, min_delta=1e-4, task_weights=[1.] * 7,
                    counts=dict(units=len(UNITS), inner_fits=len(UNITS)*INNER_COUNT,
                                final_fits=len(UNITS)*FINAL_COUNT, total_fits=len(UNITS)*(INNER_COUNT+FINAL_COUNT)))
    return Campaign(dict(campaign_id=archived.digest(identity), identity=identity), dataset, triads, release, features)


def request_for(plan, unit, arm, strength, *, seed, inner=None, epochs=None):
    backbone, outer = UNITS[unit]
    bundle = dict(backbone=backbone, outer_fold=outer, inner_fold=inner, selection_id=unit, seed=seed)
    request = archived.fit_request(plan, bundle, arm, strength, epochs)
    if arm in EXTRA_ARMS:
        request["cell"] = arm
    del request["fit_id"]
    request["fit_id"] = "UNI-" + archived.digest(request)[:24]
    return request


def inner_requests(plan, unit):
    return [request_for(plan, unit, arm, strength, seed=42, inner=fold)
            for fold in FOLDS for arm, strength in INNER_SETTINGS]


def final_requests(plan, unit, selection):
    strength = selection["selected_lambda"]
    epochs = selection["zero_arm_epochs"]
    selected = next(row for row in selection["grid"] if row["lambda_value"] == strength)
    settings = [(arm, 0., epochs[arm]["median_epoch"]) for arm in ZERO_ARMS]
    settings.extend((arm, strength, selected["arms"][arm]["median_epoch"]) for arm in archived.ARMS)
    return [request_for(plan, unit, arm, value, seed=seed, epochs=duration)
            for seed in SEEDS for arm, value, duration in settings]


def selection_record(campaign, unit, fits, marker_hashes):
    backbone, outer = UNITS[unit]
    core = [(request, outputs) for request, outputs in fits if request["feature_arm"] in archived.ARMS]
    sources = {request["fit_id"]: marker_hashes[request["fit_id"]] for request, _ in core}
    validation_ids = set(campaign.plan["identity"]["splits"]["inner"][str(outer)])
    selection = archived.select_from_calibration(campaign.plan["campaign_id"], unit, backbone, outer,
                                                 validation_ids, campaign.triads, core, sources)
    epochs = defaultdict(list)
    for request, (summary, _) in fits:
        if not request["lambda_value"]:
            epochs[request["feature_arm"]].append(dict(inner_fold=request["inner_fold"], fit_id=request["fit_id"],
                                                        best_epoch=summary["best_epoch"]))
    if set(epochs) != set(ZERO_ARMS) or any([r["inner_fold"] for r in part] != list(FOLDS)
                                          for part in epochs.values()):
        raise ValueError(f"Incomplete zero-loss epoch calibration for {unit}")
    selection["zero_arm_epochs"] = {
        arm: dict(epochs=part, median_epoch=int(statistics.median(row["best_epoch"] for row in part)))
        for arm, part in epochs.items()
    }
    return selection


def checked_fits(directory, requests):
    fits, hashes, paired = [], {}, defaultdict(list)
    for request in requests:
        fit_id = request["fit_id"]
        result = archived.validate_fit(directory / fit_id, request)
        fits.append((request, result))
        hashes[fit_id] = archived.sha(directory / fit_id / "complete.json")
        paired[request["inner_fold"], request["seed"]].append((request, result[0]))
    for part in paired.values():
        requests_in_pair, summaries = zip(*part, strict=True)
        archived.validate_pairing(directory, requests_in_pair, summaries)
        fits_with_ph = [summary for summary in summaries if summary["feature_arm"] != "P0"]
        if len({summary["initial_state_hash"] for summary in fits_with_ph}) != 1:
            raise ValueError(f"The width-matched pH arms have different full initial states: {directory}")
    return fits, hashes


def unit_marker(campaign, unit, path, fits, hashes):
    environments = {outputs[0]["environment_hash"] for _, outputs in fits}
    if len(environments) != 1:
        raise ValueError(f"Unit spans multiple runtime environments: {unit}")
    seed_initializations, ph_initializations = defaultdict(set), defaultdict(set)
    for request, (summary, _) in fits:
        seed_initializations[request["seed"]].add(summary["common_state_hash"])
        if request["feature_arm"] != "P0":
            ph_initializations[request["seed"]].add(summary["initial_state_hash"])
    if any(len(states) != 1 for states in (*seed_initializations.values(), *ph_initializations.values())):
        raise ValueError(f"Nominally identical seeds used different shared initial states: {unit}")
    backbone, outer = UNITS[unit]
    return dict(campaign_id=campaign.plan["campaign_id"], unit_id=unit, backbone=backbone, outer_fold=outer,
                inner_fits=INNER_COUNT, final_fits=FINAL_COUNT, unique_fits=len(hashes),
                environment_hash=environments.pop(), selection_sha256=archived.sha(path / "selection.json"),
                fit_markers=hashes)


def read_unit(campaign, unit, path):
    directory = path / "fits"
    inner, inner_markers = checked_fits(directory, inner_requests(campaign.plan, unit))
    selection = selection_record(campaign, unit, inner, inner_markers)
    if archived.read_json(path / "selection.json") != selection:
        raise ValueError(f"Invalid saved epoch or lambda selection for {unit}")
    final, final_markers = checked_fits(directory, final_requests(campaign.plan, unit, selection))
    marker = unit_marker(campaign, unit, path, inner + final, inner_markers | final_markers)
    if archived.read_json(path / "complete_unit.json") != marker:
        raise ValueError(f"Invalid unit completion marker for {unit}")
    return marker, selection, inner, final


def run_unit(campaign, unit, output, device):
    from combined_train import base, run_fit

    path = output / unit
    path.mkdir(parents=True, exist_ok=True)
    directory = path / "fits"
    with base._fit_lock(path):
        if (path / "complete_unit.json").exists():
            return read_unit(campaign, unit, path)[0]
        identity = dict(campaign_id=campaign.plan["campaign_id"], unit_id=unit)
        unit_path = path / "unit.json"
        if unit_path.exists() and archived.read_json(unit_path) != identity:
            raise ValueError(f"Unit output belongs to a different campaign: {path}")
        archived.write_json(unit_path, identity)

        def train(requests):
            for request in requests:
                run_fit(request, campaign.release, campaign.features, directory / request["fit_id"], device)
            return checked_fits(directory, requests)

        inner, inner_markers = train(inner_requests(campaign.plan, unit))
        selection = selection_record(campaign, unit, inner, inner_markers)
        archived.write_json(path / "selection.json", selection)
        final, final_markers = train(final_requests(campaign.plan, unit, selection))
        marker = unit_marker(campaign, unit, path, inner + final, inner_markers | final_markers)
        archived.write_json(path / "complete_unit.json", marker)
        return marker


def ionisation_row(request, prediction):
    return dict(prediction, cohort="numerical", backbone=request["backbone"], outer_fold=request["outer_fold"],
                procedure="primary", arm=request["feature_arm"], seed=request["seed"],
                source_campaign_id=request["campaign_id"], source_fit_id=request["fit_id"])


def export(campaign, output, destination):
    combined = defaultdict(list)
    ionisation = []
    selections, environments, fit_markers = [], set(), set()
    for unit in UNITS:
        marker, selection, inner, final = read_unit(campaign, unit, output / unit)
        environments.add(marker["environment_hash"])
        if fit_markers.intersection(marker["fit_markers"]):
            raise ValueError(f"Repeated fit IDs in export unit {unit}")
        fit_markers.update(marker["fit_markers"])
        selections.append(selection)
        for stage, fits in (("inner", inner), ("final", final)):
            for request, (_, predictions) in fits:
                if request["feature_arm"] in archived.ARMS:
                    combined[stage].extend(archived.export_row(request, row, campaign.dataset, campaign.triads)
                                           for row in predictions)
                if stage == "final" and not request["lambda_value"]:
                    ionisation.extend(ionisation_row(request, row) for row in predictions)
    if len(environments) != 1 or len(fit_markers) != campaign.plan["identity"]["counts"]["total_fits"]:
        raise ValueError("Unified export needs exactly one common runtime environment and all planned fit records")

    combo_dir, ion_dir = destination / "combined", destination / "ionisation"
    core_final, core_inner = combined["final"], combined["inner"]
    p1_zero = [row for row in core_inner if row["cell"] == "10"]
    refs = [row for row in ionisation if row["arm"] in archived.ARMS]
    ph = [dict(row, arm=PH_ALIASES.get(row["arm"], row["arm"])) for row in ionisation]
    if (len(core_final), len(core_inner), len(p1_zero), len(refs), len(ph)) != (35316, 54936, 3924, 17658, 52974):
        raise ValueError("Unexpected frozen prediction export coverage")
    for name, records in (("final_predictions.csv", core_final), ("inner_grid_predictions.csv", core_inner),
                          ("inner_p1_zero_predictions.csv", p1_zero)):
        archived.write_csv(combo_dir / name, archived.EXPORT_FIELDS, records)
    archived.write_json(combo_dir / "selections.json", selections)
    for name, records in (("outer_predictions.csv", refs), ("reference_and_ph_predictions.csv", ph)):
        archived.write_csv(ion_dir / name, ION_FIELDS, records)

    archived.write_json(destination / "plan.json", campaign.plan)
    metadata = dict(campaign_id=campaign.plan["campaign_id"], release_fingerprint=archived.FINGERPRINT,
                    plan_sha256=archived.sha(destination / "plan.json"), environment_hash=environments.pop(),
                    unified_fit_counts=campaign.plan["identity"]["counts"])
    combined_files = ("final_predictions.csv", "inner_p1_zero_predictions.csv", "inner_grid_predictions.csv", "selections.json")
    ionisation_files = ("outer_predictions.csv", "reference_and_ph_predictions.csv")
    archived.write_json(combo_dir / "manifest.json", dict(
        metadata, final_rows=len(core_final), inner_grid_rows=len(core_inner), inner_p1_zero_rows=len(p1_zero),
        inner_fits=1050, final_fits=540, files={name: archived.sha(combo_dir / name) for name in combined_files}))
    archived.write_json(ion_dir / "manifest.json", dict(
        metadata, rows=len(ph), fits=810, reference_rows=len(refs), reference_fits=270,
        files={name: archived.sha(ion_dir / name) for name in ionisation_files}))
    archived.write_json(destination / "manifest.json", dict(metadata, exports={
        name: archived.sha(destination / name / "manifest.json") for name in ("combined", "ionisation")
    }))
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("units")
    preflight = commands.add_parser("preflight")
    train = commands.add_parser("run-unit")
    collect = commands.add_parser("export")
    for command in (preflight, train, collect):
        command.add_argument("--root", type=Path, default=ROOT)
    for command in (train, collect):
        command.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--unit", required=True, choices=UNITS)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    collect.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "units":
        result = list(UNITS)
    else:
        campaign = load_campaign(args.root.resolve())
        if args.command == "preflight":
            result = dict(campaign_id=campaign.plan["campaign_id"], units=list(UNITS),
                          counts=campaign.plan["identity"]["counts"], ph_inventory=campaign.plan["identity"]["ph_inventory"])
        elif args.command == "run-unit":
            marker = run_unit(campaign, args.unit, args.output_dir.resolve(), args.device)
            result = {key: value for key, value in marker.items() if key != "fit_markers"}
        else:
            result = export(campaign, args.output_dir.resolve(), args.destination.resolve())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
