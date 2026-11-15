"""Collect frozen campaign units while comparing runtime metadata across backbones."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import combined_campaign as archived
import unified_campaign as frozen


def selection_replay_delta(saved, recomputed):
    """Compare checked platform-dependent reductions and exact selection decisions."""
    def with_normalized_scores(selection):
        normalized = dict(selection)
        normalized["grid"] = [{key: value for key, value in row.items() if key != "mean_score"}
                              for row in selection["grid"]]
        for row in normalized["grid"]:
            row["arms"] = {arm: {key: value for key, value in values.items()
                                 if key not in ("score", "endpoint_log10_mae")}
                           for arm, values in row["arms"].items()}
        return normalized

    def scores(selection):
        for row in selection["grid"]:
            yield row["mean_score"]
            for arm in archived.ARMS:
                values = row["arms"][arm]
                yield values["score"]
                for endpoint in archived.TRIAD_ENDPOINTS:
                    yield values["endpoint_log10_mae"][endpoint]

    if with_normalized_scores(saved) != with_normalized_scores(recomputed):
        raise ValueError("Saved selection decisions or calibration identities changed")
    differences = []
    for before, after in zip(scores(saved), scores(recomputed), strict=True):
        if not math.isclose(before, after, rel_tol=0., abs_tol=1e-10):
            raise ValueError("Saved selection grid scores do not reproduce within 1e-10")
        differences.append(abs(before - after))
    return max(differences, default=0.)


def read_unit(campaign, unit, path):
    directory = path / "fits"
    inner, inner_markers = frozen.checked_fits(directory, frozen.inner_requests(campaign.plan, unit))
    replay = frozen.selection_record(campaign, unit, inner, inner_markers)
    selection = archived.read_json(path / "selection.json")
    score_delta = selection_replay_delta(selection, replay)
    final, final_markers = frozen.checked_fits(directory, frozen.final_requests(campaign.plan, unit, selection))
    marker = frozen.unit_marker(campaign, unit, path, inner + final, inner_markers | final_markers)
    if archived.read_json(path / "complete_unit.json") != marker:
        raise ValueError(f"Invalid unit completion marker for {unit}")
    return marker, selection, inner, final, score_delta


def shared_environment(backbone, summary):
    environment = dict(summary["environment"])
    if archived.digest(environment) != summary["environment_hash"]:
        raise ValueError("Fit runtime metadata disagree with their saved checksum")
    version = environment.pop("chemprop", None)
    if version != ("1.6.1" if backbone == "Chemprop" else None):
        raise ValueError(f"Unexpected backbone-specific Chemprop version for {backbone}")
    return environment


def collect(campaign, output, destination):
    combined = defaultdict(list)
    ionisation, selections, unit_markers, score_deltas = [], [], {}, {}
    environments, fit_ids = {}, set()
    for unit, (backbone, _) in frozen.UNITS.items():
        marker, selection, inner, final, score_delta = read_unit(campaign, unit, output / unit)
        common = shared_environment(backbone, final[0][1][0])
        environments[archived.digest(common)] = common
        if fit_ids.intersection(marker["fit_markers"]):
            raise ValueError(f"Repeated fit IDs in export unit {unit}")
        fit_ids.update(marker["fit_markers"])
        selections.append(selection)
        unit_markers[unit] = archived.sha(output / unit / "complete_unit.json")
        score_deltas[unit] = score_delta
        for stage, fits in (("inner", inner), ("final", final)):
            for request, (_, predictions) in fits:
                if request["feature_arm"] in archived.ARMS:
                    combined[stage].extend(archived.export_row(request, row, campaign.dataset, campaign.triads)
                                           for row in predictions)
                if stage == "final" and not request["lambda_value"]:
                    ionisation.extend(frozen.ionisation_row(request, row) for row in predictions)
    counts = campaign.plan["identity"]["counts"]
    if len(environments) != 1 or len(fit_ids) != counts["total_fits"]:
        raise ValueError("Unified export requires one common runtime and every planned fit")
    environment_hash, environment = environments.popitem()

    combo_dir, ion_dir = destination / "combined", destination / "ionisation"
    core_final, core_inner = combined["final"], combined["inner"]
    p1_zero = [row for row in core_inner if row["cell"] == "10"]
    refs = [row for row in ionisation if row["arm"] in archived.ARMS]
    ph = [dict(row, arm=frozen.PH_ALIASES.get(row["arm"], row["arm"])) for row in ionisation]
    if (len(core_final), len(core_inner), len(p1_zero), len(refs), len(ph)) != (35316, 54936, 3924, 17658, 52974):
        raise ValueError("Unexpected frozen prediction export coverage")
    for name, records in (("final_predictions.csv", core_final), ("inner_grid_predictions.csv", core_inner),
                          ("inner_p1_zero_predictions.csv", p1_zero)):
        archived.write_csv(combo_dir / name, archived.EXPORT_FIELDS, records)
    archived.write_json(combo_dir / "selections.json", selections)
    for name, records in (("outer_predictions.csv", refs), ("reference_and_ph_predictions.csv", ph)):
        archived.write_csv(ion_dir / name, frozen.ION_FIELDS, records)

    plan_path = destination / "plan.json"
    archived.write_json(plan_path, campaign.plan)
    metadata = dict(campaign_id=campaign.plan["campaign_id"], release_fingerprint=archived.FINGERPRINT,
                    plan_sha256=archived.sha(plan_path), environment_hash=environment_hash,
                    environment=environment, chemprop_version="1.6.1", unified_fit_counts=counts,
                    max_selection_score_replay_delta=max(score_deltas.values()))
    combined_files = ("final_predictions.csv", "inner_p1_zero_predictions.csv", "inner_grid_predictions.csv", "selections.json")
    ionisation_files = ("outer_predictions.csv", "reference_and_ph_predictions.csv")
    archived.write_json(combo_dir / "manifest.json", dict(
        metadata, final_rows=len(core_final), inner_grid_rows=len(core_inner), inner_p1_zero_rows=len(p1_zero),
        inner_fits=1050, final_fits=540, files={name: archived.sha(combo_dir / name) for name in combined_files}))
    archived.write_json(ion_dir / "manifest.json", dict(
        metadata, rows=len(ph), fits=810, reference_rows=len(refs), reference_fits=270,
        files={name: archived.sha(ion_dir / name) for name in ionisation_files}))
    archived.write_json(destination / "manifest.json", dict(metadata, unit_markers=unit_markers,
                        selection_score_replay_deltas=score_deltas, exports={
        name: archived.sha(destination / name / "manifest.json") for name in ("combined", "ionisation")
    }))
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=frozen.ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    result = collect(frozen.load_campaign(args.root.resolve()), args.output_dir.resolve(), args.destination.resolve())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
