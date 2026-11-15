"""Compare verified unified zero-loss fits with the original numerical archive."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import combined_campaign as archived
from collect_unified import read_unit
from unified_campaign import ROOT, UNITS, load_campaign


def compare_fit(summary, predictions, original, current):
    historical = archived.read_json(original / "summary.json")
    historical_predictions = archived.rows(original / "predictions.csv")
    by_id = archived.by_id(historical_predictions)
    if by_id.keys() != {row["structure_id"] for row in predictions}:
        raise ValueError(f"Historical and unified prediction populations differ: {original}")
    maximum = max(abs(float(row[f"pred_{endpoint}"]) - float(by_id[row["structure_id"]][f"pred_{endpoint}"]))
                  for endpoint in archived.ENDPOINTS for row in predictions)
    config_match = historical["config"] == summary["config"]
    seed_match = historical["seed"] == summary["seed"]
    if not config_match or not seed_match:
        raise ValueError(f"Historical and unified matched fit recipes differ: {original}")
    old_curve, new_curve = (archived.rows(path / "learning_curve.csv") for path in (original, current))
    streams = ("batch_order_seed", "dropout_seed", "batch_order_sha256")
    overlapping_streams_equal = all(all(a[field] == b[field] for field in streams)
                                   for a, b in zip(old_curve, new_curve))
    loss_delta = abs(float(old_curve[0]["train_supervised"]) - float(new_curve[0]["train_supervised"]))
    return dict(initial_state_equal=summary["initial_state_hash"] == historical["initial_state_hash"],
                common_state_equal=summary["common_state_hash"] == historical["common_state_hash"],
                best_epoch_equal=summary["best_epoch"] == historical["best_epoch"],
                best_epoch_original=historical["best_epoch"], best_epoch_unified=summary["best_epoch"],
                predictions_equal=maximum == 0., max_transformed_prediction_difference=maximum,
                overlapping_streams_equal=overlapping_streams_equal, epoch_one_supervised_loss_equal=loss_delta == 0.,
                epoch_one_supervised_loss_difference=loss_delta,
                historical_fit=historical["fit_id"], unified_fit=summary["fit_id"],
                historical_runtime=historical["environment"], historical_device=historical["device"],
                unified_runtime=summary["environment"], unified_device=summary["device"])


def compare_unit(campaign, unit, output, original):
    _, selection, inner, final, score_delta = read_unit(campaign, unit, output / unit)
    old_unit = original / f"numerical-{unit}"
    historical_selection = archived.read_json(old_unit / "selection.json")
    old_final = {(rec["arm"], rec["seed"]): rec["fit_id"] for rec in historical_selection["final_requests"]
                 if rec["procedure"] == "primary" and rec["arm"] in archived.ARMS}
    records = []
    for stage, fit_records in (("inner", inner), ("outer", final)):
        for request, (summary, predictions) in fit_records:
            arm, seed, fold = (request[name] for name in ("feature_arm", "seed", "inner_fold"))
            if arm not in archived.ARMS or request["lambda_value"] != 0.:
                continue
            historical_id = (historical_selection["calibration_fit_ids"][arm][fold - 1] if stage == "inner"
                             else old_final[arm, seed])
            original_fit = old_unit / "fits" / historical_id
            ids = archived.read_json(original_fit / "request.json")["effective_ids"]
            if any(request[key] != ids[key] for key in ("train_ids", "val_ids", "predict_ids")):
                raise ValueError(f"Historical and unified split IDs differ: {original_fit}")
            current_fit = output / unit / "fits" / request["fit_id"]
            records.append(dict(stage=stage, arm=arm, seed=seed, inner_fold=fold,
                                **compare_fit(summary, predictions, original_fit, current_fit)))
    if len(records) != 28:
        raise ValueError(f"Expected 28 matched zero-loss fits for {unit}")
    by_stage = defaultdict(list)
    for row in records:
        by_stage[row["stage"]].append(row)
    checks = ("initial_state_equal", "common_state_equal", "best_epoch_equal", "predictions_equal",
              "overlapping_streams_equal", "epoch_one_supervised_loss_equal")
    summary = {stage: dict(count=len(part), **{name: sum(row[name] for row in part) for name in checks},
                           max_transformed_prediction_difference=max(row["max_transformed_prediction_difference"] for row in part),
                           max_epoch_one_supervised_loss_difference=max(row["epoch_one_supervised_loss_difference"] for row in part))
               for stage, part in by_stage.items()}
    return dict(unit=unit, selected_positive_lambda=selection["selected_lambda"], selection_score_replay_delta=score_delta,
                summary=summary, records=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--unit", action="append", choices=UNITS)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    campaign = load_campaign(args.root.resolve())
    units = args.unit or [unit for unit in UNITS if (args.output_dir / unit / "complete_unit.json").is_file()]
    if not units:
        raise ValueError("No completed unified units found")
    result = {unit: compare_unit(campaign, unit, args.output_dir, args.historical) for unit in units}
    if args.report:
        archived.write_json(args.report, result)
    print(json.dumps({unit: record["summary"] for unit, record in result.items()}))


if __name__ == "__main__":
    main()
