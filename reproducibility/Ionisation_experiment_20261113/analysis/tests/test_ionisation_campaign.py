"""Exercise the frozen campaign coordinator with local synthetic work units."""

from copy import deepcopy
from pathlib import Path
import shutil
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ionisation_campaign as campaign
import ionisation_data as data

from test_ionisation_data import decisions, inventory


def copy_grid(root):
    grid = root / "config" / "search_grid.json"
    grid.parent.mkdir(parents=True)
    shutil.copyfile(data.EXPERIMENT / "config" / grid.name, grid)


@pytest.fixture()
def frozen_campaign(tmp_path, inventory, decisions):
    root = tmp_path / "experiment"
    copy_grid(root)
    decisions_path = tmp_path / "synthetic_decisions.csv"
    data.write_csv(decisions_path, decisions, data.DECISION_FIELDS)
    release_root = root / "data" / "release"
    data.freeze(inventory, decisions_path, release_root, ("class", "numerical"))
    data.subset_features(
        data.MASS_BALANCE / "data", data.MASS_BALANCE / "features",
        release_root, root / "features", ("class", "numerical"),
    )
    return root


def test_audit_inventory_cannot_make_a_production_plan(tmp_path, inventory):
    copy_grid(tmp_path)
    data.prepare(inventory, tmp_path / "data" / "release" / "class")
    with pytest.raises(ValueError, match="frozen|chemical|training"):
        campaign.build_plan(tmp_path, ("class",))


def test_plan_work_counts_are_deterministic_and_gate_tampered_units(frozen_campaign):
    plan = campaign.build_plan(frozen_campaign)
    assert plan == campaign.build_plan(frozen_campaign)
    assert plan["new_fits_upper_bound"] == 5880
    assert len(plan["units"]) == 30
    assert campaign.build_plan(frozen_campaign, include_tuned=False)["new_fits_upper_bound"] == 3450
    class_only = campaign.build_plan(frozen_campaign, ("class",))
    assert class_only["new_fits_upper_bound"] == 2325
    assert len(class_only["units"]) == 15
    campaign.verify_plan(frozen_campaign, plan)
    altered = deepcopy(plan)
    # The freshly built object shares the nested and top-level unit list.
    # Construct independent lists as they would be after reading a plan file.
    altered["units"] = deepcopy(altered["units"])
    altered["units"][0]["outer_fold"] = 9
    with pytest.raises(ValueError, match="Top-level units"):
        campaign.verify_plan(frozen_campaign, altered)


def inner_result(sse, count, epoch):
    return {
        "val_squared_error_sums": dict.fromkeys(campaign.ENDPOINTS, sse),
        "val_observed_counts": dict.fromkeys(campaign.ENDPOINTS, count),
        "best_epoch": epoch,
    }


def test_choose_pools_observations_before_endpoint_average():
    # A simple fold mean picks config 1, whose MSE is always 2.
    # Pooled MSE instead selects config 0, whose difficult fold is small.
    sparse_high_error = inner_result(10, 1, 11)
    large_low_error = [inner_result(0, 100, epoch) for epoch in (5, 9, 7, 13)]
    stable = [inner_result(200, 100, 20) for _ in range(5)]
    config, epoch, scores = campaign.choose({0: [sparse_high_error, *large_low_error], 1: stable})
    assert (config, epoch) == (0, 9)
    assert scores == pytest.approx({0: 10 / 401, 1: 2})


def test_requests_keep_groups_apart_and_reuse_exact_fit_identity(frozen_campaign):
    plan = campaign.build_plan(frozen_campaign)
    _, outer, inner = campaign.verify_release(frozen_campaign, "class")
    unit = next(u for u in plan["units"] if u["cohort"] == "class")
    config = plan["identity"]["grid"][unit["backbone"]][0]
    raw_outer = campaign.rows(frozen_campaign / "data" / "release" / "class" / "outer.csv")
    group = {r["structure_id"]: r["group_id"] for r in raw_outer}
    outer_test = {sid for sid, fold in outer.items() if fold == unit["outer_fold"]}

    def fit_groups(ids):
        return {group[sid] for sid in ids}

    def request(arm, seed, inner_fold=None, epochs=None):
        return campaign.fit_request(plan, unit, arm, config, seed, outer, inner, inner_fold, epochs)

    search = request("C2", campaign.SEARCH_SEED, inner_fold=2)
    calibration = request("C2", campaign.SEARCH_SEED, inner_fold=2)
    assert search == calibration
    assert search["fit_id"] != request("C0", campaign.SEARCH_SEED, inner_fold=2)["fit_id"]
    assert search["fit_id"] != request("C2", campaign.SEARCH_SEED, inner_fold=3)["fit_id"]
    assert set(search["predict_ids"]) == set(search["val_ids"])
    assert set(search["train_ids"]) | set(search["val_ids"]) == set(outer) - outer_test
    assert fit_groups(search["train_ids"]).isdisjoint(fit_groups(search["val_ids"]))
    assert fit_groups(outer_test).isdisjoint(fit_groups(search["train_ids"] + search["val_ids"]))

    final = request("C2", campaign.SEEDS[0], epochs=15)
    assert final["fit_id"] == request("C2", campaign.SEEDS[0], epochs=15)["fit_id"]
    assert final["fit_id"] != request("C2", campaign.SEEDS[0], epochs=16)["fit_id"]
    assert set(final["predict_ids"]) == outer_test
    assert set(final["train_ids"]) == set(outer) - outer_test
    assert final["val_ids"] == []


def smoke_plan(root):
    plan = campaign.build_plan(root, ("class",))
    # Restrict the immutable plan to one class unit while keeping the normal
    # plan verifier, work-unit driver and exporter active.
    unit = next(u for u in plan["units"] if u["backbone"] == "GraphConv")
    plan["units"] = plan["identity"]["units"] = [unit]
    plan["new_fits_upper_bound"] = unit["new_fits_upper_bound"]
    plan["campaign_id"] = campaign.digest(plan["identity"])
    return plan, unit


def test_mock_unit_reuse_and_export_coverage(frozen_campaign, tmp_path, monkeypatch):
    plan, unit = smoke_plan(frozen_campaign)
    output = tmp_path / "campaign"
    executed = set()

    def fake_fit(args):
        request, _, _, output_dir, _ = args
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)
        fit_id = request["fit_id"]
        assert fit_id not in executed
        executed.add(fit_id)
        prediction = {f"{kind}_{endpoint}": 1.0 for kind in ("pred", "physical") for endpoint in campaign.ENDPOINTS}
        predictions = [{"structure_id": sid, **prediction} for sid in request["predict_ids"]]
        data.write_csv(output_dir / "predictions.csv", predictions)
        campaign.write(output_dir / "request.json", {"request": request})
        campaign.write(output_dir / "complete.json", {
            "fit_id": fit_id, "files": {name: campaign.sha(output_dir / name)
                                       for name in ("request.json", "predictions.csv")},
        })
        return inner_result(2, 2, 12)

    monkeypatch.setattr(campaign, "_execute", fake_fit)
    done = campaign.run_unit(frozen_campaign, plan, unit["unit_id"], output, device="cpu")
    # 100 search, 10 additional calibrations, 36 primary finals. The tuned C2
    # selected the same configuration and epochs, so its nine finals reuse.
    assert done["unique_fits"] == len(executed) == 146
    assert len(done["final_records"]) == 45
    units = output / unit["unit_id"]
    selection = campaign.read(units / "selection.json")
    assert selection["search_fit_ids"]["C2"]["0"] == selection["calibration_fit_ids"]["C2"]
    primary_ids = {r["fit_id"] for r in done["final_records"] if r["arm"] == "C2" and r["procedure"] == "primary"}
    tuned_ids = {r["fit_id"] for r in done["final_records"] if r["arm"] == "C2" and r["procedure"] == "tuned"}
    assert primary_ids == tuned_ids and len(primary_ids) == 9

    destination = tmp_path / "export.csv"
    expected = 45 * sum(int(r["outer_fold"]) == unit["outer_fold"] for r in campaign.rows(
        frozen_campaign / "data" / "release" / "class" / "outer.csv"))
    assert campaign.export(frozen_campaign, plan, output, destination) == expected
    manifest = campaign.read(destination.with_suffix(".manifest.json"))
    assert manifest["sha256"] == campaign.sha(destination)
    assert manifest["rows"] == expected
    assert manifest["release_fingerprints"]["class"] == plan["identity"]["datasets"]["class"]["fingerprint"]

    record = done["final_records"][0]
    fit = units / "fits" / record["fit_id"]
    request = campaign.read(fit / "request.json")
    altered_request = deepcopy(request)
    altered_request["request"]["seed"] = -1
    campaign.write(fit / "request.json", altered_request)
    with pytest.raises(ValueError, match="request|artifact|Request"):
        campaign.export(frozen_campaign, plan, output, destination)
    campaign.write(fit / "request.json", request)

    path = units / "complete_unit.json"
    campaign.write(path, done | {"final_records": done["final_records"][1:]})
    with pytest.raises(ValueError, match="selection|coverage|request|record"):
        campaign.export(frozen_campaign, plan, output, destination)
    campaign.write(path, done)
    (fit / "predictions.csv").write_text("structure_id\ninvalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="predictions|artifact"):
        campaign.export(frozen_campaign, plan, output, destination)
    marker = campaign.read(fit / "complete.json")
    marker["files"]["predictions.csv"] = campaign.sha(fit / "predictions.csv")
    campaign.write(fit / "complete.json", marker)
    done["fit_markers"][record["fit_id"]] = campaign.sha(fit / "complete.json")
    campaign.write(path, done)
    with pytest.raises(ValueError, match="coverage"):
        campaign.export(frozen_campaign, plan, output, destination)
