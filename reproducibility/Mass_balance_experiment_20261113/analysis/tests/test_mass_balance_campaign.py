"""Protocol-level checks using the immutable v6 inputs and fake trainer artifacts."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import errno
from pathlib import Path

import numpy as np
import pytest

import mass_balance_campaign as mb


ANALYSIS = Path(__file__).resolve().parents[1]
RELEASE = ANALYSIS.parent / "data"
FEATURES = ANALYSIS.parent / "features"
PROTOCOL = ANALYSIS.parent / "plans"


@pytest.fixture(scope="module")
def plan():
    return mb.Plan(RELEASE, FEATURES, PROTOCOL)


@pytest.fixture
def campaign(plan, tmp_path):
    campaign = mb.Campaign(plan, tmp_path)
    campaign.init()
    return campaign


def fake_trainer(plan, request, directory, invalid_id=None):
    directory.mkdir(parents=True, exist_ok=True)
    fold = request["inner_fold"]
    lam = request["lambda_value"]
    candidate = request["candidate_id"]
    error = {"GraphConv_01": 0.05, "GraphConv_02": 0.05 - 1e-9, "ST_03": 0.05}.get(candidate, 0.1)
    resolved_hash = mb.digest([request["fit_id"], "fake"])
    epoch = 19 + fold if fold else request["fixed_epochs"]
    mb.write_json(directory / "summary.json", {
        "resolved_hash": resolved_hash, "val_equal_mse": error if fold else None,
        "best_epoch": epoch, "actual_epochs": 80 if fold else epoch,
        "val_observed_counts": {"t_half": len(request["val_ids"])},
    })
    mb.write_json(directory / "request.json", {
        "request": request, "request_hash": mb.digest(request), "resolved_hash": resolved_hash,
        "effective_ids": {key: request[key] for key in ("train_ids", "val_ids", "predict_ids")},
    })
    for name in ("checkpoint.pt", "scaler.json", "learning_curve.csv"):
        (directory / name).write_text("test", encoding="utf-8")
    change = {0: (2, 0, 0), 0.03: (0.5, 0, 0), 0.1: (0.2, 0, 0),
              0.3: (0.2 + 1e-9, 0, 0), 1: (1, 0, 0), 3: (1.1, 0, 0), 10: (1.2, 0, 0)}[lam]
    predictions = []
    for sid in request["predict_ids"]:
        triad = plan.same_row_triads.get(sid, np.array([100, 100, 100])) + change
        if sid == invalid_id and lam == 0.03:
            triad = [float("nan"), 0, 0]
        predictions.append((sid, *triad, 0, 0, 0, 0))
    mb.write_csv(directory / "predictions.csv", ("structure_id", *mb.PRED_COLUMNS), predictions)
    mb.write_json(directory / "complete.json", {
        "fit_id": request["fit_id"], "resolved_hash": resolved_hash,
        "files": {name: mb.sha256(directory / name) for name in mb.FIT_FILES},
    })


def finish_stage(campaign, name, invalid_id=None):
    stage = next(stage for stage in campaign.plan.stages if stage.name == name)
    for fit in stage.inputs:
        assert campaign.assignment_path(fit.logical_id).is_file()
        claim = campaign.claim(fit.logical_id)
        assert claim is not None
        assert campaign.execute(*claim, "cpu", trainer=lambda request, release, features, directory, device:
                                fake_trainer(campaign.plan, request, directory, invalid_id))
    campaign.refresh()
    return stage, campaign.load_selection(name)


def test_plan_resolves_all_physical_fits_and_partitions(plan):
    resolutions = {}
    for fit in plan.fits.values():
        if fit.block in mb.SEARCH:
            resolutions[fit.logical_id] = plan.request(fit)
    for stage in plan.stages:
        for fit in stage.outputs:
            if fit.endpoint:
                candidate = "ST_03"
            elif fit.block in mb.SECONDARY:
                candidate = "GraphConv_01"
            else:
                candidate = f"{fit.backbone}_01"
            selected = {"candidate_id": candidate, "selected_lambda": 0.3,
                        "epochs": {str(value): 22 for value in mb.LAMBDAS}}
            resolutions[fit.logical_id] = plan.request(fit, selected)
    physical_fields = ("backbone", "config", "seed", "lambda_value", "train_ids", "val_ids",
                       "predict_ids", "fixed_epochs", "endpoint", "task_weights")
    physical = {mb.digest([request.get(key) for key in physical_fields]) for request in resolutions.values()}
    assert len(resolutions) == len(physical) == 4240
    assert {key for key in resolutions} == plan.fits.keys()
    for request in resolutions.values():
        train, val, predict = (set(request[key]) for key in ("train_ids", "val_ids", "predict_ids"))
        outer = request["outer_fold"]
        assert train and predict and not train & (val | predict)
        if request["inner_fold"]:
            assert predict == val
            assert all(plan.outer[sid] != outer for sid in train | val)
            assert all(plan.inner[outer][sid] == request["inner_fold"] for sid in val)
        else:
            assert not val
            assert predict == {sid for sid in plan.ids if plan.outer[sid] == outer}
        if request["endpoint"]:
            assert (train | val) <= plan.labelled[request["endpoint"]]


def test_pipeline_selections_aliases_and_inner_only_labels(campaign):
    assert campaign.status()["totals"] == {"ready": 2550, "waiting": 1690}
    assert not (campaign.selections / "GraphConv-outer1-baseline").exists()
    stage, baseline = finish_stage(campaign, "GraphConv-outer1-baseline")
    assert baseline["candidate_id"] == "GraphConv_01"  # Smaller ID within 1e-8.
    assert baseline["epochs"] == {"0.0": 22}
    assert baseline["alpha_controls"]["orthogonal"]["selected_alpha"] == 0
    assert baseline["alpha_controls"]["half_life"]["selected_alpha"] == 1
    assert all(campaign.assignment_path(fit.logical_id).exists() for fit in stage.outputs)
    assert not (campaign.selections / "GraphConv-outer2-baseline").exists()
    plan = campaign.plan
    mask_file = campaign.selections / stage.name / "alpha_selection_mask.csv"
    mask = mb.read_csv(mask_file)
    expected = set(plan.inner[1]) & plan.same_row_triads.keys()
    assert {row["structure_id"] for row in mask if row["used"] == "1"} == expected
    assert not ({row["structure_id"] for row in mask} - plan.inner[1].keys())
    oof = {row["structure_id"]: [float(row[column]) for column in mb.PRED_COLUMNS]
           for row in mb.read_csv(campaign.selections / stage.name / "baseline_oof.csv")}
    held_out = next(sid for sid in plan.same_row_triads if plan.outer[sid] == 1)
    original = plan.same_row_triads[held_out]
    try:
        plan.same_row_triads[held_out] = np.array([1e9, 1e9, 1e9])
        _, truth, preds, valid = mb.triad_arrays(plan, 1, oof)
        assert mb.alpha_selection(preds, truth, valid) == baseline["alpha_controls"]
    finally:
        plan.same_row_triads[held_out] = original

    lambda_stage, selected = finish_stage(campaign, "GraphConv-outer1-lambda", min(expected))
    assert selected["selected_lambda"] == 0.1
    assert selected["selected_zero_eligible_lambda"] == 0.1
    assert selected["n_eligible"] == len(expected)
    assert selected["lambda_scores"]["0.03"]["n_invalid"] == 1
    assert {item["n_scored"] for item in selected["lambda_scores"].values()} == {len(expected) - 1}
    aliases = selected["aliases"]
    assert len(aliases["selected_positive"]) == 9
    assert all(aliases["fixed_lambda_curve"]["0.1"][str(seed)] == aliases["selected_positive"][str(seed)]
               for seed in mb.SEEDS[:3])
    finals = [campaign.request(fit.logical_id) for fit in stage.outputs + lambda_stage.outputs
              if fit.block == "primary_final"]
    assert len(finals) == 33
    seed_first = [item for item in finals if item["seed"] == mb.SEEDS[0]]
    assert len({tuple(item["train_ids"]) for item in seed_first}) == 1
    assert {item["lambda_value"] for item in seed_first} == set(mb.LAMBDAS)
    transferred = next(fit for fit in lambda_stage.outputs if fit.block == "task_weight_sensitivity"
                       and fit.policy != "0")
    assert campaign.request(transferred.logical_id)["task_weights"] == [1, 5, 5, 1, 1, 1, 1]
    secondary_name = "task_weight_sensitivity-GraphConv-outer1-depth2-GCN_inner_selected_positive"
    secondary, secondary_choice = finish_stage(campaign, secondary_name)
    assert secondary_choice["epochs"] == {"0.1": 22}
    assert {campaign.request(fit.logical_id)["fixed_epochs"] for fit in secondary.outputs} == {22}
    assert not campaign.status(verify=True)["invalid_artifacts"]


def test_independent_single_task_and_claim_recovery(campaign):
    st, selection = finish_stage(campaign, "ST-t_half-outer1")
    assert selection["candidate_id"] == "ST_03"
    assert not (campaign.selections / "GraphConv-outer1-baseline").exists()
    logical_id = st.outputs[0].logical_id
    with ThreadPoolExecutor(max_workers=4) as executor:
        claims = [claim for claim in executor.map(campaign.claim, [logical_id] * 4) if claim]
    assert len(claims) == 1
    owner, attempt = claims[0]
    assert campaign.claim_metadata(logical_id)["token"] == owner["token"]
    with pytest.raises(ValueError, match="exact abandoned claim token"):
        campaign.retry(logical_id)
    with pytest.raises(ValueError, match="heartbeat"):
        campaign.retry(logical_id, owner["token"])

    remote_owner = {**owner, "hostname": "terminated-remote-node"}
    mb.write_json(campaign.claims / logical_id / f"{owner['token']}.json", remote_owner)
    old = datetime.now(timezone.utc) - timedelta(hours=1)
    mb.write_json(attempt / "heartbeat.json", {"time": old.isoformat()})
    campaign.retry(logical_id, owner["token"])
    assert (attempt / "recovered.json").exists()

    def broken(*args, **kwargs):
        raise RuntimeError("simulated crash")

    claim = campaign.claim(logical_id)
    assert not campaign.execute(*claim, "cpu", trainer=broken)
    assert logical_id in campaign.status()["failed_ids"]
    assert campaign.claim(logical_id) is None
    campaign.retry(logical_id)
    assert (claim[1] / "failure.json").exists()
    (campaign.claims / logical_id).mkdir()  # Interrupted after unlinking the owner.
    claim = campaign.claim(logical_id)
    assert campaign.execute(*claim, "cpu", trainer=lambda request, release, features, directory, device:
                            fake_trainer(campaign.plan, request, directory))
    assert campaign.claim(logical_id) is None
    assert campaign.status(verify=True)["totals"]["completed"] == 31
    (campaign.results / logical_id / "checkpoint.pt").write_text("bad!", encoding="utf-8")
    assert logical_id in campaign.status(verify=True)["invalid_artifacts"]


def test_atomic_publication_retries_permission_error_and_preserves_winner(tmp_path, monkeypatch):
    monkeypatch.setattr(mb, "PUBLISH_DELAYS", (0.0, 0.0, 0.0))
    rename = Path.rename
    replace = mb.os.replace
    counts = {"rename": 0, "replace": 0}

    def transient_rename(source, target):
        counts["rename"] += 1
        if counts["rename"] <= 2 or target.exists():
            raise PermissionError(errno.EACCES, "simulated Windows sharing violation")
        return rename(source, target)

    def transient_replace(source, target):
        counts["replace"] += 1
        if counts["replace"] == 1:
            raise PermissionError(errno.EACCES, "simulated Windows sharing violation")
        return replace(source, target)

    monkeypatch.setattr(Path, "rename", transient_rename)
    monkeypatch.setattr(mb.os, "replace", transient_replace)
    source = tmp_path / "staged"
    source.mkdir()
    target = tmp_path / "published"
    mb.write_json(source / "content.json", {"owner": 1})
    assert mb.publish_directory(source, target)
    assert counts == {"rename": 3, "replace": 2}
    source.mkdir()
    mb.write_json(source / "content.json", {"owner": 2})
    assert not mb.publish_directory(source, target)
    assert counts["rename"] == 4
    assert mb.read_json(target / "content.json") == {"owner": 1}
