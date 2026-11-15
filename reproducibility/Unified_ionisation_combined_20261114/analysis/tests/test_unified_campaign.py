"""Check approved fit counts, lambda selection and width-matched initialisation."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from collect_unified import collect, selection_replay_delta, shared_environment
import combined_train as trainer
import unified_campaign as unified
from unified_features import category_codes


@pytest.fixture(scope="module")
def campaign():
    return unified.load_campaign(Path(__file__).resolve().parents[2])


def test_frozen_scope_and_selection(campaign):
    assert campaign.plan["identity"]["counts"] == dict(units=15, inner_fits=1350, final_fits=1080, total_fits=2430)
    ids = set()
    for unit, (backbone, outer) in unified.UNITS.items():
        inner = unified.inner_requests(campaign.plan, unit)
        assert len(inner) == 90
        mock_fits = []
        for request in inner:
            assert request["backbone"] == backbone and request["outer_fold"] == outer and request["seed"] == 42
            assert trainer._check_request(request)[-1] is True
            assert not (set(request["train_ids"]) & set(request["predict_ids"]))
            predictions = [{"structure_id": sid, "pred_t_half": "1.0", "pred_Vd": "1.0", "pred_CL": "1.0"}
                           for sid in request["predict_ids"]]
            mock_fits.append((request, (dict(best_epoch=request["inner_fold"] * 2), predictions)))
        markers = {request["fit_id"]: request["fit_id"] for request in inner}
        selection = unified.selection_record(campaign, unit, mock_fits, markers)
        assert selection["selected_lambda"] == 0.03 and selection["zero_eligible_lambda"] == 0
        for arm in ("P0", "P1"):
            assert selection["zero_arm_epochs"][arm] == {
                field: selection["grid"][0]["arms"][arm][field] for field in ("epochs", "median_epoch")
            }
        final = unified.final_requests(campaign.plan, unit, selection)
        assert len(final) == 72
        for request in final:
            assert request["fixed_epochs"] == 6 and request["inner_fold"] is None
            assert trainer._check_request(request)[-1] is True
            assert not request["val_ids"]
        for requests in (inner, final):
            for request in requests:
                assert request["fit_id"] not in ids
                ids.add(request["fit_id"])
    assert len(ids) == 2430


def test_strict_threshold():
    rows = [dict(pka_acid="8.0", pka_base="8.0"), dict(pka_acid="7.9999", pka_base="8.0001"),
            dict(pka_acid="7.9999", pka_base="8.0"), dict(pka_acid="8.0", pka_base="8.0001")]
    assert category_codes("PH8", rows) == [2, 3, 0, 1]


def test_cross_backbone_runtime_identity(campaign):
    assert "analysis/collect_unified.py" not in campaign.plan["identity"]["code"]
    common = dict(platform="Linux", gpu_name="Tesla T4", torch="2.5.1+cu124")
    digest = unified.archived.digest
    chemprop = dict(common, chemprop="1.6.1")
    graph = shared_environment("GraphConv", dict(environment=common, environment_hash=digest(common)))
    molecule = shared_environment("Chemprop", dict(environment=chemprop, environment_hash=digest(chemprop)))
    assert graph == molecule == common
    with pytest.raises(ValueError, match="saved checksum"):
        shared_environment("Chemprop", dict(environment=chemprop, environment_hash=digest(common)))
    invalid_version = dict(common, chemprop="2.0.0")
    with pytest.raises(ValueError, match="Chemprop version"):
        shared_environment("Chemprop", dict(environment=invalid_version, environment_hash=digest(invalid_version)))


def test_selection_replay_keeps_decisions_and_hashes_exact():
    values = dict(score=0.25, endpoint_log10_mae={name: 0.25 for name in unified.archived.TRIAD_ENDPOINTS},
                  epochs=[dict(fit_id="f1", best_epoch=5)], median_epoch=5)
    saved = dict(selected_lambda=0.1, grid=[dict(lambda_value=0.1, mean_score=0.25,
                                               arms={arm: values for arm in unified.archived.ARMS})],
                 source_bundles={"f1": "source_hash"})
    replay = deepcopy(saved)
    replay["grid"][0]["mean_score"] += 1e-16
    assert 0 < selection_replay_delta(saved, replay) < 1e-15
    replay["grid"][0]["mean_score"] += 1e-8
    with pytest.raises(ValueError, match="within 1e-10"):
        selection_replay_delta(saved, replay)
    replay = deepcopy(saved)
    replay["grid"][0]["arms"]["P0"]["epochs"][0]["best_epoch"] = 6
    with pytest.raises(ValueError, match="calibration identities"):
        selection_replay_delta(saved, replay)


def test_collector_reuses_exact_zero_loss_predictions(campaign, monkeypatch, tmp_path):
    digest = unified.archived.digest
    runtime = dict(platform="Linux", gpu_name="Tesla T4", torch="2.5.1+cu124")
    output, destination = tmp_path / "units", tmp_path / "exports"
    def fake_unit(campaign, unit, path):
        backbone, _ = unified.UNITS[unit]
        common = dict(runtime, chemprop="1.6.1") if backbone == "Chemprop" else runtime
        summary = dict(environment=common, environment_hash=digest(common), best_epoch=7)

        def fake_fit(request):
            predictions = [dict(structure_id=sid, **{f"pred_{target}": "1.0" for target in trainer.ENDPOINTS},
                                **{f"physical_{target}": "10.0" for target in trainer.ENDPOINTS})
                           for sid in request["predict_ids"]]
            return request, (summary, predictions)

        inner = [fake_fit(request) for request in unified.inner_requests(campaign.plan, unit)]
        markers = {request["fit_id"]: "fixture" for request, _ in inner}
        selection = unified.selection_record(campaign, unit, inner, markers)
        final = [fake_fit(request) for request in unified.final_requests(campaign.plan, unit, selection)]
        markers.update({request["fit_id"]: "fixture" for request, _ in final})
        marker = dict(fit_markers=markers)
        unified.archived.write_json(path / "complete_unit.json", marker)
        return marker, selection, inner, final, 0.

    monkeypatch.setattr("collect_unified.read_unit", fake_unit)
    collected = collect(campaign, output, destination)
    assert collected["environment"] == runtime
    read = unified.archived.rows
    ionisation = {(row["source_fit_id"], row["structure_id"]): row
                  for row in read(destination / "ionisation/outer_predictions.csv")}
    assert len(ionisation) == 17658
    rows = [row for row in read(destination / "combined/final_predictions.csv") if row["cell"] in ("00", "10")]
    assert {(row["fit_id"], row["structure_id"]) for row in rows} == ionisation.keys()
    for row in rows:
        same = ionisation[row["fit_id"], row["structure_id"]]
        assert row["feature_arm"] == same["arm"]
        for target in trainer.ENDPOINTS:
            scale = "physical" if target in ("F", "PPB") else "pred"
            assert float(row[f"q_{target}"]) == float(same[f"{scale}_{target}"])


@pytest.fixture
def tiny_inputs(tmp_path):
    release, structural = tmp_path / "release", tmp_path / "features"
    release.mkdir()
    structural.mkdir()
    ids = [f"s{i:02d}" for i in range(12)]
    data = pd.DataFrame(dict(structure_id=ids, canonical_smiles=["CCO", "CCN", "CC"] * 4,
                             ionisation_class=["anionic", "neutral", "zwitterionic"] * 4,
                             pka_acid=[4, 7.8, 4] * 4, pka_base=[6, 6, 9] * 4))
    for endpoint in trainer.RAW_ENDPOINTS:
        data[endpoint] = [1, 2, 3, 4, 5, 6, 7, 8, 40, 50, 60, 70]
    data.to_csv(release / "model_dataset.csv", index=False)
    trainer.base._write_json(structural / "structures.json", ids)
    np.save(structural / "chemberta.npy", np.random.default_rng(42).normal(size=(12, 768)).astype(np.float32))
    request = dict(cohort="numerical", seed=42, train_ids=ids[:8], val_ids=ids[8:], predict_ids=ids[8:],
                   max_epochs=2, fixed_epochs=None, diagnostic=True, task_weights=[1.] * 7)
    return release, structural, request


@pytest.mark.parametrize("backbone", trainer.BACKBONES)
def test_cpu_archived_parity_and_shared_init(backbone, tiny_inputs, tmp_path):
    release, structural, request = tiny_inputs
    config = dict(hidden_dim=8, ffn_layers=2, dropout=0.2, lr=0.001, weight_decay=0.001)
    if backbone != "ChemBERTa_head":
        config["depth"] = 2
    request.update(config=config, backbone=backbone)
    summaries = {}
    for arm, lam in [(arm, 0.) for arm in unified.ZERO_ARMS] + [("P1", 0.3)]:
        cell = trainer.CELLS[arm, lam > 0]
        fit = dict(request, fit_id=cell, feature_arm=arm, cell=cell, lambda_value=lam)
        folder = tmp_path / cell
        result = trainer.run_fit(fit, release, structural, folder, "cpu")
        summaries[arm, lam] = result
        assert (folder / "complete.json").is_file()
        if arm in ("P0", "P1") and lam == 0:
            old = dict(request, fit_id=f"old-{cell}", arm=arm, lambda_value=0.)
            archived = trainer.frozen.run_fit(old, release, structural, tmp_path / f"old-{cell}", "cpu")
            assert (folder / "predictions.csv").read_bytes() == (tmp_path / f"old-{cell}" / "predictions.csv").read_bytes()
            assert result["initial_state_hash"] == archived["initial_state_hash"]
    assert len({result["common_state_hash"] for result in summaries.values()}) == 1
    assert len({result["initial_state_hash"] for (arm, _), result in summaries.items() if arm != "P0"}) == 1
    assert (tmp_path / "10" / "predictions.csv").read_bytes() != (tmp_path / "PH8" / "predictions.csv").read_bytes()
    assert (tmp_path / "10" / "predictions.csv").read_bytes() != (tmp_path / "11" / "predictions.csv").read_bytes()
