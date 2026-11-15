import json

import numpy as np
import pandas as pd
import pytest
import torch

import mass_balance_train as mb


def test_transforms_and_scaler_match_protocol():
    raw = np.array([[1, 10, 100, 0, 100, 10000, 1], [10, 100, 1000, 50, 25, 1, 10]])
    target = mb.transform_targets(raw)
    np.testing.assert_allclose(target[0], [0, 1, 2, np.log10(.1 / 99.9), np.log10(99.9 / .1), 4, 0])
    np.testing.assert_allclose(mb.inverse_physical(target)[0], [1, 10, 100, .1, 99.9, 10000, 1])
    assert raw[0, 3] == 0 and raw[0, 4] == 100
    mean, sd, n = mb.fit_scaler(target)
    np.testing.assert_allclose(mean, target.mean(axis=0))
    np.testing.assert_allclose(sd, np.abs(target[0] - target[1]) / 2)
    np.testing.assert_array_equal(n, np.full(7, 2))
    with pytest.raises(ValueError, match="variance"):
        mb.fit_scaler(np.repeat(target[:1], 2, axis=0))
    with pytest.raises(ValueError, match="two training"):
        mb.fit_scaler(target[:1])
    with pytest.raises(ValueError, match="Positive endpoints"):
        mb.transform_targets(np.zeros((2, 7)))


def test_masked_weighting_and_all_input_penalty_are_separate():
    mean = torch.tensor([0., 2., 1., 0., 0., 0., 0.])
    sd = torch.tensor([2., 1., 1., 1., 1., 1., 1.])
    pred = torch.zeros(3, 7, requires_grad=True)
    with torch.no_grad():
        pred[:, 0] = (mb.LOG10_LN2 + 1) / 2
        pred[2, 0] += 1
    missing = float("nan")
    target = torch.full_like(pred, missing)
    target[:, 4] = torch.tensor([1., 3., missing])
    target[0, 5] = 2
    weights = torch.tensor([.1, .1, .1, .1, .1, .2, .1])
    mse, penalty = mb.loss_terms(pred, target, mean, sd, weights)
    assert mse.item() == pytest.approx((.1 * 5 + .2 * 4) / .3)
    assert penalty.item() == pytest.approx(4 / 3)
    (mse + penalty).backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[2, 0] != 0
    assert pred.grad[2, 5] == 0


def test_validation_reduces_whole_partition_counts():
    nan = float("nan")
    batches = [
        (torch.zeros(3, 2), torch.tensor([[1., 3.], [1., nan], [1., nan]])),
        (torch.zeros(1, 2), torch.tensor([[3., 1.]])),
    ]
    score, sums, counts = mb._score_validation(torch.nn.Identity(), batches, "cpu", 2)
    assert sums == [12, 10] and counts == [4, 2]
    assert score == 4


@pytest.fixture
def tiny_bundle(tmp_path):
    release, features = tmp_path / "release", tmp_path / "features"
    release.mkdir()
    features.mkdir()
    ids = [f"s{i:02}" for i in range(12)]
    source = pd.DataFrame({
        "structure_id": ids, "canonical_smiles": ["CCO", "CC", "O", "CCN", "CCC", "CCCl"] * 2,
    })
    for i, raw in enumerate(mb.RAW_ENDPOINTS):
        source[raw] = np.arange(1., 13.) * (i + 1)
    source.loc[1, "VD(liter)"] = np.nan
    source.loc[7, "Cl(liter/hour)"] = np.nan
    source.loc[8, "PPB(percentage)"] = 0.0
    source.to_csv(release / "model_dataset.csv", index=False)
    (features / "structures.json").write_text(json.dumps(ids[::-1]), encoding="utf-8")
    matrix = np.arange(12 * 768, dtype=np.float32).reshape(12, 768) / 10000
    np.save(features / "chemberta.npy", matrix[::-1])
    request = {
        "fit_id": "tiny", "backbone": "GraphConv",
        "config": dict(hidden_dim=8, depth=2, ffn_layers=2, dropout=.1, lr=.001, weight_decay=.0001),
        "train_ids": ids[:6], "val_ids": ids[6:9], "predict_ids": ids[6:9][::-1],
        "seed": 101, "lambda_value": 0, "endpoint": None, "fixed_epochs": None, "max_epochs": 3,
    }
    return release, features, source, request


@pytest.mark.parametrize("backbone", ["GraphConv", "GIN", "GAT", "AttentiveFP", "Chemprop", "ChemBERTa_head"])
def test_real_fit_checkpoint_and_order(backbone, tiny_bundle, tmp_path):
    release, features, source, request = tiny_bundle
    request["backbone"] = backbone
    request["max_epochs"] = 1
    if backbone == "Chemprop":
        pytest.importorskip("chemprop")
        request["config"]["ffn_layers"] = 3
    if backbone == "ChemBERTa_head":
        request["config"]["depth"] = None
    out = tmp_path / "output"
    summary = mb.run_fit(request, release, features, out, device="cpu")
    assert summary["actual_epochs"] == summary["best_epoch"] == 1
    assert summary["parameter_count"] > 0 and summary["checkpoint_reload_verified"]
    assert set(json.loads((out / "complete.json").read_text())["files"]) == set(mb.OUTPUT_FILES)
    predictions = pd.read_csv(out / "predictions.csv")
    assert predictions.structure_id.tolist() == request["predict_ids"]
    values = predictions[[f"pred_{x}" for x in mb.ENDPOINTS]].to_numpy()
    assert np.isfinite(values).all()
    scaler = json.loads((out / "scaler.json").read_text())
    expected = mb.transform_targets(source.set_index("structure_id").loc[request["predict_ids"], list(mb.RAW_ENDPOINTS)])
    squared = ((expected - values) / scaler["sd"]) ** 2
    assert summary["val_equal_mse"] == pytest.approx(np.nanmean(squared, axis=0).mean(), rel=1e-6)
    assert summary["val_observed_counts"]["CL"] == 2


def test_rng_pairing_resume_and_hash_validation(tiny_bundle, tmp_path):
    release, features, _, request = tiny_bundle
    first, second = tmp_path / "first", tmp_path / "second"
    base = mb.run_fit(request, release, features, first, device="cpu")
    assert mb.run_fit(request, release, features, first, device="cpu") == base
    other = dict(request, fit_id="penalty", lambda_value=1.0, train_ids=request["train_ids"][::-1])
    penalized = mb.run_fit(other, release, features, second, device="cpu")
    assert base["initial_state_hash"] == penalized["initial_state_hash"]
    assert base["resolved_hash"] != penalized["resolved_hash"]
    for name in ("batch_order_seed", "dropout_seed", "batch_order_sha256"):
        assert pd.read_csv(first / "learning_curve.csv")[name].equals(pd.read_csv(second / "learning_curve.csv")[name])
    assert (first / "predictions.csv").read_bytes() != (second / "predictions.csv").read_bytes()
    with pytest.raises(ValueError, match="Completed fit hash mismatch"):
        mb.run_fit(dict(request, lambda_value=.1), release, features, first, device="cpu")
    (first / "predictions.csv").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="Completed artifact hash mismatch"):
        mb.run_fit(request, release, features, first, device="cpu")


def test_fixed_single_task_uses_training_labels_only(tiny_bundle, tmp_path):
    release, features, source, request = tiny_bundle
    request.update(endpoint="PPB", val_ids=[], fixed_epochs=2, predict_ids=["s09", "s08"])
    first, second = tmp_path / "first", tmp_path / "second"
    s1 = mb.run_fit(request, release, features, first, device="cpu")
    source.loc[8:, "PPB(percentage)"] = 100
    source.to_csv(release / "model_dataset.csv", index=False)
    s2 = mb.run_fit(request, release, features, second, device="cpu")
    assert s1["val_equal_mse"] is None and s1["actual_epochs"] == s1["best_epoch"] == 2
    assert s1["data_hash"] != s2["data_hash"]
    assert (first / "scaler.json").read_bytes() == (second / "scaler.json").read_bytes()
    pred = pd.read_csv(first / "predictions.csv")
    assert pred.structure_id.tolist() == ["s09", "s08"]
    assert pred[["pred_PPB", "physical_PPB"]].notna().all().all()
    assert pred.drop(columns=["structure_id", "pred_PPB", "physical_PPB"]).isna().all().all()
    assert (first / "predictions.csv").read_bytes() == (second / "predictions.csv").read_bytes()


def test_overlapping_training_and_validation_fails_without_completion(tiny_bundle, tmp_path):
    release, features, _, request = tiny_bundle
    request["val_ids"].append(request["train_ids"][0])
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="disjoint"):
        mb.run_fit(request, release, features, output, device="cpu")
    assert not (output / "complete.json").exists()


def test_pinned_features_must_match_dataset_and_manifest(tiny_bundle, tmp_path):
    release, features, _, request = tiny_bundle
    request["backbone"] = "ChemBERTa_head"
    request["config"]["depth"] = None
    request["max_epochs"] = 1
    files = {name: mb._file_hash(features / name) for name in ("structures.json", "chemberta.npy")}
    (features / "manifest.json").write_text(json.dumps({
        "data": {"model_dataset.csv": mb._file_hash(release / "model_dataset.csv"), "manifest.json": None},
        "files": files,
    }), encoding="utf-8")
    first = tmp_path / "first"
    mb.run_fit(request, release, features, first, device="cpu")
    matrix = np.load(features / "chemberta.npy")
    np.save(features / "chemberta.npy", matrix * 2)
    with pytest.raises(ValueError, match="Feature checksum mismatch"):
        mb.run_fit(request, release, features, first, device="cpu")


@pytest.fixture
def production_bundle(tiny_bundle):
    release, features, source, request = tiny_bundle
    request.update(campaign_id="production-fixture", max_epochs=1)
    fingerprint = "frozen-fixture"
    dataset_hash = mb._file_hash(release / "model_dataset.csv")
    release_manifest = release / "manifest.json"
    release_manifest.write_text(json.dumps({
        "data_frozen": True, "splits_frozen": True, "fingerprint": fingerprint,
        "files": {"model_dataset.csv": dataset_hash},
    }), encoding="utf-8")
    ids = source.structure_id.tolist()
    (features / "structures.json").write_text(json.dumps(ids), encoding="utf-8")
    torch.save([mb._make_graph(s) for s in source.canonical_smiles], features / "graphs.pt")
    feature_manifest = {
        "data": {"model_dataset.csv": dataset_hash, "manifest.json": mb._file_hash(release_manifest),
                 "fingerprint": fingerprint},
        "files": {f: mb._file_hash(features / f) for f in ("structures.json", "graphs.pt")},
    }
    (features / "manifest.json").write_text(json.dumps(feature_manifest), encoding="utf-8")
    return release, features, request, feature_manifest


@pytest.mark.parametrize("missing", ["release/manifest.json", "features/manifest.json",
                                     "features/graphs.pt", "features/structures.json"])
def test_production_refuses_missing_inputs(production_bundle, tmp_path, missing):
    release, features, request, _ = production_bundle
    (tmp_path / missing).unlink()
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="requires manifest|Required feature file"):
        mb.run_fit(request, release, features, output, device="cpu")
    assert not (output / "complete.json").exists()


def test_production_uses_cached_graphs_and_rejects_permuted_ids(production_bundle, tmp_path, monkeypatch):
    release, features, request, manifest = production_bundle

    def no_graph_fallback(_):
        pytest.fail("Production fit attempted to regenerate a graph")

    monkeypatch.setattr(mb, "_make_graph", no_graph_fallback)
    summary = mb.run_fit(request, release, features, tmp_path / "first", device="cpu")
    assert summary["actual_epochs"] == 1
    ids_path = features / "structures.json"
    ids_path.write_text(json.dumps(json.loads(ids_path.read_text())[::-1]), encoding="utf-8")
    manifest["files"]["structures.json"] = mb._file_hash(ids_path)
    (features / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="structure order must exactly match"):
        mb.run_fit(request, release, features, tmp_path / "second", device="cpu")


def test_production_enforces_pinned_adapter_versions(production_bundle, tmp_path, monkeypatch):
    release, features, request, _ = production_bundle
    with monkeypatch.context() as patch:
        patch.setattr(mb, "PRODUCTION_PYG", "0.0")
        with pytest.raises(ValueError, match="Production fits require torch-geometric"):
            mb.run_fit(request, release, features, tmp_path / "pyg", device="cpu")
    pytest.importorskip("chemprop")
    with monkeypatch.context() as patch:
        patch.setattr(mb, "PRODUCTION_RDKIT", "0.0")
        with pytest.raises(ValueError, match="Production Chemprop requires RDKit"):
            mb.run_fit(dict(request, backbone="Chemprop"), release, features, tmp_path / "rdkit", device="cpu")
