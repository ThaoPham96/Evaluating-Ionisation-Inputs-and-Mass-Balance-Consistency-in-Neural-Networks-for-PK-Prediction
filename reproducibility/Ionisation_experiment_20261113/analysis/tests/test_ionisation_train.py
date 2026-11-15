import json

import numpy as np
import pandas as pd
import pytest
import torch

import ionisation_train as ion


@pytest.fixture
def bundle(tmp_path):
    release, features = tmp_path / "release", tmp_path / "features"
    release.mkdir()
    features.mkdir()
    ids = [f"s{i:02}" for i in range(20)]
    smiles = ["CCO", "CC", "O", "CCN", "CCC", "CCCl", "C#N", "CC(=O)O", "C1CC1", "CCF"] * 2
    data = pd.DataFrame({
        "canonical_smiles": smiles, "ionisation_class": list(ion.CLASS_NAMES) * 5,
        "pka_acid": np.linspace(2., 12., len(ids)), "pka_base": np.linspace(9., 3., len(ids)),
    }, index=pd.Index(ids, name="structure_id"))
    for i, raw in enumerate(ion.RAW_ENDPOINTS):
        data[raw] = np.arange(1., len(ids) + 1.) * (i + 1) / 2
    data.loc["s02", "VD(liter)"] = np.nan
    data.loc["s10", "Cl(liter/hour)"] = np.nan
    data.loc["s08", "PPB(percentage)"] = 0.0
    data.to_csv(release / "model_dataset.csv")
    (features / "structures.json").write_text(json.dumps(ids), encoding="utf-8")
    matrix = np.arange(20 * 768, dtype=np.float32).reshape(20, 768) / 10000
    np.save(features / "chemberta.npy", matrix)
    request = {
        "fit_id": "tiny", "diagnostic": True, "backbone": "GraphConv", "arm": "C3", "cohort": "class",
        "config": dict(hidden_dim=8, depth=2, ffn_layers=2, dropout=.1, lr=.001, weight_decay=.0001),
        "train_ids": ids[:8], "val_ids": ids[8:12], "predict_ids": ids[8:12][::-1],
        "seed": 101, "lambda_value": 0, "task_weights": [1] * 7, "fixed_epochs": None, "max_epochs": 2,
    }
    return release, features, data, request


def test_hh_profile_order_threshold_direction_and_stability():
    pka = [[7.4, 7.4], [-1e308, 1e308], [1e308, -1e308], [2, 5]]
    result = ion.ionisation_propensities(pka, ion.PROFILE_PH)
    np.testing.assert_allclose(result[0, -2:], [.5, .5])
    np.testing.assert_array_equal(result[1], np.ones(8))
    np.testing.assert_array_equal(result[2], np.zeros(8))
    np.testing.assert_allclose(result[3], [1 / (1 + 10 ** (2 - h)) if i % 2 == 0
                                          else 1 / (1 + 10 ** (h - 5))
                                          for i, h in enumerate(np.repeat(ion.PROFILE_PH, 2))])
    assert np.all(np.diff(result[0, ::2]) > 0)
    assert np.all(np.diff(result[0, 1::2]) < 0)
    with pytest.raises(ValueError, match="finite"):
        ion.ionisation_propensities([[np.nan, 1.]], ion.PROFILE_PH)


def test_categorical_descriptor_map(bundle):
    _, _, data, request = bundle
    data = data.assign(pka_acid=np.nan, pka_base=np.nan)
    train = request["train_ids"]
    c1, _ = ion.fit_descriptors("C1", data, train)
    c2, _ = ion.fit_descriptors("C2", data, train)
    c3, _ = ion.fit_descriptors("C3", data, train)
    p1, _ = ion.fit_descriptors("P1", data, train)
    torch.testing.assert_close(c1[:4], torch.tensor([[-1.], [1.], [0.], [0.]]))
    torch.testing.assert_close(c2[:4], torch.eye(4))
    torch.testing.assert_close(c3[:4], torch.arange(4))
    torch.testing.assert_close(c2, p1)
    for arm in ("C0", "P0"):
        tensor, metadata = ion.fit_descriptors(arm, data, train)
        assert tensor.shape == (len(data), 0) and metadata["input_dim"] == 0


def test_numerical_feature_scaling_and_width(bundle):
    _, _, data, request = bundle
    train = request["train_ids"]
    descriptors = {}
    for arm in ("P2", "P3", "P4", "P5"):
        actual, recipe = ion.fit_descriptors(arm, data, train)
        heldout = data.copy()
        heldout.loc[~heldout.index.isin(train), ["pka_acid", "pka_base"]] = [12., 14.]
        perturbed, new_recipe = ion.fit_descriptors(arm, heldout, train)
        assert recipe == new_recipe
        torch.testing.assert_close(actual[:len(train)], perturbed[:len(train)])
        assert not torch.equal(actual[len(train):], perturbed[len(train):])
        torch.testing.assert_close(actual[:len(train)].square().sum(dim=1).mean(), torch.tensor(2.))
        descriptors[arm] = actual, recipe
    p2, recipe = descriptors["P2"]
    np.testing.assert_allclose(recipe["mean"], data.loc[train, ["pka_acid", "pka_base"]].mean(axis=0))
    torch.testing.assert_close(p2[:len(train)].mean(dim=0), torch.zeros(2), atol=1e-7, rtol=0.)
    p3, _ = descriptors["P3"]
    p4, _ = descriptors["P4"]
    p5, recipe = descriptors["P5"]
    torch.testing.assert_close(p4[:, -2:], p3 / 2)
    torch.testing.assert_close(p5, p3.repeat(1, 4) / 2)
    assert recipe["width_divisor"] == 2. and recipe["repeats"] == 4
    data.loc[train, "pka_acid"] = 3.3
    p4, recipe = ion.fit_descriptors("P4", data, train)
    assert recipe["zero_variance"] == [True, False] * 4
    assert recipe["sd"][::2] == [0.] * 4
    torch.testing.assert_close(p4[:, ::2], torch.zeros_like(p4[:, ::2]))


def test_equal_masked_endpoint_loss():
    nan = float("nan")
    target = torch.tensor([[1., 2., nan, nan, nan, nan, nan], [3., nan, nan, nan, nan, nan, nan]])
    prediction = torch.zeros_like(target, requires_grad=True)
    loss = ion.supervised_loss(prediction, target)
    assert loss.item() == pytest.approx(4.5)
    loss.backward()
    assert torch.equal(prediction.grad[~target.isfinite()], torch.zeros(11))


@pytest.mark.parametrize("backbone", ion.BACKBONES)
def test_real_fits_pair_initialization_and_affine_fusion(backbone, bundle, tmp_path):
    release, features, data, request = bundle
    request["backbone"] = backbone
    request["config"]["depth"] = None if backbone == "ChemBERTa_head" else 2
    if backbone == "Chemprop":
        request["config"]["ffn_layers"] = 3
    base_request = dict(request, arm="C0", fit_id="tiny-base", train_ids=request["train_ids"][::-1])
    first, second = tmp_path / "c0", tmp_path / "c3"
    base = ion.run_fit(base_request, release, features, first, device="cpu")
    embedded = ion.run_fit(request, release, features, second, device="cpu")
    assert base["common_state_hash"] == embedded["common_state_hash"]
    assert base["initial_state_hash"] != embedded["initial_state_hash"]
    assert embedded["shape_dependent_parameters"] == ["fusion.weight", "embedding.weight"]
    assert embedded["descriptor_input_dim"] == 16
    for name in ("batch_order_seed", "dropout_seed", "batch_order_sha256"):
        pd.testing.assert_series_equal(pd.read_csv(first / "learning_curve.csv")[name],
                                       pd.read_csv(second / "learning_curve.csv")[name])

    assert embedded["checkpoint_reload_verified"] and embedded["actual_epochs"] == 2
    assert set(json.loads((second / "complete.json").read_text())["files"]) == set(ion.OUTPUT_FILES)
    pred = pd.read_csv(second / "predictions.csv")
    assert pred.structure_id.tolist() == request["predict_ids"]
    observed = ion.base.transform_targets(data.loc[request["predict_ids"], list(ion.RAW_ENDPOINTS)])
    values = pred[[f"pred_{x}" for x in ion.ENDPOINTS]].to_numpy()
    scaler = json.loads((second / "scaler.json").read_text())
    assert embedded["scaler_hash"] == ion.base._object_hash(scaler)
    assert embedded["val_equal_mse"] == pytest.approx(np.nanmean(((observed - values) / scaler["sd"]) ** 2,
                                                                  axis=0).mean(), rel=1e-6)
    assert embedded["val_observed_counts"]["CL"] == 3

    # For each structure, recover exactly the learned category offset from one-hot.
    codes, _ = ion.fit_descriptors("C3", data, request["train_ids"])
    inputs = ion.FitInputs(backbone, data, features, codes)
    batch_ids = data.index[:4][::-1].tolist()
    model, info = ion._make_model(backbone, request["config"], "C3", 101, inputs, batch_ids[0], torch.device("cpu"))
    state = torch.load(second / "checkpoint.pt", weights_only=True)["state_dict"]
    model.load_state_dict(state)
    model.train()  # Exercise the active pre-affine structural dropout.
    seen = []
    handle = model.fusion.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach()))
    structural_inputs, batch_codes = inputs.batch(batch_ids, "cpu")
    torch.testing.assert_close(batch_codes, torch.tensor([3, 2, 1, 0]))
    model((structural_inputs, batch_codes))
    handle.remove()
    structural_dim = info["structural_input_dim"]
    embed = model.embedding(batch_codes)
    torch.testing.assert_close(seen[0][:, structural_dim:], embed)
    e_weight = model.fusion.weight[:, structural_dim:]
    one_hot = torch.nn.functional.one_hot(batch_codes, num_classes=4).float()
    torch.testing.assert_close(embed @ e_weight.T, one_hot @ (model.embedding.weight @ e_weight.T))


def test_descriptor_batch_parity(bundle):
    _, features, data, request = bundle
    matrix, _ = ion.fit_descriptors("P4", data, request["train_ids"])
    batch_ids = ["s13", "s06", "s08"]
    expected = matrix[[13, 6, 8]]
    for backbone in ion.BACKBONES:
        inputs = ion.FitInputs(backbone, data, features, matrix)
        _, features_seen = inputs.batch(batch_ids, "cpu")
        torch.testing.assert_close(features_seen, expected)


@pytest.mark.parametrize("change, message", [
    ({"lambda_value": .01}, "lambda_value=0"),
    ({"task_weights": [1, 5, 1, 1, 1, 1, 1]}, "task weights"),
    ({"task_weights": [1, 1, True, 1, 1, 1, 1]}, "task weights"),
    ({"endpoint": "Vd"}, "all seven endpoints"),
    ({"cohort": "numerical"}, "Arm and cohort"),
    ({"fixed_epochs": True}, "Invalid fixed epoch"),
    ({"diagnostic": False}, "diagnostic=True"),
    ({"campaign_id": "production"}, "diagnostic flag"),
    ({"val_ids": ["s00"]}, "disjoint"),
])
def test_invalid_fit_request_fails_before_completion(bundle, tmp_path, change, message):
    release, features, _, request = bundle
    out = tmp_path / "output"
    with pytest.raises(ValueError, match=message):
        ion.run_fit(dict(request, **change), release, features, out, device="cpu")
    assert not (out / "complete.json").exists()


def _freeze_bundle(release, features, data, cohort, **flags):
    release_manifest = release / "manifest.json"
    data_hash = ion.base._file_hash(release / "model_dataset.csv")
    fingerprint = "frozen-test"
    release_manifest.write_text(json.dumps({
        "data_frozen": True, "splits_frozen": True, "training_ready": True, **flags,
        "cohort": cohort, "fingerprint": fingerprint, "files": {"model_dataset.csv": data_hash},
    }), encoding="utf-8")
    torch.save([ion.base._make_graph(s) for s in data.canonical_smiles], features / "graphs.pt")
    files = ("structures.json", "graphs.pt", "chemberta.npy")
    (features / "manifest.json").write_text(json.dumps({
        "data": {"model_dataset.csv": data_hash, "manifest.json": ion.base._file_hash(release_manifest),
                 "fingerprint": fingerprint},
        "files": {name: ion.base._file_hash(features / name) for name in files},
    }), encoding="utf-8")


def test_production_gate_fold_counts_and_input_hashes(bundle, tmp_path):
    release, features, data, request = bundle
    request.pop("diagnostic")
    request.update(campaign_id="production-fixture", max_epochs=1)
    out = tmp_path / "fit"
    with pytest.raises(ValueError, match="requires manifest"):
        ion.run_fit(request, release, features, out, device="cpu")
    _freeze_bundle(release, features, data, "class", training_ready=False)
    with pytest.raises(ValueError, match="training-ready"):
        ion.run_fit(request, release, features, out, device="cpu")
    _freeze_bundle(release, features, data, "numerical")
    with pytest.raises(ValueError, match="exact cohort"):
        ion.run_fit(request, release, features, out, device="cpu")
    _freeze_bundle(release, features, data, "class")
    with pytest.raises(ValueError, match="at least 10"):
        ion.run_fit(request, release, features, out, device="cpu")
    ids = data.index[8:].tolist()
    request.update(val_ids=ids, predict_ids=ids[::-1])
    summary = ion.run_fit(request, release, features, out, device="cpu")
    assert summary["actual_epochs"] == 1 and not summary["diagnostic"]
    assert ion.run_fit(request, release, features, out, device="cpu") == summary
    with pytest.raises(ValueError, match="Completed fit hash mismatch"):
        ion.run_fit(dict(request, seed=999), release, features, out, device="cpu")
    (out / "scaler.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Completed artifact hash mismatch"):
        ion.run_fit(request, release, features, out, device="cpu")
    with (features / "graphs.pt").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="Feature checksum mismatch"):
        ion.run_fit(request, release, features, out, device="cpu")


def test_fixed_numerical_fit_heldout_targets_and_artifacts(bundle, tmp_path):
    release, features, data, request = bundle
    request.update(arm="P5", cohort="numerical", val_ids=[], fixed_epochs=1)
    first, second = tmp_path / "before", tmp_path / "after"
    before = ion.run_fit(request, release, features, first, device="cpu")
    data.loc[request["predict_ids"], list(ion.RAW_ENDPOINTS)] = 95.
    data.to_csv(release / "model_dataset.csv")
    after = ion.run_fit(request, release, features, second, device="cpu")
    assert before["val_equal_mse"] is None and before["best_epoch"] == 1
    assert before["data_hash"] != after["data_hash"] and before["scaler_hash"] == after["scaler_hash"]
    assert (first / "predictions.csv").read_bytes() == (second / "predictions.csv").read_bytes()
    data.loc["s15", "pka_acid"] = np.nan
    data.to_csv(release / "model_dataset.csv")
    with pytest.raises(ValueError, match="finite pKa"):
        ion.run_fit(dict(request, arm="P0"), release, features, tmp_path / "missing", device="cpu")
