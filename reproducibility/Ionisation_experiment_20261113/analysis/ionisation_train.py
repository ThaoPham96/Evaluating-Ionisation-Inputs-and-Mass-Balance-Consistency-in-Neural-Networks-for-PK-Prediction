"""Train one protocol-v2 ionisation fit and verify its committed artifacts."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from _vendor import ensure_vendor

VENDOR = ensure_vendor()
from vendor_mass_balance import mass_balance_train as base

RAW_ENDPOINTS = base.RAW_ENDPOINTS
ENDPOINTS = base.ENDPOINTS
OUTPUT_FILES = base.OUTPUT_FILES
BACKBONES = ("GraphConv", "Chemprop", "ChemBERTa_head")
CLASS_NAMES = ("anionic", "cationic", "neutral", "zwitterionic")
PROFILE_PH = (2.0, 5.0, 6.5, 7.4)
EMBEDDING_DIM = 16
WIDTHS = {"C0": 0, "C1": 1, "C2": 4, "C3": EMBEDDING_DIM,
          "P0": 0, "P1": 4, "P2": 2, "P3": 2, "P4": 8, "P5": 8}
MIN_VALIDATION_COUNT = 10


def ionisation_propensities(pka, ph_values):
    """Return alternating acid/base propensities for each requested pH."""
    pka = np.asarray(pka, dtype=np.float64)
    ph = np.asarray(ph_values, dtype=np.float64)
    if pka.ndim != 2 or pka.shape[1] != 2 or not np.isfinite(pka).all():
        raise ValueError("Expected two finite acid/base pKa values per structure")
    if ph.ndim != 1 or not ph.size or not np.isfinite(ph).all():
        raise ValueError("Expected a finite one-dimensional pH sequence")
    with np.errstate(over="ignore", under="ignore"):
        exponent = np.stack((pka[:, :1] - ph, ph - pka[:, 1:]), axis=-1) * math.log(10)
        return np.exp(-np.logaddexp(0, exponent)).reshape(len(pka), 2 * ph.size)


def fit_descriptors(arm, data, train_ids):
    """Encode a cohort and return its aligned tensor and JSON-compatible recipe."""
    width = WIDTHS[arm]
    recipe = {"arm": arm, "input_dim": width, "method": "none"}
    if width == 0:
        return torch.empty((len(data), 0), dtype=torch.float32), recipe
    if arm.startswith("C") or arm == "P1":
        class_index = data["ionisation_class"].map(dict(zip(CLASS_NAMES, range(len(CLASS_NAMES)))))
        if class_index.isna().any():
            raise ValueError("Unsupported ionisation_class in the cohort")
        codes = torch.tensor(class_index.to_numpy(), dtype=torch.long)
        recipe["class_order"] = list(CLASS_NAMES)
        if arm == "C1":
            recipe.update(method="scalar_charge_sign", class_values=[-1, 1, 0, 0])
            return torch.tensor(recipe["class_values"], dtype=torch.float32)[codes, None], recipe
        if arm == "C3":
            recipe.update(method="learned_category_embedding", embedding_init_sd=1 / math.sqrt(width))
            return codes, recipe
        recipe["method"] = "one_hot"
        return nn.functional.one_hot(codes, num_classes=len(CLASS_NAMES)).float(), recipe

    pka = data[["pka_acid", "pka_base"]].to_numpy(dtype=np.float64)
    if not np.isfinite(pka).all():
        raise ValueError("Numerical cohort requires two finite pKa values per structure")
    ph = () if arm == "P2" else PROFILE_PH if arm == "P4" else (7.4,)
    values = ionisation_propensities(pka, ph) if ph else pka
    training = values[data.index.get_indexer(train_ids)]
    mean = training.mean(axis=0)
    sd = training.std(axis=0, ddof=0)
    # Comparing the original values detects constants even with roundoff in the mean.
    constant = np.all(training == training[0], axis=0) | (sd == 0)
    if not np.isfinite(mean).all() or not np.isfinite(sd).all():
        raise ValueError("Nonfinite fitted descriptor moments")
    denominator = np.where(constant, 1., sd)
    scaled = (values - mean) / denominator
    scaled[:, constant] = 0.0
    if arm == "P5":
        scaled = np.tile(scaled, (1, len(PROFILE_PH)))
    divisor = math.sqrt(len(PROFILE_PH)) if arm in {"P4", "P5"} else 1.
    features = torch.tensor(scaled / divisor, dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise ValueError("Nonfinite standardised descriptor input")
    recipe.update(
        method="fit_training_only_population_sd", ph=list(ph),
        scaler_channels=([f"{site}({h:g})" for h in ph for site in ("a", "b")]
                         if ph else ["pka_acid", "pka_base"]),
        mean=mean.tolist(), sd=np.where(constant, 0., sd).tolist(),
        scale_denominator=denominator.tolist(), zero_variance=constant.tolist(),
        zero_variance_rule="map to zero on every partition",
        repeats=len(PROFILE_PH) if arm == "P5" else 1, width_divisor=divisor,
        n_fit_training=len(train_ids),
    )
    return features, recipe


class FitInputs:
    """Align the one shared descriptor matrix with the native structural inputs."""

    def __init__(self, backbone, data, feature_dir, descriptors):
        self.structures = base.FitInputs(backbone, data, feature_dir)
        self.indices = {sid: index for index, sid in enumerate(data.index)}
        self.descriptors = descriptors

    def batch(self, ids, device):
        positions = [self.indices[sid] for sid in ids]
        return self.structures.batch(ids, device), self.descriptors[positions].to(device)


class GraphConvPool(nn.Module):
    def __init__(self, convs):
        super().__init__()
        self.convs = convs

    def forward(self, graph):
        x = graph.x
        for conv in self.convs:
            x = torch.relu(conv(x, graph.edge_index))
        return global_mean_pool(x, graph.batch)


class PooledRegressor(nn.Module):
    def __init__(self, encoder, structural_dropout, fusion, head, embedding):
        super().__init__()
        self.encoder = encoder
        self.structural_dropout = structural_dropout
        self.fusion = fusion
        self.head = head
        self.embedding = embedding

    def forward(self, features):
        structure, descriptors = features
        structure = self.structural_dropout(self.encoder(structure))
        if self.embedding is not None:
            descriptors = self.embedding(descriptors)
        fused = self.fusion(torch.cat((structure, descriptors), dim=1))
        return self.head(fused)


def _make_model(backbone, config, arm, seed, inputs, first_id, device):
    # Build the native zero-descriptor model before initialising any new parameters.
    init_seed = base._seed_for(seed, "initialization")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(init_seed)
        native = base._make_model(backbone, config, ENDPOINTS, torch.device("cpu"), inputs.structures, first_id)
        if backbone == "GraphConv":
            encoder = GraphConvPool(native.convs)
            structural_dropout, first = native.dropout, native.fc1
            head = nn.Sequential(nn.ReLU(), native.dropout, native.fc2)
        elif backbone == "Chemprop":
            encoder = native.encoder
            structural_dropout, first = native.readout[0], native.readout[1]
            head = native.readout[2:]
            # MPNEncoder uses this device attribute for its cached graph tensors.
            for component in encoder.modules():
                if hasattr(component, "device"):
                    component.device = device
        else:
            encoder, structural_dropout = nn.Identity(), nn.Identity()
            first, head = native[0], native[1:]

        width = WIDTHS[arm]
        fusion, embedding = first, None
        if width:
            fusion = nn.Linear(first.in_features + width, first.out_features)
            generator = torch.Generator().manual_seed(base._seed_for(seed, "descriptor_columns"))
            # A common maximum-width draw pairs overlapping descriptor columns as well.
            shape = (first.out_features, EMBEDDING_DIM)
            if backbone == "Chemprop":
                std = math.sqrt(2 / (first.in_features + first.out_features))
                columns = torch.randn(shape, generator=generator) * std
            else:
                bound = 1 / math.sqrt(first.in_features)
                columns = torch.empty(shape).uniform_(-bound, bound, generator=generator)
            with torch.no_grad():
                fusion.weight[:, :first.in_features].copy_(first.weight)
                fusion.weight[:, first.in_features:].copy_(columns[:, :width])
                fusion.bias.copy_(first.bias)
            if arm == "C3":
                generator = torch.Generator().manual_seed(base._seed_for(seed, "category_embedding"))
                embedding = nn.Embedding(len(CLASS_NAMES), EMBEDDING_DIM)
                with torch.no_grad():
                    embedding.weight.normal_(std=1 / math.sqrt(EMBEDDING_DIM), generator=generator)
    model = PooledRegressor(encoder, structural_dropout, fusion, head, embedding)
    # Compare the actual common parameters, including the original structural block.
    common_state = {k: v for k, v in model.state_dict().items() if k != "embedding.weight"}
    common_state["fusion.weight"] = fusion.weight[:, :first.in_features].detach()
    initialization = {
        "common_state_hash": base._state_hash(common_state),
        "shape_dependent_parameters": (["fusion.weight"] if width else []) +
                                      (["embedding.weight"] if arm == "C3" else []),
        "structural_input_dim": first.in_features,
        "initialization_seed": init_seed,
        "descriptor_columns_seed": base._seed_for(seed, "descriptor_columns") if width else None,
        "category_embedding_seed": base._seed_for(seed, "category_embedding") if arm == "C3" else None,
        "descriptor_weight_rule": (
            "xavier_normal_on_structure_only_fan" if backbone == "Chemprop" else
            "uniform_on_structure_only_fan_in"
        ) if width else None,
        "pre_affine_dropout": "structural_channels_only" if backbone != "ChemBERTa_head" else "none",
    }
    return model.to(device), initialization


def supervised_loss(prediction, target):
    """Average the masked, training-scaled endpoint MSEs with equal weights."""
    sums, counts = base.squared_errors(prediction, target)
    return base.mean_endpoint_error(sums, counts, torch.ones_like(sums))


def _check_request(request):
    if not isinstance(request, dict):
        raise ValueError("Expected a JSON fit request object")
    for key in ("fit_id", "arm", "cohort", "backbone", "config", "seed",
                "train_ids", "val_ids", "predict_ids"):
        if key not in request:
            raise ValueError(f"Missing fit request field: {key}")
    if not isinstance(request["fit_id"], str) or not request["fit_id"].strip():
        raise ValueError("fit_id must be a nonempty string")
    arm, cohort, backbone = (request[k] for k in ("arm", "cohort", "backbone"))
    if not isinstance(arm, str) or arm not in WIDTHS:
        raise ValueError("Unknown ionisation arm")
    if cohort != ("class" if arm.startswith("C") else "numerical"):
        raise ValueError("Arm and cohort do not match")
    if backbone not in BACKBONES:
        raise ValueError("Unknown ionisation backbone")
    production = "campaign_id" in request
    if production:
        if not isinstance(request["campaign_id"], str) or not request["campaign_id"].strip():
            raise ValueError("Production campaign_id must be a nonempty string")
        if "diagnostic" in request:
            raise ValueError("Production requests cannot include a diagnostic flag")
    elif request.get("diagnostic") is not True:
        raise ValueError("Nonproduction fits require diagnostic=True")
    lam = request.get("lambda_value", 0)
    if type(lam) not in (int, float) or lam != 0:
        raise ValueError("Ionisation fits require lambda_value=0")
    if request.get("endpoint") is not None:
        raise ValueError("Ionisation fits require all seven endpoints")
    weights = request.get("task_weights", [1] * len(ENDPOINTS))
    if (not isinstance(weights, (list, tuple)) or len(weights) != len(ENDPOINTS)
            or any(type(weight) not in (int, float) or weight != 1 for weight in weights)):
        raise ValueError("Ionisation fits require seven task weights of one")

    config = request["config"]
    required = {"hidden_dim", "ffn_layers", "dropout", "lr", "weight_decay"}
    allowed = required | {"depth"}
    if (not isinstance(config, dict) or not required <= config.keys()
            or not config.keys() <= allowed):
        raise ValueError("Invalid fit configuration fields")
    for key in ("hidden_dim", "ffn_layers"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if backbone == "ChemBERTa_head":
        if config.get("depth") is not None:
            raise ValueError("Frozen ChemBERTa does not have a trainable encoder depth")
    elif type(config.get("depth")) is not int or config["depth"] < 1:
        raise ValueError("Encoder depth must be a positive integer")
    if backbone != "Chemprop" and config["ffn_layers"] != 2:
        raise ValueError("GraphConv and ChemBERTa adapters require two FFN layers")
    for key in ("dropout", "lr", "weight_decay"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid model configuration: {key}")
    if config["dropout"] >= 1 or config["lr"] == 0:
        raise ValueError("Dropout must be below one and learning rate positive")
    if type(request["seed"]) is not int or request["seed"] < 0:
        raise ValueError("Fit seed must be a nonnegative integer")

    max_epochs, fixed = request.get("max_epochs", base.MAX_EPOCHS), request.get("fixed_epochs")
    if type(max_epochs) is not int or not 1 <= max_epochs <= base.MAX_EPOCHS:
        raise ValueError("Invalid maximum epoch count")
    if fixed is not None and (type(fixed) is not int or not 1 <= fixed <= max_epochs):
        raise ValueError("Invalid fixed epoch count")
    ids = {}
    for key in ("train_ids", "val_ids", "predict_ids"):
        values = request[key]
        if (not isinstance(values, list) or any(not isinstance(sid, str) or not sid.strip() for sid in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"Expected unique structure IDs in {key}")
        ids[key] = values.copy()
    train, val, predict = (ids[k] for k in ("train_ids", "val_ids", "predict_ids"))
    if not train or not predict or (fixed is None and not val):
        raise ValueError("Training, prediction and early-stopping validation partitions must exist")
    if set(train) & (set(val) | set(predict)):
        raise ValueError("Training must be disjoint from validation and prediction")
    if production and (set(val) != set(predict) if fixed is None else bool(val)):
        raise ValueError("Production requires inner validation predictions or fixed-epoch outer prediction")
    ids["train_ids"].sort()
    return ids, max_epochs, fixed, production


def _load_cohort(request, release_dir, feature_dir, ids, production):
    manifest_path = release_dir / "manifest.json"
    if production:
        if not manifest_path.is_file():
            raise ValueError("Production fit requires a release manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("cohort") != request["cohort"] or
                any(manifest.get(k) is not True for k in ("data_frozen", "splits_frozen", "training_ready"))):
            raise ValueError("Production fit requires a frozen, training-ready release for its exact cohort")

    data = pd.read_csv(release_dir / "model_dataset.csv", dtype={"structure_id": str})
    required = {"structure_id", "canonical_smiles", "ionisation_class", "pka_acid", "pka_base", *RAW_ENDPOINTS}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing cohort columns: {sorted(missing)}")
    data = data.set_index("structure_id")
    if data.empty or data.index.hasnans or not data.index.is_unique:
        raise ValueError("Cohort must contain unique, nonmissing structure IDs")
    if not data["canonical_smiles"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Every structure requires canonical_smiles")
    if not data["ionisation_class"].isin(CLASS_NAMES).all():
        raise ValueError("Unsupported ionisation_class in the cohort")
    if request["cohort"] == "numerical" and not np.isfinite(data[["pka_acid", "pka_base"]].to_numpy(float)).all():
        raise ValueError("Numerical cohort requires two finite pKa values per structure")
    for key, values in ids.items():
        if not set(values) <= set(data.index):
            raise ValueError(f"Unknown cohort structure IDs in {key}")
    if production:
        feature_ids = json.loads((feature_dir / "structures.json").read_text(encoding="utf-8"))
        if feature_ids != data.index.tolist():
            raise ValueError("Production feature structure order must exactly match the frozen dataset")
    return data


def _prepare_targets(data, ids, production):
    train = base.transform_targets(data.loc[ids["train_ids"], list(RAW_ENDPOINTS)].to_numpy())
    val = base.transform_targets(data.loc[ids["val_ids"], list(RAW_ENDPOINTS)].to_numpy())
    mean, sd, counts = base.fit_scaler(train)
    minimum = MIN_VALIDATION_COUNT if production else 1
    if len(val) and np.any(np.isfinite(val).sum(axis=0) < minimum):
        raise ValueError(f"Each endpoint requires at least {minimum} validation observations")
    scaler = {"endpoints": list(ENDPOINTS), "mean": mean.tolist(), "sd": sd.tolist(),
              "counts": counts.tolist(), "method": "fit_training_only_population_sd"}
    return (train - mean) / sd, (val - mean) / sd, scaler


def _hashes(request, release_dir, feature_dir, device):
    hashes, environment = base._hashes(request, release_dir, feature_dir, request["backbone"])
    source = Path(__file__).resolve()
    vendor_manifest = VENDOR / "SOURCE_MANIFEST.json"
    vendored = json.loads(vendor_manifest.read_text(encoding="utf-8"))["files"]
    if any(base._file_hash(VENDOR / name) != digest for name, digest in vendored.items()):
        raise ValueError("Vendored mass-balance source checksum mismatch")
    hashes["code_hash"] = base._object_hash({
        source.name: base._file_hash(source), "_vendor.py": base._file_hash(source.with_name("_vendor.py")),
        "vendor_manifest": base._file_hash(vendor_manifest), "vendor_trainer": hashes["code_hash"],
    })
    if request.get("diagnostic") is True:
        environment["diagnostic_device"] = str(device)
        if device.type == "cuda":
            environment["diagnostic_gpu_name"] = torch.cuda.get_device_name(device)
        hashes["environment_hash"] = base._object_hash(environment)
    hashes.pop("resolved_hash")
    return hashes, environment


def _train_fit(request, ids, max_epochs, fixed, data, descriptors, train, val, scaler,
               feature_dir, output_dir, device, hashes, environment, start):
    backbone, arm, seed = (request[k] for k in ("backbone", "arm", "seed"))
    config = request["config"]
    train_ids, val_ids, predict_ids = (ids[k] for k in ("train_ids", "val_ids", "predict_ids"))
    z_train = torch.as_tensor(train, dtype=torch.float32, device=device)
    z_val = torch.as_tensor(val, dtype=torch.float32, device=device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = FitInputs(backbone, data, feature_dir, descriptors)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, initialization = _make_model(backbone, config, arm, seed, inputs, train_ids[0], device)
    init_hash = base._state_hash(model.state_dict())
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    validation = [(inputs.batch(batch, device), z_val[k:k + len(batch)])
                  for k, batch in zip(range(0, len(val_ids), base.BATCH_SIZE), base._batches(val_ids))]
    start_training = time.perf_counter()
    best_score, best_epoch, best_state = math.inf, 0, None
    history = []
    for epoch in range(1, (fixed or max_epochs) + 1):
        begin_epoch = time.perf_counter()
        batch_seed = base._seed_for(seed, "batch_order", epoch)
        dropout_seed = base._seed_for(seed, "dropout", epoch)
        generator = torch.Generator().manual_seed(batch_seed)
        order = torch.randperm(len(train_ids), generator=generator).tolist()
        torch.random.default_generator.manual_seed(dropout_seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(dropout_seed)
        model.train()
        total = torch.zeros((), device=device)
        for index in base._batches(order):
            features = inputs.batch([train_ids[j] for j in index], device)
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_loss(model(features), z_train[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach() * len(index)
        supervised = float(total / len(train_ids))
        if not math.isfinite(supervised):
            raise ValueError(f"Nonfinite training loss at epoch {epoch}")
        score = None
        if fixed is None:
            score, _, _ = base._score_validation(model, validation, device, len(ENDPOINTS))
            if score < best_score - base.MIN_DELTA:
                best_score, best_epoch, best_state = score, epoch, base._model_state(model)
        history.append(dict(
            epoch=epoch, train_supervised=supervised, train_consistency=0.0, train_objective=supervised,
            val_equal_mse=score, batch_order_seed=batch_seed, dropout_seed=dropout_seed,
            batch_order_sha256=base._object_hash(order), epoch_seconds=time.perf_counter() - begin_epoch,
        ))
        if fixed is None and epoch - best_epoch >= base.PATIENCE:
            break
    if fixed is not None:
        best_epoch, best_state = fixed, base._model_state(model)
    end_training = time.perf_counter()

    model.load_state_dict(best_state, strict=True)
    probe = predict_ids[:base.BATCH_SIZE]
    expected = base._evaluate(model, inputs, probe, device)
    checkpoint_path = output_dir / "checkpoint.pt"
    base._atomic_write(checkpoint_path, lambda path: torch.save({
        "state_dict": best_state, "fit_id": request["fit_id"], "best_epoch": best_epoch,
        "resolved_hash": hashes["resolved_hash"], "backbone": backbone, "arm": arm,
        "cohort": request["cohort"], "config": config, "seed": seed, **scaler,
    }, path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["resolved_hash"] != hashes["resolved_hash"] or
            any(checkpoint[key] != value for key, value in scaler.items())):
        raise ValueError("Checkpoint fit/scaler hash mismatch")
    del optimizer, model
    model, _ = _make_model(checkpoint["backbone"], checkpoint["config"], checkpoint["arm"],
                           checkpoint["seed"], inputs, train_ids[0], device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    torch.testing.assert_close(base._evaluate(model, inputs, probe, device), expected, rtol=1e-5, atol=1e-6)
    end_checkpoint = time.perf_counter()

    score, sums, val_counts = (base._score_validation(model, validation, device, len(ENDPOINTS))
                               if validation else (None, [], []))
    pred = base._evaluate(model, inputs, predict_ids, device).numpy().astype(np.float64)
    transformed = pred * np.asarray(scaler["sd"]) + np.asarray(scaler["mean"])
    physical = base.inverse_physical(transformed)
    if not np.isfinite(transformed).all() or not np.isfinite(physical).all():
        raise ValueError("Nonfinite predictions")
    frame = pd.DataFrame({"structure_id": predict_ids})
    for j, name in enumerate(ENDPOINTS):
        frame["pred_" + name] = transformed[:, j]
    for j, name in enumerate(ENDPOINTS):
        frame["physical_" + name] = physical[:, j]
    base._atomic_write(output_dir / "predictions.csv", lambda p: frame.to_csv(p, index=False))
    base._atomic_write(output_dir / "learning_curve.csv", lambda p: pd.DataFrame(history).to_csv(p, index=False))
    base._write_json(output_dir / "scaler.json", scaler)
    base._write_json(output_dir / "request.json", {"request": request, "effective_ids": ids, **hashes})
    artifact_hashes = {name: base._file_hash(output_dir / name) for name in OUTPUT_FILES if name != "summary.json"}
    end = time.perf_counter()
    timing = dict(
        prep_seconds=start_training - start, train_seconds=end_training - start_training,
        checkpoint_seconds=end_checkpoint - end_training, prediction_and_artifact_seconds=end - end_checkpoint,
        total_seconds=end - start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    )
    summary = dict(
        fit_id=request["fit_id"], backbone=backbone, arm=arm, cohort=request["cohort"],
        diagnostic=request.get("diagnostic", False), campaign_id=request.get("campaign_id"),
        config=config, seed=seed, lambda_value=0.0, endpoint=None, task_weights=[1] * len(ENDPOINTS),
        best_epoch=best_epoch, actual_epochs=len(history), val_equal_mse=score,
        val_squared_error_sums=dict(zip(ENDPOINTS, sums)), val_observed_counts=dict(zip(ENDPOINTS, val_counts)),
        initial_state_hash=init_hash, **initialization, descriptor_spec=scaler["descriptor"],
        descriptor_input_dim=WIDTHS[arm], parameter_count=params, trainable_parameter_count=trainable,
        timing=timing, environment=environment,
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        n_train=len(train_ids), n_validation=len(val_ids), n_predict=len(predict_ids),
        aggregation="mean" if backbone != "ChemBERTa_head" else None,
        checkpoint_reload_verified=True, **hashes,
    )
    base._write_json(output_dir / "summary.json", summary)
    artifact_hashes["summary.json"] = base._file_hash(output_dir / "summary.json")
    base._write_json(output_dir / "complete.json", {
        "fit_id": request["fit_id"], "resolved_hash": hashes["resolved_hash"], "files": artifact_hashes,
    })
    return summary


def run_fit(request: dict, release_dir, feature_dir, output_dir, device="cuda") -> dict:
    """Execute or verify an isolated fit with train-only target and descriptor scaling."""
    start = time.perf_counter()
    ids, max_epochs, fixed, production = _check_request(request)
    release_dir, feature_dir, output_dir = map(Path, (release_dir, feature_dir, output_dir))
    if any(output_dir.resolve().is_relative_to(root.resolve()) for root in (release_dir, feature_dir)):
        raise ValueError("Fit artifacts must be outside the frozen data and feature bundles")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Fit device must be CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    hashes, environment = _hashes(request, release_dir, feature_dir, device)
    data = _load_cohort(request, release_dir, feature_dir, ids, production)
    descriptors, recipe = fit_descriptors(request["arm"], data, ids["train_ids"])
    train, val, scaler = _prepare_targets(data, ids, production)
    scaler["descriptor"] = recipe
    hashes["scaler_hash"] = base._object_hash(scaler)
    hashes["resolved_hash"] = base._object_hash(hashes)
    output_dir.mkdir(parents=True, exist_ok=True)
    with base._fit_lock(output_dir):
        existing = base._resume_completed(output_dir, request["fit_id"], hashes)
        if existing is not None:
            return existing
        return _train_fit(request, ids, max_epochs, fixed, data, descriptors, train, val, scaler,
                          feature_dir, output_dir, device, hashes, environment, start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    summary = run_fit(request, args.release_dir, args.feature_dir, args.output_dir, args.device)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
