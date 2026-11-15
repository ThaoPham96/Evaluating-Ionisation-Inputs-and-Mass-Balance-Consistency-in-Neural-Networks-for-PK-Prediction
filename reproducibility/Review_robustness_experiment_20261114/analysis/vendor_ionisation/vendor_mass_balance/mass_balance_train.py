"""Shared supervised trainer for the frozen v6 mass-balance experiment."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from torch import nn
from torch_geometric.data import Batch, Data

from mass_balance_controls import LOG10_LN2
from pipknet.featurizers import atom_to_feature_vector, smiles_to_pyg
from pipknet.models import (
    ATTENTIVEFP_TIMESTEPS,
    BACKBONES,
    BOND_FEATURE_DIM,
    GAT_HEADS,
    GNNMultitask,
)

RAW_ENDPOINTS = (
    "t1/2(hour)", "VD(liter)", "Cl(liter/hour)", "F(percentage)",
    "PPB(percentage)", "Cmax_uM", "Tmax",
)
ENDPOINTS = ("t_half", "Vd", "CL", "F", "PPB", "Cmax", "Tmax")
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 30
MIN_DELTA = 1e-4
PRODUCTION_PYG = "2.7.0"
PRODUCTION_RDKIT = "2025.09.6"
OUTPUT_FILES = (
    "summary.json", "predictions.csv", "checkpoint.pt", "scaler.json",
    "request.json", "learning_curve.csv",
)


def transform_targets(raw):
    """Apply v6 transformations and retain missing cells."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(ENDPOINTS) or np.isinf(raw).any():
        raise ValueError("Expected seven PK columns with finite observed values")
    positive = raw[:, [0, 1, 2, 5, 6]]
    percent = raw[:, [3, 4]]
    if np.any(positive <= 0) or np.any((percent < 0) | (percent > 100)):
        raise ValueError("Positive endpoints must be >0 and percentages in [0, 100]")
    transformed = raw.copy()
    transformed[:, [0, 1, 2, 5, 6]] = np.log10(positive)
    clipped = np.clip(percent, 0.1, 99.9)
    transformed[:, [3, 4]] = np.log10(clipped / (100 - clipped))
    return transformed


def fit_scaler(train):
    """Estimate means and population SDs using only fit-training observations."""
    counts = np.isfinite(train).sum(axis=0)
    if np.any(counts < 2):
        raise ValueError("Each modelled endpoint needs at least two training observations")
    mean, sd = np.nanmean(train, axis=0), np.nanstd(train, axis=0, ddof=0)
    if np.any(~np.isfinite(sd) | (sd <= 0)):
        raise ValueError("Each modelled endpoint needs positive training variance")
    return mean, sd, counts


def squared_errors(prediction, target):
    """Squared-error sums and observed counts by modelled endpoint."""
    mask = torch.isfinite(target)
    difference = torch.where(mask, prediction - torch.nan_to_num(target), 0.0)
    return difference.square().sum(dim=0), mask.sum(dim=0)


def mean_endpoint_error(sums, counts, weights):
    present_weights = weights * (counts > 0)
    total_weight = present_weights.sum()
    denominator = torch.where(total_weight > 0, total_weight, 1.0)
    return ((sums / counts.clamp_min(1)) * present_weights).sum() / denominator


def loss_terms(prediction, target, mean, sd, weights):
    sums, counts = squared_errors(prediction, target)
    supervised = mean_endpoint_error(sums, counts, weights)
    if prediction.shape[1] == 1:
        return supervised, prediction.new_zeros(())
    q = prediction[:, :3] * sd[:3] + mean[:3]
    residual = q[:, 0] - q[:, 1] + q[:, 2] - LOG10_LN2
    return supervised, residual.square().mean()


def inverse_physical(transformed):
    raw = transformed.astype(np.float64, copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        raw[:, [0, 1, 2, 5, 6]] = np.power(10.0, raw[:, [0, 1, 2, 5, 6]])
        logits = raw[:, [3, 4]] * math.log(10)
        raw[:, [3, 4]] = 100 * np.exp(-np.logaddexp(0, -logits))
    return raw


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _atomic_write(path, writer):
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, value):
    _atomic_write(path, lambda p: p.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    ))


@contextmanager
def _fit_lock(output_dir):
    """A process-scoped advisory lock, released by the OS on worker exit."""
    with (output_dir / ".fit.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            yield
        finally:
            handle.seek(0)
            unlock()


def _resume_completed(output_dir, fit_id, hashes):
    marker_path = output_dir / "complete.json"
    if not marker_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker["fit_id"] != fit_id or marker["resolved_hash"] != hashes["resolved_hash"]:
        raise ValueError(f"Completed fit hash mismatch in {output_dir}")
    if set(marker["files"]) != set(OUTPUT_FILES):
        raise ValueError(f"Incomplete artifact manifest in {output_dir}")
    for name, digest in marker["files"].items():
        if _file_hash(output_dir / name) != digest:
            raise ValueError(f"Completed artifact hash mismatch: {output_dir / name}")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if any(summary[k] != value for k, value in hashes.items()):
        raise ValueError(f"Completed fit summary hash mismatch in {output_dir}")
    return summary


def _make_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"Invalid reviewed SMILES: {smiles}")
    if mol.GetNumBonds():
        graph = smiles_to_pyg(smiles, "neutral")
        del graph.ion_feat
        return graph
    return Data(
        x=torch.stack([atom_to_feature_vector(a) for a in mol.GetAtoms()]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32),
    )


class FitInputs:
    """Order-preserving adapters over reviewed graph or frozen feature inputs."""

    def __init__(self, backbone, data, feature_dir):
        self.backbone = backbone
        if backbone == "Chemprop":
            from chemprop.features import MolGraph, reset_featurization_parameters
            reset_featurization_parameters()
            self.values = {sid: MolGraph(smi) for sid, smi in data.canonical_smiles.items()}
            return
        if backbone == "ChemBERTa_head" or (feature_dir / "graphs.pt").is_file():
            ids = json.loads((feature_dir / "structures.json").read_text(encoding="utf-8"))
            if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != set(data.index):
                raise ValueError("Feature IDs must match the frozen modelling cohort")
            if backbone == "ChemBERTa_head":
                matrix = np.load(feature_dir / "chemberta.npy", allow_pickle=False)
                if matrix.dtype != np.float32 or matrix.shape != (len(ids), 768) or not np.isfinite(matrix).all():
                    raise ValueError("Expected finite float32 ChemBERTa features of shape (N, 768)")
                self.values = dict(zip(ids, torch.from_numpy(matrix)))
            else:
                # A PyG object cache is a trusted local bundle artifact.
                graphs = torch.load(feature_dir / "graphs.pt", map_location="cpu", weights_only=False)
                if not isinstance(graphs, list) or len(graphs) != len(ids) or not all(isinstance(g, Data) for g in graphs):
                    raise ValueError("Expected one PyG Data graph per feature ID")
                self.values = dict(zip(ids, graphs))
        else:
            self.values = {sid: _make_graph(smi) for sid, smi in data.canonical_smiles.items()}

    def batch(self, ids, device):
        inputs = [self.values[sid] for sid in ids]
        if self.backbone == "Chemprop":
            from chemprop.features import BatchMolGraph
            return [BatchMolGraph(inputs)]
        if self.backbone == "ChemBERTa_head":
            return torch.stack(inputs).to(device)
        return Batch.from_data_list(inputs).to(device)


def _make_model(backbone, config, endpoints, device, inputs, first_id):
    width, dropout = config["hidden_dim"], config["dropout"]
    if backbone == "Chemprop":
        from chemprop.args import TrainArgs
        from chemprop.models import MoleculeModel
        args = TrainArgs()
        args.device = device
        args.dataset_type = "regression"
        args.loss_function = "mse"
        args.hidden_size = args.ffn_hidden_size = width
        args.depth = config["depth"]
        args.ffn_num_layers = config["ffn_layers"]
        args.dropout = dropout
        args.task_names = list(endpoints)
        return MoleculeModel(args).float().to(device)
    if backbone == "ChemBERTa_head":
        return nn.Sequential(
            nn.Linear(768, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, len(endpoints)),
        ).float().to(device)
    model = GNNMultitask(
        backbone=backbone, hidden_dim=width, dropout=dropout, n_layers=config["depth"],
        out_dim=len(endpoints), use_ion=False,
    ).float()
    # Materialize the lazy first GraphConv layer on CPU before any train/dropout RNG.
    if backbone == "GraphConv":
        model.eval()
        with torch.no_grad():
            model(inputs.batch([first_id], "cpu"))
    return model.to(device)


def _seed_for(seed, purpose, epoch=0):
    return int(_object_hash([seed, purpose, epoch])[:15], 16)


def _model_state(model):
    return {name: tensor.detach().to(device="cpu", copy=True) for name, tensor in model.state_dict().items()}


def _state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _batches(ids):
    return (ids[i:i + BATCH_SIZE] for i in range(0, len(ids), BATCH_SIZE))


def _score_validation(model, batches, device, width):
    model.eval()
    sums = torch.zeros(width, dtype=torch.float64, device=device)
    counts = torch.zeros(width, dtype=torch.int64, device=device)
    with torch.inference_mode():
        for features, labels in batches:
            squared, seen = squared_errors(model(features).to(torch.float64), labels)
            sums += squared
            counts += seen
    if (counts == 0).any():
        raise ValueError("Each modelled endpoint needs validation observations")
    error = float((sums / counts).mean())
    if not math.isfinite(error):
        raise ValueError("Nonfinite validation score")
    return error, sums.tolist(), counts.tolist()


def _evaluate(model, inputs, ids, device):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(inputs.batch(batch, device)).cpu() for batch in _batches(ids)])


def _check_request(request, data):
    backbone = request["backbone"]
    if backbone == "GraphConv_single_task":
        backbone = "GraphConv"
        if request.get("endpoint") is None:
            raise ValueError("GraphConv_single_task requires an endpoint")
    if backbone not in (*BACKBONES, "Chemprop", "ChemBERTa_head"):
        raise ValueError(f"Unknown backbone: {backbone}")
    endpoint = request.get("endpoint")
    if endpoint is not None and (endpoint not in ENDPOINTS or backbone != "GraphConv"):
        raise ValueError("Single-task fits require GraphConv and one known endpoint")
    lam = float(request["lambda_value"])
    if not math.isfinite(lam) or lam < 0 or (endpoint is not None and lam != 0):
        raise ValueError("Invalid lambda for this fit")
    config = dict(request["config"])
    config["hidden_dim"] = int(config["hidden_dim"])
    config["ffn_layers"] = int(config["ffn_layers"])
    if backbone != "ChemBERTa_head":
        config["depth"] = int(config["depth"])
    if config["hidden_dim"] < 1 or (backbone != "ChemBERTa_head" and config["depth"] < 1):
        raise ValueError("Model dimensions must be positive")
    if config["ffn_layers"] < 1 or (backbone != "Chemprop" and config["ffn_layers"] != 2):
        raise ValueError("This adapter requires two FFN layers")
    for name in ("dropout", "lr", "weight_decay"):
        config[name] = float(config[name])
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"Invalid model configuration: {name}")
    if config["dropout"] >= 1 or config["lr"] == 0:
        raise ValueError("Dropout must be below one and learning rate positive")
    seed = request["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Fit seed must be a nonnegative integer")
    max_epochs = request.get("max_epochs", MAX_EPOCHS)
    fixed = request.get("fixed_epochs")
    if not isinstance(max_epochs, int) or not 1 <= max_epochs <= MAX_EPOCHS:
        raise ValueError("Invalid maximum epoch count")
    if fixed is not None and (not isinstance(fixed, int) or not 1 <= fixed <= max_epochs):
        raise ValueError("Invalid fixed epoch count")
    ids = {}
    for key in ("train_ids", "val_ids", "predict_ids"):
        ids[key] = list(request[key])
        if len(set(ids[key])) != len(ids[key]) or not set(ids[key]) <= set(data.index):
            raise ValueError(f"Duplicate or unknown structure IDs in {key}")
    if not ids["train_ids"] or not ids["predict_ids"] or (fixed is None and not ids["val_ids"]):
        raise ValueError("Training, prediction and early-stopping validation partitions must exist")
    if set(ids["train_ids"]) & (set(ids["val_ids"]) | set(ids["predict_ids"])):
        raise ValueError("Training must be disjoint from validation and prediction")
    task_weights = request.get("task_weights")
    if task_weights is not None and endpoint is not None:
        raise ValueError("Task weights apply only to multitask fits")
    if task_weights is not None:
        weights = np.asarray(task_weights, dtype=np.float32)
        if weights.shape != (len(ENDPOINTS),) or not np.all(np.isfinite(weights) & (weights > 0)):
            raise ValueError("Task weights must be seven finite positive values")
    else:
        weights = np.ones(1 if endpoint is not None else len(ENDPOINTS), dtype=np.float32)
    ids["train_ids"].sort()
    return backbone, config, endpoint, lam, seed, max_epochs, fixed, ids, weights


def _hashes(request, release_dir, feature_dir, backbone):
    source = Path(__file__).resolve()
    code_files = [source, source.with_name("mass_balance_controls.py"),
                  source.parent / "pipknet/models.py", source.parent / "pipknet/featurizers.py"]
    production = "campaign_id" in request
    release_manifest = release_dir / "manifest.json"
    feature_manifest = feature_dir / "manifest.json"
    if production:
        if not isinstance(request["campaign_id"], str) or not request["campaign_id"]:
            raise ValueError("Production campaign_id must be a nonempty string")
        for path in (release_manifest, feature_manifest):
            if not path.is_file():
                raise ValueError(f"Production fit requires manifest: {path}")
    feature_files = ["structures.json"] if production else []
    if backbone == "ChemBERTa_head":
        feature_files = ["structures.json", "chemberta.npy"]
    elif backbone != "Chemprop" and (production or (feature_dir / "graphs.pt").is_file()):
        feature_files = ["structures.json", "graphs.pt"]
    dataset_hash = _file_hash(release_dir / "model_dataset.csv")
    release_manifest_hash = _file_hash(release_manifest) if release_manifest.is_file() else None
    frozen = None
    if release_manifest.is_file():
        frozen = json.loads(release_manifest.read_text(encoding="utf-8"))
        if frozen["files"].get("model_dataset.csv") != dataset_hash:
            raise ValueError("Dataset does not match the frozen release manifest")
        if production and (frozen.get("data_frozen") is not True or frozen.get("splits_frozen") is not True
                           or not frozen.get("fingerprint")):
            raise ValueError("Production fit requires a frozen data and split release")
    for filename in feature_files:
        if not (feature_dir / filename).is_file():
            raise ValueError(f"Required feature file is missing: {filename}")
    feature_digests = {f: _file_hash(feature_dir / f) for f in feature_files}
    if feature_manifest.is_file():
        manifest = json.loads(feature_manifest.read_text(encoding="utf-8"))
        if (manifest["data"]["model_dataset.csv"] != dataset_hash or
                manifest["data"]["manifest.json"] != release_manifest_hash):
            raise ValueError("Feature bundle does not match the frozen data release")
        if production and manifest["data"].get("fingerprint") != frozen["fingerprint"]:
            raise ValueError("Feature bundle does not match the frozen release fingerprint")
        for filename, digest in feature_digests.items():
            if manifest["files"].get(filename) != digest:
                raise ValueError(f"Feature checksum mismatch: {filename}")
        feature_digests["manifest.json"] = _file_hash(feature_manifest)
    environment = dict(
        python=platform.python_version(), torch=torch.__version__, torch_geometric=version("torch-geometric"),
        numpy=np.__version__, pandas=pd.__version__, rdkit=rdBase.rdkitVersion,
    )
    if production and environment["torch_geometric"] != PRODUCTION_PYG:
        raise ValueError(f"Production fits require torch-geometric {PRODUCTION_PYG}")
    if backbone == "Chemprop":
        environment["chemprop"] = version("chemprop")
        if environment["chemprop"] != "1.6.1":
            raise ValueError("Chemprop 1.6.1 is required")
        if production and environment["rdkit"] != PRODUCTION_RDKIT:
            raise ValueError(f"Production Chemprop requires RDKit {PRODUCTION_RDKIT}")
    hashes = dict(
        request_hash=_object_hash(request),
        code_hash=_object_hash({str(f.relative_to(source.parent)): _file_hash(f) for f in code_files}),
        data_hash=dataset_hash,
        release_manifest_hash=release_manifest_hash,
        feature_hash=_object_hash(feature_digests),
        environment_hash=_object_hash(environment),
    )
    hashes["resolved_hash"] = _object_hash(hashes)
    return hashes, environment


def _train_fit(request, data, feature_dir, output_dir, device, hashes, environment, start):
    backbone, config, endpoint, lam, seed, max_epochs, fixed, ids, weight = _check_request(request, data)
    columns = list(range(len(ENDPOINTS))) if endpoint is None else [ENDPOINTS.index(endpoint)]
    endpoints = [ENDPOINTS[i] for i in columns]
    train_ids, val_ids, predict_ids = (ids[k] for k in ("train_ids", "val_ids", "predict_ids"))
    train = transform_targets(data.loc[train_ids, list(RAW_ENDPOINTS)].to_numpy())[:, columns]
    val = transform_targets(data.loc[val_ids, list(RAW_ENDPOINTS)].to_numpy())[:, columns]
    if endpoint is not None and not np.isfinite(train).all():
        raise ValueError("Single-task training IDs must all have that endpoint")
    mean, sd, counts = fit_scaler(train)
    z_train = torch.as_tensor((train - mean) / sd, dtype=torch.float32, device=device)
    z_val = torch.as_tensor((val - mean) / sd, dtype=torch.float32, device=device)
    weights = torch.as_tensor(weight, device=device)
    mean_dev = torch.as_tensor(mean, dtype=torch.float32, device=device)
    sd_dev = torch.as_tensor(sd, dtype=torch.float32, device=device)

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(_seed_for(seed, "initialization"))
    random.seed(_seed_for(seed, "initialization"))
    inputs = FitInputs(backbone, data, feature_dir)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = _make_model(backbone, config, endpoints, device, inputs, train_ids[0])
    init_hash = _state_hash(_model_state(model))
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    validation = [(inputs.batch(batch, device), z_val[k:k + len(batch)])
                  for k, batch in zip(range(0, len(val_ids), BATCH_SIZE), _batches(val_ids))]
    start_training = time.perf_counter()
    best_score, best_epoch, best_state = math.inf, 0, None
    history = []
    for epoch in range(1, (fixed or max_epochs) + 1):
        begin_epoch = time.perf_counter()
        batch_seed = _seed_for(seed, "batch_order", epoch)
        dropout_seed = _seed_for(seed, "dropout", epoch)
        generator = torch.Generator().manual_seed(batch_seed)
        order = torch.randperm(len(train_ids), generator=generator).tolist()
        torch.manual_seed(dropout_seed)
        model.train()
        totals = torch.zeros(3, device=device)
        for index in _batches(order):
            features = inputs.batch([train_ids[j] for j in index], device)
            optimizer.zero_grad(set_to_none=True)
            supervised, consistency = loss_terms(model(features), z_train[index], mean_dev, sd_dev, weights)
            loss = supervised + lam * consistency if lam else supervised
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            totals += torch.stack((supervised.detach(), consistency.detach(), loss.detach())) * len(index)
        sup, cons, total = (totals / len(train_ids)).tolist()
        if not math.isfinite(total):
            raise ValueError(f"Nonfinite training loss at epoch {epoch}")
        score = None
        if fixed is None:
            score, _, _ = _score_validation(model, validation, device, len(endpoints))
            if score < best_score - MIN_DELTA:
                best_score, best_epoch, best_state = score, epoch, _model_state(model)
        history.append(dict(
            epoch=epoch, train_supervised=sup, train_consistency=cons, train_objective=total,
            val_equal_mse=score, batch_order_seed=batch_seed, dropout_seed=dropout_seed,
            batch_order_sha256=_object_hash(order),
            epoch_seconds=time.perf_counter() - begin_epoch,
        ))
        if fixed is None and epoch - best_epoch >= PATIENCE:
            break
    if fixed is not None:
        best_epoch, best_state = fixed, _model_state(model)
    end_training = time.perf_counter()

    model.load_state_dict(best_state, strict=True)
    probe = predict_ids[:BATCH_SIZE]
    expected = _evaluate(model, inputs, probe, device)
    checkpoint_path = output_dir / "checkpoint.pt"
    _atomic_write(checkpoint_path, lambda path: torch.save({
        "state_dict": best_state, "fit_id": request["fit_id"], "best_epoch": best_epoch,
        "resolved_hash": hashes["resolved_hash"], "backbone": backbone, "config": config,
        "endpoints": endpoints, "mean": mean.tolist(), "sd": sd.tolist(),
    }, path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["resolved_hash"] != hashes["resolved_hash"]:
        raise ValueError("Checkpoint fit hash mismatch")
    del optimizer, model
    model = _make_model(
        checkpoint["backbone"], checkpoint["config"], checkpoint["endpoints"],
        device, inputs, train_ids[0],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    # Reloaded predictions must agree, allowing ordinary floating-point reduction noise.
    torch.testing.assert_close(_evaluate(model, inputs, probe, device), expected, rtol=1e-5, atol=1e-6, equal_nan=True)
    end_checkpoint = time.perf_counter()
    score, sums, val_counts = _score_validation(model, validation, device, len(endpoints)) if validation else (None, [], [])
    predicted = _evaluate(model, inputs, predict_ids, device).numpy().astype(np.float64) * sd + mean
    full = np.full((len(predict_ids), len(ENDPOINTS)), np.nan)
    full[:, columns] = predicted
    physical = inverse_physical(full)
    frame = pd.DataFrame({"structure_id": predict_ids})
    for j, name in enumerate(ENDPOINTS):
        frame["pred_" + name] = full[:, j]
    for j, name in enumerate(ENDPOINTS):
        frame["physical_" + name] = physical[:, j]
    _atomic_write(output_dir / "predictions.csv", lambda p: frame.to_csv(p, index=False))
    _atomic_write(output_dir / "learning_curve.csv", lambda p: pd.DataFrame(history).to_csv(p, index=False))
    _write_json(output_dir / "scaler.json", {
        "endpoints": endpoints, "mean": mean.tolist(), "sd": sd.tolist(), "counts": counts.tolist(),
        "method": "fit_training_only_population_sd",
    })
    _write_json(output_dir / "request.json", {"request": request, "effective_ids": ids, **hashes})
    artifact_hashes = {
        name: _file_hash(output_dir / name) for name in OUTPUT_FILES if name != "summary.json"
    }
    end = time.perf_counter()
    timing = dict(
        prep_seconds=start_training - start, train_seconds=end_training - start_training,
        checkpoint_seconds=end_checkpoint - end_training, prediction_and_artifact_seconds=end - end_checkpoint,
        total_seconds=end - start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    )
    summary = dict(
        fit_id=request["fit_id"], backbone=backbone, config=config, seed=seed, lambda_value=lam,
        endpoint=endpoint, task_weights=weight.tolist(), best_epoch=best_epoch, actual_epochs=len(history),
        val_equal_mse=score, val_squared_error_sums=dict(zip(endpoints, sums)),
        val_observed_counts=dict(zip(endpoints, val_counts)), initial_state_hash=init_hash,
        parameter_count=params, trainable_parameter_count=trainable, timing=timing, environment=environment,
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        n_train=len(train_ids), n_validation=len(val_ids), n_predict=len(predict_ids),
        aggregation="native_attentive_readout" if backbone == "AttentiveFP" else
                    "mean" if backbone != "ChemBERTa_head" else None,
        gat_attention_heads=GAT_HEADS if backbone == "GAT" else None,
        attentivefp_readout_iterations=ATTENTIVEFP_TIMESTEPS if backbone == "AttentiveFP" else None,
        checkpoint_reload_verified=True, **hashes,
    )
    _write_json(output_dir / "summary.json", summary)
    artifact_hashes["summary.json"] = _file_hash(output_dir / "summary.json")
    _write_json(output_dir / "complete.json", {
        "fit_id": request["fit_id"], "resolved_hash": hashes["resolved_hash"],
        "files": artifact_hashes,
    })
    return summary


def run_fit(request: dict, release_dir, feature_dir, output_dir, device="cuda") -> dict:
    """Execute or verify a v6 fit in its own artifact directory."""
    start = time.perf_counter()
    release_dir, feature_dir, output_dir = map(Path, (release_dir, feature_dir, output_dir))
    if output_dir.resolve().is_relative_to(release_dir.resolve()) or output_dir.resolve().is_relative_to(feature_dir.resolve()):
        raise ValueError("Fit artifacts must be outside the frozen data and feature bundles")
    if not isinstance(request.get("fit_id"), str) or not request["fit_id"]:
        raise ValueError("fit_id must be a nonempty string")
    hashes, environment = _hashes(request, release_dir, feature_dir, request["backbone"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with _fit_lock(output_dir):
        existing = _resume_completed(output_dir, request["fit_id"], hashes)
        if existing is not None:
            return existing
        data = pd.read_csv(release_dir / "model_dataset.csv", dtype={"structure_id": str}).set_index("structure_id")
        if not data.index.is_unique:
            raise ValueError("Duplicate structure IDs in frozen modelling data")
        if "campaign_id" in request:
            feature_ids = json.loads((feature_dir / "structures.json").read_text(encoding="utf-8"))
            if feature_ids != data.index.tolist():
                raise ValueError("Production feature structure order must exactly match the frozen dataset")
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return _train_fit(request, data, feature_dir, output_dir, device, hashes, environment, start)
