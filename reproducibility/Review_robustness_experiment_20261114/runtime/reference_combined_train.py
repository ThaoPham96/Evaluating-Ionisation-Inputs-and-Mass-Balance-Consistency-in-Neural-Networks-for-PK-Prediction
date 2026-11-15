"""Train a paired feature/loss cell on the frozen numerical cohort."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from combined_features import COHORT, FOLDS, load_splits, verify_manifests
from unified_features import EXTRA_ARMS, descriptors as ph_descriptors
from vendor_ionisation import ionisation_train as frozen


base = frozen.base
BACKBONES = frozen.BACKBONES
ENDPOINTS = frozen.ENDPOINTS
RAW_ENDPOINTS = frozen.RAW_ENDPOINTS
OUTPUT_FILES = frozen.OUTPUT_FILES
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909)
CELLS = {('P0', False): '00', ('P1', False): '10', ('P0', True): '01', ('P1', True): '11'}
CELLS.update({(arm, False): arm for arm in EXTRA_ARMS})


def _check_request(request):
    if not isinstance(request, dict) or 'arm' in request:
        raise ValueError('Expected a fit request with feature_arm and cell')
    arm, lam = request.get('feature_arm'), request.get('lambda_value')
    if (not isinstance(arm, str) or type(lam) not in (float, int) or not math.isfinite(lam)
            or lam < 0 or (arm, lam > 0) not in CELLS):
        raise ValueError('Expected a frozen feature arm and a compatible nonnegative lambda_value')
    if request.get('cell') != CELLS[arm, lam > 0] or request.get('cohort') != COHORT:
        raise ValueError('Cell, feature arm, lambda and numerical cohort do not agree')
    # The archived request validator checks all shared fields and accepts only its own zero-loss arm.
    compatible = dict(request, arm='P1' if arm in EXTRA_ARMS else arm, lambda_value=0)
    ids, max_epochs, fixed, production = frozen._check_request(compatible)
    if production:
        outer, inner = request.get('outer_fold'), request.get('inner_fold')
        if type(outer) is not int or outer not in FOLDS:
            raise ValueError('Production fit requires a frozen outer fold')
        if max_epochs != base.MAX_EPOCHS or request.get('task_weights') != [1.] * len(ENDPOINTS):
            raise ValueError('Production fits require explicit equal endpoint weights and the frozen maximum epochs')
        if fixed is None:
            if type(inner) is not int or inner not in FOLDS or request['seed'] != 42:
                raise ValueError('Inner calibration requires a frozen inner fold and seed 42')
        elif inner is not None or request['seed'] not in SEEDS:
            raise ValueError('Fixed-epoch final fits require an outer fold and a frozen final seed')
    return ids, max_epochs, fixed, production


def _check_splits(request, ids, release_dir):
    outer, inner, _ = load_splits(release_dir)
    fold, subfold = request['outer_fold'], request.get('inner_fold')
    assignments = inner[fold]
    expected = dict(train_ids=sorted(sid for sid in assignments if assignments[sid] != subfold),
                    val_ids=sorted(sid for sid in assignments if assignments[sid] == subfold),
                    predict_ids=sorted(sid for sid in outer if outer[sid] == fold) if subfold is None
                    else sorted(sid for sid in assignments if assignments[sid] == subfold))
    if ids != expected:
        raise ValueError('Production fit IDs differ from the frozen inner or outer partition')


def _hashes(request, release_dir, feature_dir, device):
    hashes, environment = frozen._hashes(request, release_dir, feature_dir, device)
    analysis = Path(__file__).resolve().parent
    package = analysis/'vendor_ionisation'
    manifest = package/'SOURCE_MANIFEST.json'
    vendored = json.loads(manifest.read_text(encoding='utf-8'))['files']
    if any(base._file_hash(package/name) != checksum for name, checksum in vendored.items()):
        raise ValueError('Vendored ionisation source checksum mismatch')
    hashes['code_hash'] = base._object_hash({
        'frozen_trainer': hashes['code_hash'], 'vendor_manifest': base._file_hash(manifest),
        **{name:base._file_hash(analysis/name) for name in
           ('combined_train.py', 'combined_features.py', 'unified_features.py', 'vendor_ionisation/__init__.py')},
    })
    environment.update(platform=platform.system(), device_type=device.type, cuda_version=torch.version.cuda)
    if device.type == 'cuda':
        environment.update(gpu_name=torch.cuda.get_device_name(device),
                           gpu_capability=list(torch.cuda.get_device_capability(device)))
    hashes['environment_hash'] = base._object_hash(environment)
    return hashes, environment


def _train_fit(request, ids, max_epochs, fixed, data, descriptors, train, val, scaler,
               feature_dir, output_dir, device, hashes, environment, start):
    backbone, arm, seed = (request[k] for k in ("backbone", "feature_arm", "seed"))
    model_arm = "P1" if arm in EXTRA_ARMS else arm
    lam = request["lambda_value"]
    config = request["config"]
    train_ids, val_ids, predict_ids = (ids[k] for k in ("train_ids", "val_ids", "predict_ids"))
    z_train = torch.as_tensor(train, dtype=torch.float32, device=device)
    z_val = torch.as_tensor(val, dtype=torch.float32, device=device)
    mean = torch.as_tensor(scaler["mean"], dtype=torch.float32, device=device)
    sd = torch.as_tensor(scaler["sd"], dtype=torch.float32, device=device)
    weights = torch.ones(len(ENDPOINTS), dtype=torch.float32, device=device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = frozen.FitInputs(backbone, data, feature_dir, descriptors)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, initialization = frozen._make_model(backbone, config, model_arm, seed, inputs, train_ids[0], device)
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
        totals = torch.zeros(3, device=device)
        for index in base._batches(order):
            features = inputs.batch([train_ids[j] for j in index], device)
            optimizer.zero_grad(set_to_none=True)
            supervised, consistency = base.loss_terms(model(features), z_train[index], mean, sd, weights)
            loss = supervised + lam * consistency if lam else supervised
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            totals += torch.stack((supervised.detach(), consistency.detach(), loss.detach())) * len(index)
        supervised, consistency, objective = (totals / len(train_ids)).tolist()
        if not all(map(math.isfinite, (supervised, consistency, objective))):
            raise ValueError(f"Nonfinite training loss at epoch {epoch}")
        score = None
        if fixed is None:
            score, _, _ = base._score_validation(model, validation, device, len(ENDPOINTS))
            if score < best_score - base.MIN_DELTA:
                best_score, best_epoch, best_state = score, epoch, base._model_state(model)
        history.append(dict(
            epoch=epoch, train_supervised=supervised, train_consistency=consistency, train_objective=objective,
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
        "cohort": request["cohort"], "config": config, "seed": seed,
        "cell": request["cell"], "lambda_value": lam, **scaler,
    }, path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["resolved_hash"] != hashes["resolved_hash"] or checkpoint["cell"] != request["cell"] or
            checkpoint["lambda_value"] != lam or
            any(checkpoint[key] != value for key, value in scaler.items())):
        raise ValueError("Checkpoint fit/scaler hash mismatch")
    del optimizer, model
    model, _ = frozen._make_model(checkpoint["backbone"], checkpoint["config"], model_arm,
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
        fit_id=request["fit_id"], backbone=backbone, arm=arm, feature_arm=arm, cell=request["cell"],
        outer_fold=request.get("outer_fold"), inner_fold=request.get("inner_fold"), cohort=request["cohort"],
        diagnostic=request.get("diagnostic", False), campaign_id=request.get("campaign_id"),
        config=config, seed=seed, lambda_value=lam, endpoint=None, task_weights=[1] * len(ENDPOINTS),
        best_epoch=best_epoch, actual_epochs=len(history), val_equal_mse=score,
        val_squared_error_sums=dict(zip(ENDPOINTS, sums)), val_observed_counts=dict(zip(ENDPOINTS, val_counts)),
        initial_state_hash=init_hash, **initialization, descriptor_spec=scaler["descriptor"],
        descriptor_input_dim=frozen.WIDTHS[model_arm], parameter_count=params, trainable_parameter_count=trainable,
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


def run_fit(request, release_dir, feature_dir, output_dir, device='cuda'):
    """Run a single cell or verify all artifacts for a previously completed request."""
    start = time.perf_counter()
    ids, max_epochs, fixed, production = _check_request(request)
    release_dir, feature_dir, output_dir = map(Path, (release_dir, feature_dir, output_dir))
    if any(output_dir.resolve().is_relative_to(path.resolve()) for path in (release_dir, feature_dir)):
        raise ValueError('Fit artifacts must be outside the frozen input bundles')
    device = torch.device(device)
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('Fit device must be CPU or CUDA')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    if production:
        verify_manifests(release_dir, feature_dir)
        _check_splits(request, ids, release_dir)
    hashes, environment = _hashes(request, release_dir, feature_dir, device)
    data = frozen._load_cohort(request, release_dir, feature_dir, ids, production)
    arm = request['feature_arm']
    if arm in EXTRA_ARMS:
        with (release_dir / 'model_dataset.csv').open(encoding='utf-8', newline='') as stream:
            records = list(csv.DictReader(stream))
        if [record['structure_id'] for record in records] != data.index.tolist():
            raise ValueError('Single-pH feature rows differ from the model dataset order')
        values, recipe = ph_descriptors(arm, records)
    else:
        values, recipe = frozen.fit_descriptors(arm, data, ids['train_ids'])
    train, val, scaler = frozen._prepare_targets(data, ids, production)
    scaler['descriptor'] = recipe
    hashes['scaler_hash'] = base._object_hash(scaler)
    hashes['resolved_hash'] = base._object_hash(hashes)
    output_dir.mkdir(parents=True, exist_ok=True)
    with base._fit_lock(output_dir):
        existing = base._resume_completed(output_dir, request['fit_id'], hashes)
        if existing is not None:
            return existing
        return _train_fit(request, ids, max_epochs, fixed, data, values, train, val, scaler,
                          feature_dir, output_dir, device, hashes, environment, start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('request', 'release-dir', 'feature-dir', 'output-dir'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding='utf-8'))
    summary = run_fit(request, args.release_dir, args.feature_dir, args.output_dir, args.device)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
