"""Fit one four-channel single-pH arm using the original ionisation fit loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

import ionisation_train as original
from ph_ionisation_features import FINGERPRINT, load
from ph_ionisation_train import _with_namespace
from single_ph_features import ARM_PH, AUDITED_CLASSES, CATEGORIES, WIDTHS, feature_matrix, threshold_codes


_namespace = dict(vars(original), WIDTHS=WIDTHS)
_namespace['_make_model'] = _with_namespace(original._make_model, _namespace)
_check_request = _with_namespace(original._check_request, _namespace)
_train_fit = _with_namespace(original._train_fit, _namespace)
base = original.base


def descriptors(arm, data, ledger_dir, production):
    if production:
        manifest, codes = load(ledger_dir, data.index.tolist(), FINGERPRINT)
        if any(AUDITED_CLASSES[row[0]] != cls for row, cls in zip(codes, data.ionisation_class, strict=True)):
            raise ValueError('Categorical pH ledger does not reproduce the audited pH 7.4 categories')
        fingerprint = manifest['fingerprint']
    else:
        codes = [[block[2] for block in threshold_codes(a, b)]
                 for a, b in zip(data.pka_acid, data.pka_base, strict=True)]
        fingerprint = None
    spec = dict(arm=arm, input_dim=WIDTHS[arm], method='strict_threshold_fixed_one_hot_blocks',
                acidic_rule='pka_acid < ph', basic_rule='pka_base > ph', category_order=list(CATEGORIES),
                ph_blocks=[ARM_PH[arm]], width_divisor=1.0, feature_ledger_fingerprint=fingerprint)
    return torch.from_numpy(feature_matrix(arm, codes)), spec


def run_fit(request, release_dir, feature_dir, output_dir, device='cuda', *, ledger_dir=None):
    start = time.perf_counter()
    ids, max_epochs, fixed, production = _check_request(request)
    if production and ledger_dir is None:
        raise ValueError('Production single-pH fits require the frozen feature ledger')
    release_dir, feature_dir, output_dir = map(Path, (release_dir, feature_dir, output_dir))
    protected = [release_dir, feature_dir] + ([Path(ledger_dir)] if ledger_dir is not None else [])
    if any(output_dir.resolve().is_relative_to(path.resolve()) for path in protected):
        raise ValueError('Fit artifacts must be outside all frozen input directories')
    device = torch.device(device)
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('Fit device must be CPU or CUDA')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    hashes, environment = original._hashes(request, release_dir, feature_dir, device)
    if production:
        release = json.loads((release_dir/'manifest.json').read_text(encoding='utf-8'))
        if release['fingerprint'] != FINGERPRINT or release.get('structure_count') != 327:
            raise ValueError('Expected the original frozen 327-structure numerical release')
    source = Path(__file__).resolve()
    dependencies = (source.name, 'single_ph_features.py', 'ph_ionisation_features.py', 'ph_ionisation_train.py')
    hashes['code_hash'] = base._object_hash({'prior_trainer': hashes['code_hash'], **{
        name: base._file_hash(source.with_name(name)) for name in dependencies
    }})
    data = original._load_cohort(request, release_dir, feature_dir, ids, production)
    values, descriptor = descriptors(request['arm'], data, ledger_dir, production)
    hashes['categorical_feature_hash'] = descriptor['feature_ledger_fingerprint']
    train, val, scaler = original._prepare_targets(data, ids, production)
    scaler['descriptor'] = descriptor
    hashes['scaler_hash'] = base._object_hash(scaler)
    hashes['resolved_hash'] = base._object_hash(hashes)
    output_dir.mkdir(parents=True, exist_ok=True)
    with base._fit_lock(output_dir):
        existing = base._resume_completed(output_dir, request['fit_id'], hashes)
        if existing is not None:
            return existing
        return _train_fit(request, ids, max_epochs, fixed, data, values, train, val, scaler,
                          feature_dir, output_dir, device, hashes, environment, start)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('request', 'release-dir', 'feature-dir', 'output-dir'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--ledger-dir', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding='utf-8'))
    result = run_fit(request, args.release_dir, args.feature_dir, args.output_dir,
                     args.device, ledger_dir=args.ledger_dir)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
