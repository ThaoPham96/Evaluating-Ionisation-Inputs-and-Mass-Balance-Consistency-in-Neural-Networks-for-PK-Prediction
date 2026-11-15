"""Fit a categorical pH arm with the frozen ionisation training procedure."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from types import FunctionType

import torch

import ionisation_train as original
from ph_ionisation_features import AUDITED_CLASSES, BLOCKS, CATEGORIES, FINGERPRINT, PH, WIDTHS, feature_matrix, load


def _with_namespace(function, namespace):
    """Bind the immutable implementation to this experiment's arm registry."""
    return FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)


# The first campaign is a frozen dependency. Rebind only these three private
# functions in a separate namespace so the actual fit loop and initialisation
# remain identical without modifying the prior code or its process globals.
_namespace = dict(vars(original), WIDTHS=WIDTHS)
_namespace['_make_model'] = _with_namespace(original._make_model, _namespace)
_check_request = _with_namespace(original._check_request, _namespace)
_train_fit = _with_namespace(original._train_fit, _namespace)
base = original.base


def descriptors(arm, data, ledger_dir, production):
    if production:
        manifest, codes = load(ledger_dir, data.index.tolist(), FINGERPRINT)
        if any(AUDITED_CLASSES[entry[0]] != cls for entry, cls in zip(codes, data.ionisation_class, strict=True)):
            raise ValueError('Categorical pH ledger does not reproduce the audited pH 7.4 categories')
        fingerprint = manifest['fingerprint']
    else:
        from ph_ionisation_features import threshold_codes

        codes = [[part[2] for part in threshold_codes(a, b)] for a, b in zip(data.pka_acid, data.pka_base, strict=True)]
        fingerprint = None
    blocks = BLOCKS[arm]
    spec = dict(arm=arm, input_dim=WIDTHS[arm], method='strict_threshold_fixed_one_hot_blocks',
                acidic_rule='pka_acid < ph', basic_rule='pka_base > ph', category_order=list(CATEGORIES),
                ph_blocks=[PH[i] for i in blocks], width_divisor=math.sqrt(len(blocks)),
                feature_ledger_fingerprint=fingerprint)
    return torch.from_numpy(feature_matrix(arm, codes)), spec


def run_fit(request, release_dir, feature_dir, output_dir, device='cuda', *, ledger_dir=None):
    start = time.perf_counter()
    ids, max_epochs, fixed, production = _check_request(request)
    if production and ledger_dir is None:
        raise ValueError('Production categorical pH fits require a frozen feature ledger')
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
    hashes['code_hash'] = base._object_hash({
        'prior_trainer': hashes['code_hash'], source.name: base._file_hash(source),
        'ph_ionisation_features.py': base._file_hash(source.with_name('ph_ionisation_features.py')),
    })
    data = original._load_cohort(request, release_dir, feature_dir, ids, production)
    values, recipe = descriptors(request['arm'], data, ledger_dir, production)
    hashes['categorical_feature_hash'] = recipe['feature_ledger_fingerprint']
    train, val, scaler = original._prepare_targets(data, ids, production)
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
    for field in ('request', 'release-dir', 'feature-dir', 'output-dir'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--ledger-dir', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding='utf-8'))
    result = run_fit(request, args.release_dir, args.feature_dir, args.output_dir,
                     args.device, ledger_dir=args.ledger_dir)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
