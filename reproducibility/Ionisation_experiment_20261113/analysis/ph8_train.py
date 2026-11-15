"""Bind the frozen single-pH trainer to the pH 8.0 descriptor."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

import ionisation_train as original
from ph_ionisation_train import _with_namespace
import single_ph_train as single
from ph8_features import ARM, CATEGORIES, PH, WIDTH, feature_matrix, load, threshold_code


def descriptors(arm, data, ledger_dir, production):
    if arm != ARM:
        raise ValueError('The pH 8.0 adapter accepts only PH8')
    if production:
        raise ValueError('Production descriptor binding requires the frozen dataset path')
    codes = [threshold_code(a, b) for a, b in zip(data.pka_acid, data.pka_base, strict=True)]
    return torch.from_numpy(feature_matrix(codes)), descriptor_spec(None)


def descriptor_spec(fingerprint):
    return dict(arm=ARM, input_dim=WIDTH, method='strict_threshold_fixed_one_hot_blocks',
                acidic_rule='pka_acid < ph', basic_rule='pka_base > ph', category_order=list(CATEGORIES),
                ph_blocks=[PH], width_divisor=1.0, feature_ledger_fingerprint=fingerprint)


def _hashes(request, release_dir, feature_dir, device):
    hashes, environment = original._hashes(request, release_dir, feature_dir, device)
    source = Path(__file__).resolve()
    dependencies = ('ph8_features.py', 'single_ph_train.py')
    hashes['code_hash'] = single.base._object_hash({'prior_trainer': hashes['code_hash'], **{
        name: single.base._file_hash(source.with_name(name)) for name in dependencies
    }})
    return hashes, environment


# Bind the same original fit loop and initialization in an isolated namespace.
_namespace = dict(vars(original), WIDTHS={ARM:WIDTH})
_namespace['_make_model'] = _with_namespace(original._make_model, _namespace)
_check_request = _with_namespace(original._check_request, _namespace)
_train_fit = _with_namespace(original._train_fit, _namespace)
_original = SimpleNamespace(_hashes=_hashes, _load_cohort=original._load_cohort,
                            _prepare_targets=original._prepare_targets)


def run_fit(request, release_dir, feature_dir, output_dir, device='cuda', *, ledger_dir=None):
    dataset = Path(release_dir)/'model_dataset.csv'

    def bound_descriptors(arm, data, ledger, production):
        if not production:
            return descriptors(arm, data, ledger, production)
        manifest, codes = load(ledger, data.index.tolist(), dataset)
        return torch.from_numpy(feature_matrix(codes)), descriptor_spec(manifest['fingerprint'])

    namespace = dict(vars(single), __file__=__file__, original=_original, _check_request=_check_request,
                     _train_fit=_train_fit, descriptors=bound_descriptors)
    return _with_namespace(single.run_fit, namespace)(request, release_dir, feature_dir, output_dir,
                                                       device, ledger_dir=ledger_dir)


if __name__ == '__main__':
    _with_namespace(single.main, dict(vars(single), run_fit=run_fit, __doc__=__doc__))()
