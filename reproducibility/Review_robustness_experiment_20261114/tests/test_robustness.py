"""Executable compatibility gate against the actual previous combined trainer."""
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
import combined_train as new


@pytest.fixture(scope='module')
def old():
    path = ROOT / 'runtime/reference_combined_train.py'
    spec = importlib.util.spec_from_file_location('reference_combined_train', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny(tmp_path):
    release, features = tmp_path / 'release', tmp_path / 'features'
    release.mkdir(); features.mkdir()
    ids = [f's{i:02d}' for i in range(12)]
    data = pd.DataFrame(dict(structure_id=ids, canonical_smiles=['CCO', 'CCN', 'CC'] * 4,
        ionisation_class=['anionic', 'neutral', 'zwitterionic'] * 4,
        pka_acid=[4, 7.8, 4] * 4, pka_base=[6, 6, 9] * 4))
    for endpoint in new.RAW_ENDPOINTS:
        data[endpoint] = np.arange(1, 13)
    data.to_csv(release / 'model_dataset.csv', index=False)
    new.base._write_json(features / 'structures.json', ids)
    np.save(features / 'chemberta.npy', np.random.default_rng(42).normal(size=(12, 768)).astype('float32'))
    return release, features, data.set_index('structure_id'), dict(cohort='numerical', seed=42,
        train_ids=ids[:8], val_ids=ids[8:], predict_ids=ids[8:], max_epochs=2,
        fixed_epochs=None, diagnostic=True, task_weights=[1.] * 7)


@pytest.mark.parametrize('backbone', new.BACKBONES)
def test_actual_reference_execution(old, tiny, tmp_path, backbone):
    release, features, data, request = tiny
    # Hashes include the path's adjacent vendor tree, so use old execution with the
    # unchanged shared hashing helper; assertions compare computation, not fit IDs.
    old._hashes = new._hashes
    config = dict(hidden_dim=8, ffn_layers=2, dropout=.2, lr=.001, weight_decay=.001)
    if backbone != 'ChemBERTa_head': config['depth'] = 2
    common = []
    for arm in ('P0', 'P1'):
        for lam in (0., .3):
            req = dict(request, backbone=backbone, config=config, feature_arm=arm,
                cell=new.CELLS[arm, lam > 0], lambda_value=lam, fit_id=f'{arm}-{lam}')
            paths = [tmp_path / f'{label}-{arm}-{lam}' for label in ('old', 'new')]
            results = [trainer.run_fit(req, release, features, path, 'cpu')
                       for trainer, path in zip((old, new), paths)]
            assert (paths[0] / 'predictions.csv').read_bytes() == (paths[1] / 'predictions.csv').read_bytes()
            assert (paths[0] / 'scaler.json').read_bytes() == (paths[1] / 'scaler.json').read_bytes()
            for key in ('initial_state_hash', 'common_state_hash', 'best_epoch', 'actual_epochs'):
                assert results[0][key] == results[1][key]
            curves = [pd.read_csv(p / 'learning_curve.csv').drop(columns='epoch_seconds') for p in paths]
            pd.testing.assert_frame_equal(*curves, check_exact=True)
            states = [torch.load(p / 'checkpoint.pt', weights_only=True)['state_dict'] for p in paths]
            for key in states[0]: torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
            common.append(results[1]['common_state_hash'])
    assert len(set(common)) == 1
    # The same unchanged native forward and common initialization also extends
    # to both dimension-two arms; +/-lambda use exactly the same initial state.
    for arm in ('P2', 'P3'):
        summaries = []
        for lam in (0., .3):
            req = dict(request, backbone=backbone, config=config, feature_arm=arm,
                cell=new.CELLS[arm, lam > 0], lambda_value=lam, fit_id=f'{arm}-{lam}')
            summaries.append(new.run_fit(req, release, features, tmp_path / f'new-{arm}-{lam}', 'cpu'))
        assert summaries[0]['initial_state_hash'] == summaries[1]['initial_state_hash']
        assert summaries[0]['common_state_hash'] == common[0]


def test_missing_labels_sigma_and_batch_average():
    prediction = torch.tensor([[.2, .5, .7, .1, .3, .4, .6], [.1, .6, .4, .3, .2, .7, .5]], dtype=torch.float64, requires_grad=True)
    target = torch.full_like(prediction, float('nan')); target[0, 0] = 1.
    mean = torch.arange(7, dtype=torch.float64) * .2
    sd = torch.arange(1, 8, dtype=torch.float64)
    supervised, penalty = new.base.loss_terms(prediction, target, mean, sd, torch.ones(7))
    sg = torch.autograd.grad(supervised, prediction, retain_graph=True)[0]
    assert torch.count_nonzero(sg).item() == 1
    q = prediction[:, :3] * sd[:3] + mean[:3]
    residual = q[:, 0] - q[:, 1] + q[:, 2] - math.log10(math.log(2))
    torch.testing.assert_close(penalty, residual.square().mean())
    grad = torch.autograd.grad(penalty, prediction)[0]
    expected = 2 / len(prediction) * residual[:, None] * sd[:3] * torch.tensor([1., -1., 1.])
    torch.testing.assert_close(grad[:, :3], expected)
    assert torch.count_nonzero(grad[:, 3:]).item() == 0
    separate = [new.base.loss_terms(prediction[i:i+1], target[i:i+1], mean, sd, torch.ones(7))[1] for i in range(2)]
    torch.testing.assert_close(penalty, sum(separate) / 2)


def test_scalers_train_only(tiny):
    _, _, data, req = tiny
    for arm in ('P2', 'P3'):
        first, recipe = new.frozen.fit_descriptors(arm, data, req['train_ids'])
        changed = data.copy()
        changed.loc[req['val_ids'], ['pka_acid', 'pka_base']] = [50., -50.]
        second, other = new.frozen.fit_descriptors(arm, changed, req['train_ids'])
        assert recipe == other
        torch.testing.assert_close(first[:8], second[:8], rtol=0, atol=0)
        torch.testing.assert_close(first[:8].mean(0), torch.zeros(2), atol=1e-6, rtol=0)
    ids = {k: req[k] for k in ('train_ids', 'val_ids', 'predict_ids')}
    _, _, recipe = new.frozen._prepare_targets(data, ids, False)
    changed = data.copy(); changed.loc[req['val_ids'], list(new.RAW_ENDPOINTS)] = 99.
    _, _, other = new.frozen._prepare_targets(changed, ids, False)
    assert recipe == other


def test_early_stopping_uses_unpenalized_validation(tiny, tmp_path, monkeypatch):
    release, features, _, request = tiny
    scores = iter([1., .99995, 1.1, .8, .8])
    monkeypatch.setattr(new.base, '_score_validation', lambda *args: (next(scores), [1.] * 7, [4] * 7))
    request.update(backbone='ChemBERTa_head', config=dict(hidden_dim=8, ffn_layers=2, dropout=0., lr=.001, weight_decay=0.),
        feature_arm='P3', cell='P3_1', lambda_value=10., fit_id='early-stop', max_epochs=4)
    result = new.run_fit(request, release, features, tmp_path / 'fit', 'cpu')
    assert result['best_epoch'] == 4
    curve = pd.read_csv(tmp_path / 'fit/learning_curve.csv')
    assert curve.val_equal_mse.tolist() == [1., .99995, 1.1, .8]
