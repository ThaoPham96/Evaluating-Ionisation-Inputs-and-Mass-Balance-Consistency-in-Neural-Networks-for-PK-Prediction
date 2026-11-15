"""Exercise the frozen single-pH recipe and local unit provenance."""
from collections import Counter
from pathlib import Path

import pytest

import single_ph_campaign as campaign


def test_real_dry_plan_reuses_exact_original_recipe_and_reference_fits():
    root = Path(__file__).resolve().parents[2]
    plan = campaign.read(root/'runtime/single_ph_plan.json')
    outer, inner = campaign.verify_plan(root, plan)
    assert len(outer) == 327
    assert plan['new_fits_upper_bound'] == 630
    assert Counter(unit['backbone'] for unit in plan['units']) == dict.fromkeys(campaign.BACKBONES, 5)
    assert sum(unit['new_fits_upper_bound'] for unit in plan['units']) == 630
    assert len(plan['identity']['reference_fits']) == 270
    assert {ref['source_campaign_id'] for ref in plan['identity']['reference_fits']} == {campaign.previous.OLD_CAMPAIGN}
    assert {ref['source_arm'] for ref in plan['identity']['reference_fits']} == {'P0', 'P1'}
    refs = {ref['source_fit_id'] for ref in plan['identity']['reference_fits']}
    assert len(refs) == 270
    counts = Counter()
    fids = set()
    for unit in plan['units']:
        calibration = campaign._calibration_requests(plan, unit, outer, inner)
        summaries = {req['fit_id']: {'best_epoch': req['inner_fold']} for req in calibration}
        finals, _, _ = campaign._final_requests(plan, unit, outer, inner, calibration, summaries)
        key = f"{unit['backbone']}-outer{unit['outer_fold']}"
        recipe = plan['identity']['anchor_recipes'][key]['config']
        for req in calibration+finals:
            assert req['config'] == recipe
            assert req['lambda_value'] == 0.0 and req['task_weights'] == [1.0]*7
            assert not set(req['train_ids']).intersection(req['predict_ids'])
            assert req['fit_id'] not in fids | refs
            fids.add(req['fit_id'])
            counts[req['arm']] += 1
        for req in calibration:
            assert req['seed'] == 42 and req['fixed_epochs'] is None
            assert req['predict_ids'] == req['val_ids']
            assert all(outer[sid] != unit['outer_fold'] for sid in req['predict_ids'])
        for req in finals:
            assert req['seed'] in campaign.SEEDS and req['fixed_epochs'] == 3
            assert req['val_ids'] == []
            assert all(outer[sid] == unit['outer_fold'] for sid in req['predict_ids'])
    assert counts == dict.fromkeys(campaign.ARM_PH, 210)


def test_completed_unit_reconstructs_calibration_and_refuses_tampering(tmp_path, monkeypatch):
    unit = campaign.unit_spec()[0]
    outer = {'heldout': 1, **{f's{i}': 2 for i in campaign.INNER_FOLDS}}
    inner = {1: {f's{i}': i for i in campaign.INNER_FOLDS}}
    plan = dict(campaign_id='test-campaign', units=[unit], identity={
        'anchor_recipes': {'Chemprop-outer1': {'candidate_index': 2, 'config': {'hidden_dim': 8}}},
        'categorical_features_fingerprint': 'test-feature-ledger',
    })
    monkeypatch.setattr(campaign, 'verify_plan', lambda root, plan: (outer, inner))
    calls = []

    def fit(args):
        req, _, _, _, folder, _ = args
        calls.append(req['fit_id'])
        folder.mkdir(parents=True)
        arm = req['arm']
        best = req['fixed_epochs'] or (req['inner_fold'] + list(campaign.ARM_PH).index(arm))
        descriptor = dict(ph_blocks=[campaign.ARM_PH[arm]], input_dim=4, width_divisor=1,
                          feature_ledger_fingerprint='test-feature-ledger')
        scaler = dict(descriptor=descriptor)
        summary = {key:req[key] for key in ('fit_id','arm','cohort','backbone','seed','config',
                                          'campaign_id','lambda_value','task_weights')}
        summary.update(descriptor_spec=descriptor, scaler_hash=campaign.digest(scaler),
                       resolved_hash='fake-resolved', checkpoint_reload_verified=True, actual_epochs=best,
                       best_epoch=best)
        campaign.write(folder/'summary.json', summary)
        campaign.write(folder/'scaler.json', scaler)
        effective = {key:req[key] for key in ('train_ids','val_ids','predict_ids')}
        campaign.write(folder/'request.json', dict(request=req, effective_ids=effective,
                       request_hash=campaign.digest(req), resolved_hash='fake-resolved', scaler_hash=campaign.digest(scaler)))
        for name in ('predictions.csv', 'checkpoint.pt', 'learning_curve.csv'):
            (folder/name).write_text(name, encoding='utf-8')
        campaign.write(folder/'complete.json', dict(fit_id=req['fit_id'], resolved_hash='fake-resolved',
                       files={name:campaign.sha(folder/name) for name in campaign.previous.ARTIFACTS}))
        return summary

    monkeypatch.setattr(campaign, '_execute', fit)
    output = tmp_path/'work'
    marker, = campaign.run(tmp_path, plan, output, device='cpu', unit_id=unit['unit_id'])
    assert marker['unique_fits'] == len(set(calls)) == 42
    assert {r['arm']:r['epochs'] for r in marker['final_records']} == {'PH2':3, 'PH5':4, 'PH65':5}
    assert campaign.run(tmp_path, plan, output, device='cpu') == [marker]
    assert len(calls) == 42
    path = output/unit['unit_id']/'fits'/calls[0]/'predictions.csv'
    path.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='artifact checksum'):
        campaign.run(tmp_path, plan, output, device='cpu')
