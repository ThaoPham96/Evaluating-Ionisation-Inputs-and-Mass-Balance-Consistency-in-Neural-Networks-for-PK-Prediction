"""Verify pH 8.0 features, frozen requests, and the full export/scoring contract."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ionisation_score as common
import ph8_campaign as campaign
import ph8_features as features
import ph8_score as score


ROOT = Path(__file__).resolve().parents[2]


def test_threshold_four_channels_and_real_inventory(tmp_path):
    codes = [features.threshold_code(a,b) for a,b in
             [('7.999','8.0'), ('8.0','8.001'), ('8.0','8.0'), ('7.999','8.001')]]
    np.testing.assert_array_equal(features.feature_matrix(codes), np.eye(4, dtype=np.float32))
    assert features.threshold_code('7.3','7.5') == features.threshold_code('7.5','7.3') == 0
    with pytest.raises(ValueError, match='finite pKas'):
        features.threshold_code('NaN','8.0')
    with pytest.raises(ValueError, match='one integer'):
        features.feature_matrix([[0,1]])
    dataset = ROOT/'data/release/numerical/model_dataset.csv'
    directory = tmp_path/'ledger'
    manifest = features.prepare(dataset, directory)
    frozen = campaign.read(directory/'codes.json')
    assert [Counter(frozen.values())[code] for code in range(4)] == [82,75,128,42]
    assert manifest['switchers_vs_74'] == 32
    ids = list(reversed(frozen))
    assert features.load(directory, ids, dataset)[1] == [frozen[sid] for sid in ids]
    frozen[ids[0]] = (frozen[ids[0]] + 1) % 4
    campaign.write(directory/'codes.json', frozen)
    with pytest.raises(ValueError, match='changed'):
        features.prepare(dataset, directory)


def test_dry_plan_exact_frozen_recipe_and_folds():
    plan = campaign.read(ROOT/'runtime/ph8_plan.json')
    outer, inner = campaign.verify_plan(ROOT, plan)
    assert plan['new_fits_upper_bound'] == 210
    assert len(plan['identity']['reference_fits']) == 270
    requests = []
    for unit in plan['units']:
        calibration = campaign._calibration_requests(plan, unit, outer, inner)
        summaries = {req['fit_id']: {'best_epoch': req['inner_fold']} for req in calibration}
        finals, selection, records = campaign._final_requests(plan, unit, outer, inner, calibration, summaries)
        key = f"{unit['backbone']}-outer{unit['outer_fold']}"
        recipe = plan['identity']['anchor_recipes'][key]
        assert selection == {'PH8': {'candidate_index': recipe['candidate_index'], 'epochs':3}}
        assert len(calibration) == 5 and len(finals) == len(records) == 9
        assert {req['seed'] for req in calibration} == {42}
        assert {req['seed'] for req in finals} == set(campaign.SEEDS)
        for req in calibration+finals:
            assert req['arm'] == 'PH8' and req['config'] == recipe['config']
            assert req['task_weights'] == [1.0]*7 and req['lambda_value'] == 0
            assert not set(req['train_ids']).intersection(req['predict_ids'])
        assert all(req['predict_ids'] == req['val_ids'] for req in calibration)
        assert all(req['fixed_epochs'] == 3 and not req['val_ids'] for req in finals)
        requests.extend(calibration+finals)
    assert len({req['fit_id'] for req in requests}) == 210
    assert not ({req['fit_id'] for req in requests} & {ref['source_fit_id'] for ref in plan['identity']['reference_fits']})


def test_fit_paired_initialization_and_resume(tmp_path):
    import ionisation_train as original
    import ph8_train as trainer
    import single_ph_train as single

    release, structural = tmp_path/'release', tmp_path/'features'
    release.mkdir()
    structural.mkdir()
    ids = [f's{i}' for i in range(12)]
    frame = pd.DataFrame(dict(structure_id=ids, canonical_smiles=['CCO','CCN','CC']*4,
                              ionisation_class=['anionic','neutral','zwitterionic']*4,
                              pka_acid=[4,8,4]*4, pka_base=[6,6,9]*4))
    for endpoint in original.RAW_ENDPOINTS:
        frame[endpoint] = np.arange(1,13)
    frame.to_csv(release/'model_dataset.csv', index=False)
    (structural/'structures.json').write_text(json.dumps(ids), encoding='utf-8')
    np.save(structural/'chemberta.npy', np.ones((len(ids),768),dtype=np.float32))
    request = dict(fit_id='ref', arm='P1', cohort='numerical', backbone='ChemBERTa_head', seed=42,
                   config=dict(hidden_dim=8,ffn_layers=2,dropout=.1,lr=.001,weight_decay=.001),
                   train_ids=ids[:7],val_ids=ids[7:10],predict_ids=ids[7:10],max_epochs=1,diagnostic=True,
                   fixed_epochs=None,task_weights=[1.]*7,lambda_value=0.)
    ref = original.run_fit(request, release, structural, tmp_path/'ref', 'cpu')
    request.update(arm='PH8', fit_id='extension')
    outcome = trainer.run_fit(request, release, structural, tmp_path/'fit', 'cpu')
    assert outcome['initial_state_hash'] == ref['initial_state_hash']
    assert outcome['common_state_hash'] == ref['common_state_hash']
    assert outcome['descriptor_input_dim'] == 4 and outcome['checkpoint_reload_verified']
    assert outcome['descriptor_spec']['ph_blocks'] == ['8.0']
    assert outcome['task_weights'] == [1.]*7 and outcome['lambda_value'] == 0.
    assert 'PH8' not in single.WIDTHS and 'PH8' not in original.WIDTHS
    assert trainer.run_fit(request, release, structural, tmp_path/'fit', 'cpu') == outcome
    with pytest.raises(ValueError, match='task weights'):
        trainer.run_fit(dict(request,task_weights=[2,1,1,1,1,1,1]), release, structural, tmp_path/'invalid', 'cpu')
    path = tmp_path/'fit'/'predictions.csv'
    path.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        trainer.run_fit(request, release, structural, tmp_path/'fit', 'cpu')


def test_export_and_full_score_reproduce_original_references(tmp_path, monkeypatch):
    """Use synthetic PH8 fits with real reference predictions and exact bootstrap checks."""
    plan = campaign.read(ROOT/'runtime/ph8_plan.json')
    outer, inner = campaign.verify_plan(ROOT, plan)
    data = common.load_cohort(ROOT/'data/release', campaign.COHORT)
    output = tmp_path/'fits'
    computed = {}
    for unit in plan['units']:
        calibration = campaign._calibration_requests(plan, unit, outer, inner)
        summaries = {req['fit_id']: {'best_epoch': 1} for req in calibration}
        finals, _, _ = campaign._final_requests(plan, unit, outer, inner, calibration, summaries)
        computed[unit['unit_id']] = (dict(fit_markers={req['fit_id']:'synthetic' for req in finals}), finals)
        for req in finals:
            destination = output/unit['unit_id']/'fits'/req['fit_id']/'predictions.csv'
            destination.parent.mkdir(parents=True)
            ids = req['predict_ids']
            prediction = np.full((len(ids),7), req['seed'] / 1000)
            frame = pd.DataFrame(prediction, columns=common.PRED_COLUMNS)
            frame.insert(0, 'structure_id', ids)
            frame[list(common.PHYSICAL_COLUMNS)] = common.inverse_physical(prediction)
            frame.to_csv(destination, index=False)

    def verify(plan, unit, outer, inner, folder, *, full):
        assert full
        return computed[unit['unit_id']]

    monkeypatch.setattr(campaign, '_verify_complete', verify)
    destination = tmp_path/'export'/'outer_predictions.csv'
    assert campaign.export(ROOT, plan, output, destination) == 26487
    frame = common.read_csv(destination)
    old = common.read_csv(ROOT/campaign.SINGLE_EXPORT)
    keys = ['arm','backbone','outer_fold','seed','structure_id']
    pd.testing.assert_frame_equal(frame.loc[frame.arm.isin(campaign.single.REFERENCES)].sort_values(keys).reset_index(drop=True),
                                  old.loc[old.arm.isin(campaign.single.REFERENCES)].sort_values(keys).reset_index(drop=True),
                                  check_exact=True)
    _, sources = score.validate_predictions(frame, data, plan['campaign_id'])
    assert Counter(sources.arm) == {'REF0':135, 'REF74':135, 'PH8':135}
    with pytest.raises(ValueError, match='new destination'):
        campaign.export(ROOT, plan, output, destination)
    results = tmp_path/'scoring'
    summary = score.score_export(ROOT/'data/release', destination, results)
    assert summary['n_contrasts'] == 63 and summary['n_seed_rows'] == 567
    assert summary['verified_exact_reference_score_rows'] == {'contrasts.csv':21, 'contrast_seeds.csv':189}
    assert summary['n_new_prediction_fits'] == 135 and summary['n_reused_prediction_fits'] == 270
    contrasts = common.read_csv(results/'contrasts.csv')
    assert contrasts.valid_bootstrap_draws.eq(10000).all()
    assert set(contrasts.endpoint) == set(common.ENDPOINTS)
    assert not contrasts.primary_test.any()
