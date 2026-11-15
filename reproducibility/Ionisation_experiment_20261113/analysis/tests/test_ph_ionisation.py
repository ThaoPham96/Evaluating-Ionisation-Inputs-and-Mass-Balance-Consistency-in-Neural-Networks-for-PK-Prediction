"""The pH follow-up retains the frozen trainer and pairs feature-width controls."""
from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

import ionisation_campaign as previous
import ph_ionisation_campaign as campaign
import ph_ionisation_features as features


def test_strict_threshold_block_order_and_width_matched_norm():
    threshold = features.threshold_codes('7.4', '7.4')
    assert threshold[0][:3] == (False, False, 2)
    assert all(row[:3] == (False, True, 1) for row in threshold[1:])
    selected = [features.threshold_codes(a,b) for a,b in [('4','5'),('8','1'),('3','9'),('5','4')]]
    codes = np.asarray([[r[2] for r in matrix] for matrix in selected])
    np.testing.assert_array_equal(codes[0], [0,1,0,0])
    for arm, blocks in features.BLOCKS.items():
        matrix = features.feature_matrix(arm, codes)
        assert matrix.shape == (4, 4*len(blocks))
        np.testing.assert_allclose((matrix**2).sum(axis=1), 1, rtol=2e-7)
        np.testing.assert_array_equal(matrix.reshape(len(codes),len(blocks),4).argmax(axis=2), codes[:,blocks])
    assert features.WIDTHS['REP2'] == features.WIDTHS['ADD2'] == features.WIDTHS['ADD65'] == features.WIDTHS['ADD5'] == 8
    assert features.WIDTHS['REP4'] == features.WIDTHS['ALL4'] == 16


def test_decimal_cutoff_and_all_stress_scenarios():
    original = features.threshold_codes('9.0','7.90')
    assert str(original[0][4]) == '0.50'
    summary, records = features.perturbations('9.0','7.90', original)
    assert len(records) == len({(r['acid_shift'],r['base_shift']) for r in records}) == 24
    assert summary['original_profile_can_change'] == 1
    assert summary['base_changes_7.4'] == 10
    assert all(r['base_changes_7.4'] for r in records if r['base_shift'] == '-0.5')


def test_dag_counts_and_exact_request_reuse():
    units = campaign.unit_spec()
    assert Counter(u['stage'] for u in units) == {'core':15, 'tune':15}
    assert sum(u['new_fits_upper_bound'] for u in units if u['stage']=='core') == 1260
    assert sum(u['new_fits_upper_bound'] for u in units if u['stage']=='tune') == 1620
    assert set(campaign.TUNED) == {'ALL4','REP4'}
    assert len({u['unit_id'] for u in units}) == 30
    core, tune = units[:2]
    assert tune['depends_on'] == core['unit_id']
    plan = {'campaign_id':'synthetic'}
    outer = {'train':2, 'valid':2, 'heldout':1}
    inner = {1:{'train':1, 'valid':2}}
    key = (plan, core, 'ALL4', {'hidden_dim':8}, 42, outer, inner)
    calibration = previous.fit_request(*key, inner=2)
    tuning = previous.fit_request(plan, tune, *key[2:], inner=2)
    assert calibration == tuning
    assert calibration['task_weights'] == [1.]*7 and calibration['lambda_value'] == 0.
    assert calibration['train_ids'] == ['train'] and calibration['val_ids'] == ['valid']
    assert previous.fit_request(*key, inner=1)['fit_id'] != calibration['fit_id']
    assert previous.fit_request(*key, epochs=2)['fit_id'] != calibration['fit_id']


@pytest.fixture
def fixture(tmp_path):
    release, structural = tmp_path/'release', tmp_path/'features'
    release.mkdir()
    structural.mkdir()
    ids = [f's{i}' for i in range(12)]
    frame = pd.DataFrame(dict(structure_id=ids, canonical_smiles=['CCO','CCN','CC']*4,
                         ionisation_class=['neutral']*12, pka_acid=np.linspace(3,9,12), pka_base=np.linspace(2,9,12)))
    from ionisation_train import RAW_ENDPOINTS
    for endpoint in RAW_ENDPOINTS:
        frame[endpoint] = np.arange(1,13)
    frame.loc[0,RAW_ENDPOINTS[3]] = np.nan
    frame.to_csv(release/'model_dataset.csv',index=False)
    (structural/'structures.json').write_text(json.dumps(ids),encoding='utf-8')
    np.save(structural/'chemberta.npy', np.ones((len(ids),768),dtype=np.float32))
    request = dict(fit_id='ph-test', cohort='numerical', backbone='GraphConv', arm='ALL4', seed=42,
                   config=dict(hidden_dim=8,depth=2,ffn_layers=2,dropout=.1,lr=.001,weight_decay=.001),
                   train_ids=ids[:7],val_ids=ids[7:10],predict_ids=ids[7:10],max_epochs=1,diagnostic=True,
                   fixed_epochs=None,task_weights=[1.]*7,lambda_value=0.)
    return release, structural, request


@pytest.mark.parametrize('backbone', previous.BACKBONES)
def test_cpu_fit_equal_width_paired_init_and_reload(fixture,tmp_path,backbone):
    from ph_ionisation_train import run_fit

    release, structural, request = fixture
    request['backbone'] = backbone
    if backbone == 'ChemBERTa_head':
        request['config']['depth'] = None
    result = run_fit(request, release, structural, tmp_path/'all', device='cpu')
    repeat_request = dict(request,fit_id='ph-repeat',arm='REP4')
    repeat = run_fit(repeat_request,release,structural,tmp_path/'repeat',device='cpu')
    assert result['initial_state_hash'] == repeat['initial_state_hash']
    assert result['common_state_hash'] == repeat['common_state_hash']
    assert result['descriptor_input_dim'] == 16 and result['checkpoint_reload_verified']
    assert result['task_weights'] == [1.]*7 and result['lambda_value'] == 0.
    assert result['val_observed_counts'] == dict.fromkeys(previous.ENDPOINTS,3)
    assert run_fit(request,release,structural,tmp_path/'all',device='cpu') == result
    assert result['scaler_hash'] != repeat['scaler_hash']
    original = __import__('ionisation_train')
    assert 'ALL4' not in original.WIDTHS
    assert original.WIDTHS['P1'] == 4


@pytest.mark.parametrize('change,match', [({'lambda_value':.01},'lambda_value'),({'task_weights':[1,1,1,2,1,1,1]},'task weights')])
def test_no_loss_change(fixture,tmp_path,change,match):
    from ph_ionisation_train import run_fit
    release, structural, request = fixture
    with pytest.raises(ValueError,match=match):
        run_fit(dict(request,**change),release,structural,tmp_path/'out',device='cpu')


def test_tuning_reuses_complete_core_fits_and_rejects_tampering(tmp_path, monkeypatch):
    """Exercise production request generation and both unit paths without real fits."""
    core, tune = campaign.unit_spec()[:2]
    outer = {'heldout':1, **{f's{i}':2 for i in range(1,6)}}
    inner = {1:{f's{i}':i for i in range(1,6)}}
    identity = {'grid':{'Chemprop':[{'value':i} for i in range(10)]},
                'anchor_recipes':{'Chemprop-outer1':{'candidate_index':0}}}
    plan = {'campaign_id':'mock-campaign','identity':identity,'units':[core,tune]}
    monkeypatch.setattr(campaign,'verify_plan',lambda root,plan:(outer,inner))
    seen = []

    def execute(args):
        request, _, _, _, destination, _ = args
        seen.append(request['fit_id'])
        destination.mkdir(parents=True)
        config = request['config']
        scaler = {'descriptor':{'arm':request['arm']}}
        request_hash = previous.digest(request)
        resolved_hash = previous.digest([request_hash,scaler])
        summary = {k:request[k] for k in ('fit_id','arm','cohort','backbone','seed','config','campaign_id','lambda_value','task_weights')}
        summary.update(resolved_hash=resolved_hash,scaler_hash=previous.digest(scaler),
                       descriptor_spec=scaler['descriptor'],checkpoint_reload_verified=True,
                       actual_epochs=2,best_epoch=2,val_squared_error_sums=dict.fromkeys(previous.ENDPOINTS,float(config['value'])),
                       val_observed_counts=dict.fromkeys(previous.ENDPOINTS,1))
        previous.write(destination/'summary.json',summary)
        previous.write(destination/'scaler.json',scaler)
        previous.write(destination/'request.json',dict(request=request,request_hash=request_hash,resolved_hash=resolved_hash,
                       scaler_hash=previous.digest(scaler),effective_ids={k:request[k] for k in ('train_ids','val_ids','predict_ids')}))
        for name in ('predictions.csv','checkpoint.pt','learning_curve.csv'):
            (destination/name).write_text(name,encoding='utf-8')
        previous.write(destination/'complete.json',dict(fit_id=request['fit_id'],resolved_hash=resolved_hash,
                       files={name:previous.sha(destination/name) for name in campaign.ARTIFACTS}))
        return summary

    monkeypatch.setattr(campaign,'_execute',execute)
    core_marker = campaign.run_unit(tmp_path,plan,core['unit_id'],tmp_path/'output',device='cpu')
    tune_marker = campaign.run_unit(tmp_path,plan,tune['unit_id'],tmp_path/'output',device='cpu')
    assert core_marker['unique_fits'] == 84
    assert tune_marker['unique_fits'] == 90
    assert len(tune_marker['reused_fits']) == 28  # Ten inner anchor fits and 18 matching finals.
    assert len(seen) == len(set(seen)) == 174
    assert all(record['source_unit_id'] == core['unit_id'] for record in tune_marker['final_records'])
    fit = next(iter(tune_marker['reused_fits']))
    summary = tmp_path/'output'/core['unit_id']/'fits'/fit/'summary.json'
    summary.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='artifact checksum'):
        campaign.run_unit(tmp_path,plan,tune['unit_id'],tmp_path/'output',device='cpu')
