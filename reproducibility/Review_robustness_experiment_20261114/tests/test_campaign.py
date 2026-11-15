"""Exercise pooled selection, complete logical coverage and request deduplication."""
from pathlib import Path
import sys
import math
from collections import defaultdict
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
import robustness_campaign as r


@pytest.fixture(scope='module')
def campaign(): return r.load_campaign(ROOT)


def fake_output(c, req):
    predictions=[]
    for sid in req['predict_ids']:
        predictions.append(dict(structure_id=sid,
            **{'pred_'+ep: .3 for ep in r.a.ENDPOINTS},
            **{'physical_'+ep: 10**.3 for ep in r.a.ENDPOINTS}))
    # Nonuniform endpoint/fold counts expose mean-of-fold-means mistakes.
    inner=req['inner_fold'] or 1
    counts={ep: inner+j+1 for j,ep in enumerate(r.a.ENDPOINTS)}
    sums={ep: (inner+j+1)*(inner+j+.2) for j,ep in enumerate(r.a.ENDPOINTS)}
    summary=dict(best_epoch=inner+2,val_squared_error_sums=sums,val_observed_counts=counts,
                 initial_state_hash='same',environment=c.references['environments'][req['backbone']])
    return summary,predictions


def synthetic_unit(c, unit):
    fits={}
    requests={}
    for group in r.architecture_requests(c,unit):
        for req in group:
            fits[req['fit_id']]=fake_output(c,req); requests[req['fit_id']]=req
    architecture=r.architecture_selection(c,unit,fits)
    for group in r.lambda_requests(c,unit,architecture):
        for req in group:
            fits[req['fit_id']]=fake_output(c,req); requests[req['fit_id']]=req
    selection=r.selection_record(c,unit,fits)
    conditions=r.final_conditions(c,unit,selection)
    for row in conditions:
        req=row['request']; fits[req['fit_id']]=fake_output(c,req); requests[req['fit_id']]=req
    marker=dict(fit_markers={fid:'synthetic' for fid in fits},unique_fits=len(fits))
    return marker,selection,conditions,fits


def test_pooled_selection_and_all_request_counts(campaign):
    architecture_ids=set(); positive_ids=set(); finals=set(); logical=0
    from combined_train import _check_request
    for unit in r.UNITS:
        marker,selection,conditions,fits=synthetic_unit(campaign,unit)
        assert selection['architecture']['selected_config_id']==1
        assert selection['selected_lambda']==.03 and selection['zero_eligible_lambda']==0
        assert selection['grid'][0]['median_epoch']==5
        groups=r.architecture_requests(campaign,unit)
        expected=sum(sum(fake_output(campaign,req)[0]['val_squared_error_sums'][ep] for req in groups[0]) /
                     sum(fake_output(campaign,req)[0]['val_observed_counts'][ep] for req in groups[0]) for ep in r.a.ENDPOINTS)/7
        assert math.isclose(selection['architecture']['grid'][0]['val_equal_mse'],expected)
        for group in groups:
            for req in group:
                _check_request(req); architecture_ids.add(req['fit_id'])
        for group in r.lambda_requests(campaign,unit,selection['architecture']):
            for req in group: _check_request(req); positive_ids.add(req['fit_id'])
        for row in conditions: _check_request(row['request']); finals.add(row['request']['fit_id']); logical+=1
        if unit.endswith(('P2','P3')):
            recipe=campaign.plan['identity']['recipes'][unit.rsplit('-',1)[0]]
            assert selection['R0']['config']==recipe['config']
            assert selection['R0']['config_id']==recipe['candidate_index']+1
    assert len(architecture_ids)==3000 and len(positive_ids)==1800
    assert logical==1080+270+108
    assert len(finals)<=logical
    assert not architecture_ids & positive_ids
    assert not (architecture_ids|positive_ids)&finals
    assert len(architecture_ids|positive_ids|finals)<=6258


@pytest.mark.parametrize('arm',r.ARMS)
def test_run_unit_state_machine(campaign,arm,tmp_path,monkeypatch):
    import combined_train as trainer
    unit='ChemBERTa_head-outer1-'+arm
    model='ChemBERTa_head'
    monkeypatch.setattr(trainer,'_hashes',lambda *args: ({},campaign.references['environments'][model]))
    produced={}
    def fake_run(req,release,features,path,device):
        path.mkdir(parents=True,exist_ok=True)
        produced[req['fit_id']]=fake_output(campaign,req)
        r.a.write_json(path/'complete.json',dict(fit_id=req['fit_id']))
        r.a.write_csv(path/'learning_curve.csv',['epoch','batch_order_seed','dropout_seed','batch_order_sha256'],
                      [dict(epoch=1,batch_order_seed=1,dropout_seed=2,batch_order_sha256='x')])
    monkeypatch.setattr(trainer,'run_fit',fake_run)
    monkeypatch.setattr(r.a,'validate_fit',lambda path,req: produced[req['fit_id']])
    actual=r.execute_unit(campaign,unit,tmp_path,device='cpu')
    replay=r.execute_unit(campaign,unit,tmp_path,read_only=True)
    assert actual[0]==replay[0]
    assert actual[0]['inner_fits']==80
    assert 18<=actual[0]['final_fits']<=27


def test_complete_export_and_reference_aliases(campaign,tmp_path,monkeypatch):
    monkeypatch.setattr(r,'execute_unit',lambda c,u,*args,**kwargs: synthetic_unit(c,u))
    manifest=r.export(campaign,tmp_path/'units',tmp_path/'export')
    assert manifest['final_rows']==327*3*9*16
    rows=r.a.rows(tmp_path/'export/final_predictions.csv')
    saved=r.a.rows(campaign.root/'runtime/references/final_predictions.csv')
    lookup={(row['fit_id'],row['structure_id']):row for row in saved}
    for row in rows:
        assert row['reference_fit_id']
        if row['protocol']=='F' and row['penalty'] in ('off','shared') and row['feature_arm'] in ('P0','P1'):
            original=lookup[row['fit_id'],row['structure_id']]
            for ep in r.a.ENDPOINTS: assert row['q_'+ep]==original['q_'+ep]


def test_runtime_fails_closed_before_training(campaign,tmp_path,monkeypatch):
    import combined_train as trainer
    monkeypatch.setattr(trainer,'_hashes',lambda *args: ({},{'torch':'wrong'}))
    monkeypatch.setattr(trainer,'run_fit',lambda *args: pytest.fail('Must not train'))
    with pytest.raises(ValueError,match='runtime differs'):
        r.execute_unit(campaign,'ChemBERTa_head-outer1-P2',tmp_path,device='cpu')
