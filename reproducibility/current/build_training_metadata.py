"""Rebuild supplementary training metadata and compare archived F/T calibration.

Run scripts/pipeline.py hydrate first. No model fitting or network access occurs.
"""
from pathlib import Path
from io import StringIO
import argparse
import gzip
import hashlib
import json
import numpy as np
import pandas as pd

DATA=Path(__file__).resolve().parents[1]
EXP=DATA/'Review_robustness_experiment_20261114'
ENDPOINTS={'t_half':'t1/2(hour)','Vd':'VD(liter)','CL':'Cl(liter/hour)','Cmax':'Cmax_uM','Tmax':'Tmax','F':'F(percentage)','PPB':'PPB(percentage)'}

def build(output):
    output=output.resolve()
    if output==DATA or DATA in output.parents:
        raise ValueError('Use an output directory outside reproducibility/')
    output.mkdir(parents=True,exist_ok=True)
    selections=json.loads((EXP/'outputs/export/selections.json').read_text())
    comparisons=[]
    archive=DATA/'current/paired_calibration.jsonl.gz'
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['checksums'][str(archive.relative_to(DATA)).replace('\\','/')]
    with gzip.open(archive,'rt',encoding='utf-8') as handle:
        for line in handle:
            x=json.loads(line);o,n=x['F'],x['T']
            assert o['request']==n['request']
            for key in ['initial_state_hash','data_hash','release_manifest_hash','feature_hash','scaler_hash','environment_hash']:
                assert o['summary'][key]==n['summary'][key]
            for side in [o,n]:
                assert hashlib.sha256(side['curve_csv'].encode()).hexdigest()==side['curve_sha256']
            oc,nc=[pd.read_csv(StringIO(side['curve_csv'])) for side in [o,n]]
            k=min(len(oc),len(nc));a,b=oc.iloc[:k],nc.iloc[:k]
            for col in ['batch_order_seed','dropout_seed','batch_order_sha256']:
                assert a[col].equals(b[col])
            cols=['train_supervised','train_consistency','train_objective','val_equal_mse']
            difference=np.abs(a[cols].to_numpy()-b[cols].to_numpy())
            changed=np.where(np.any(difference!=0,axis=1))[0]
            selection=selections[f"{x['model']}-outer{x['outer_fold']}-P0"]
            fg={v['lambda_value']:v for v in selection['R2']['grid']}
            tg={v['lambda_value']:v for v in selection['grid']}
            comparisons.append(dict(model=x['model'],outer_fold=x['outer_fold'],inner_fold=x['inner_fold'],lambda_value=x['lambda_value'],F_fit=o['fit_id'],T_fit=n['fit_id'],request_matched=True,initial_state_matched=True,scaler_matched=True,environment_matched=True,batch_dropout_streams_matched=True,F_epoch=o['summary']['best_epoch'],T_epoch=n['summary']['best_epoch'],first_divergent_epoch=int(changed[0]+1) if len(changed) else 0,first_divergent_max_abs=float(difference[changed[0]].max()) if len(changed) else 0.,F_score=fg[x['lambda_value']]['score'],T_score=tg[x['lambda_value']]['score']))
    assert len(comparisons)==525
    choices=[]
    for unit,s in selections.items():
        if not unit.endswith('-P0'):continue
        model,rest=unit.split('-outer');outer=int(rest.split('-')[0]);f=s['R2']
        choices.append(dict(model=model,outer_fold=outer,config_id=s['architecture']['selected_config_id'],F_positive=f['selected_lambda'],T_positive=s['selected_lambda'],F_zero=f['zero_eligible_lambda'],T_zero=s['zero_eligible_lambda'],F_score_range=max(x['score'] for x in f['grid'])-min(x['score'] for x in f['grid']),T_score_range=max(x['score'] for x in s['grid'])-min(x['score'] for x in s['grid'])))
    cohorts={'full952':DATA/'Mass_balance_experiment_20261113/data','numerical327':EXP/'data/release/numerical','class494':DATA/'Ionisation_experiment_20261113/data/release/class'}
    metadata=[]
    for cohort,p in cohorts.items():
        df=pd.read_csv(p/'model_dataset.csv').set_index('structure_id')
        inner=pd.read_csv(p/'inner.csv');outer=pd.read_csv(p/'outer.csv')
        for o in range(1,6):
            inn=inner[inner.outer_fold==o]
            for f in [0,1,2,3,4,5]:
                trids=inn.structure_id if f==0 else inn.loc[inn.inner_fold!=f,'structure_id']
                vaids=outer.loc[outer.outer_fold==o,'structure_id'] if f==0 else inn.loc[inn.inner_fold==f,'structure_id']
                tr,va=df.loc[trids],df.loc[vaids]
                row=dict(cohort=cohort,outer_fold=o,inner_fold=f,partition='outer' if f==0 else 'inner',n_train=len(tr),n_validation=len(va))
                for name,col in ENDPOINTS.items():
                    row['validation_n_'+name]=int(va[col].notna().sum())
                    row['training_n_'+name]=int(tr[col].notna().sum())
                    if name in ['t_half','Vd','CL']:
                        row['training_log10_sd_'+name]=float(np.log10(tr[col].dropna()).std(ddof=0))
                metadata.append(row)
    r2=json.loads((EXP/'runtime/references/r2_calibration.json').read_text())
    epochs=[]
    for unit,x in sorted(r2.items()):
        model,rest=unit.split('-outer');fold,arm=rest.split('-');fold=int(fold)
        g={q['lambda_value']:q for q in x['grid']}
        epochs.append(dict(model=model,input=arm,outer_fold=fold,shared_lambda=x['shared_lambda'],separate_positive_lambda=x['selected_lambda'],zero_inclusive_lambda=x['zero_eligible_lambda'],off_epochs=g[0.]['median_epoch'],shared_epochs=g[x['shared_lambda']]['median_epoch'],separate_positive_epochs=g[x['selected_lambda']]['median_epoch'],zero_inclusive_epochs=g[x['zero_eligible_lambda']]['median_epoch']))
    tables={'Supplementary_Data_3_F_policy_epochs.csv':epochs,'Supplementary_Data_4_training_metadata.csv':metadata,'Supplementary_Data_4_matched_calibration.csv':comparisons,'Supplementary_Data_4_selection_comparison.csv':choices}
    checks={}
    for name,rows in tables.items():
        p=output/name;pd.DataFrame(rows).to_csv(p,index=False)
        expected=pd.read_csv(DATA/'supporting_data'/name)
        pd.testing.assert_frame_equal(pd.read_csv(p),expected,check_exact=False,rtol=1e-12,atol=1e-14)
        checks[name]=len(rows)
    result={'status':'PASS','verified_rows':checks,'training_performed':False}
    (output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DATA.parent/'outputs/training-metadata')
    build(parser.parse_args().output)
