"""Offline Python-standard-library reconstruction and independent saved-prediction scoring.

python verify_release.py --root candidate-v0 --output validation-v0.json
Does not train models, render figures, or claim to reproduce pending revision CIs.
"""
import argparse
from collections import defaultdict, Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

MASS='Mass_balance_experiment_20261113'; ION='Ionisation_experiment_20261113'; UNI='Unified_ionisation_combined_20261114'
E={'t_half':'t1/2(hour)','Vd':'VD(liter)','CL':'Cl(liter/hour)','F':'F(percentage)','PPB':'PPB(percentage)','Cmax':'Cmax_uM','Tmax':'Tmax'}
PERCENT={'F','PPB'}
def csvrows(p):
    op=gzip.open if p.suffix=='.gz' else open
    with op(p,'rt',encoding='utf-8-sig',newline='') as f: yield from csv.DictReader(f)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def eq(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-11),(a,b)
def num(x):return float(x) if x!='' else None
def mean(xs):return math.fsum(xs)/len(xs)

def reconstruct(root):
    d=root/MASS/'data'; model={r['structure_id']:r for r in csvrows(d/'model_dataset.csv')}
    original={r['ORIG_INDEX']:r for r in csvrows(d/'reviewed_source_records.csv')}
    masked=0
    for m in csvrows(d/'endpoint_identity_masks.csv'):
        r=original[m['row_idx']]; eq(r[m['endpoint']],m['archived_value']); r[m['endpoint']]=''; masked+=1
    grouped=defaultdict(list)
    for r in original.values():
        if r['canonical_smiles'] and r['identity_exclusion'].lower()!='true' and r['provenance_alias_only'].lower()!='true':
            grouped[r['canonical_smiles']].append(r)
    expected={}; triads=set(); expected_provenance=set()
    for smiles,records in grouped.items():
        sid=hashlib.sha256(smiles.encode()).hexdigest()[:16]; values={}
        for endpoint in E.values():
            vals={float(r[endpoint]) for r in records if r[endpoint]!=''}; assert len(vals)<=1,(sid,endpoint,vals)
            values[endpoint]=next(iter(vals)) if vals else None
            for r in records:
                if r[endpoint]!='': expected_provenance.add((sid,endpoint,r['ORIG_INDEX'],float(r[endpoint])))
        if any(v is not None for v in values.values()):
            expected[sid]=values
            assert sid in model
            assert set(json.loads(model[sid]['source_rows']))=={int(r['ORIG_INDEX']) for r in records}
            assert model[sid]['canonical_smiles']==smiles
            for endpoint,value in values.items():assert value==num(model[sid][endpoint]),(sid,endpoint,value)
            if any(all(r[E[e]]!='' for e in ['t_half','Vd','CL']) for r in records):triads.add(sid)
    assert set(expected)==set(model) and len(model)==952
    actual_provenance={(r['structure_id'],r['endpoint'],r['row_idx'],float(r['value'])) for r in csvrows(d/'endpoint_provenance.csv')}
    assert expected_provenance==actual_provenance
    assert triads=={r['structure_id'] for r in csvrows(d/'same_row_complete_triads.csv')} and len(triads)==674
    assert len({r['group_id'] for r in model.values()})==552
    for rel,n in [(Path(MASS)/'data',952),(Path(ION)/'data/release/numerical',327),(Path(ION)/'data/release/class',494)]:
        cohort={r['structure_id']:r for r in csvrows(root/rel/'model_dataset.csv')};assert len(cohort)==n
        outer=list(csvrows(root/rel/'outer.csv')); assert len(outer)==n
        assert {r['structure_id'] for r in outer}==set(cohort)
        groups=defaultdict(set)
        for r in outer:
            assert r['group_id']==model[r['structure_id']]['group_id'];assert r['outer_fold']==model[r['structure_id']]['outer_fold']
            groups[r['group_id']].add(r['outer_fold'])
        assert all(len(v)==1 for v in groups.values())
        inner=list(csvrows(root/rel/'inner.csv')); assert len(inner)==4*n
        for outer_fold in map(str,range(1,6)):
            subset=[r for r in inner if r['outer_fold']==outer_fold]
            expect={sid for sid,r in cohort.items() if r['outer_fold']!=outer_fold}
            assert len(subset)==len(expect) and {r['structure_id'] for r in subset}==expect
            ig=defaultdict(set)
            for r in subset:ig[r['group_id']].add(r['inner_fold'])
            assert all(len(v)==1 for v in ig.values())
    numerical={r['structure_id']:r for r in csvrows(root/ION/'data/release/numerical/model_dataset.csv')}
    classes={r['structure_id']:r for r in csvrows(root/ION/'data/release/class/model_dataset.csv')}
    assert set(numerical)<=set(classes)<=set(model)
    si2=list(csvrows(root/'output/supporting_data_20261114/Table_S2_archived_pKa.csv'))
    si3=list(csvrows(root/'output/supporting_data_20261114/Table_S3_ionisation_classes.csv'))
    archived={r['ORIG_INDEX']:r for r in csvrows(root/'revision_work/analysis/data/df_ref.csv')}
    assert len(si2)==327 and len(si3)==494
    for r in si2:
        n=numerical[r['structure_id']]
        eq(r['CX_MOST_APKA'],n['pka_acid']);eq(r['CX_MOST_BPKA'],n['pka_base'])
        assert r['ionisation_class']==n['ionisation_class']
        matches=[archived[str(i)] for i in json.loads(r['selected_source_rows'])]
        assert any(x['ChEMBL_ID']==r['archived_chembl_id'] and
                   float(x['pKa_acidic_cx'])==float(r['CX_MOST_APKA']) and
                   float(x['pKa_basic_cx'])==float(r['CX_MOST_BPKA']) for x in matches)
    for r in si3:assert r['ionisation_class']==classes[r['structure_id']]['ionisation_class']
    # Explicit observed masks and membership table must exactly encode source missingness.
    masks=list(csvrows(root/'derived/observation_masks_and_membership.csv'));assert len(masks)==952
    for r in masks:
        sid=r['structure_id']
        for e in E.values():assert int(r[f'observed_{e}'])==int(model[sid][e]!='')
        assert int(r['numerical_327'])==int(sid in numerical);assert int(r['class_494'])==int(sid in classes)
    return model,{'structures':952,'groups':552,'complete_triads':674,'identity_masks':masked,
                  'endpoint_counts':{e:sum(r[e]!='' for r in model.values()) for e in E.values()},
                  'nested_group_splits':'PASS for 952/327/494','exact_S2_S3_values':'PASS','source_reconstruction':'PASS'}

def score_mass(root,model,rows=None):
    acc=defaultdict(list);coverage=defaultdict(set);n=0
    for r in csvrows(root/MASS/'figure_data/primary_outer_predictions.csv.gz'):
        key=(r['backbone'],r['policy'],r['seed']);sid=r['structure_id']; assert sid not in coverage[key]; coverage[key].add(sid); n+=1
        assert r['group_id']==model[sid]['group_id'] and r['outer_fold']==model[sid]['outer_fold']
        pred={}; errors={}
        for e,col in E.items():
            suffix='percent' if e in PERCENT else 'log10'; pred[e]=float(r[f'pred_{e}_{suffix}']);assert math.isfinite(pred[e])
            if model[sid][col]!='':
                truth=float(model[sid][col]);truth=truth if e in PERCENT else math.log10(truth)
                eq(truth,r[f'observed_endpoint_{e}_{suffix}'])
                err=abs(pred[e]-truth);errors[e]=err
                domain=f'{e}_percentage_point' if e in PERCENT else f'{e}_log10'
                acc[key+(domain,'mae')].append(err)
        residual=pred['t_half']-pred['Vd']+pred['CL']-math.log10(math.log(2));eq(residual,r['signed_residual_log10'])
        acc[key+('coherence_all_structures','within_1_25')].append(float(abs(residual)<=math.log10(1.25)))
        if r['same_row_triad_eligible']=='1':acc[key+('same_row_triad_log10','mae')].append(mean([errors[e] for e in ['t_half','Vd','CL']]))
    assert len(coverage)==189 and all(ids==set(model) for ids in coverage.values())
    verified=0
    for r in csvrows(root/MASS/'results/summary/seed_metrics.csv'):
        k=(r['backbone'],r['policy'],r['seed'],r['domain'],r['metric'])
        if k in acc:
            value=mean(acc[k]);eq(value,r['value']);verified+=1
            if rows is not None: rows.append(dict(family='mass' if len(k)==5 else 'unified', key='|'.join(k), n=len(acc[k]), computed=value, archived=float(r['value']), difference=value-float(r['value'])))
    assert verified==len(acc)
    return {'prediction_rows':n,'complete_policy_seed_sets':len(coverage),'archived_metrics_verified':verified,'maximum_tolerance':1e-11}

def score_unified(root,model,rows=None):
    acc=defaultdict(list);coverage=defaultdict(set);refs={};n=0;fit_ids=set()
    for r in csvrows(root/UNI/'outputs/export/combined/final_predictions.csv'):
        key=(r['backbone'],r['cell'],r['seed']);sid=r['structure_id'];assert sid not in coverage[key];coverage[key].add(sid);n+=1;fit_ids.add(r['fit_id'])
        p={e:float(r[f'q_{e}']) for e in E};assert all(math.isfinite(v) for v in p.values())
        if r['cell'] in ['00','10']:refs[(r['backbone'],r['seed'],sid,'P0' if r['cell']=='00' else 'P1')]=(p,r['fit_id'])
        errs={}
        for e,col in E.items():
            assert int(r[f'observed_{e}'])==int(model[sid][col]!='')
            if model[sid][col]!='':
                truth=float(model[sid][col]);truth=truth if e in PERCENT else math.log10(truth);eq(truth,r[f'true_{e}'])
                errs[e]=abs(p[e]-truth);acc[key+(f'{e}_mae',)].append(errs[e])
        residual=p['t_half']-p['Vd']+p['CL']-math.log10(math.log(2))
        acc[key+('mean_abs_log10_residual',)].append(abs(residual))
        if r['same_row_triad_eligible']=='1':acc[key+('triad_macro_log10_mae',)].append(mean([errs[e] for e in ['t_half','Vd','CL']]))
    assert len(coverage)==108 and all(len(v)==327 for v in coverage.values())
    verified=0
    for r in csvrows(root/UNI/'reporting/output/scores/seed_metrics.csv'):
        if r.get('ionisation_class'):continue
        k=(r['backbone'],r['cell'],r['seed'],r['metric'])
        if k in acc:
            value=mean(acc[k]);eq(value,r['value']);verified+=1
            if rows is not None: rows.append(dict(family='mass' if len(k)==5 else 'unified', key='|'.join(k), n=len(acc[k]), computed=value, archived=float(r['value']), difference=value-float(r['value'])))
    assert verified==len(acc),(verified,len(acc))
    reference_count=0
    for r in csvrows(root/UNI/'outputs/export/ionisation/outer_predictions.csv'):
        k=(r['backbone'],r['seed'],r['structure_id'],r['arm']);p,fit=refs[k]
        assert r['source_fit_id']==fit
        for e in E:eq(p[e],r[f'physical_{e}' if e in PERCENT else f'pred_{e}'])
        reference_count+=1
    fit_metadata={json.loads(line)['fit_id'] for line in gzip.open(root/'fit_metadata.jsonl.gz','rt',encoding='utf-8')}
    assert fit_ids<=fit_metadata
    return {'prediction_rows':n,'complete_arm_seed_sets':len(coverage),'archived_metrics_verified':verified,
            'shared_reference_rows_verified':reference_count,'fit_ids_with_request_epoch_config_metadata':len(fit_ids)}

def score_class(root,model,rows=None):
    acc=defaultdict(list);coverage=defaultdict(set);n=0
    for r in csvrows(root/ION/'outputs/outer_predictions.csv'):
        if r['cohort']!='class':continue
        key=(r['procedure'],r['backbone'],r['arm'],r['seed']);sid=r['structure_id'];assert sid not in coverage[key];coverage[key].add(sid);n+=1
        for e,col in E.items():
            if model[sid][col]!='':
                t=float(model[sid][col]);t=t if e in PERCENT else math.log10(t)
                pred=float(r[f'physical_{e}' if e in PERCENT else f'pred_{e}']);acc[key+(e,)].append(abs(pred-t))
    verified=0
    for r in csvrows(root/ION/'outputs/scoring/endpoint_metrics.csv'):
        if r['cohort']!='class' or r['procedure']=='control':continue
        k=(r['procedure'],r['backbone'],r['arm'],r['seed'],r['endpoint']);value=mean(acc[k]);eq(value,r['mae']);verified+=1
        if rows is not None: rows.append(dict(family='class', key='|'.join(k), n=len(acc[k]), computed=value, archived=float(r['mae']), difference=value-float(r['mae'])))
    assert verified==len(acc) and all(len(v)==494 for v in coverage.values())
    return {'prediction_rows':n,'complete_arm_seed_sets':len(coverage),'archived_MAEs_verified':verified}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    root=a.root.resolve();manifest=json.loads((root/'manifest.json').read_text())
    for p,h in manifest['checksums'].items():assert sha(root/p)==h,p
    model,data=reconstruct(root)
    report={'status':'PASS','release_version':manifest['release_version'],'python':sys.version,
            'checksum_files_verified':len(manifest['checksums']),'data':data,'mass_scoring':score_mass(root,model),
            'unified_scoring':score_unified(root,model),'class_scoring':score_class(root,model),
            'scope':'Offline data reconstruction and independent historical point-metric scoring only. No training or pending revision interval/figure reproduction claimed.'}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
