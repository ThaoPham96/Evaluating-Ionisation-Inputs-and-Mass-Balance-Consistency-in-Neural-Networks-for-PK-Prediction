"""Immutable ionisation campaign plans and resumable local/remote work units.

One unit = cohort x backbone x outer fold. Inner fits never see outer labels.
Units are independent and can be distributed over hosts without a shared disk.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import time

BACKBONES = ('GraphConv', 'Chemprop', 'ChemBERTa_head')
ENDPOINTS = ('t_half', 'Vd', 'CL', 'F', 'PPB', 'Cmax', 'Tmax')
SEEDS = (101,202,303,404,505,606,707,808,909)
ARMS = {'class': ('C0','C1','C2','C3'), 'numerical': ('P0','P1','P2','P3','P4','P5')}
RETUNED = {'class': ('C2',), 'numerical': ('P3','P4')}
SEARCH_SEED = 42


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def code_identity(root):
    analysis=Path(root)/'analysis'
    return {str(p.relative_to(root)).replace('\\','/'):sha(p)
            for p in sorted(analysis.rglob('*.py')) if 'tests' not in p.parts}


def verify_release(root,cohort):
    root=Path(root);base=root/'data/release'/cohort;features=root/'features'/cohort
    from ionisation_data import read_release
    _, meta = read_release(base,cohort)
    if any(meta.get(k) is not True for k in ('data_frozen','splits_frozen','training_ready')):
        raise ValueError(f'{cohort}: chemical/data gates are not complete')
    if meta.get('cohort')!=cohort:raise ValueError('Cohort manifest mismatch')
    for name,h in meta['files'].items():
        p=(base/name).resolve()
        if not p.is_relative_to(base.resolve()) or sha(p)!=h:raise ValueError(f'Changed release {name}')
    fm=read(features/'manifest.json')
    if not {'structures.json','graphs.pt','chemberta.npy'} <= set(fm['files']):raise ValueError('Required feature checksums absent')
    if fm['data']['model_dataset.csv']!=sha(base/'model_dataset.csv') or fm['data']['manifest.json']!=sha(base/'manifest.json'):
        raise ValueError('Feature bundle does not match cohort')
    if fm['data'].get('fingerprint')!=meta['fingerprint']:raise ValueError('Feature fingerprint mismatch')
    for name,h in fm['files'].items():
        p=(features/name).resolve()
        if not p.is_relative_to(features.resolve()) or sha(p)!=h:raise ValueError(f'Changed features {name}')
    dataset=rows(base/'model_dataset.csv');ids=[x['structure_id'] for x in dataset]
    outer=rows(base/'outer.csv');inner=rows(base/'inner.csv')
    if len(ids)!=len(set(ids)) or read(features/'structures.json')!=ids:raise ValueError('Feature IDs/order mismatch')
    if len(outer)!=len(ids) or {r['structure_id'] for r in outer}!=set(ids):raise ValueError('Outer coverage mismatch')
    om={r['structure_id']:int(r['outer_fold']) for r in outer}
    groups={r['structure_id']:r['group_id'] for r in outer}
    if set(om.values())!={1,2,3,4,5}:raise ValueError('Five outer folds required')
    for group in set(groups.values()):
        if len({om[s] for s in ids if groups[s]==group})!=1:raise ValueError('Outer group leakage')
    if {int(r['outer_fold']) for r in inner}!={1,2,3,4,5}:raise ValueError('Unexpected inner outer-fold index')
    im={}
    for fold in range(1,6):
        part=[r for r in inner if int(r['outer_fold'])==fold]
        im[fold]={r['structure_id']:int(r['inner_fold']) for r in part}
        if len(part)!=len(im[fold]) or set(im[fold])!={s for s in ids if om[s]!=fold}:raise ValueError('Inner coverage/leakage')
        if set(im[fold].values())!={1,2,3,4,5}:raise ValueError('Five inner folds required')
        for group in set(groups[s] for s in im[fold]):
            if len({im[fold][s] for s in im[fold] if groups[s]==group})!=1:raise ValueError('Inner group leakage')
    return {'release':sha(base/'manifest.json'),'features':sha(features/'manifest.json'),
            'fingerprint':meta['fingerprint'],'structures':len(ids)},om,im


def build_plan(root,cohorts=('class','numerical'),include_tuned=True):
    root=Path(root)
    cfg=read(root/'config/search_grid.json')
    if set(cfg)!=set(BACKBONES) or any(len(v)!=10 for v in cfg.values()):raise ValueError('Ten configurations per backbone required')
    for b,items in cfg.items():
        if len({digest(c) for c in items})!=10:raise ValueError(f'Duplicate configurations in {b}')
    datasets={c:verify_release(root,c)[0] for c in cohorts}
    units=[]
    for cohort in cohorts:
        for backbone in BACKBONES:
            for outer in range(1,6):
                units.append({'unit_id':f'{cohort}-{backbone}-outer{outer}', 'cohort':cohort,'backbone':backbone,
                              'outer_fold':outer,'new_fits_upper_bound':(155 if include_tuned else 101) if cohort=='class' else (237 if include_tuned else 129)})
    # Heavier numerical/Chemprop work starts first; later units are taken dynamically.
    units.sort(key=lambda u:(u['cohort']!='numerical',u['backbone']!='Chemprop',u['backbone']=='ChemBERTa_head',u['outer_fold']))
    identity={'protocol':'ionisation_v2','datasets':datasets,'code':code_identity(root),'grid':cfg,
              'seeds':list(SEEDS),'search_seed':SEARCH_SEED,'include_tuned':include_tuned,
              'max_epochs':200,'patience':30,'endpoint_weights':[1]*7,'lambda':0,'units':units}
    return {'campaign_id':digest(identity),'identity':identity,'training_ready':True,
            'new_fits_upper_bound':sum(u['new_fits_upper_bound'] for u in units),'units':units}


def verify_plan(root,plan):
    ident=plan['identity']
    if plan['campaign_id']!=digest(ident) or plan.get('training_ready') is not True:raise ValueError('Invalid campaign identity')
    if plan['units']!=ident['units']:raise ValueError('Top-level units differ from immutable identity')
    if ident['seeds']!=list(SEEDS) or ident['search_seed']!=SEARCH_SEED or ident['max_epochs']!=200:raise ValueError('Unsupported frozen training specification')
    if code_identity(root)!=ident['code']:raise ValueError('Code changed after planning; create a new campaign')
    for cohort,expected in ident['datasets'].items():
        if verify_release(root,cohort)[0]!=expected:raise ValueError('Data/features changed after planning')


def choose(summaries):
    """Pool per-endpoint inner squared-error sums/counts, then average endpoints."""
    scores={}
    for candidate,items in summaries.items():
        if len(items)!=5:raise ValueError('Five inner summaries required')
        sums={e:sum(float(s['val_squared_error_sums'][e]) for s in items) for e in ENDPOINTS}
        counts={e:sum(int(s['val_observed_counts'][e]) for s in items) for e in ENDPOINTS}
        if min(counts.values())<=0:raise ValueError('Missing validation endpoint')
        scores[candidate]=sum(sums[e]/counts[e] for e in ENDPOINTS)/7
    if not scores or not all(__import__('math').isfinite(x) for x in scores.values()):raise ValueError('Invalid selection scores')
    best=min(scores,key=lambda c:(scores[c],c))
    epochs=int(statistics.median(int(s['best_epoch']) for s in summaries[best]))
    return best,epochs,scores


def fit_request(plan,unit,arm,config,seed,outer_map,inner_map,inner=None,epochs=None):
    fold=unit['outer_fold'];part=inner_map[fold]
    request={'campaign_id':plan['campaign_id'],'cohort':unit['cohort'],'arm':arm,'backbone':unit['backbone'],
             'outer_fold':fold,'inner_fold':inner,'config':config,'seed':seed,'max_epochs':200,
             'fixed_epochs':epochs,'lambda_value':0.0,'task_weights':[1.0]*7,
             'train_ids':sorted(s for s in part if inner is None or part[s]!=inner),
             'val_ids':sorted(s for s in part if inner is not None and part[s]==inner),
             'predict_ids':sorted(s for s in part if part[s]==inner) if inner is not None else sorted(s for s in outer_map if outer_map[s]==fold)}
    request['fit_id']='ION-'+digest(request)[:24]
    return request


def _execute(args):
    request,release,features,output,device=args
    from ionisation_train import run_fit
    return run_fit(request,release,features,output,device)


def run_unit(root,plan,unit_id,output,workers=1,device='cuda'):
    root=Path(root).resolve();output=Path(output).resolve()
    if workers<1 or workers>8:raise ValueError('Workers must be 1..8; use a pilot to choose concurrency')
    verify_plan(root,plan)
    matching=[u for u in plan['units'] if u['unit_id']==unit_id]
    if len(matching)!=1:raise ValueError('Unknown unit')
    unit=matching[0];cohort=unit['cohort'];base=output/unit_id;base.mkdir(parents=True,exist_ok=True)
    from _vendor import ensure_vendor
    ensure_vendor()
    from mass_balance_train import _fit_lock
    with _fit_lock(base):
        started=time.monotonic();_,outer,inner=verify_release(root,cohort)
        arms=ARMS[cohort];anchor=arms[0];grid=plan['identity']['grid'][unit['backbone']]
        records=[];summaries={};requests={}
        def execute(reqs,pool):
            unique={r['fit_id']:r for r in reqs}
            args=[(r,str(root/'data/release'/cohort),str(root/'features'/cohort),str(base/'fits'/fid),device) for fid,r in unique.items()]
            results=map(_execute,args) if pool is None else pool.map(_execute,args,chunksize=1)
            for (fid,req),result in zip(unique.items(),results):
                summaries[fid]=result;requests[fid]=req
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) if workers>1 else None
        try:
            tuned_arms=(anchor,)+ (RETUNED[cohort] if plan['identity']['include_tuned'] else ())
            search={}
            reqs=[]
            for arm in tuned_arms:
                search[arm]={}
                for ci,config in enumerate(grid):
                    req=[fit_request(plan,unit,arm,config,SEARCH_SEED,outer,inner,j) for j in range(1,6)]
                    search[arm][ci]=[r['fit_id'] for r in req];reqs.extend(req)
            execute(reqs,pool)
            selected={}
            for arm in tuned_arms:
                ci,epochs,scores=choose({c:[summaries[fid] for fid in ids] for c,ids in search[arm].items()})
                selected[arm]={'candidate_index':ci,'epochs':epochs,'scores':scores}
            anchor_ci=selected[anchor]['candidate_index'];calibrations={};reqs=[]
            for arm in arms:
                req=[fit_request(plan,unit,arm,grid[anchor_ci],SEARCH_SEED,outer,inner,j) for j in range(1,6)]
                calibrations[arm]=[r['fit_id'] for r in req]
                reqs.extend(r for r in req if r['fit_id'] not in summaries)
            execute(reqs,pool)
            reqs=[]
            for arm in arms:
                epochs=int(statistics.median(summaries[fid]['best_epoch'] for fid in calibrations[arm]))
                for seed in SEEDS:
                    req=fit_request(plan,unit,arm,grid[anchor_ci],seed,outer,inner,epochs=epochs)
                    reqs.append(req);records.append({'procedure':'primary','arm':arm,'seed':seed,'fit_id':req['fit_id'],'candidate_index':anchor_ci,'epochs':epochs})
            if plan['identity']['include_tuned']:
                for arm in RETUNED[cohort]:
                    sel=selected[arm]
                    for seed in SEEDS:
                        req=fit_request(plan,unit,arm,grid[sel['candidate_index']],seed,outer,inner,epochs=sel['epochs'])
                        reqs.append(req);records.append({'procedure':'tuned','arm':arm,'seed':seed,'fit_id':req['fit_id'],'candidate_index':sel['candidate_index'],'epochs':sel['epochs']})
            write(base/'selection.json',{'campaign_id':plan['campaign_id'],'unit':unit,'selected':selected,
                  'primary_anchor_index':anchor_ci,'calibration_fit_ids':calibrations,'search_fit_ids':search,'final_requests':records})
            execute(reqs,pool)
        finally:
            if pool:pool.shutdown(wait=True,cancel_futures=True)
        # Full verification before publishing the unit completion marker.
        artifacts={}
        for fid in requests:
            fit=base/'fits'/fid;marker=read(fit/'complete.json')
            if marker['fit_id']!=fid:raise ValueError('Fit marker mismatch')
            for name,h in marker['files'].items():
                if sha(fit/name)!=h:raise ValueError('Fit artifact corruption')
            artifacts[fid]=sha(fit/'complete.json')
        result={'campaign_id':plan['campaign_id'],'unit':unit,'final_records':records,
                'selection_sha256':sha(base/'selection.json'),'fit_markers':artifacts,
                'unique_fits':len(artifacts),'elapsed_seconds':time.monotonic()-started,'workers':workers}
        write(base/'complete_unit.json',result)
        return result


def export(root,plan,output,destination):
    import pandas as pd
    verify_plan(root,plan);frames=[]
    outer_maps={c:verify_release(root,c)[1] for c in plan['identity']['datasets']}
    for unit in plan['units']:
        base=Path(output)/unit['unit_id'];done=read(base/'complete_unit.json')
        if done['campaign_id']!=plan['campaign_id'] or done['unit']!=unit:raise ValueError('Wrong completed unit')
        if sha(base/'selection.json')!=done['selection_sha256']:raise ValueError('Changed selection')
        if done['final_records']!=read(base/'selection.json')['final_requests']:raise ValueError('Final records differ from selection')
        expected_records={('primary',a,s) for a in ARMS[unit['cohort']] for s in SEEDS}
        if plan['identity']['include_tuned']:expected_records|={('tuned',a,s) for a in RETUNED[unit['cohort']] for s in SEEDS}
        actual_records=[(r['procedure'],r['arm'],r['seed']) for r in done['final_records']]
        if len(actual_records)!=len(set(actual_records)) or set(actual_records)!=expected_records:raise ValueError('Incomplete final arm/seed coverage')
        for record in done['final_records']:
            fit=base/'fits'/record['fit_id'];marker=read(fit/'complete.json')
            if sha(fit/'complete.json')!=done['fit_markers'][record['fit_id']]:raise ValueError('Changed fit marker')
            if sha(fit/'predictions.csv')!=marker['files']['predictions.csv']:raise ValueError('Changed predictions')
            if sha(fit/'request.json')!=marker['files']['request.json']:raise ValueError('Changed request')
            frame=pd.read_csv(fit/'predictions.csv',dtype={'structure_id':str})
            request=read(fit/'request.json')['request']
            if (any(request[k]!=unit[k] for k in ('cohort','backbone','outer_fold')) or
                any(request[k]!=record[k] for k in ('arm','seed','fit_id')) or
                request['campaign_id']!=plan['campaign_id']):raise ValueError('Request metadata does not match final record')
            expected={s for s,f in outer_maps[unit['cohort']].items() if f==unit['outer_fold']}
            if set(frame.structure_id)!=expected or set(request['predict_ids'])!=expected or frame.structure_id.duplicated().any():raise ValueError('Prediction coverage mismatch')
            for key in ('cohort','backbone','outer_fold'):frame[key]=unit[key]
            for key in ('procedure','arm','seed'):frame[key]=record[key]
            frames.append(frame)
    combined=pd.concat(frames,ignore_index=True)
    if combined.duplicated(['cohort','backbone','procedure','arm','seed','structure_id']).any():raise ValueError('Duplicate outer predictions')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    combined.to_csv(destination,index=False)
    write(destination.with_suffix('.manifest.json'),{'campaign_id':plan['campaign_id'],'sha256':sha(destination),'rows':len(combined),'units':len(plan['units']),
          'release_fingerprints':{c:m['fingerprint'] for c,m in plan['identity']['datasets'].items()}})
    return len(combined)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    subs=parser.add_subparsers(dest='cmd',required=True)
    p=subs.add_parser('plan');p.add_argument('--out',type=Path,required=True);p.add_argument('--cohorts',default='class,numerical');p.add_argument('--primary-only',action='store_true')
    p=subs.add_parser('run-unit');p.add_argument('--plan',type=Path,required=True);p.add_argument('--unit',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--device',default='cuda')
    p=subs.add_parser('export');p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=subs.add_parser('status');p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.cmd=='plan':
        cohorts=tuple(args.cohorts.split(','))
        if not cohorts or len(set(cohorts))!=len(cohorts) or not set(cohorts)<=set(ARMS):raise ValueError('Invalid cohorts')
        plan=build_plan(args.root,cohorts,not args.primary_only);write(args.out,plan)
        print(json.dumps({'campaign_id':plan['campaign_id'],'units':len(plan['units']),'fits':plan['new_fits_upper_bound']}))
    elif args.cmd=='run-unit':print(json.dumps(run_unit(args.root,read(args.plan),args.unit,args.output,args.workers,args.device)))
    elif args.cmd=='export':print('Exported rows:',export(args.root,read(args.plan),args.output,args.out))
    else:
        plan=read(args.plan);complete=[];invalid=[]
        for u in plan['units']:
            p=args.output/u['unit_id']/'complete_unit.json'
            if p.exists():
                if read(p).get('campaign_id')==plan['campaign_id']:complete.append(u['unit_id'])
                else:invalid.append(u['unit_id'])
        print(json.dumps({'total':len(plan['units']),'complete':len(complete),'invalid':invalid,'remaining':[u['unit_id'] for u in plan['units'] if u['unit_id'] not in complete]},indent=2))


if __name__=='__main__':main()
