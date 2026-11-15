"""Timing-only diagnostic on provisional inputs; never a scientific fit release."""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import time
import numpy as np
import pandas as pd
import torch
from ionisation_campaign import read,write,sha
from _vendor import ensure_vendor
ensure_vendor()


def prepare(root,mass_balance,out):
    candidate=pd.read_csv(root/'data/candidate/candidate_ledger.csv')
    candidate=candidate[candidate.historical_candidate.astype(str).str.lower()=='true']
    src=pd.read_csv(mass_balance/'data/model_dataset.csv')
    all_ids=read(mass_balance/'features/structures.json')
    graphs=torch.load(mass_balance/'features/graphs.pt',map_location='cpu',weights_only=False)
    embeddings=np.load(mass_balance/'features/chemberta.npy',allow_pickle=False)
    for cohort in ('class','numerical'):
        frame=candidate if cohort=='class' else candidate[candidate.historical_two_pkas.astype(str).str.lower()=='true']
        labels=frame[['structure_id','historical_class','historical_pka_acid','historical_pka_base']].rename(columns={
            'historical_class':'ionisation_class','historical_pka_acid':'pka_acid','historical_pka_base':'pka_base'})
        data=src.merge(labels,on='structure_id',how='inner',validate='one_to_one')
        data_dir=out/'data'/cohort;feat=out/'features'/cohort
        data_dir.mkdir(parents=True,exist_ok=True);feat.mkdir(parents=True,exist_ok=True)
        data.to_csv(data_dir/'model_dataset.csv',index=False)
        ids=data.structure_id.tolist();ix=[all_ids.index(s) for s in ids]
        write(feat/'structures.json',ids)
        np.save(feat/'chemberta.npy',embeddings[ix])
        torch.save([graphs[i] for i in ix],feat/'graphs.pt')
    write(out/'PILOT_ONLY.json',{'diagnostic':True,'training_ready':False,
          'warning':'Provisional labels used solely for software/timing checks. No model result may enter scientific analysis.',
          'candidate_sha256':sha(root/'data/candidate/candidate_ledger.csv')})


def execute(job):
    from ionisation_train import run_fit
    req,data,features,out,device=job
    return run_fit(req,data,features,out,device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--mass-balance',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--epochs',type=int,default=10);p.add_argument('--device',default='cuda')
    p.add_argument('--all-configs',action='store_true',help='Time every frozen search configuration; no performance selection')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    prepare(args.root,args.mass_balance,args.out)
    grid=read(args.root/'config/search_grid.json');jobs=[]
    for cohort,arm in [('class','C3'),('numerical','P4')]:
        ids=read(args.out/'features'/cohort/'structures.json')
        # A fixed large training partition approximates final-fit load; targets never select a recipe.
        split=int(len(ids)*0.8);train=ids[:split];test=ids[split:]
        for backbone in grid:
            for ci,config in enumerate(grid[backbone] if args.all_configs else grid[backbone][:1]):
                req={'fit_id':f'pilot-{cohort}-{backbone}-w{args.workers}-e{args.epochs}-c{ci}',
                     'diagnostic':True,'cohort':cohort,'arm':arm,'backbone':backbone,'config':config,
                     'seed':101,'max_epochs':args.epochs,'fixed_epochs':args.epochs,'lambda_value':0,'task_weights':[1]*7,
                     'train_ids':train,'val_ids':[],'predict_ids':test}
                jobs.append((req,args.out/'data'/cohort,args.out/'features'/cohort,args.out/'fits'/req['fit_id'],args.device))
    start=time.monotonic()
    if args.workers==1:results=list(map(execute,jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:results=list(pool.map(execute,jobs))
    report={'diagnostic':True,'scientific_results':False,'workers':args.workers,'wall_seconds':time.monotonic()-start,
            'epochs':args.epochs,'gpu':torch.cuda.get_device_name() if args.device=='cuda' else None,
            'fits':[{'fit_id':s['fit_id'],'backbone':s['backbone'],'cohort':s['cohort'],'parameters':s['parameter_count'],
                     'config':s['config'],'timing':s['timing']} for s in results]}
    write(args.out/'timing.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
