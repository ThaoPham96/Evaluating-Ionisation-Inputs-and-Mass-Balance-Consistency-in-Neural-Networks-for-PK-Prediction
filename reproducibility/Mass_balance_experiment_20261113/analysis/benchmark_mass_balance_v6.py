"""Representative v6 fit throughput; all outputs isolated from campaign fits."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json
import multiprocessing
import time
import pandas as pd


def execute(payload):
    from mass_balance_train import run_fit
    requests, release, features, out, device = payload
    results=[]
    for request in requests:
        start=time.perf_counter()
        result=run_fit(request,release,features,Path(out)/request['fit_id'],device=device)
        results.append({'fit_id':request['fit_id'],'backbone':request['backbone'],'seconds':time.perf_counter()-start,'summary':result})
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--concurrency',type=int,nargs='+',default=[1,2,4])
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--device',default='cuda')
    p.add_argument('--backbones',nargs='+',default=['GraphConv','ChemBERTa_head','Chemprop'])
    p.add_argument('--early-stopping',action='store_true')
    args=p.parse_args()
    inner=pd.read_csv(args.release/'inner.csv',dtype={'structure_id':str})
    split=inner[inner.outer_fold.eq(1)]
    train=split.loc[split.inner_fold.ne(1),'structure_id'].tolist()
    valid=split.loc[split.inner_fold.eq(1),'structure_id'].tolist()
    configs=pd.read_csv(args.plan)
    base=[]
    for backbone in args.backbones:
        candidates=configs[configs.backbone.eq(backbone)].sort_values(['hidden_dim','candidate_id'])
        for size,row in [('small',candidates.iloc[0]),('large',candidates.iloc[-1])]:
            cfg={k:(int(row[k]) if k in ['hidden_dim','depth','ffn_layers'] else float(row[k])) for k in ['hidden_dim','depth','ffn_layers','dropout','lr','weight_decay'] if pd.notna(row[k])}
            for penalty in [0.0,1.0]:
                base.append(dict(fit_id=f'{backbone}_{size}_lambda{penalty:g}',backbone=backbone,config=cfg,train_ids=train,val_ids=valid,predict_ids=valid,seed=101,lambda_value=penalty,endpoint=None,fixed_epochs=None if args.early_stopping else args.epochs,max_epochs=200))
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for concurrency in args.concurrency:
        out=args.output/f'concurrency_{concurrency}'
        if out.exists(): raise FileExistsError(f'Fresh benchmark output required: {out}')
        groups=[base[i::concurrency] for i in range(concurrency)]
        payloads=[(g,str(args.release),str(args.features),str(out),args.device) for g in groups if g]
        start=time.perf_counter()
        with ProcessPoolExecutor(max_workers=concurrency,mp_context=multiprocessing.get_context('spawn')) as pool:
            fits=[fit for batch in pool.map(execute,payloads) for fit in batch]
        seconds=time.perf_counter()-start
        entry={'concurrency':concurrency,'wall_seconds_including_process_startup':seconds,'completed_fits':len(fits),'fits_per_hour_including_startup':len(fits)*3600/seconds,'fits':fits}
        results.append(entry)
        (args.output/'timing.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in entry.items() if k!='fits'}),flush=True)


if __name__=='__main__': main()
