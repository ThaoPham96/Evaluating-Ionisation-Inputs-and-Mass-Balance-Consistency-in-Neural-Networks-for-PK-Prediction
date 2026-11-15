"""Saved-prediction S/H evaluator. No fitting and no modification of source archives."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EPS = ('t_half', 'Vd', 'CL', 'Cmax', 'Tmax', 'F', 'PPB')
RAW = ('t1/2(hour)', 'VD(liter)', 'Cl(liter/hour)', 'Cmax_uM', 'Tmax', 'F(percentage)', 'PPB(percentage)')
SEEDS = tuple(range(101, 910, 101))
K = np.log10(np.log(2))
HASHES = {}

def record(path):
    path = Path(path).resolve()
    with path.open('rb') as f:
        HASHES[str(path)] = hashlib.file_digest(f, 'sha256').hexdigest()
    return path

def read(path):
    return pd.read_csv(record(path), dtype={'structure_id': str, 'group_id': str, 'cell': str}, float_precision='round_trip')

def require(ok, message):
    if not ok:
        raise ValueError(message)

@dataclass
class Cohort:
    name: str
    rows: pd.DataFrame
    truth: np.ndarray
    triads: np.ndarray
    group: np.ndarray

def cohort(name, data, triads):
    rows = read(data / 'model_dataset.csv').set_index('structure_id', verify_integrity=True).sort_index()
    outer = read(data / 'outer.csv').set_index('structure_id', verify_integrity=True).sort_index()
    pd.testing.assert_frame_equal(rows[['group_id','outer_fold']], outer[['group_id','outer_fold']], check_dtype=False)
    require(rows.groupby('group_id').outer_fold.nunique().max() == 1, 'Group crosses outer folds')
    inner = read(data / 'inner.csv')
    for f, d in inner.groupby('outer_fold'):
        expected = rows.index[rows.outer_fold.ne(f)]
        require(len(d) == len(expected) and set(d.structure_id) == set(expected), 'Inner includes outer test or omits training structure')
        require(not d.structure_id.duplicated().any(), 'Duplicate inner assignment')
        require(d.groupby('group_id').inner_fold.nunique().max() == 1, 'Group crosses inner folds')
        require(np.array_equal(d.group_id, d.structure_id.map(rows.group_id)), 'Inner group mismatch')
    tr = read(triads).set_index('structure_id', verify_integrity=True)
    require(tr.index.isin(rows.index).all(), 'Triads outside cohort')
    np.testing.assert_allclose(tr[list(RAW[:3])], rows.loc[tr.index,list(RAW[:3])], rtol=0, atol=0)
    y = rows[list(RAW)].to_numpy(float)
    require(not np.isinf(y).any() and not (y[:,:5] <= 0).any(), 'Invalid observed target domain')
    y[:,:5] = np.log10(y[:,:5])
    _, g = np.unique(rows.group_id, return_inverse=True)
    return Cohort(name, rows, y, rows.index.isin(tr.index), g)

class Bootstrap:
    def __init__(self, c, seeds, draws=10000, seed=None):
        self.c, self.seeds, self.n = c, tuple(seeds), draws
        self.ng = int(c.group.max()) + 1
        # Stable across model families and seed counts for a given cohort.
        seed = seed if seed is not None else (20261112 if c.name=='full952' else 20261113)
        rng = np.random.Generator(np.random.PCG64(seed))
        if c.name=='full952':
            ids = rng.integers(self.ng, size=(draws,self.ng), dtype=np.int32)
            self.counts = np.zeros((draws,self.ng), dtype=float)
            np.add.at(self.counts, (np.arange(draws)[:,None], ids), 1)
        else:
            self.counts = rng.multinomial(self.ng,np.full(self.ng,1/self.ng),size=draws).astype(float)
        self.seed_counts = rng.multinomial(len(seeds), np.full(len(seeds),1/len(seeds)),size=draws)

    def means(self, values, mask):
        require(values.shape == (len(self.seeds),len(self.c.rows)), 'Array alignment/seed mismatch')
        require(np.isfinite(values[:,mask]).all(), 'Nonfinite scored values')
        require(mask.any(), 'Empty metric mask')
        sums = np.zeros((self.ng, len(self.seeds)))
        np.add.at(sums, self.c.group[mask], values[:,mask].T)
        counts = np.bincount(self.c.group[mask], minlength=self.ng)
        denom = self.counts @ counts
        require((denom > 0).all(), 'Bootstrap draw has no eligible observations')
        return values[:,mask].mean(axis=1), (self.counts @ sums)/denom[:,None]

def hard_derive(q, endpoint):
    p = q.copy()
    if endpoint == 't_half': p[:,:,0] = K+p[:,:,1]-p[:,:,2]
    elif endpoint == 'Vd': p[:,:,1] = p[:,:,0]+p[:,:,2]-K
    elif endpoint == 'CL': p[:,:,2] = K+p[:,:,1]-p[:,:,0]
    else: raise ValueError(endpoint)
    np.testing.assert_allclose(p[:,:,0]-p[:,:,1]+p[:,:,2]-K, 0, atol=2e-14)
    return p

def metrics(q,c):
    errors = np.abs(q-c.truth)
    residual = np.abs(q[:,:,0]-q[:,:,1]+q[:,:,2]-K)
    observed_r = np.abs(c.truth[:,0]-c.truth[:,1]+c.truth[:,2]-K)
    strata = {'all':np.ones(len(c.rows),bool), 'complete_triad':c.triads,
              'observed_le_1.5':c.triads & (observed_r<=np.log10(1.5)),
              'observed_gt_1.5':c.triads & (observed_r>np.log10(1.5))}
    out = {}
    for name,mask in strata.items():
        # Full seven-endpoint profiles in every prespecified stratum.
        for j,ep in enumerate(EPS):
            out[(name,ep)] = (errors[:,:,j], mask & np.isfinite(c.truth[:,j]), 'percentage_points' if j>=5 else 'log10')
        out[(name,'triad_mae')] = (errors[:,:,:3].mean(axis=2),mask & c.triads,'log10')
        out[(name,'mean_abs_residual')] = (residual,mask,'log10')
        out[(name,'within_1.25')] = (100*(residual<=np.log10(1.25)),mask,'percentage_points')
    return out

def bounds(x):
    require(np.isfinite(x).all(), 'Nonfinite bootstrap distribution')
    return np.quantile(x,[.025,.975],method='linear')

def direction(lo,hi):
    return 'negative' if hi<0 else 'positive' if lo>0 else 'spans_zero'

class Evaluator:
    def __init__(self, draws=10000):
        self.draws=draws; self.contrasts=[]; self.summary=[]; self.seed_rows=[]; self.scatter=[]; self.audit=[]

    def suite(self,c,model,protocol,arms,seeds,comparisons,hard=False,legacy='group_recomputed'):
        boot=Bootstrap(c,seeds,self.draws)
        stats={}
        for arm,q in arms.items():
            require(q.shape==(len(seeds),len(c.rows),7) and np.isfinite(q).all(), f'Incomplete predictions: {arm}')
            stats[arm]={}
            for (stratum,metric),(v,mask,unit) in metrics(q,c).items():
                p,d=boot.means(v,mask)
                stats[arm][(stratum,metric)]=(p,d,int(mask.sum()),unit)
                lo,hi=bounds(d.mean(axis=1))
                self.summary.append(dict(cohort=c.name,protocol=protocol,model=model,condition=arm,stratum=stratum,metric=metric,unit=unit,n=int(mask.sum()),n_seeds=len(seeds),mean=p.mean(),seed_sd=p.std(ddof=1),primary_low=lo,primary_high=hi,geometric_mean_fold=10**p.mean() if metric=='mean_abs_residual' else np.nan))
                for seed,value in zip(seeds,p):
                    self.seed_rows.append(dict(cohort=c.name,protocol=protocol,model=model,condition=arm,stratum=stratum,metric=metric,seed=seed,value=value))
            for j,ep in enumerate(EPS):
                mask=np.isfinite(c.truth[:,j]); y=c.truth[mask,j]; pred=q[:,mask,j]; ensemble=pred.mean(axis=0)
                self.scatter.append(dict(cohort=c.name,protocol=protocol,model=model,condition=arm,endpoint=ep,n=len(y),n_seeds=len(seeds),individual_seed_mae_mean=np.abs(pred-y).mean(),ensemble_mae=np.abs(ensemble-y).mean(),ensemble_predictive_r2=1-((ensemble-y)**2).sum()/((y-y.mean())**2).sum()))
        for label,terms,denominator in comparisons:
            self.contrast(c,model,protocol,label,terms,denominator,stats,boot,legacy)
        if hard:
            for arm,q in list(arms.items()):
                for endpoint in EPS[:3]:
                    derived=hard_derive(q,endpoint); h=f'{arm}__derive_{endpoint}'
                    hs={}
                    for key,(v,mask,unit) in metrics(derived,c).items():
                        p,d=boot.means(v,mask); hs[key]=(p,d,int(mask.sum()),unit)
                    stats[h]=hs
                    self.contrast(c,model,protocol,f'H:{h}-original',{h:1,arm:-1},arm,stats,boot,legacy,only_triads=True)
                    del stats[h]

    def contrast(self,c,model,protocol,label,terms,denominator,stats,boot,legacy,only_triads=False):
        rels={}
        for key in stats[denominator]:
            stratum,metric=key
            if only_triads and metric not in (*EPS[:3],'triad_mae','mean_abs_residual','within_1.25'): continue
            p=sum(w*stats[a][key][0] for a,w in terms.items())
            d=sum(w*stats[a][key][1] for a,w in terms.items())
            bp,bd,n,unit=stats[denominator][key]
            meta=dict(cohort=c.name,protocol=protocol,model=model,contrast=label,terms=json.dumps(terms,sort_keys=True),denominator=denominator,stratum=stratum,metric=metric,unit=unit,n=n,n_groups=int(np.unique(c.group[metrics_mask(c,key)]).size),n_seeds=len(boot.seeds),seeds=','.join(map(str,boot.seeds)),denominator_mean=bp.mean(),absolute_delta=p.mean(),absolute_seed_sd=p.std(ddof=1))
            lo,hi=bounds(d.mean(axis=1)); meta.update(absolute_low=lo,absolute_high=hi)
            crossed=(d*boot.seed_counts).sum(axis=1)/len(boot.seeds)
            if metric in (*EPS,'triad_mae'):
                require(bp.mean()>0 and (bd.mean(axis=1)>0).all(),'Nonpositive accuracy denominator')
                point=100*p/bp.mean(); primary=100*d.mean(axis=1)/bd.mean(axis=1)
                fixed=100*d.mean(axis=1)/bp.mean(); cross=100*crossed/bp.mean()
                rels[key]=(point,primary,fixed,cross)
                low,high=bounds(primary); fl,fh=bounds(fixed); cl,ch=bounds(cross)
                old=(cl,ch) if legacy=='crossed_fixed' else (fl,fh) if legacy=='group_fixed' else (low,high)
                meta.update(relative_pct=point.mean(),relative_seed_sd=point.std(ddof=1),primary_low=low,primary_high=high,fixed_group_low=fl,fixed_group_high=fh,crossed_fixed_low=cl,crossed_fixed_high=ch,legacy_method=legacy,legacy_low=old[0],legacy_high=old[1],primary_direction=direction(low,high),legacy_direction=direction(*old),interval_direction_changed=direction(low,high)!=direction(*old))
            self.contrasts.append(meta)
        if not only_triads:
            for stratum in ('all','complete_triad','observed_le_1.5','observed_gt_1.5'):
                arrays=[rels[(stratum,ep)] for ep in EPS]
                point,primary,fixed,cross=[np.mean([a[j] for a in arrays],axis=0) for j in range(4)]
                low,high=bounds(primary); fl,fh=bounds(fixed); cl,ch=bounds(cross)
                old=(cl,ch) if legacy=='crossed_fixed' else (fl,fh) if legacy=='group_fixed' else (low,high)
                self.contrasts.append(dict(cohort=c.name,protocol=protocol,model=model,contrast=label,terms=json.dumps(terms,sort_keys=True),denominator=denominator,stratum=stratum,metric='seven_endpoint_average',unit='mean_endpoint_relative_pct',n_seeds=len(boot.seeds),seeds=','.join(map(str,boot.seeds)),relative_pct=point.mean(),relative_seed_sd=point.std(ddof=1),primary_low=low,primary_high=high,fixed_group_low=fl,fixed_group_high=fh,crossed_fixed_low=cl,crossed_fixed_high=ch,legacy_method=legacy,legacy_low=old[0],legacy_high=old[1],primary_direction=direction(low,high),legacy_direction=direction(*old),interval_direction_changed=direction(low,high)!=direction(*old)))

def metrics_mask(c,key):
    s,m=key; r=np.abs(c.truth[:,0]-c.truth[:,1]+c.truth[:,2]-K)
    mask=np.ones(len(c.rows),bool) if s=='all' else c.triads.copy()
    if s=='observed_le_1.5': mask &= r<=np.log10(1.5)
    if s=='observed_gt_1.5': mask &= r>np.log10(1.5)
    if m in EPS: mask &= np.isfinite(c.truth[:,EPS.index(m)])
    if m=='triad_mae': mask &= c.triads
    return mask

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--draws',type=int,default=10000); parser.add_argument('--skip-supplementary',action='store_true'); parser.add_argument('--new-export',type=Path); parser.add_argument('--output',type=Path,default=HERE/'outputs'); args=parser.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    ev=Evaluator(args.draws)
    with threadpool_limits(limits=1):
        from sources import existing, new_export
        existing(ev,args.skip_supplementary)
        if args.new_export: new_export(ev,args.new_export)
    for row in ev.contrasts:
        row['denominator_condition']=row['denominator']
        common_scale='structure_scale' in row['contrast'] or 'interaction' in row['contrast']
        row['relative_unit']='percentage_point_difference_on_structure_only_MAE_scale' if common_scale else 'percent_change_vs_comparator_MAE'
        row['scale']='common_structure_only' if common_scale else 'comparator_specific'
        if row['metric']=='seven_endpoint_average': row['relative_unit']='equal_endpoint_mean_of_'+row['relative_unit']
        if row['metric'] not in (*EPS,'triad_mae','seven_endpoint_average'):
            row['relative_unit']='not_applicable'; row['scale']='not_applicable'
    tables={'contrasts':ev.contrasts,'condition_metrics':ev.summary,'individual_seed_metrics':ev.seed_rows,'ensemble_scatter_metrics':ev.scatter}
    for name,rows in tables.items(): pd.DataFrame(rows).to_csv(args.output/f'{name}.csv',index=False)
    contrasts=pd.DataFrame(ev.contrasts)
    changes=contrasts.loc[contrasts.interval_direction_changed.eq(True)]
    changes.to_csv(args.output/'interval_direction_changes.csv',index=False)
    claim=contrasts.loc[contrasts.stratum.isin(['all','observed_le_1.5','observed_gt_1.5'])].copy()
    claim.to_csv(args.output/'machine_claim_table.csv',index=False)
    for path in HERE.glob('*.py'): record(path)
    manifest=dict(bootstrap_draws=args.draws,bootstrap_seeds={'full952':20261112,'unified327':20261113,'class494':20261113},rng='numpy PCG64; full952 integer draws, other cohorts multinomial counts; sorted groups; same RNG continues for crossed seed counts',primary='Fixed fitted seeds averaged; paired whole-group resampling; comparator mean MAE recomputed in every draw. Ratio of seed-mean MAEs, preserving original point/seed estimands. Equal seven-endpoint mean of per-endpoint relative changes within each draw.',sensitivity='Whole fitted seed trajectories jointly resampled using independent multinomial seed counts; fixed observed mean denominator. Also fixed denominator with groups only. Original archive bounds retained separately when available.',interval='Nominal exploratory equal-tailed percentile 95%, conditional on fitted models/data/folds/selection. Does not refit models or reproduce training/selection uncertainty.',hard='Three prespecified directions, log10; post-processing original held-out predictions, not trained hard models. Exact consistency is algebraically guaranteed.',audit=ev.audit,source_sha256=HASHES,n_contrasts=len(contrasts),n_interval_direction_changes=len(changes),python=sys.version,numpy=np.__version__,pandas=pd.__version__)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'contrasts':len(contrasts),'interval_direction_changes':len(changes),'output':str(args.output)}))

if __name__=='__main__':
    sys.modules['evaluate']=sys.modules[__name__]
    main()
