"""Subgroup and encoding comparisons from archived out-of-fold predictions only."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def main(root, supporting, output):
    root, supporting, output = map(Path, (root, supporting, output))
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / 'Review_statistics_20261114'))
    import evaluate as ev
    from sources import aligned
    ev.HASHES.clear()
    mass = root / 'Mass_balance_experiment_20261113'
    uni = root / 'Unified_ionisation_combined_20261114'
    full = ev.cohort('full952', mass / 'data', mass / 'data/same_row_complete_triads.csv')
    num = ev.cohort('unified327', uni / 'data/release/numerical', uni / 'runtime/same_row_complete_triads.csv')

    def summary(c, boot, arms, mask, model, stratum, metric, a, b):
        mask = np.asarray(mask, bool)
        ng = len(np.unique(c.group[mask]))
        counts = np.bincount(c.group[mask], minlength=boot.ng)
        draw_n = boot.counts @ counts
        valid = draw_n > 0
        points, draws = {}, {}
        for name in (a, b):
            v = arms[name]
            assert v.shape == (9, len(c.rows)) and np.isfinite(v[:, mask]).all()
            points[name] = v[:, mask].mean() if mask.any() else np.nan
            sums = np.zeros(boot.ng)
            np.add.at(sums, c.group[mask], v[:, mask].mean(axis=0))
            draws[name] = (boot.counts @ sums)[valid] / draw_n[valid]
        delta = points[a] - points[b]
        pct = 100 * delta / points[b]
        # Tiny strata are descriptive; no percentile interval is reported below five groups.
        eligible = ng >= 5 and valid.sum() >= 9500
        lo, hi = ev.bounds(100*(draws[a]-draws[b])/draws[b]) if eligible else (np.nan, np.nan)
        alo, ahi = ev.bounds(draws[a]-draws[b]) if eligible else (np.nan, np.nan)
        return dict(cohort=c.name, model=model, stratum=stratum, metric=metric,
                    comparison=a+'-'+b, condition=a, reference=b, n=int(mask.sum()), n_groups=ng,
                    n_seeds=9, reference_mae=points[b], condition_mae=points[a], absolute_delta=delta,
                    absolute_low=alo, absolute_high=ahi, relative_pct=pct, primary_low=lo, primary_high=hi,
                    bootstrap_draws=boot.n, nonempty_draws=int(valid.sum()), interval_reported=eligible)

    # Full structure-level target and grouping inventory, retaining the original scalar values.
    inventory = full.rows.copy()
    inventory['same_row_complete_triad'] = full.triads
    inventory['observed_triad_endpoint_count'] = np.isfinite(full.truth[:, :3]).sum(axis=1)
    inventory.reset_index().to_csv(output/'Supplementary_Data_6_full_structure_inventory.csv', index=False)
    flags = ev.read(supporting/'Supplementary_Data_5_structure_flags.csv').set_index('structure_id')
    flags = flags.loc[num.rows.index]
    masks = {
        'single_component_no_charged_atoms': flags.component_count.eq(1) & flags.charged_atom_count.eq(0),
        'single_component_charged_atoms': flags.component_count.eq(1) & flags.charged_atom_count.gt(0),
        'multiple_components': flags.component_count.gt(1),
    }
    assert np.array_equal(sum(v.astype(int) for v in masks.values()), np.ones(len(num.rows)))
    preds = ev.read(uni/'outputs/export/combined/final_predictions.csv')
    boot = ev.Bootstrap(num, ev.SEEDS)
    composition = []
    for model in ('GraphConv', 'Chemprop', 'ChemBERTa_head'):
        arms = {arm: aligned(preds[preds.backbone.eq(model) & preds.cell.eq(cell)], num, ev.SEEDS)
                for arm, cell in [('structure', '00'), ('ionisation', '10')]}
        for stratum, stratum_mask in masks.items():
            for j, ep in enumerate(ev.EPS):
                errors = {k: np.abs(q[:, :, j]-num.truth[:, j]) for k, q in arms.items()}
                composition.append(summary(num, boot, errors, stratum_mask & np.isfinite(num.truth[:, j]),
                                           model, stratum, ep, 'ionisation', 'structure'))
    pd.DataFrame(composition).to_csv(output/'Supplementary_Data_6_input_composition.csv', index=False)

    preds = ev.read(mass/'figure_data/primary_outer_predictions.csv.gz')
    boot = ev.Bootstrap(full, ev.SEEDS)
    nobs = np.isfinite(full.truth[:, :3]).sum(axis=1)
    partial = []
    for model in ('GraphConv', 'Chemprop', 'ChemBERTa_head'):
        arms = {}
        for policy in ('baseline', 'selected_positive'):
            d = preds[preds.backbone.eq(model) & preds.policy.eq(policy)].copy()
            for ep in ev.EPS:
                suffix = ep + ('_percent' if ep in ('F','PPB') else '_log10')
                d['q_'+ep] = d['pred_'+suffix]
                d['true_'+ep] = d['observed_endpoint_'+suffix]
                d['observed_'+ep] = d['true_'+ep].notna()
            arms[policy] = aligned(d, full, ev.SEEDS)
        for name, mask in [('one_observed', nobs==1), ('two_observed', nobs==2),
                           ('one_or_two_observed', (nobs>=1)&(nobs<=2))]:
            for j, ep in enumerate(ev.EPS[:3]):
                errors = {k: np.abs(q[:, :, j]-full.truth[:, j]) for k, q in arms.items()}
                partial.append(summary(full, boot, errors, mask & np.isfinite(full.truth[:, j]),
                                       model, name, ep, 'selected_positive', 'baseline'))
    pd.DataFrame(partial).to_csv(output/'Supplementary_Data_6_partial_triad_accuracy.csv', index=False)

    old = root/'Ionisation_experiment_20261113'
    rows = ev.read(old/'data/release/class/model_dataset.csv').set_index('structure_id', verify_integrity=True).sort_index()
    pd.testing.assert_frame_equal(rows[['group_id','outer_fold']],full.rows.loc[rows.index,['group_id','outer_fold']],check_dtype=False)
    np.testing.assert_allclose(rows[list(ev.RAW)],full.rows.loc[rows.index,list(ev.RAW)],rtol=0,atol=0,equal_nan=True)
    truth = rows[list(ev.RAW)].to_numpy(float); truth[:,:5] = np.log10(truth[:,:5])
    _, group = np.unique(rows.group_id, return_inverse=True)
    c = ev.Cohort('class494',rows,truth,rows.index.isin(full.rows.index[full.triads]),group)
    boot = ev.Bootstrap(c, ev.SEEDS)
    preds = ev.read(old/'outputs/outer_predictions.csv')
    preds = preds[preds.cohort.eq('class') & preds.procedure.eq('primary')]
    encoding = []
    for model in ('GraphConv','Chemprop','ChemBERTa_head'):
        arms = {arm:aligned(preds[preds.backbone.eq(model)&preds.arm.eq(arm)],c,ev.SEEDS,prefix='')
                for arm in ('C0','C1','C2','C3')}
        for b,a in itertools.combinations(arms,2):
            for j,ep in enumerate(ev.EPS):
                errors = {k:np.abs(q[:,:,j]-c.truth[:,j]) for k,q in arms.items()}
                encoding.append(summary(c,boot,errors,np.isfinite(c.truth[:,j]),model,'all',ep,a,b))
    pd.DataFrame(encoding).to_csv(output/'Supplementary_Data_6_encoding_contrasts.csv',index=False)

    archive=ev.read(supporting/'Table_S2_archived_pKa.csv')
    thresholds=[]
    boundary=[]
    for ph in (2.0,5.0,6.5,7.4,8.0):
        acid=archive.CX_MOST_APKA.to_numpy();base=archive.CX_MOST_BPKA.to_numpy()
        ae,be=acid==ph,base==ph
        code=(acid<ph).astype(int)+2*(base>ph).astype(int)
        thresholds.append(dict(pH=ph,acid_equal=int(ae.sum()),base_equal=int(be.sum()),both_equal=int((ae&be).sum()),
                              neutral=int((code==0).sum()),acid_only=int((code==1).sum()),base_only=int((code==2).sum()),both=int((code==3).sum())))
        for i in np.flatnonzero(ae|be):
            boundary.append(dict(structure_id=archive.structure_id.iloc[i],pH=ph,acid_pKa=acid[i],base_pKa=base[i],acid_equal=bool(ae[i]),base_equal=bool(be[i]),class_code=int(code[i])))
    pd.DataFrame(thresholds).to_csv(output/'Supplementary_Data_6_threshold_counts.csv',index=False)
    pd.DataFrame(boundary).to_csv(output/'Supplementary_Data_6_threshold_boundary_records.csv',index=False)
    # Reuse the verified original factorial stratum statistics; do not change estimators.
    stats=ev.read(supporting/'Supplementary_Data_1_statistics.csv')
    selected=stats[stats.cohort.eq('unified327') & stats.protocol.eq('F') & stats.metric.eq('triad_mae') &
                   stats.stratum.isin(['observed_le_1.5','observed_gt_1.5']) &
                   stats.contrast.isin(['10-00','01-00','11-00','11-10_structure_scale','interaction'])]
    assert len(selected)==30
    selected.to_csv(output/'Supplementary_Data_6_factorial_strata.csv',index=False)
    epochs=[]
    for model in ('GraphConv','Chemprop','ChemBERTa_head'):
        for fold in range(1,6):
            p=root/f'fit_selections/full_models/{model}-outer{fold}-lambda/selection.json'
            d=json.loads(ev.record(p).read_text())
            for arm,weight in [('baseline',0.0),('positive',d['selected_lambda'])]:
                epochs.append(dict(cohort='full952',model=model,outer_fold=fold,arm=arm,
                                   configuration_id=int(d['candidate_id'].rsplit('_',1)[1]),
                                   penalty_weight=weight,median_best_epoch=d['epochs'][str(float(weight))]))
            p=root/f'fit_selections/ionisation_class/class-{model}-outer{fold}/selection.json'
            d=json.loads(ev.record(p).read_text())
            for arm in ('C0','C1','C2','C3'):
                fitted=[r for r in d['final_requests'] if r['procedure']=='primary' and r['arm']==arm]
                assert len(fitted)==9 and len({r['epochs'] for r in fitted})==1
                epochs.append(dict(cohort='class494',model=model,outer_fold=fold,arm=arm,
                                   configuration_id=d['primary_anchor_index']+1,penalty_weight=0,
                                   median_best_epoch=fitted[0]['epochs']))
    pd.DataFrame(epochs).to_csv(output/'Supplementary_Data_6_stopping_epochs.csv',index=False)
    manifest=dict(training_performed=False,bootstrap_draws=10000,
                  estimator='Difference of nine-seed mean MAEs; paired whole-group percentile intervals; comparator denominator recomputed in each draw.',
                  bootstrap_seeds={'full952':20261112,'unified327':20261113,'class494':20261113},
                  small_strata='Intervals omitted below five eligible groups or fewer than 9500 nonempty draws; all point estimates retained. Empty draws omitted jointly from numerator and denominator.',
                  input_sha256={('supporting_data/'+Path(p).name if Path(p).is_relative_to(supporting) else Path(p).relative_to(root).as_posix()):v for p,v in ev.HASHES.items()},
                  outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('Supplementary_Data_6_*.csv')})
    (output/'Supplementary_Data_6_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(input_composition_rows=len(composition),partial_rows=len(partial),encoding_rows=len(encoding),factorial_rows=len(selected),thresholds=thresholds),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--supporting-data',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):
        main(a.root.resolve(),(a.supporting_data or a.root/'supporting_data').resolve(),a.output.resolve())
