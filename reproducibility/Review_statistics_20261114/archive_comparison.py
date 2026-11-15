"""Verify unchanged reported points and retain exact original interval/SD values."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from evaluate import HERE,ROOT,read,HASHES,direction,EPS,record

def run(output=HERE/'outputs'):
    new=pd.read_csv(output/'contrasts.csv')
    records=[]
    def add(source,row,protocol,cohort,model,contrast,metric,stratum,point,lo,hi,kind='CI95'):
        hit=new.loc[new.protocol.eq(protocol)&new.cohort.eq(cohort)&new.model.eq(model)&new.contrast.eq(contrast)&new.metric.eq(metric)&new.stratum.eq(stratum)]
        assert len(hit)==1,(protocol,model,contrast,metric,stratum,len(hit))
        current=hit.iloc[0]
        np.testing.assert_allclose(current.relative_pct,point,rtol=1e-10,atol=1e-10)
        # Unified original crossed fixed-denominator intervals must reproduce exactly.
        if kind=='CI95' and cohort=='unified327':
            np.testing.assert_allclose([current.crossed_fixed_low,current.crossed_fixed_high],[lo,hi],rtol=1e-10,atol=1e-10)
        if kind=='CI95' and cohort=='full952':
            np.testing.assert_allclose([current.primary_low,current.primary_high],[lo,hi],rtol=1e-10,atol=1e-10)
        records.append(dict(source=source.resolve().relative_to(ROOT.resolve()).as_posix(),source_row=int(row),cohort=cohort,protocol=protocol,model=model,contrast=contrast,metric=metric,stratum=stratum,original_kind=kind,original_point=point,new_point=current.relative_pct,original_low=lo,original_high=hi,primary_low=current.primary_low,primary_high=current.primary_high,original_direction=direction(lo,hi) if kind=='CI95' else 'not_CI',primary_direction=current.primary_direction,interval_direction_changed=direction(lo,hi)!=current.primary_direction if kind=='CI95' else False,denominator_condition=current.denominator))
    source=ROOT/'Unified_ionisation_combined_20261114/reporting/output/scores/contrasts.csv'
    frame=read(source)
    for i,r in frame.iterrows():
        if r.metric.endswith('excluding_central') or r.metric=='mean_abs_log10_residual':continue
        metric='triad_mae' if r.metric=='triad_macro_log10_mae' else r.metric.removesuffix('_mae')
        name=r.contrast.replace('_minus_','-')
        if name in ('11-10','11-01','zero_eligible_11-10'): name+='_structure_scale'
        protocol='F_zero_policy' if name.startswith('zero_eligible') else 'F'
        add(source,i,protocol,'unified327',r.backbone,name,metric,'all',r.relative_change_pct,r.relative_crossed95_low_pct,r.relative_crossed95_high_pct)
    base=ROOT/'Unified_ionisation_combined_20261114/reporting/output'
    for filename in ['fig2/panel_c_estimates.csv','figS2/single_ph_estimates.csv']:
        source=base/filename; frame=read(source)
        for i,r in frame.iterrows():
            add(source,i,'F_pH','unified327',r.backbone,f'{r.arm}-{r.comparator_arm}',r.endpoint,'all',r.relative_delta_pct,r.crossed95_low_pct,r.crossed95_high_pct)
    source=ROOT/'Mass_balance_experiment_20261113/supplementary_completion/main_figure_layout_v3/inputs/split_1_5_summary.csv'
    for i,r in read(source).iterrows():
        add(source,i,'F','full952',r.backbone,'selected_positive-baseline','triad_mae','observed_le_1.5' if r.stratum=='within 1.5-fold' else 'observed_gt_1.5',r.relative_percent,r.relative_ci95_lower,r.relative_ci95_upper)
    source=ROOT/'Mass_balance_experiment_20261113/supplementary_completion/main_figure_layout_v3/inputs/endpoint_summary.csv'
    for i,r in read(source).iterrows():
        ep=next((e for e in EPS if r.metric.startswith(e+'_')),'seven_endpoint_average')
        add(source,i,'F','full952',r.backbone,'selected_positive-baseline',ep,'all',r['mean'],r.lower_sd,r.upper_sd,'seed_SD')
    source=ROOT/'Mass_balance_experiment_20261113/supplementary_completion/supplementary_selected_v1/encoder_contrasts.csv'
    for i,r in read(source).iterrows():
        model='GraphConv_depth2' if r.block=='primary_final' else f'{r.backbone}_depth{r.depth}_{r.block}'
        add(source,i,'F_supp3','full952',model,'loss-baseline','triad_mae','all',r.estimate,r.ci95_lower,r.ci95_upper)
    source=ROOT/'Mass_balance_experiment_20261113/supplementary_completion/supplementary_selected_v1/single_task_contrasts.csv'
    for i,r in read(source).iterrows():
        metric='triad_mae' if r.metric.startswith('triad_') else next(e for e in EPS if r.metric.startswith(e+'_'))
        comparator='baseline' if r.contrast.endswith('baseline') else 'loss'
        add(source,i,'F_supp3','full952','GraphConv_single_vs_multi',f'single_task-{comparator}',metric,'all',r.estimate,r.ci95_lower,r.ci95_upper)
    df=pd.DataFrame(records)
    df.to_csv(output/'exact_archived_comparison.csv',index=False)
    df.loc[df.interval_direction_changed].to_csv(output/'exact_archived_interval_changes.csv',index=False)
    manifest=json.loads((output/'manifest.json').read_text())
    for path in HERE.glob('*.py'):record(path)
    manifest['source_sha256'].update(HASHES)
    manifest['exact_archived_comparison']={'rows':len(df),'changed_interval_directions':int(df.interval_direction_changed.sum()),'all_original_points_preserved':True,'original_CIs_reproduced_with_same_draws':True}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(manifest['exact_archived_comparison'])
    return df

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'outputs')
    run(parser.parse_args().output)
