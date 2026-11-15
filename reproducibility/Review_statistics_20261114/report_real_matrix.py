"""Independent arithmetic/identity gates and evidence for the real matrix."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
OUT=HERE/'outputs_with_real_matrix';EXPORT=ROOT/'Review_robustness_experiment_20261114/outputs/export'

def main():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(EXPORT/'final_predictions.csv')=='73375d27caae320fdc98e403c0960a43451ce4085a2935eb9022a6d1cb7419b5'
    d=pd.read_csv(OUT/'contrasts.csv',float_precision='round_trip')
    old=pd.read_csv(HERE/'outputs/contrasts.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(d.iloc[:len(old)].reset_index(drop=True),old,check_dtype=False)
    p=pd.read_csv(EXPORT/'final_predictions.csv',low_memory=False)
    key=['protocol','input','backbone','outer_fold','seed','structure_id']
    off=p.loc[p.penalty.eq('off'),key+['fit_id']].rename(columns={'fit_id':'expected_reference'})
    aligned=p.merge(off,on=key,validate='many_to_one')
    assert len(aligned)==141264
    assert (aligned.reference_fit_id==aligned.expected_reference).all()
    assert d.loc[d.metric.ne('seven_endpoint_average'),['absolute_delta','absolute_low','absolute_high']].notna().all().all()
    s=pd.read_csv(OUT/'condition_metrics.csv').set_index(['protocol','model','condition','stratum','metric'])
    new=d[d.protocol.isin(['F_new','T_new','F_new_zero_policy','T_new_zero_policy'])]
    for r in new.loc[new.metric.ne('seven_endpoint_average')].itertuples():
        value=sum(w*s.loc[(r.protocol,r.model,c,r.stratum,r.metric),'mean'] for c,w in json.loads(r.terms).items())
        np.testing.assert_allclose(value,r.absolute_delta,rtol=0,atol=1e-12)
        if pd.notna(r.relative_pct):np.testing.assert_allclose(100*value/r.denominator_mean,r.relative_pct,rtol=0,atol=1e-10)
    t=d.loc[d.protocol.eq('T_new')&d.stratum.eq('all')]
    descriptor=t.loc[t.contrast.isin([f'P{i}_off-P0_off'for i in [1,2,3]])&t.metric.isin(['Vd','CL'])]
    triad=t.loc[t.contrast.isin([f'P{i}_positive-P{i}_off'for i in range(4)])&t.metric.eq('triad_mae')]
    residual=t.loc[t.contrast.isin([f'P{i}_positive-P{i}_off'for i in range(4)])&t.metric.eq('mean_abs_residual')]
    reductions=-100*residual.absolute_delta/residual.denominator_mean
    r2=d.loc[d.protocol.eq('F_new')&d.stratum.eq('all')&d.contrast.isin(['P0_separate-P0_shared','P1_separate-P1_shared'])&d.metric.eq('seven_endpoint_average')]
    result=dict(complete_contrasts=len(d),new_contrasts=len(new),old_contrasts_unchanged=len(old),same_input_reference_rows=len(aligned),T_descriptor_Vd_CL_negative_intervals=int((descriptor.primary_high<0).sum()),T_triad_range=[triad.relative_pct.min(),triad.relative_pct.max()],T_residual_reduction_percent_range=[reductions.min(),reductions.max()],R2_average_range=[r2.relative_pct.min(),r2.relative_pct.max()],R2_average_intervals_include_zero=int(((r2.primary_low<=0)&(r2.primary_high>=0)).sum()),campaign_id=json.loads((EXPORT/'manifest.json').read_text())['campaign_id'],prediction_sha256=sha(EXPORT/'final_predictions.csv'))
    assert result['T_descriptor_Vd_CL_negative_intervals']==18 and result['R2_average_intervals_include_zero']==6
    (OUT/'real_matrix_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['REAL F/T MATRIX: 15 November 2026','',json.dumps(result,indent=2),'','All comparisons below use all available endpoint labels, fixed nine fitted seeds, 10,000 paired whole-group draws; declared comparator MAE is recomputed within each draw. See manifest.json for draw seeds and conditional interpretation. F_new and T_new references remain distinct. reference_fit_id is the same-input off fit; denominator_condition is the actual contrast denominator. All strata and both positive-only/zero-inclusive policies remain in contrasts.csv.','', 'COMPLETE NEW ALL-LABEL ACCURACY CONTRASTS (percent changes unless common_structure_only, then percentage-point differences):']
    for r in new.loc[new.stratum.eq('all')&new.metric.isin(['Vd','CL','triad_mae','seven_endpoint_average'])].itertuples():
        lines.append(f'{r.protocol} | {r.model} | {r.contrast} | {r.metric}: {r.relative_pct:.6f} [{r.primary_low:.6f}, {r.primary_high:.6f}]; scale={r.scale}; denominator={r.denominator_condition}')
    (OUT/'REAL_MATRIX_EVIDENCE.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    evidence=OUT/'EVIDENCE.txt'
    text=evidence.read_text(encoding='utf-8').replace('T will remain a distinct tuning policy and reuse its own structure-only denominator.','The completed T matrix remains a distinct tuning policy and uses its own declared comparators; see REAL_MATRIX_EVIDENCE.txt for all new results.')
    evidence.write_text(text,encoding='utf-8')
    manifest=json.loads((OUT/'manifest.json').read_text())
    manifest['real_matrix_validation']=result
    manifest['output_sha256']={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
    manifest['real_matrix_report_code_sha256']=sha(Path(__file__))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--export',type=Path,default=EXPORT)
    args=parser.parse_args();OUT=args.output;EXPORT=args.export
    main()
