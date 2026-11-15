"""Render only Fig2/3 with harmonised primary intervals; original artists/styles retained."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
STAGE=None
OLD=ROOT/'Unified_ionisation_combined_20261114/reporting/output'
NEW=ROOT/'Review_statistics_20261114/outputs'
OUT=HERE/'rendered_S'
HASHES={}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    HASHES[str(path)]=sha(path)
    return pd.read_csv(path,float_precision='round_trip')
def module(name,path):
    HASHES[str(path)]=sha(path);sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj

def save(fig,path,stem):
    for text in fig.findobj(Text):
        text.set_text(text.get_text().replace('ChemBERTA','ChemBERTa').replace('MAE relative to structure only (%)','MAE vs structure only (%)'))
        label=text.get_text()
        if stem=='combined_main' and label in ('Concatenate','Prediction head'):
            x,y=text.get_position()
            text.set_position((x+(-2 if label=='Concatenate' else 2),y))
        if stem=='ionisation_main' and label=='Neither-active':
            text.set_fontsize(6.8)
    if stem=='combined_main':
        for ax in fig.axes:
            if any('%' in t.get_text() for t in ax.get_xticklabels()):
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'.replace('-','\u2212')))
    fig.canvas.draw()
    for ext,meta in [('pdf',{'CreationDate':None}),('png',None),('svg',{'Date':None})]:
        fig.savefig(path/f'{stem}.{ext}',dpi=300,facecolor='white',metadata=meta)
    plt.close(fig)

def main():
    OUT.mkdir()
    contrasts=read(NEW/'contrasts.csv')
    c=contrasts.loc[contrasts.cohort.eq('unified327')&contrasts.stratum.eq('all')].set_index(['protocol','model','contrast','metric'],verify_integrity=True)
    # Legacy drawing APIs express interval endpoints in an absolute-coordinate
    # carrier then scale them by100/reference MAE. These carriers are rendering
    # coordinates only, not absolute-MAE confidence intervals. Scientific tables
    # remain the independent evaluator outputs with explicit primary fields.
    factorial={p.stem:read(p) for p in (OLD/'scores').glob('*.csv')}
    fc=factorial['contrasts'].copy()
    for i,row in fc.iterrows():
        if row.metric not in [e+'_mae' for e in ('t_half','Vd','CL','Cmax','Tmax','F','PPB')]:continue
        if row.contrast not in ('10_minus_00','01_minus_00','11_minus_00'):continue
        n=c.loc[('F',row.backbone,row.contrast.replace('_minus_','-'),row.metric.removesuffix('_mae'))]
        np.testing.assert_allclose([row.relative_change_pct,row.baseline_00_mean],[n.relative_pct,n.denominator_mean],atol=1e-12,rtol=0)
        fc.loc[i,'crossed95_low']=n.primary_low*row.baseline_00_mean/100
        fc.loc[i,'crossed95_high']=n.primary_high*row.baseline_00_mean/100
        fc.loc[i,'relative_crossed95_low_pct']=n.primary_low
        fc.loc[i,'relative_crossed95_high_pct']=n.primary_high
    factorial['contrasts']=fc
    averages=read(OLD/'fig3/average_summary.csv')
    for i,row in averages.iterrows():
        n=c.loc[('F',row.backbone,row.contrast.replace('_minus_','-'),'seven_endpoint_average')]
        np.testing.assert_allclose(row.mean_pct,n.relative_pct,atol=1e-12,rtol=0)
        averages.loc[i,['low95_pct','high95_pct']]=[n.primary_low,n.primary_high]
    average_seeds=read(OLD/'fig3/average_seeds.csv')
    ph=read(OLD/'figS2/single_ph_estimates.csv')
    # This archived file covers all inputs versus REF0; add direct REF74
    # comparisons only if available in the reporting tables, not needed to draw.
    for i,row in ph.iterrows():
        key=('F_pH',row.backbone,f'{row.arm}-{row.comparator_arm}',row.endpoint)
        n=c.loc[key]
        np.testing.assert_allclose(row.relative_delta_pct,n.relative_pct,atol=1e-12,rtol=0)
        ph.loc[i,['crossed95_low_pct','crossed95_high_pct']]=[n.primary_low,n.primary_high]
        ph.loc[i,['crossed95_low','crossed95_high']]=np.array([n.primary_low,n.primary_high])*row.comparator_mae/100
    ph_seeds=read(OLD/'figS2/single_ph_seed_values.csv')
    # Use exact same authoritative table values for shared Fig2/Fig3 references.
    for i,row in ph.loc[ph.arm.eq('REF74')&ph.comparator_arm.eq('REF0')].iterrows():
        n=c.loc[('F',row.backbone,'10-00',row.endpoint)]
        np.testing.assert_allclose([row.crossed95_low_pct,row.crossed95_high_pct],[n.primary_low,n.primary_high],rtol=0,atol=1e-12)
        ph.loc[i,['crossed95_low_pct','crossed95_high_pct']]=[n.primary_low,n.primary_high]
        ph.loc[i,['crossed95_low','crossed95_high']]=np.array([n.primary_low,n.primary_high])*row.comparator_mae/100
    adapter=module('S_render_adapter',ROOT/'Unified_ionisation_combined_20261114/reporting/render_staged.py')
    adapter.save=save
    limits=json.loads((OLD/'fig2/scatter_inputs/manifest.json').read_text())['axis_limits_log10']
    with plt.rc_context():
        ion,_=adapter.render_ion(OUT/'fig2',ROOT/'Ionisation_experiment_20261113',ph,ph_seeds,averages,average_seeds,
            read(OLD/'figS2/scatter_points.csv'),read(OLD/'figS2/scatter_metrics.csv'),limits,HASHES)
    with plt.rc_context():
        plotted,layout=adapter.render_combined(OUT/'fig3',ROOT/'Combined_ionisation_mass_balance_20261113',factorial,averages,average_seeds,HASHES)
    # Builder multiplies the absolute-coordinate carriers by100/reference, so
    # last-bit differences can occur. Compare plotted CI/point identity to1e-12.
    p=plotted.loc[plotted.plot_kind.eq('summary')].copy()
    for row in p.itertuples():
        metric='seven_endpoint_average' if row.metric=='average_7_endpoint_mae_pct' else row.metric.removesuffix('_mae')
        n=c.loc[('F',row.backbone,row.contrast.replace('_minus_','-'),metric)]
        np.testing.assert_allclose([row.mean_pct,row.low95_pct,row.high95_pct],[n.relative_pct,n.primary_low,n.primary_high],atol=1e-12,rtol=0)
    shared=ion[0].set_index(['backbone','endpoint'])
    for row in p.loc[p.contrast.eq('10_minus_00')&p.metric.ne('average_7_endpoint_mae_pct')].itertuples():
        n=shared.loc[(row.backbone,row.metric.removesuffix('_mae'))]
        np.testing.assert_allclose([row.mean_pct,row.low95_pct,row.high95_pct],[n.relative_delta_pct,n.crossed95_low_pct,n.crossed95_high_pct],atol=1e-12,rtol=0)
    preserved={}
    # Do not retain the archived generated captions with outdated interval text.
    for caption in OUT.glob('*/caption.txt'):
        caption.write_text('Primary intervals are paired whole-group percentile95% intervals with fitted seeds fixed and the declared comparator mean MAE recomputed in each draw. Point estimates and seed dots retain the original full-cohort denominators. Final reader-facing legends are in staged main.tex.\n')
    HASHES[str(Path(__file__))]=sha(Path(__file__))
    receipt=dict(primary_CI_source=str(NEW/'contrasts.csv'),source_sha256=HASHES,preserved_figures=preserved,modified_figures=['fig2_ionisation.pdf','fig3_combined.pdf'],figure2_figure3_shared_points_and_intervals='PASS to1e-12',comparison=f'All {len(p)} displayed Fig3 endpoint/average means and bounds checked against the primary score table.',layout=layout,render_adapter='Legacy crossed fields are private drawing-coordinate carriers; primary scientific uncertainty is defined by contrasts.csv, not those carrier names.',outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()})
    (OUT/'figure_S_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Rendered Fig2/Fig3; preserved Fig1/S1/S2; score-to-artist checks PASS')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    OUT=parser.parse_args().output
    main()
