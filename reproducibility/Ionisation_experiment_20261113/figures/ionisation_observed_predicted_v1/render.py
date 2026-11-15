"""Observed versus predicted Vd and clearance, coloured by pH 7.4 category."""
from hashlib import sha256
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
EXP=HERE.parents[1]
PRED=EXP/'outputs/ph_followup/outer_predictions.csv'
DATA=EXP/'data/release/numerical/model_dataset.csv'
METRICS=EXP/'outputs/ph_followup/scoring/endpoint_metrics.csv'
MODELS={'GraphConv':'GraphConv','Chemprop':'Chemprop','ChemBERTa_head':'Frozen ChemBERTa + MLP'}
ARMS={'REF74':'pH 7.4 ionisation category','REF0':'Structure only',
      'REP4':'Repeated pH 7.4 category (matched-width control)',
      'ALL4':'Four-pH ionisation profile'}
CLASSES={'anionic':('Acid-only (anionic)','#0072B2'),
         'cationic':('Base-only (cationic)','#D55E00'),
         'neutral':('Neither-active','#7C8A92'),
         'zwitterionic':('Dual-active','#8552A0')}
ENDPOINTS={'Vd':('Volume of distribution','VD(liter)','L',259),
           'CL':('Clearance','Cl(liter/hour)','L/h',246)}
INK,MUTED,GRID='#263541','#586874','#E5EBEF'
SEEDS=[101,202,303,404,505,606,707,808,909]
plt.rcParams.update({'font.family':'Arial','font.size':10,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':INK,'axes.edgecolor':'#89959D',
    'axes.spines.top':True,'axes.spines.right':True,'axes.linewidth':.65,
    'pdf.fonttype':42,'svg.fonttype':'none'})


def digest(p):return sha256(p.read_bytes()).hexdigest()


if __name__=='__main__':
    HERE.mkdir(parents=True,exist_ok=True)
    export=json.loads(PRED.with_suffix('.manifest.json').read_text())
    release=json.loads((DATA.parent/'manifest.json').read_text())
    score=json.loads((METRICS.parent/'scoring_manifest.json').read_text())
    assert digest(PRED)==export['sha256']==score['predictions_sha256']
    assert digest(DATA)==release['files'][DATA.name]
    assert digest(METRICS)==score['files'][METRICS.name]
    data=pd.read_csv(DATA,dtype={'structure_id':str})
    assert len(data)==327 and data.structure_id.nunique()==327
    pred=pd.read_csv(PRED,usecols=['structure_id','backbone','arm','procedure','seed','outer_fold','pred_Vd','pred_CL'],dtype={'structure_id':str})
    pred=pred.loc[pred.procedure.eq('primary') & pred.arm.isin(ARMS)]
    keys=['arm','backbone','structure_id']
    assert len(pred)==4*3*327*9 and not pred.duplicated(keys+['seed']).any()
    assert pred.groupby(keys).seed.apply(lambda x: sorted(x)==SEEDS).all()
    assert pred.outer_fold.eq(pred.structure_id.map(data.set_index('structure_id').outer_fold)).all()
    avg=pred.groupby(keys,as_index=False)[['pred_Vd','pred_CL']].mean()
    avg=avg.merge(data[['structure_id','ionisation_class','VD(liter)','Cl(liter/hour)','outer_fold']],on='structure_id',validate='many_to_one')
    scores=pd.read_csv(METRICS)
    points=[];summary=[];limits={}
    for ep,(_,raw,unit,n) in ENDPOINTS.items():
        obs=avg[raw].to_numpy();mask=np.isfinite(obs)&(obs>0)
        d=avg.loc[mask].copy()
        d['observed_log10']=np.log10(d[raw]);d['predicted_log10']=d[f'pred_{ep}']
        d['endpoint']=ep;d['physical_unit']=unit
        d['predicted_geometric_mean']=10**d.predicted_log10
        assert d.groupby(['arm','backbone']).size().eq(n).all()
        for (arm,model),rows in d.groupby(['arm','backbone']):
            y=rows.observed_log10.to_numpy();x=rows.predicted_log10.to_numpy()
            r2=1-np.square(y-x).sum()/np.square(y-y.mean()).sum()
            rho=float(spearmanr(x,y).statistic)
            seed=scores.loc[scores.procedure.eq('primary')&scores.arm.eq(arm)&scores.backbone.eq(model)&scores.endpoint.eq(ep)]
            assert len(seed)==9
            summary.append({'arm':arm,'backbone':model,'endpoint':ep,'n':n,
                'r2_displayed_seed_mean':r2,'spearman_displayed_seed_mean':rho,
                'mae_log10_displayed_seed_mean':np.abs(y-x).mean(),
                'r2_individual_seed_mean':seed.r2.mean(),'r2_individual_seed_sd':seed.r2.std(ddof=1),
                'spearman_individual_seed_mean':seed.spearman.mean()})
        lo=min(d.observed_log10.min(),d.predicted_log10.min())
        hi=max(d.observed_log10.max(),d.predicted_log10.max())
        lo=np.floor(lo*2)/2-.15;hi=np.ceil(hi*2)/2+.15
        limits[ep]=(lo,hi)
        points.append(d[['arm','backbone','structure_id','outer_fold','ionisation_class','endpoint','physical_unit',
                         'observed_log10','predicted_log10','predicted_geometric_mean',raw]].rename(columns={raw:'observed_physical'}))
    points=pd.concat(points,ignore_index=True);summary=pd.DataFrame(summary)
    with PdfPages(HERE/'vd_clearance_observed_predicted.pdf') as pdf:
        for arm,arm_label in ARMS.items():
            fig,axes=plt.subplots(2,3,figsize=(12.8,9.6))
            fig.subplots_adjust(left=.075,right=.975,bottom=.135,top=.82,hspace=.32,wspace=.22)
            fig.text(.045,.965,arm_label,fontsize=19,weight='bold',va='top')
            fig.text(.045,.918,'One point per drug · held-out predictions averaged in log10 space over nine training seeds',fontsize=10.5,color=MUTED)
            handles=[Line2D([],[],marker='o',color=color,linestyle='none',markersize=6,label=label)
                     for label,color in CLASSES.values()]
            fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.04,.899),frameon=False,ncol=4,
                       fontsize=10,handletextpad=.35,columnspacing=1.5)
            for i,(ep,(name,raw,unit,n)) in enumerate(ENDPOINTS.items()):
                for j,(model,model_label) in enumerate(MODELS.items()):
                    ax=axes[i,j]
                    d=points.loc[points.arm.eq(arm)&points.backbone.eq(model)&points.endpoint.eq(ep)]
                    r=summary.loc[summary.arm.eq(arm)&summary.backbone.eq(model)&summary.endpoint.eq(ep)].iloc[0]
                    lo,hi=limits[ep]
                    ax.plot([lo,hi],[lo,hi],color='#9AA5AD',ls=(0,(4,3)),lw=.85,zorder=1)
                    for label,(_,color) in CLASSES.items():
                        c=d.loc[d.ionisation_class.eq(label)]
                        ax.scatter(c.predicted_log10,c.observed_log10,s=18,color=color,alpha=.78,
                                   edgecolor='white',linewidth=.25,zorder=3)
                    ax.set(xlim=(lo,hi),ylim=(lo,hi));ax.set_aspect('equal',adjustable='box')
                    ax.grid(color=GRID,lw=.65,zorder=0);ax.xaxis.set_major_locator(MultipleLocator(1));ax.yaxis.set_major_locator(MultipleLocator(1))
                    ax.set_xlabel(f'Predicted log10 {ep} ({unit})',fontsize=9.6)
                    ax.set_ylabel(f'Observed log10 {ep} ({unit})' if j==0 else '',fontsize=9.6)
                    ax.tick_params(labelsize=9)
                    ax.set_title(f'{chr(65+i)}{j+1}  {model_label}\n'+r'$R^2$'+f' = {r.r2_displayed_seed_mean:.3f}  |  n = {n}',
                                 fontsize=10.3,weight='bold',pad=7)
            fig.text(.045,.074,'Top row: volume of distribution (Vd). Bottom row: clearance (CL). Dashed line: predicted = observed.',fontsize=10,color=MUTED)
            fig.text(.045,.046,'Colours use the same pH 7.4 categories on every page. R² is calculated from the displayed nine-seed mean predictions.',fontsize=9.5,color=MUTED)
            fig.text(.975,.021,f'{list(ARMS).index(arm)+1} / 4',fontsize=8.5,color=MUTED,ha='right')
            pdf.savefig(fig)
            for ext in ['png','svg']:fig.savefig(HERE/f'vd_clearance_{arm.lower()}.{ext}',dpi=240,facecolor='white')
            plt.close(fig)
    points.to_csv(HERE/'plotted_points.csv',index=False);summary.to_csv(HERE/'metrics.csv',index=False)
    (HERE/'caption.txt').write_text('''Observed versus predicted volume of distribution and clearance, coloured by ionisation category.
Each page shows GraphConv, Chemprop and frozen ChemBERTa with a trained MLP head. The top row shows Vd in litres (259 observed labels), and the bottom shows clearance in litres/hour (246 labels), within the 327-structure numerical-pKa cohort. Predictions are held out from five grouped outer folds. The horizontal axis is predicted log10 endpoint and the vertical axis is observed log10 endpoint, matching the supplied example. One point represents one structure, with its predicted log10 value averaged across nine training seeds. This is the log10 of the geometric mean prediction on the physical scale. The dashed line denotes equality. All arms and backbones use the same endpoint-specific x and y limits.
Colours retain the original pH 7.4 operational categories across all pages, including the four-pH profile. They are acid-only (anionic category), base-only (cationic category), neither-active and dual-active. The labels derive from the acidic/basic pKa thresholds and do not denote individual resolved microspecies.
R-squared is recomputed from the displayed seed-averaged predictions as 1 - sum((observed - predicted)^2)/sum((observed - mean(observed))^2), in log10 space. It is not a fitted-line correlation squared. These displayed-prediction scores are distinct from the mean of nine per-seed scores used in the main comparison figures. metrics.csv retains both definitions explicitly, along with Spearman correlation and MAE for the displayed predictions.
Page 1: pH 7.4 one-hot category (REF74). Page 2: structure only (REF0). Page 3: repeated pH 7.4 one-hot category (REP4). Page 4: the four-pH categorical profile (ALL4; pH 7.4, 2.0, 6.5, 5.0). For the additional-pH comparison, REP4 is the width-matched control for ALL4. Both contain 16 channels, divided by two. The four-pH versus single-category comparison also changes input parameterisation.
All pages use the shared training recipe, equal explicit endpoint weights, and zero mass-balance penalty. No new training was performed for these plots.
''',encoding='utf-8')
    outputs={p.name:digest(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ['.pdf','.png','.svg','.csv','.txt']}
    (HERE/'manifest.json').write_text(json.dumps({'sources':{str(p):digest(p) for p in [PRED,DATA,METRICS]},
        'outputs':outputs,'axis_limits_log10':limits,'pages':list(ARMS),'n_seeds':9},indent=2),encoding='utf-8')
    print(summary.loc[summary.arm.eq('REF74'),['backbone','endpoint','n','r2_displayed_seed_mean']].to_string(index=False))
