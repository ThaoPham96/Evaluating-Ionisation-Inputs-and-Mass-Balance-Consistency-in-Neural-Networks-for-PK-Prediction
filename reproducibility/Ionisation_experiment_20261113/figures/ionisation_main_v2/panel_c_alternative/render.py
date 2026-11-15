"""A simpler proposed panel C: size of Vd prediction changes within each seed."""
from pathlib import Path
from hashlib import sha256
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
EXP=HERE.parents[2]
SOURCE=EXP/'outputs/ph_followup/outer_predictions.csv'
MODELS={'GraphConv':'GraphConv','Chemprop':'Chemprop','ChemBERTa_head':'Frozen ChemBERTa + MLP'}
SEEDS=[101,202,303,404,505,606,707,808,909]
INK,MUTED,GRID,COLOR='#263541','#586874','#E5EBEF','#D55E00'
digest=sha256(SOURCE.read_bytes()).hexdigest()
assert digest==json.loads(SOURCE.with_suffix('.manifest.json').read_text())['sha256']
d=pd.read_csv(SOURCE,usecols=['structure_id','backbone','procedure','arm','seed','outer_fold','physical_Vd'])
d=d.query("procedure=='primary' and arm in ['ALL4','REP4']")
keys=['structure_id','backbone','seed','procedure','outer_fold']
a=d.query("arm=='ALL4'").merge(d.query("arm=='REP4'"),on=keys,suffixes=('_profile','_control'),validate='one_to_one')
assert len(a)==327*9*3
assert a.groupby(['backbone','seed']).size().eq(327).all()
assert (a[['physical_Vd_profile','physical_Vd_control']].to_numpy()>0).all()
a['signed_prediction_change_pct']=100*(a.physical_Vd_profile/a.physical_Vd_control-1)
a['absolute_prediction_change_pct']=a.signed_prediction_change_pct.abs()
s=a.groupby(['backbone','seed'],as_index=False).absolute_prediction_change_pct.median().rename(
    columns={'absolute_prediction_change_pct':'median_absolute_change_pct'})
summary=s.groupby('backbone').median_absolute_change_pct.agg(['mean','std','min','max']).reset_index()
plt.rcParams.update({'font.family':'Arial','font.size':10,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':INK,'axes.edgecolor':'#85929A',
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(8.2,4.65),facecolor='white')
fig.text(.035,.94,'C',weight='bold',fontsize=18,va='top')
fig.text(.075,.94,'How much do Vd predictions change?',weight='bold',fontsize=17,va='top')
fig.text(.075,.845,'Four-pH profile versus pH 7.4 categories repeated four times',fontsize=10.5,color=MUTED)
ax=fig.add_axes([.30,.29,.55,.46])
jitter=np.array([-.075,.07,-.04,.09,.015,-.06,.045,-.012,.06])
for y,(model,label) in enumerate(MODELS.items()):
    row=summary.set_index('backbone').loc[model]
    vals=s.loc[s.backbone.eq(model)].sort_values('seed')
    assert vals.seed.tolist()==SEEDS
    ax.scatter(vals.median_absolute_change_pct,y+jitter,s=30,color=COLOR,alpha=.55,edgecolor='white',linewidth=.4,zorder=3)
    ax.plot([row['mean']-row['std'],row['mean']+row['std']],[y,y],color=COLOR,lw=1.5,zorder=2)
    ax.plot(row['mean'],y,'D',ms=6.5,color=COLOR,mec=INK,mew=.6,zorder=4)
    ax.text(1.04,y,f"{row['mean']:.1f}%",transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=12,weight='bold')
ax.set(xlim=(0,70),ylim=(2.5,-.5),yticks=range(3),xticks=[0,20,40,60])
ax.set_yticklabels(list(MODELS.values()),fontsize=10)
ax.tick_params(axis='y',length=0,pad=8);ax.spines['left'].set_visible(False)
ax.grid(axis='x',color=GRID,lw=.8,zorder=0)
ax.set_xlabel('Median absolute change in predicted Vd (%)',labelpad=8,fontsize=11)
fig.text(.075,.145,'Dots: nine training seeds     Diamond: mean     Bar: ±1 seed SD',fontsize=10,color=MUTED)
fig.text(.075,.088,'Within each seed: calculate each drug’s absolute % change, then take the median over 327 drugs.',fontsize=9,color=MUTED)
fig.text(.075,.048,'0% = identical predictions. This measures movement in predictions; it does not measure prediction error.',fontsize=9,color=MUTED)
HERE.mkdir(parents=True,exist_ok=True)
for ext in ['pdf','png','svg']:fig.savefig(HERE/f'panel_c_vd_shift.{ext}',dpi=250,facecolor='white')
plt.close(fig)
a.to_csv(HERE/'paired_prediction_changes.csv',index=False)
s.to_csv(HERE/'seed_medians.csv',index=False);summary.to_csv(HERE/'summary.csv',index=False)
(HERE/'caption.txt').write_text('''Proposed replacement for panel C: size of Vd prediction changes.
For each drug and matched seed, absolute percentage change is 100 * abs(Vd_four_pH / Vd_repeated_pH7.4 - 1). Each seed contributes the median across all 327 structures, including those without observed Vd. Points show the nine seed-specific medians; diamonds show their arithmetic mean and bars show one sample SD across seeds, not confidence intervals. All predictions are held-out outer-fold predictions from the shared-recipe comparison; both inputs have 16 channels and matched normalisation. This diagnostic measures prediction changes, not error against observations.
This differs intentionally from the earlier panel's statistic: the previous plot averaged signed drug changes across seeds before describing the drug distribution. The new plot summarises each seed first so opposing changes across training seeds do not cancel. The means are GraphConv 20.2824%, Chemprop 38.3896%, and frozen ChemBERTa 7.3210%. Vd relative percentage changes are measured against the repeated-pH comparator and are not symmetric fold factors.
The original full figure is preserved; this file is a proposed replacement panel.
''',encoding='utf-8')
outputs={p.name:sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file() and p.suffix in ['.pdf','.png','.svg','.csv','.txt']}
(HERE/'manifest.json').write_text(json.dumps({'source':str(SOURCE),'source_sha256':digest,'outputs':outputs},indent=2),encoding='utf-8')
print(summary.to_string(index=False))
