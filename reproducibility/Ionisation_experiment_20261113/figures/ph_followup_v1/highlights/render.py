"""Three focused views of the frozen pH results, without recomputing training."""
from pathlib import Path
from hashlib import sha256
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
EXP=ROOT.parents[1]
INK,MUTED,GRID='#25343F','#566875','#E7ECEF'
CYAN,ORANGE='#4298B5','#D55E00'
plt.rcParams.update({'font.family':'Arial','font.size':11,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':INK,
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#85929A',
    'pdf.fonttype':42,'svg.fonttype':'none'})

paths=[ROOT/'prediction_shifts/structure_shifts.csv',ROOT/'prediction_shifts/shift_summary.csv',
       EXP/'outputs/ph_followup/scoring/contrasts.csv']
structure,summary,contrasts=[pd.read_csv(p) for p in paths]
mask=lambda d: d.procedure.eq('primary') & d.arm.eq('ALL4') & d.comparator_arm.eq('REP4')
structure,summary,contrasts=[d.loc[mask(d)].copy() for d in [structure,summary,contrasts]]
f=structure.loc[structure.backbone.eq('Chemprop') & structure.endpoint.eq('F')].sort_values('structure_id')
vd=structure.loc[structure.backbone.eq('ChemBERTa_head') & structure.endpoint.eq('Vd')].sort_values('structure_id')
assert len(f)==len(vd)==327
assert not f.structure_id.duplicated().any() and not vd.structure_id.duplicated().any()

fig=plt.figure(figsize=(15.5,7.4),facecolor='white')
fig.text(.045,.95,'Three findings worth focusing on',fontsize=22,weight='bold',va='top')
fig.text(.045,.895,'Four-pH categories (7.4, 2.0, 6.5, 5.0) versus pH 7.4 repeated four times; same input width.',
         fontsize=12,color=MUTED,va='top')
positions=[(.065,.31,.24,.43),(.405,.31,.24,.43),(.755,.31,.22,.43)]
axes=[fig.add_axes(pos) for pos in positions]
headings=[('A','Individual changes can cancel','Chemprop · bioavailability'),
          ('B','Vd shifts differ by ionisation class','Frozen ChemBERTa + MLP'),
          ('C','What happens to accuracy?','All three models · Vd and bioavailability')]
for ax,(letter,heading,sub) in zip(axes,headings):
    x=ax.get_position().x0
    fig.text(x,.815,f'{letter}  {heading}',fontsize=12.5,weight='bold')
    fig.text(x,.779,sub,fontsize=10.5,color=MUTED)

# A: show the complete distribution, including both tails.
ax=axes[0]
counts,bins=np.histogram(f.mean_delta,bins=np.arange(-30,33,3))
assert counts.sum()==327
ax.bar(bins[:-1],counts,width=3,align='edge',color=CYAN,alpha=.83,edgecolor='white',linewidth=.7,zorder=2)
ax.axvline(0,color=INK,lw=1.1,ls='--',zorder=3)
ax.set(xlim=(-30,30),xlabel='Change in predicted bioavailability\n(percentage points)',ylabel='Number of drugs')
ax.set_xticks([-30,-15,0,15,30]);ax.set_ylim(0,max(counts)*1.14)
ax.grid(axis='y',color=GRID,zorder=0)
ax.text(.03,.94,f'{int((f.mean_delta<0).sum())} lower',transform=ax.transAxes,va='top',fontsize=10)
ax.text(.97,.94,f'{int((f.mean_delta>0).sum())} higher',transform=ax.transAxes,va='top',ha='right',fontsize=10)
fig.text(.065,.205,f'Median size of change: {np.median(abs(f.mean_delta)):.2f} points',fontsize=12,weight='bold')
fig.text(.065,.175,f'Average signed change: {f.mean_delta.mean():+.2f} points\nUpward and downward changes balance out.',fontsize=10.7,color=MUTED,linespacing=1.45,va='top')

# B: all drugs in all four classes, with a geometric mean ratio per class.
ax=axes[1]
class_defs=[('anionic','Acid-only',65),('cationic','Base-only',90),('neutral','Neither',119),('zwitterionic','Dual-active',53)]
rng=np.random.default_rng(20261113)
ax.axvline(1,color=INK,lw=1.1,ls='--',zorder=1)
means=[]
for y,(code,label,n) in enumerate(class_defs):
    d=vd.loc[vd['class_7.4'].eq(code)]
    assert len(d)==n
    ratios=np.exp2(d.mean_delta.to_numpy());mean=np.exp2(d.mean_delta.mean());means.append(mean)
    assert ((ratios>=.75)&(ratios<=1.35)).all()
    ax.scatter(ratios,y+rng.uniform(-.16,.16,n),s=12,color=ORANGE,alpha=.32,edgecolor='none',zorder=2)
    ax.scatter(mean,y,s=48,marker='D',facecolor=ORANGE,edgecolor=INK,lw=.75,zorder=3)
ax.set(xlim=(.75,1.35),ylim=(3.55,-.55),xlabel='Predicted Vd relative to control\n(1 = unchanged)')
ax.set_xticks([.8,1,1.2]);ax.set_xticklabels(['0.8×','1×','1.2×'])
ax.set_yticks(range(4),[f'{label}\n(n={n})' for _,label,n in class_defs]);ax.tick_params(axis='y',length=0,pad=6)
ax.spines['left'].set_visible(False);ax.grid(axis='x',color=GRID,zorder=0)
base=vd.loc[vd['class_7.4'].eq('cationic')]
fig.text(.405,.205,f'{int(base.mean_delta.lt(0).sum())}/90 base-only drugs shift down',fontsize=12,weight='bold')
fig.text(.405,.175,f'Geometric mean ratio: {means[1]:.2f}×\nTheir input categories remain unchanged.',fontsize=10.7,color=MUTED,linespacing=1.45,va='top')

# C: only the six predeclared primary tests; use their adjusted intervals.
ax=axes[2]
primary=contrasts.loc[contrasts.primary_test & contrasts.scope.eq('all')].copy()
assert len(primary)==6
ax.axvline(0,color=INK,lw=1.1,ls='--',zorder=1)
models=[('GraphConv','GraphConv'),('Chemprop','Chemprop'),('ChemBERTa_head','Frozen\nChemBERTa')]
for y,(model,label) in enumerate(models):
    for ep,offset,color in [('Vd',-.13,ORANGE),('F',.13,CYAN)]:
        r=primary.loc[primary.backbone.eq(model)&primary.endpoint.eq(ep)].iloc[0]
        lo,hi=100*np.array([r.crossed_adjusted_low,r.crossed_adjusted_high])/r.comparator_mae
        assert lo>=-11 and hi<=11
        ax.plot([lo,hi],[y+offset,y+offset],color=color,lw=1.7,zorder=2)
        ax.scatter(r.relative_delta_pct,y+offset,s=38,marker='D',color=color,edgecolor=INK,lw=.6,zorder=3)
ax.set(xlim=(-11,11),ylim=(2.65,-.65),xlabel='Change in prediction error (%)\n← lower error     higher error →')
ax.set_xticks([-10,-5,0,5,10]);ax.set_yticks(range(3),[label for _,label in models]);ax.tick_params(axis='y',length=0,pad=6)
ax.spines['left'].set_visible(False);ax.grid(axis='x',color=GRID,zorder=0)
handles=[Line2D([],[],color=color,marker='D',markersize=4,label=label) for color,label in [(ORANGE,'Vd'),(CYAN,'Bioavailability')]]
ax.legend(handles=handles,frameon=False,loc='lower left',fontsize=9.5,handlelength=1.3)
fig.text(.755,.205,'Large shifts need not change average error',fontsize=11.2,weight='bold')
fig.text(.755,.175,'Most intervals cross zero.\nFrozen ChemBERTa Vd error rises 2.5%.',fontsize=10.7,color=MUTED,linespacing=1.45,va='top')

fig.text(.045,.065,'A/B: 327 held-out structures; each drug’s change averages nine paired seeds. B diamonds: geometric means. These are descriptive prediction shifts.',fontsize=9.5,color=MUTED)
fig.text(.045,.035,'C: MAE on observed labels (Vd n=259; F n=172); bars are adjusted 99.17% seed/group bootstrap intervals. Models are retrained; this is not a pH-exposure simulation.',fontsize=9.5,color=MUTED)

HERE.mkdir(parents=True,exist_ok=True)
stem='ionisation_three_key_plots'
for suffix in ['pdf','png','svg']:
    fig.savefig(HERE/f'{stem}.{suffix}',dpi=220,facecolor='white')
plt.close(fig)
f.to_csv(HERE/'panel_a_points.csv',index=False);vd.to_csv(HERE/'panel_b_points.csv',index=False)
primary.to_csv(HERE/'panel_c_estimates.csv',index=False)
(HERE/'caption.txt').write_text('''Three selected views of shared-recipe ALL4 versus REP4, a matched-width comparison.
A: distribution of per-structure, nine-seed-mean changes in predicted bioavailability for Chemprop; all 327 structures, including those without measured F labels.
B: each point is the ratio implied by a structure's nine-seed mean log2 prediction change for Vd in frozen ChemBERTa. Diamonds exponentiate the average log2 ratio within each pH 7.4 class. They are geometric mean ratios, not arithmetic percentage changes.
C: the six predeclared primary MAE contrasts. Relative changes and adjusted bootstrap bounds divide absolute MAE changes by the same fixed nine-seed mean comparator MAE. Lower is better. Bars are Bonferroni-adjusted 99.1667% crossed seed/group bootstrap intervals.
These selected descriptive patterns are not an organ-specific simulation. A and B characterize prediction changes, not gains in accuracy or validated mechanisms. The original full atlas contains all endpoints and comparisons.
''',encoding='utf-8')
outputs={p.name:sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file() and p.suffix in ['.pdf','.png','.svg','.csv','.txt']}
(HERE/'manifest.json').write_text(json.dumps({'sources':{str(p):sha256(p.read_bytes()).hexdigest() for p in paths},'outputs':outputs},indent=2),encoding='utf-8')
print('Created',HERE/f'{stem}.pdf')
