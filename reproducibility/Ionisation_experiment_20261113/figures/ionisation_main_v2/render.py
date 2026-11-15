"""Narrative ionisation figure from completed predictions and scoring tables."""
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor

HERE=Path(__file__).resolve().parent
EXP=HERE.parents[1]
OLD=EXP/'outputs/scoring'
NEW=EXP/'outputs/ph_followup/scoring'
SHIFTS=EXP/'figures/ph_followup_v1/prediction_shifts'
INK,MUTED,GRID,BOX='#263541','#586874','#E5EBEF','#F1F3F4'
MODELS={'GraphConv':'GraphConv','Chemprop':'Chemprop','ChemBERTa_head':'Frozen ChemBERTa + MLP'}
SHORT={'GraphConv':'GraphConv','Chemprop':'Chemprop','ChemBERTa_head':'Frozen ChemBERTa'}
ENDPOINTS={
    't_half':('Half-life','#0072B2',316),
    'Vd':('Volume of distribution','#D55E00',259),
    'CL':('Clearance','#168875',246),
    'Cmax':('Peak concentration','#8552A0',190),
    'Tmax':('Time to peak','#A17A16',249),
    'F':('Bioavailability','#4298B5',172),
    'PPB':('Protein binding','#C45D87',292),
}
SEEDS=[101,202,303,404,505,606,707,808,909]
JITTER=np.array([-.105,.085,-.05,.12,.015,-.08,.055,-.015,.09])
SOURCES={}
FIGURE_HEIGHT=17.8
SCATTERS=EXP/'figures/ionisation_observed_predicted_v1'
PH_ARMS={
    'REF74':'pH 7.4',
    'REP2':'pH 7.4 repeated twice',
    'ADD2':'pH 7.4 + 2.0',
    'ADD5':'pH 7.4 + 5.0',
    'ADD65':'pH 7.4 + 6.5',
    'REP4':'pH 7.4 repeated four times',
    'ALL4':'All four pHs',
}
plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':8.5,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':INK,'axes.edgecolor':'#85929A',
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','axes.linewidth':.7})


def read_csv(path,check_manifest=None):
    digest=sha256(path.read_bytes()).hexdigest()
    SOURCES[str(path.relative_to(EXP))]=digest
    if check_manifest:
        manifest=json.loads(check_manifest.read_text())
        assert manifest['files'][path.name]==digest,path.name
    return pd.read_csv(path)


def subset(d,**criteria):
    mask=pd.Series(True,index=d.index)
    for k,v in criteria.items():mask &= d[k].eq(v)
    return d.loc[mask]


def one(d,**criteria):
    x=subset(d,**criteria)
    assert len(x)==1,(criteria,len(x))
    return x.iloc[0]


def heading(fig,letter,text,x,y,size=10.4):
    fig.text(x,y,letter,fontsize=14.5,weight='bold',va='baseline')
    fig.text(x+.03,y+.001,text,fontsize=size,weight='bold',va='baseline')


def original_top_block(fig,draw,*args):
    """Keep the original A/B font sizes and physical geometry on a taller page."""
    old_axes=set(fig.axes);old_text=set(fig.texts)
    draw(fig,*args)
    scale=12.8/FIGURE_HEIGHT;offset=(FIGURE_HEIGHT-12.8)/FIGURE_HEIGHT
    for ax in set(fig.axes)-old_axes:
        x,y,w,h=ax.get_position().bounds
        ax.set_position([x,y*scale+offset,w,h*scale])
    for text in set(fig.texts)-old_text:
        x,y=text.get_position();text.set_position((x,y*scale+offset))


def arrow(ax,start,end,color=INK,lw=1.2):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,
                                lw=lw,color=color,shrinkA=0,shrinkB=0,zorder=3))


def molecule(ax):
    mol=Chem.MolFromSmiles('CC(=O)Nc1ccc(O)cc1')
    Chem.Kekulize(mol,clearAromaticFlags=True);rdDepictor.Compute2DCoords(mol)
    xy=np.array(mol.GetConformer().GetPositions())[:,:2]
    xy-=(xy.min(axis=0)+xy.max(axis=0))/2
    xy*=16/np.ptp(xy[:,0]);xy+=[10,22]
    for bond in mol.GetBonds():
        a,b=xy[bond.GetBeginAtomIdx()],xy[bond.GetEndAtomIdx()]
        v=b-a;n=np.array([-v[1],v[0]])/np.linalg.norm(v)
        for off in ([-.17,.17] if bond.GetBondTypeAsDouble()==2 else [0]):
            aa,bb=a+off*n,b+off*n
            ax.plot([aa[0],bb[0]],[aa[1],bb[1]],color=INK,lw=1.05,zorder=2)
    for atom,(x,y) in zip(mol.GetAtoms(),xy):
        if atom.GetSymbol()!='C':
            label=atom.GetSymbol()+('H' if atom.GetTotalNumHs() else '')
            ax.text(x,y,label,ha='center',va='center',fontsize=8.2,
                color='#AD4952' if atom.GetSymbol()=='O' else '#406E96',
                bbox=dict(fc='white',ec='none',pad=.25),zorder=5)


def schematic(fig,ledger):
    heading(fig,'A','Ionisation as a physiology-informed input',.035,.978)
    ax=fig.add_axes([.045,.793,.925,.166]);ax.set(xlim=(0,101),ylim=(-1,32));ax.axis('off')
    for x,label in [(10,'Molecular structure'),(35,'Structural encoder'),(64.5,'Prediction head'),(87,'PK endpoints')]:
        ax.text(x,30.2,label,ha='center',fontsize=8.3,weight='bold')
    molecule(ax);arrow(ax,(20,22),(26,22))
    layers=[(28,[18.5,22,25.5]),(34,[17.3,20.4,23.6,26.7]),(40,[18.5,22,25.5])]
    for (x1,ys1),(x2,ys2) in zip(layers[:-1],layers[1:]):
        for y1 in ys1:
            for y2 in ys2:ax.plot([x1,x2],[y1,y2],color='#CCD9E1',lw=.65,zorder=0)
    for x,ys in layers:
        for y in ys:ax.add_patch(Ellipse((x,y),1.35,1.9,fc='#EAF2F6',ec='#507C96',lw=.8,zorder=2))
    ax.text(35,12.7,'GraphConv · Chemprop\nFrozen ChemBERTa',ha='center',va='center',fontsize=7.6,color=MUTED,linespacing=1.25)
    arrow(ax,(42.5,22),(52.3,22))
    ax.add_patch(Ellipse((54,22),3.2,4.5,fc=BOX,ec='#8CA1AD',lw=.8))
    ax.text(54,22,'+',ha='center',va='center',fontsize=12)
    ax.text(54,27.7,'Concatenate',ha='center',fontsize=7.5,color=MUTED)
    arrow(ax,(55.8,22),(60,22))
    ax.add_patch(FancyBboxPatch((60,18.1),9,7.8,boxstyle='round,pad=.15,rounding_size=.6',fc=BOX,ec='#CCD4D9',lw=.8))
    ax.text(64.5,22,'MLP',ha='center',va='center',fontsize=9.5)
    arrow(ax,(69.3,22),(74,22))
    symbols=[(75,26,r'$t_{1/2}$','t_half'),(84,26,r'$V_d$','Vd'),(93,26,r'$CL$','CL'),
             (74.5,16.8,r'$C_{\max}$','Cmax'),(82,16.8,r'$T_{\max}$','Tmax'),
             (89.5,16.8,r'$F$','F'),(97,16.8,r'$PPB$','PPB')]
    for x,y,sym,ep in symbols:
        ax.add_patch(FancyBboxPatch((x-3.3,y-3.2),6.6,6.4,boxstyle='round,pad=.05,rounding_size=.5',fc=BOX,ec='none'))
        ax.text(x,y,sym,ha='center',va='center',fontsize=12 if ep in ['t_half','Vd','CL'] else 9.7,color=ENDPOINTS[ep][1])
    # pKa-derived one-hot inputs enter the fused representation before the head.
    ax.add_patch(FancyBboxPatch((2,.3),47,9.9,boxstyle='round,pad=.2,rounding_size=.7',fc=BOX,ec='#CCD4D9',lw=.8))
    ax.text(25.5,8.5,r'Acidic/basic p$K_a$ + pH → one-hot category',ha='center',va='center',fontsize=8.0,weight='bold')
    ax.text(25.5,5.9,'Acid-only · base-only · neither · dual-active',ha='center',va='center',fontsize=7.6,color=MUTED)
    for j,(p,label) in enumerate([(7.4,'7.4'),(2.,'2.0'),(5.,'5.0'),(6.5,'6.5')]):
        x=6.5+j*10.8
        ax.text(x,2.65,label,ha='center',va='center',fontsize=7.7)
        for k in range(4):
            ax.add_patch(FancyBboxPatch((x+2.2+k*1.2,1.7),1.,1.85,boxstyle='round,pad=.02,rounding_size=.1',
                                      fc='white',ec='#8BA2B2',lw=.55))
    ax.plot([49.2,54,54],[5,5,19.4],color=INK,lw=1.2)
    arrow(ax,(54,17.8),(54,19.5))
    ax.text(69.7,5.5,'pH 7.4 alone: 4 inputs\nFour-pH profile: 16 inputs',ha='center',va='center',fontsize=7.9,color=MUTED,linespacing=1.45)
    counts={ph:int(ledger[f'code_{ph}'].ne(ledger['code_7.4']).sum()) for ph in ['2.0','5.0','6.5']}
    assert counts=={'2.0':206,'5.0':85,'6.5':38}
    fig.text(.065,.777,'Category changes from pH 7.4:  '+ '   ·   '.join(f'pH {ph}: {n}/327' for ph,n in counts.items()),
             fontsize=8.2,color=MUTED)
    pd.DataFrame([{'pH':k,'changed_category':v,'n':327} for k,v in counts.items()]).to_csv(HERE/'panel_a_counts.csv',index=False)


def ionisation_performance(fig,contrasts,seeds,averages,average_seeds,ph_estimates):
    heading(fig,'B',r'Ionisation at pH 7.4 lowers $V_d$ and clearance error',.035,.746)
    fig.text(.065,.723,'Category vs structure only   ·   Dots: nine seeds   ·   Diamonds: means   ·   Bars: exploratory 95% intervals',fontsize=8.1,color=MUTED)
    rows=subset(contrasts,cohort='numerical',procedure='primary',arm='P1',comparator_arm='P0',scope='all',stratum='all').copy()
    legacy_rows=rows.copy()
    for idx,r in rows.iterrows():
        direct=one(ph_estimates,arm='REF74',backbone=r.backbone,endpoint=r.endpoint)
        assert np.isclose(direct.relative_delta_pct,r.relative_delta_pct)
        rows.loc[idx,'crossed95_low']=direct.crossed95_low_pct*r.comparator_mae/100
        rows.loc[idx,'crossed95_high']=direct.crossed95_high_pct*r.comparator_mae/100
    ss=subset(seeds,cohort='numerical',procedure='primary',arm='P1',comparator_arm='P0',scope='all',stratum='all')
    assert len(rows)==21 and len(ss)==189
    plotted=[];all_values=[]
    for _,r in rows.iterrows():
        s=subset(ss,backbone=r.backbone,endpoint=r.endpoint).sort_values('seed');assert s.seed.tolist()==SEEDS
        fac=100/r.comparator_mae;values=fac*s.delta.to_numpy()
        all_values.extend([*values,fac*r.crossed95_low,fac*r.crossed95_high])
        for seed,value in zip(SEEDS,values):plotted.append({'backbone':r.backbone,'endpoint':r.endpoint,'seed':seed,'relative_delta_pct':value})
    all_values.extend(average_seeds.relative_delta_pct)
    all_values.extend(averages.crossed95_low_pct)
    all_values.extend(averages.crossed95_high_pct)
    left=min(-20.,5*np.floor(min(all_values)/5));right=max(5.,5*np.ceil(max(all_values)/5))
    for j,(model,name) in enumerate(MODELS.items()):
        ax=fig.add_axes([.244+j*.242,.500,.211,.191])
        for y in [1,2]:ax.axhspan(y-.43,y+.43,color='#F3F6F7',zorder=0)
        ax.axhspan(6.65,7.43,color='#EDF1F3',zorder=0)
        ax.axhline(6.55,color='#CCD5DA',lw=.6)
        ax.axvline(0,color='#8B9AA5',lw=.8,ls='--',zorder=1);ax.grid(axis='x',color=GRID,lw=.6,zorder=0)
        for y,(ep,(_,color,n)) in enumerate(ENDPOINTS.items()):
            r=one(rows,backbone=model,endpoint=ep);assert r.n_structures==n
            s=subset(ss,backbone=model,endpoint=ep).sort_values('seed');factor=100/r.comparator_mae
            ax.scatter(s.delta.to_numpy()*factor,y+JITTER,s=10,color=color,alpha=.6,edgecolor='none',zorder=2)
            ax.plot(factor*np.array([r.crossed95_low,r.crossed95_high]),[y,y],color=color,lw=1.15,zorder=3)
            ax.plot(r.relative_delta_pct,y,'D',ms=4,color=color,mec=INK,mew=.5,zorder=4)
        avg=one(averages,backbone=model)
        avg_seeds=subset(average_seeds,backbone=model).sort_values('seed')
        assert avg_seeds.seed.tolist()==SEEDS
        expected=pd.DataFrame(plotted).query('backbone == @model').groupby('seed').relative_delta_pct.mean().reindex(SEEDS)
        assert np.allclose(expected,avg_seeds.relative_delta_pct)
        ax.scatter(avg_seeds.relative_delta_pct,7+JITTER,s=10,color=MUTED,alpha=.65,edgecolor='none',zorder=2)
        ax.plot([avg.crossed95_low_pct,avg.crossed95_high_pct],[7,7],color=INK,lw=1.2,zorder=3)
        ax.plot(avg.relative_delta_pct,7,'D',ms=4.3,color=INK,mec=INK,mew=.5,zorder=4)
        ax.set(xlim=(left,right),ylim=(7.65,-.6),yticks=range(8))
        labels=[f'{label} (n={n})' for label,_,n in ENDPOINTS.values()]+['Average (7 endpoints)']
        ax.set_yticklabels(labels if j==0 else [],fontsize=8.2)
        if j==0:ax.get_yticklabels()[-1].set_weight('bold')
        ax.set_title(name,fontsize=8.6,weight='bold',pad=8)
        ax.tick_params(axis='y',length=0,pad=5);ax.spines['left'].set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5));ax.tick_params(axis='x',labelsize=7.8)
    fig.text(.60,.469,'Change in MAE (% of structure-only MAE)     ← lower error     higher error →',ha='center',fontsize=8.5)
    legacy_rows.to_csv(HERE/'panel_b_estimates.csv',index=False)
    rows.to_csv(HERE/'panel_b_displayed_estimates.csv',index=False)
    pd.DataFrame(plotted).to_csv(HERE/'panel_b_seeds.csv',index=False)


def vd_scatter_row(fig):
    manifest=json.loads((SCATTERS/'manifest.json').read_text())
    for name in ['plotted_points.csv','metrics.csv']:
        assert sha256((SCATTERS/name).read_bytes()).hexdigest()==manifest['outputs'][name]
    data=read_csv(SCATTERS/'plotted_points.csv')
    data=subset(data,arm='REF74',endpoint='Vd')
    metrics=read_csv(SCATTERS/'metrics.csv')
    classes={'anionic':('Acid-only (anionic)','#0072B2'),
             'cationic':('Base-only (cationic)','#D55E00'),
             'neutral':('Neither-active','#7C8A92'),
             'zwitterionic':('Dual-active','#8552A0')}
    fig.text(.065,10.72/FIGURE_HEIGHT,r'Observed versus predicted $V_d$ with the pH 7.4 input',fontsize=9.6,weight='bold')
    handles=[Line2D([],[],marker='o',color=colour,ls='none',ms=4,label=label)
             for label,colour in classes.values()]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.235,10.56/FIGURE_HEIGHT),
               ncol=4,frameon=False,fontsize=6.8,handletextpad=.2,columnspacing=.9,borderaxespad=0)
    lo,hi=manifest['axis_limits_log10']['Vd']
    for j,(model,name) in enumerate(MODELS.items()):
        ax=fig.add_axes([.244+j*.242,7.94/FIGURE_HEIGHT,.211,(.211*9.4)/FIGURE_HEIGHT])
        d=subset(data,backbone=model);assert len(d)==259
        r=one(metrics,arm='REF74',endpoint='Vd',backbone=model)
        y,x=d.observed_log10.to_numpy(),d.predicted_log10.to_numpy()
        assert np.isclose(1-((y-x)**2).sum()/((y-y.mean())**2).sum(),r.r2_displayed_seed_mean)
        ax.plot([lo,hi],[lo,hi],color='#9AA5AD',ls=(0,(4,3)),lw=.7,zorder=1)
        for cat,(_,colour) in classes.items():
            c=subset(d,ionisation_class=cat)
            ax.scatter(c.predicted_log10,c.observed_log10,s=9,color=colour,alpha=.78,
                       edgecolor='white',linewidth=.15,zorder=3)
        ax.set(xlim=(lo,hi),ylim=(lo,hi));ax.set_aspect('equal',adjustable='box')
        ax.spines[['top','right']].set_visible(True)
        ax.set_xticks([1,2,3,4,5]);ax.set_yticks([1,2,3,4,5]);ax.tick_params(labelsize=7.5)
        ax.grid(color=GRID,lw=.5)
        ax.set_title(name+'\n'+r'$R^2$'+f' = {r.r2_displayed_seed_mean:.3f}  |  n = 259',
                     fontsize=8.2,weight='bold',pad=7)
        ax.set_xlabel(r'Predicted $\log_{10} V_d$ (L)',fontsize=8.0,labelpad=3)
        if j==0:ax.set_ylabel(r'Observed $\log_{10} V_d$ (L)',fontsize=8.0,labelpad=4)
    fig.text(.244,7.37/FIGURE_HEIGHT,'Each point: nine-seed mean prediction in log10 space. Baseline comparisons: Figures S12-S13.',fontsize=7.5,color=MUTED)
    data.to_csv(HERE/'panel_b_vd_scatter_points.csv',index=False)


def ph_accuracy_panels(fig,estimates,seeds):
    rows=estimates.loc[estimates.endpoint.isin(['Vd','CL'])]
    ss=seeds.loc[seeds.endpoint.isin(['Vd','CL'])]
    assert len(rows)==42 and len(ss)==378
    values=np.r_[rows.crossed95_low_pct,rows.crossed95_high_pct,ss.relative_delta_pct]
    left=min(-20.,5*np.floor(values.min()/5));right=max(5.,5*np.ceil(values.max()/5))
    for letter,ep,title,title_y,axes_y in [
        ('C','Vd',r'$V_d$ accuracy with different pH inputs',7.10,4.47),
        ('D','CL','Clearance accuracy with different pH inputs',3.77,1.14)]:
        heading(fig,letter,title,.035,title_y/FIGURE_HEIGHT)
        fig.text(.065,(title_y-.25)/FIGURE_HEIGHT,
                 'All changes vs structure only  ·  Dots: nine seeds  ·  Diamonds: means  ·  Bars: exploratory 95% intervals',
                 fontsize=8.0,color=MUTED)
        for j,(model,name) in enumerate(MODELS.items()):
            ax=fig.add_axes([.244+j*.242,axes_y/FIGURE_HEIGHT,.211,2.12/FIGURE_HEIGHT])
            ax.axvline(0,color='#8B9AA5',lw=.8,ls='--',zorder=1)
            for y,arm in enumerate(PH_ARMS):
                colour=MUTED if arm in ['REP2','REP4'] else ENDPOINTS[ep][1]
                if arm in ['REP2','REP4']:ax.axhspan(y-.43,y+.43,color='#F1F3F4',zorder=0)
                r=one(rows,arm=arm,backbone=model,endpoint=ep)
                s=subset(ss,arm=arm,backbone=model,endpoint=ep).sort_values('seed')
                assert s.seed.tolist()==SEEDS and r.n_structures==ENDPOINTS[ep][2]
                ax.scatter(s.relative_delta_pct,y+JITTER,s=9,color=colour,alpha=.5,edgecolor='none',zorder=2)
                ax.plot([r.crossed95_low_pct,r.crossed95_high_pct],[y,y],color=colour,lw=1.05,zorder=3)
                ax.plot(r.relative_delta_pct,y,'D',ms=3.8,color=colour,mec=INK,mew=.45,zorder=4)
            ax.set(xlim=(left,right),ylim=(6.6,-.6),yticks=range(7))
            ax.set_yticklabels(list(PH_ARMS.values()) if j==0 else [],fontsize=8.0)
            ax.tick_params(axis='y',length=0,pad=5);ax.spines['left'].set_visible(False)
            ax.grid(axis='x',color=GRID,lw=.6)
            ax.set_title(name,fontsize=8.6,weight='bold',pad=8)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5));ax.tick_params(axis='x',labelsize=7.8)
        fig.text(.60,(axes_y-.34)/FIGURE_HEIGHT,
                 'Change in MAE (% of structure-only MAE)     ← lower error     higher error →',ha='center',fontsize=8.5)
    fig.text(.065,.44/FIGURE_HEIGHT,
             'Grey rows: repeated pH 7.4 controls. Added pHs: 2.0, 5.0, 6.5. Full seven-endpoint results: Figure S14.',fontsize=7.8,color=MUTED)
    fig.text(.50,.18/FIGURE_HEIGHT,
             'Grouped five-fold outer cross-validation · 327 structures · nine paired training seeds',ha='center',fontsize=8.2,color=MUTED)


def redistribution(fig,structure,summary,seed_shifts):
    heading(fig,'C','The pH profile redistributes predictions',.035,.455,size=9.5)
    heading(fig,'D',r'$V_d$ shifts differ by category stability',.555,.455,size=9.5)
    fig.text(.065,.435,'Bioavailability: four-pH profile vs repeated pH 7.4',fontsize=7.9,color=MUTED)
    fig.text(.585,.435,'Four-pH profile vs repeated pH 7.4',fontsize=7.9,color=MUTED)
    d=subset(structure,procedure='primary',arm='ALL4',comparator_arm='REP4',endpoint='F')
    ax=fig.add_axes([.173,.286,.327,.130]);rng=np.random.default_rng(20261113)
    ax.axvline(0,color='#8B9AA5',lw=.8,ls='--');medabs=[]
    low=min(-30,5*np.floor(d.mean_delta.min()/5));high=max(30,5*np.ceil(d.mean_delta.max()/5))
    for y,(model,_) in enumerate(MODELS.items()):
        vals=subset(d,backbone=model).sort_values('structure_id').mean_delta.to_numpy();assert len(vals)==327
        p10,p25,med,p75,p90=np.quantile(vals,[.1,.25,.5,.75,.9]);medabs.append(np.median(abs(vals)))
        ax.scatter(vals,y+rng.uniform(-.22,.22,len(vals)),s=4.8,color=ENDPOINTS['F'][1],alpha=.25,edgecolor='none',zorder=2)
        ax.plot([p10,p90],[y,y],color=ENDPOINTS['F'][1],lw=1.1,zorder=3)
        ax.plot([p25,p75],[y,y],color=ENDPOINTS['F'][1],lw=3,zorder=3)
        ax.plot(med,y,'o',ms=4,markerfacecolor='white',markeredgecolor=INK,markeredgewidth=.6,zorder=4)
    ax.set(xlim=(low,high),ylim=(2.6,-.6),yticks=range(3));ax.set_yticklabels(list(SHORT.values()),fontsize=7.8)
    ax.set_xticks([-30,-15,0,15,30]);ax.tick_params(axis='y',length=0,pad=5)
    ax.spines['left'].set_visible(False);ax.grid(axis='x',color=GRID,lw=.6)
    ax.set_xlabel('Change in predicted F (percentage points)',fontsize=8.2,labelpad=3)
    fig.text(.065,.246,'Median |change|: '+ ' / '.join(f'{v:.1f}' for v in medabs)+' points (top to bottom)',fontsize=7.8,color=MUTED)
    # Plot subgroup-geometric mean ratios per seed, then the pooled ratio.
    sd=subset(seed_shifts,procedure='primary',arm='ALL4',comparator_arm='REP4',endpoint='Vd',scope='switch_status')
    sr=subset(summary,procedure='primary',arm='ALL4',comparator_arm='REP4',endpoint='Vd',scope='switch_status')
    ax=fig.add_axes([.700,.286,.258,.130]);ax.axvline(0,color='#8B9AA5',lw=.8,ls='--')
    vals_all=100*np.expm1(sd.mean_delta.to_numpy()*np.log(2))
    lim=max(15,5*np.ceil(np.abs(vals_all).max()/5))
    out=[]
    for y,(model,_) in enumerate(MODELS.items()):
        for status,offset,marker,face in [('stable',-.15,'o','white'),('switcher',.15,'D',ENDPOINTS['Vd'][1])]:
            ss=subset(sd,backbone=model,stratum=status).sort_values('seed');assert ss.seed.tolist()==SEEDS
            r=one(sr,backbone=model,stratum=status)
            expected_n=121 if status=='stable' else 206;assert r.n_structures==expected_n
            values=100*np.expm1(ss.mean_delta.to_numpy()*np.log(2));value=100*np.expm1(r.mean_signed*np.log(2))
            ax.scatter(values,y+offset+JITTER*.45,s=8,color=ENDPOINTS['Vd'][1],alpha=.4,edgecolor='none',zorder=2)
            ax.plot(value,y+offset,marker=marker,ms=4.4,mfc=face,mec=INK,mew=.65,linestyle='none',zorder=3)
            out.append({'backbone':model,'stratum':status,'n':expected_n,'geometric_mean_prediction_change_pct':value})
    ax.set(xlim=(-lim,lim),ylim=(2.6,-.6),yticks=range(3));ax.set_yticklabels(list(SHORT.values()),fontsize=7.8)
    ax.tick_params(axis='y',length=0,pad=5);ax.spines['left'].set_visible(False);ax.grid(axis='x',color=GRID,lw=.6)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4));ax.set_xlabel('Change in predicted Vd (%)',fontsize=8.2,labelpad=3)
    handles=[Line2D([],[],marker='o',mfc='white',mec=INK,lw=0,ms=4,label='Stable (121)'),
             Line2D([],[],marker='D',mfc=ENDPOINTS['Vd'][1],mec=INK,lw=0,ms=4,label='Switching (206)')]
    fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.585,.249),frameon=False,ncol=2,fontsize=7.5,
               handletextpad=.35,columnspacing=1.0,borderaxespad=0)
    d.to_csv(HERE/'panel_c_points.csv',index=False)
    sr.to_csv(HERE/'panel_d_summary.csv',index=False);sd.to_csv(HERE/'panel_d_seeds.csv',index=False)
    pd.DataFrame(out).to_csv(HERE/'panel_d_plotted.csv',index=False)


def profile_accuracy(fig,contrasts):
    heading(fig,'E','Accuracy after adding the four-pH profile',.035,.213)
    fig.text(.065,.191,'Four-pH profile vs repeated pH 7.4   ·   Diamonds: means   ·   Bars: adjusted 99.17% intervals',fontsize=7.9,color=MUTED)
    d=subset(contrasts,procedure='primary',arm='ALL4',comparator_arm='REP4',scope='all',stratum='all')
    d=d.loc[d.endpoint.isin(['Vd','F'])];assert len(d)==6 and d.primary_test.all()
    for j,(ep,label,x) in enumerate([('Vd','Volume of distribution (n=259)',.218),('F','Bioavailability (n=172)',.681)]):
        ax=fig.add_axes([x,.067,.277,.100]);ax.axvline(0,color='#8B9AA5',lw=.8,ls='--')
        for y,(model,_) in enumerate(MODELS.items()):
            r=one(d,backbone=model,endpoint=ep);color=ENDPOINTS[ep][1];factor=100/r.comparator_mae
            lo,hi=factor*np.array([r.crossed_adjusted_low,r.crossed_adjusted_high]);assert lo>=-11 and hi<=11
            ax.plot([lo,hi],[y,y],color=color,lw=1.2)
            ax.plot(r.relative_delta_pct,y,'D',ms=4,color=color,mec=INK,mew=.5)
        ax.set(xlim=(-11,11),ylim=(2.55,-.55),yticks=range(3));ax.set_yticklabels(list(SHORT.values()),fontsize=7.8)
        ax.tick_params(axis='y',length=0,pad=5);ax.spines['left'].set_visible(False);ax.grid(axis='x',color=GRID,lw=.6)
        ax.set_xticks([-10,-5,0,5,10]);ax.set_title(label,fontsize=8.3,pad=6,weight='bold')
        ax.set_xlabel('Change in MAE (%)',fontsize=8.3,labelpad=3)
    fig.text(.50,.009,'Grouped five-fold outer cross-validation · 327 structures · nine paired training seeds',ha='center',fontsize=8.2,color=MUTED)
    d.to_csv(HERE/'panel_e_estimates.csv',index=False)


CAPTION='''Effects of ionisation inputs on PK prediction.

A, Molecular structural representations are concatenated with a categorical ionisation input before the prediction head. The four operational categories are acid-only, base-only, neither-active and dual-active, defined using the acidic and basic pKa values at each pH. A single pH contributes a four-dimensional one-hot vector; the four-pH profile contributes four such blocks. Empty cells in the schematic indicate input dimensions rather than assignments to the illustrative molecule. Categories differ from their pH 7.4 assignment for 206, 85 and 38 of the 327 structures at pH 2.0, 5.0 and 6.5, respectively. The diagram orders pHs for presentation; the implemented profile order is 7.4, 2.0, 6.5, 5.0.

B, Adding the pH 7.4 category to structure-only models on the numerical-pKa cohort. Dots show nine paired seed differences in held-out MAE; diamonds are means and bars are exploratory 95% crossed seed/group bootstrap intervals. All observed endpoint labels enter their respective comparisons. Shading identifies Vd and clearance. Five positive endpoints use log10 MAE; F and PPB use percentage-point MAE, retaining zero labels. Each absolute paired change and interval is divided by the fixed nine-seed mean comparator MAE. The Average row gives equal weight to the seven endpoint percentage changes within each seed, followed by the mean across seeds. Its interval resamples groups and seed trajectories jointly across all seven endpoints, preserving their dependence and using each endpoint's resampled labelled count. Average changes are -3.79%, -5.47% and -1.49% for GraphConv, Chemprop and frozen ChemBERTa, respectively. These numerical-cohort comparisons are secondary analyses of the ionisation representation experiment. The 494-structure categorical-cohort results and their multiplicity-adjusted intervals appear in Supplementary Figure S1; the main figure uses its 327-structure subset throughout.

B, lower row: observed versus predicted Vd for the 259 labelled structures, with the pH 7.4 category supplied to each model. Horizontal coordinates are mean held-out log10 predictions across nine training seeds, corresponding to geometric mean predictions on the litre scale. Vertical coordinates are observed log10 Vd. R-squared is computed from the displayed averages as 1 - sum((observed - predicted)^2)/sum((observed - mean(observed))^2); it is distinct from averaging nine separately computed seed scores. Colour indicates the operational pH 7.4 category. All three models use identical axes and all observed labels are retained. The dashed line denotes equality. Matched structure-only and ionisation scatter plots for Vd and clearance appear in Figures S12 and S13.

C-D, Relative changes in held-out MAE for Vd (259 labels) and clearance (246 labels) under alternative categorical pH inputs, each compared with the same structure-only model. Rows show pH 7.4 alone, two repeats of pH 7.4, pH 7.4 plus 2.0, 5.0 or 6.5, four repeats of pH 7.4, and all four pHs. Dots are nine paired seed differences, diamonds are means, and bars are exploratory unadjusted 95% crossed seed/group bootstrap intervals. Every effect is normalised by the fixed original nine-seed mean structure-only endpoint MAE. The grey rows are repeated-input controls: they retain pH 7.4 information while changing the input parameterisation. Two-block and four-block vectors have eight and sixteen channels, scaled by 1/sqrt(2) and 1/2, respectively. Added pH information is assessed against its corresponding repeated-input control in the supplementary analyses. Benefits versus structure only include the effect of the pH 7.4 category and cannot, by themselves, be attributed to the additional pHs. The pH 6.5 analysis is descriptive because only 38 structures change category. Full seven-endpoint comparisons are in Figure S14; width-matched four-pH results and multiplicity-adjusted primary intervals are in Figure S6.

Predictions are pooled across five parent/scaffold-grouped held-out outer folds within each seed. All panels use the same 327 structures in 238 groups with nine paired seeds. Bootstrap intervals use 10,000 draws of whole groups and whole seed trajectories, conditional on the observed data, folds and selected settings. The ChemBERTa transformer is frozen and its MLP head is trained. Supervised endpoint weights are all one and the mass-balance penalty is zero. Numeric pH conditions define input features for prediction of the original PK endpoints.
'''


if __name__=='__main__':
    HERE.mkdir(parents=True,exist_ok=True)
    old=read_csv(OLD/'contrasts.csv',OLD/'scoring_manifest.json')
    seeds=read_csv(OLD/'contrast_seeds.csv',OLD/'scoring_manifest.json')
    averages=read_csv(HERE/'panel_b_average_estimates.csv',HERE/'panel_b_average_manifest.json')
    average_seeds=read_csv(HERE/'panel_b_average_seeds.csv',HERE/'panel_b_average_manifest.json')
    ph=read_csv(HERE/'panel_ph_vs_baseline_estimates.csv',HERE/'panel_ph_vs_baseline_manifest.json')
    ph_seeds=read_csv(HERE/'panel_ph_vs_baseline_seeds.csv',HERE/'panel_ph_vs_baseline_manifest.json')
    ledger=read_csv(EXP/'runtime/ph_category_features/ledger.csv')
    assert len(ledger)==327 and ledger.structure_id.nunique()==327 and ledger.group_id.nunique()==238
    fig=plt.figure(figsize=(9.4,FIGURE_HEIGHT),facecolor='white')
    original_top_block(fig,schematic,ledger)
    original_top_block(fig,ionisation_performance,old,seeds,averages,average_seeds,ph)
    vd_scatter_row(fig)
    ph_accuracy_panels(fig,ph,ph_seeds)
    for ext in ['pdf','png','svg']:fig.savefig(HERE/f'ionisation_main.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    (HERE/'caption.txt').write_text(CAPTION,encoding='utf-8')
    (HERE/'README.txt').write_text('''IONISATION MAIN FIGURE
Open ionisation_main.pdf or ionisation_main.png.
Editable vector output: ionisation_main.svg. Render source: render.py.
The main figure uses the same 327-structure numerical-pKa cohort throughout.
Panel B includes an equally weighted seven-endpoint average and Vd scatter plots.
Panels C and D compare pH representations with structure-only Vd and clearance predictions.
Figures S12-S13 show the matched baseline scatter plots; S14 covers all seven endpoints.
Larger-cohort, representation, baseline and sensitivity results are in supplementary/.
caption.txt defines all comparisons, units, uncertainty and reporting scope.
panel_*.csv contain plotted values. manifest.json records source/output hashes.
Reproduce with revision_work/.venv/Scripts/python.exe from the PIPKNET workspace.
No new training or paid computation was used to make this figure.
''',encoding='utf-8')
    outputs={p.name:sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir()
             if p.is_file() and p.suffix in ['.pdf','.png','.svg','.csv','.txt']}
    (HERE/'manifest.json').write_text(json.dumps({'sources':SOURCES,'outputs':outputs,
        'main_cohort_structures':327,'main_cohort_groups':238,'seeds':SEEDS},indent=2),encoding='utf-8')
    print('Created',HERE/'ionisation_main.pdf')
