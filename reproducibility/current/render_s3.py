"""Render the complete prespecified robustness overview from verified real scores.

Only numpy, pandas and matplotlib; no model imports or historical builders.
"""
from pathlib import Path
import argparse, hashlib, json, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MODELS=['GraphConv','Chemprop','ChemBERTa_head']
COLORS=['#27808e','#d58b32','#9c658b']
NAMES=['GraphConv','Chemprop','ChemBERTa']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scores',type=Path,default=ROOT/'Review_statistics_20261114/outputs_with_real_matrix')
    ap.add_argument('--output',type=Path,default=HERE/'rendered_S3')
    ap.add_argument('--stage',type=Path)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.scores/'contrasts.csv',float_precision='round_trip')
    q=d.loc[d.cohort.eq('unified327')&d.stratum.eq('all')].set_index(['protocol','model','contrast','metric'],verify_integrity=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':10,'axes.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(3,2,figsize=(10.5,11.5),layout='constrained')
    used=[]
    def panel(ax,letter,title,protocol,rows,xlabel,absolute=False):
        ax.set_title(title,loc='left',pad=10)
        ax.text(-.16,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=14)
        for k,(label,contrast,metric) in enumerate(rows):
            for j,model in enumerate(MODELS):
                r=q.loc[(protocol,model,contrast,metric)]
                v,lo,hi=(r.absolute_delta,r.absolute_low,r.absolute_high) if absolute else (r.relative_pct,r.primary_low,r.primary_high)
                y=k+(j-1)*.21
                ax.errorbar(v,y,xerr=np.array([[v-lo],[hi-v]]),fmt='o',ms=3.5,lw=.9,capsize=2,color=COLORS[j])
                used.append(dict(panel=letter,protocol=protocol,model=model,contrast=contrast,metric=metric,estimate=v,low=lo,high=hi,scale=r.scale,denominator_condition=r.denominator_condition))
            if k%2==0:ax.axhspan(k-.43,k+.43,color='#f3f4f5',zorder=-2)
        ax.axvline(0,c='.5',lw=.7,zorder=-1)
        ax.set_yticks(range(len(rows)),[r[0] for r in rows]);ax.invert_yaxis()
        ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.15)
    labels={'Vd':r'$V_d$','CL':r'$CL$','triad_mae':'triad'}
    desc=[(f'{name}: {labels[ep]}',f'P{i}_off-P0_off',ep) for i,name in [(1,'Categories'),(2,r'Raw p$K_a$'),(3,'Fractions')] for ep in ['Vd','CL']]
    panel(axs[0,0],'A','Fixed capacity (F): descriptor accuracy','F_new',desc,'MAE change vs F structure only (%)')
    panel(axs[0,1],'B','Input-tuned (T): descriptor accuracy','T_new',desc,'MAE change vs T structure only (%)')
    names=['Structure','Categories',r'Raw p$K_a$','Fractions']
    panel(axs[1,0],'C','T: penalty effect on triad accuracy','T_new',[(n,f'P{i}_positive-P{i}_off','triad_mae') for i,n in enumerate(names)],'Triad MAE change vs same-input loss off (%)')
    panel(axs[1,1],'D','T: penalty effect on internal consistency','T_new',[(n,f'P{i}_positive-P{i}_off','mean_abs_residual') for i,n in enumerate(names)],r'Change in mean absolute residual (log$_{10}$)',True)
    panel(axs[2,0],'E','F: separate versus shared penalty selection','F_new',[(f'{n}: {labels[ep]}',f'P{i}_separate-P{i}_shared',ep) for i,n in [(0,'Structure'),(1,'Categories')] for ep in ['Vd','CL','triad_mae']],'MAE change vs shared penalty (%)')
    panel(axs[2,1],'F','T: input by penalty interactions','T_new',[(f'{n}: {labels[ep]}',f'interaction_{c}',ep) for n,c in [('Categories','category'),(r'Raw p$K_a$','raw_pka'),('Fractions','fractions')] for ep in ['Vd','CL']],'Interaction on T structure-only scale (pp)')
    handles=[plt.Line2D([],[],marker='o',ls='',color=c,label=n,ms=5) for c,n in zip(COLORS,NAMES)]
    fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False)
    for ext in ['pdf','png','svg']:
        fig.savefig(a.output/f'figS3_robustness.{ext}',dpi=240,metadata={'CreationDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None)
    plt.close(fig)
    pd.DataFrame(used).to_csv(a.output/'plotted_values.csv',index=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(source_sha256=sha(a.scores/'contrasts.csv'),builder_sha256=sha(Path(__file__)),n_plotted_rows=len(used),interval='10,000 paired whole-group draws; nine fixed seeds; recomputed declared denominator',artifacts={p.name:sha(p) for p in a.output.iterdir() if p.is_file() and p.name!='manifest.json'})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if a.stage:
        (a.stage/'figures').mkdir(parents=True,exist_ok=True)
        shutil.copyfile(a.output/'figS3_robustness.pdf',a.stage/'figures/figS3_robustness.pdf')
    print(json.dumps({'plotted_rows':len(used),'output':str(a.output)}))

if __name__=='__main__':main()
