"""Summarise paired prediction changes from the frozen categorical-pH campaign."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ionisation_style import AXES, DIVERGING, FONT, GRID, INK, MUTED, ENDPOINT_COLORS
EXP = OUT.parents[2]
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909)
MODELS = {'GraphConv': 'GraphConv', 'Chemprop': 'Chemprop', 'ChemBERTa_head': 'Frozen ChemBERTa + MLP'}
ENDPOINTS = {
    't_half': ('Half-life', ENDPOINT_COLORS["t_half"]), 'Vd': ('Volume of distribution', ENDPOINT_COLORS["Vd"]),
    'CL': ('Clearance', ENDPOINT_COLORS["CL"]), 'Cmax': ('Peak concentration', ENDPOINT_COLORS["Cmax"]),
    'Tmax': ('Time to peak', ENDPOINT_COLORS["Tmax"]), 'F': ('Bioavailability', ENDPOINT_COLORS["F"]),
    'PPB': ('Protein binding', ENDPOINT_COLORS["PPB"]),
}
CLASSES = ('anionic', 'cationic', 'neutral', 'zwitterionic')
PH = ('7.4', '2.0', '6.5', '5.0')
CONTRASTS = (
    ('primary', 'ALL4', 'REP4', 'any added pH'), ('tuned', 'ALL4', 'REP4', 'any added pH'),
    ('primary', 'ADD2', 'REP2', '2.0'), ('primary', 'ADD65', 'REP2', '6.5'),
    ('primary', 'ADD5', 'REP2', '5.0'), ('primary', 'REF74', 'REF0', 'any added pH'),
    ('primary', 'REP4', 'REF74', 'any added pH'),
)
plt.rcParams.update({'font.family': FONT, 'font.size': 10, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': INK,
                     'axes.edgecolor': AXES, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load_inputs():
    out = EXP/'outputs/ph_followup'
    ledger_dir = EXP/'runtime/ph_category_features'
    release_dir = EXP/'data/release/numerical'
    pred_path = out/'outer_predictions.csv'
    pred_manifest_path = out/'outer_predictions.manifest.json'
    provenance_path = out/'outer_predictions.provenance.json'
    score_manifest_path = out/'scoring/scoring_manifest.json'
    release_manifest_path = release_dir/'manifest.json'
    ledger_manifest_path = ledger_dir/'manifest.json'
    pred_manifest, score = read(pred_manifest_path), read(score_manifest_path)
    release, ledger_manifest = read(release_manifest_path), read(ledger_manifest_path)
    fingerprint = release['fingerprint']
    release_identity = {k:v for k,v in release.items() if k not in ('fingerprint','data_frozen','splits_frozen','training_ready')}
    if (digest(release_identity) != fingerprint or pred_manifest['release_fingerprints'] != {'numerical':fingerprint} or
            score['release_fingerprints'] != {'numerical':fingerprint} or not all(release[k] is True for k in ('data_frozen','splits_frozen','training_ready'))):
        raise ValueError('Numerical release fingerprint or ready flags disagree')
    if (pred_manifest['campaign_id'] != score['campaign_id'] or score['status'] != 'report_ready' or
            score['predictions_sha256'] != pred_manifest['sha256'] or sha(pred_path) != pred_manifest['sha256'] or
            sha(pred_manifest_path) != score['collector_manifest_sha256'] or
            sha(provenance_path) != score['source_fit_ledger_sha256'] or sha(provenance_path) != pred_manifest['provenance_sha256']):
        raise ValueError('Prediction export and provenance checksums disagree')
    if (digest({k:v for k,v in ledger_manifest.items() if k!='fingerprint'}) != ledger_manifest['fingerprint'] or
            ledger_manifest['fingerprint'] != pred_manifest['categorical_feature_fingerprint'] or
            ledger_manifest['release_fingerprint'] != fingerprint or ledger_manifest['release']['release'] != sha(release_manifest_path)):
        raise ValueError('Categorical feature ledger fingerprint mismatch')
    for directory, manifest in ((release_dir, release), (ledger_dir, ledger_manifest)):
        if any(sha(directory/name)!=expected for name,expected in manifest['files'].items()):
            raise ValueError(f'Frozen file checksum mismatch under {directory}')
    data = pd.read_csv(release_dir/'model_dataset.csv',dtype={'structure_id':str}).set_index('structure_id')
    ledger = pd.read_csv(ledger_dir/'ledger.csv',dtype={'structure_id':str}).set_index('structure_id')
    codes = read(ledger_dir/'codes.json')
    if (len(data)!=327 or len(ledger)!=327 or not data.index.is_unique or not ledger.index.is_unique or
            data.group_id.nunique()!=238 or set(ledger.index)!=set(data.index) or set(codes)!=set(data.index)):
        raise ValueError('Structure or group coverage differs from the frozen release')
    ledger = ledger.loc[data.index].copy()
    for col in ('group_id','pka_acid','pka_base','ionisation_class'):
        if not ledger[col].equals(data[col]):
            raise ValueError(f'Frozen ledger {col} differs from the modelling release')
    code_array = np.array([codes[sid] for sid in ledger.index],dtype=int)
    if not np.array_equal(code_array,ledger[[f'code_{ph}' for ph in PH]].to_numpy()) or not np.array_equal(np.array(CLASSES)[code_array[:,0]],ledger.ionisation_class):
        raise ValueError('pH code and audited class lookup disagree')
    for j,ph in enumerate(PH[1:],1):
        ledger[f'switch_{ph}'] = code_array[:,j] != code_array[:,0]
    ledger['switch_any added pH'] = (code_array[:,1:] != code_array[:,:1]).any(axis=1)
    if (int(ledger['switch_any added pH'].sum())!=206 or ledger.loc[ledger.ionisation_class.eq('cationic'),'switch_any added pH'].any() or
            int(ledger.ionisation_class.eq('cationic').sum())!=90):
        raise ValueError('The stable/cationic feature identity check failed')
    predictions = pd.read_csv(pred_path,dtype={'structure_id':str})
    keys = ['cohort','procedure','arm','backbone','seed','structure_id']
    if (len(predictions)!=pred_manifest['rows'] or predictions.duplicated(keys).any() or
            set(predictions.seed)!=set(SEEDS) or set(predictions.structure_id)!=set(data.index) or set(predictions.cohort)!={'numerical'} or
            not predictions.outer_fold.eq(predictions.structure_id.map(data.outer_fold)).all()):
        raise ValueError('Unexpected prediction coverage, pairing or frozen fold assignment')
    provenance = read(provenance_path)
    lookup = pd.DataFrame(provenance['fits'])
    provenance_key = ['procedure','arm','backbone','outer_fold','seed']
    actual = predictions[provenance_key+['source_campaign_id','source_fit_id']].drop_duplicates()
    expected = lookup[actual.columns]
    if not actual.sort_values(provenance_key).reset_index(drop=True).equals(expected.sort_values(provenance_key).reset_index(drop=True)):
        raise ValueError('Prediction source-fit identities disagree with provenance mapping')
    inputs = {p.relative_to(EXP).as_posix(): sha(p) for p in
              (pred_path,pred_manifest_path,provenance_path,score_manifest_path,release_manifest_path,
               release_dir/'model_dataset.csv',ledger_manifest_path,ledger_dir/'ledger.csv',ledger_dir/'codes.json')}
    return predictions, ledger, inputs, pred_manifest['campaign_id']


def paired_deltas(predictions, ledger):
    observed = predictions.set_index(['procedure','arm','backbone','seed','structure_id']).sort_index()
    index = pd.MultiIndex.from_product([SEEDS,ledger.index],names=['seed','structure_id'])
    for procedure, arm, comparator, ph in CONTRASTS:
        for backbone in MODELS:
            a = observed.loc[(procedure,arm,backbone)].reindex(index)
            b = observed.loc[(procedure,comparator,backbone)].reindex(index)
            for endpoint in ENDPOINTS:
                log = endpoint not in ('F','PPB')
                col = ('pred_' if log else 'physical_')+endpoint
                first, second = (v[col].to_numpy().reshape(len(SEEDS),len(ledger)) for v in (a,b))
                if not np.isfinite(first).all() or not np.isfinite(second).all():
                    raise ValueError(f'Missing or invalid seed pairs for {procedure}/{arm}/{comparator}/{backbone}/{endpoint}')
                if not log and (np.any((first < 0) | (first > 100)) or np.any((second < 0) | (second > 100))):
                    raise ValueError('Percentage prediction outside [0,100]')
                delta = (first-second)*np.log2(10) if log else first-second
                yield (procedure,arm,comparator,backbone,endpoint), ph, delta


def summarise(predictions, ledger):
    structure, summary, seeds = [], [], []
    for key, ph, delta in paired_deltas(predictions,ledger):
        procedure,arm,comparator,backbone,endpoint = key
        unit = 'percentage points' if endpoint in ('F','PPB') else 'log2 fold ratio'
        meta = dict(procedure=procedure,arm=arm,comparator_arm=comparator,backbone=backbone,endpoint=endpoint,unit=unit,
                    switch_definition=ph)
        mean, mean_abs = delta.mean(axis=0), np.abs(delta).mean(axis=0)
        status = ledger[f'switch_{ph}'].to_numpy()
        detail = pd.DataFrame(meta | {
            'structure_id':ledger.index,'group_id':ledger.group_id.to_numpy(),'class_7.4':ledger.ionisation_class.to_numpy(),
            'profile':ledger.transition_pattern.to_numpy(),'switch_status':np.where(status,'switcher','stable'),
            'mean_delta':mean,'mean_abs_delta_across_seeds':mean_abs,'seed_sd':delta.std(axis=0,ddof=1),
            'n_seeds_higher':(delta>0).sum(axis=0),'n_seeds_lower':(delta<0).sum(axis=0),
            'prediction_fold_ratio':np.exp2(mean) if unit=='log2 fold ratio' else np.nan,
        })
        structure.append(detail)
        scopes = [('all','all',np.ones(len(ledger),dtype=bool)),('switch_status','stable',~status),('switch_status','switcher',status)]
        scopes.extend(('class',cls,ledger.ionisation_class.eq(cls).to_numpy()) for cls in CLASSES)
        for scope,stratum,keep in scopes:
            x = mean[keep]
            seed_means = delta[:,keep].mean(axis=1)
            quantiles = dict(zip(('p10','p25','median','p75','p90'),np.quantile(x,[.10,.25,.50,.75,.90])))
            summary.append(meta | dict(scope=scope,stratum=stratum,n_structures=len(x),n_groups=ledger.loc[keep,'group_id'].nunique(),
                mean_signed=x.mean(),mean_absolute=np.abs(x).mean(),median_absolute=np.median(np.abs(x)),
                mean_abs_delta_across_seeds=mean_abs[keep].mean(),n_higher=int((x>0).sum()),n_lower=int((x<0).sum()),
                n_ge_2x_magnitude=int((np.abs(x)>=1).sum()) if unit=='log2 fold ratio' else np.nan,
                n_ge_5pp_magnitude=int((np.abs(x)>=5).sum()) if unit=='percentage points' else np.nan,
                geometric_mean_fold_ratio=np.exp2(x.mean()) if unit=='log2 fold ratio' else np.nan,
                median_fold_change_magnitude=np.exp2(np.median(np.abs(x))) if unit=='log2 fold ratio' else np.nan,
                p10_fold_ratio=np.exp2(quantiles['p10']) if unit=='log2 fold ratio' else np.nan,
                p90_fold_ratio=np.exp2(quantiles['p90']) if unit=='log2 fold ratio' else np.nan,
                seed_sd_mean=seed_means.std(ddof=1),seeds_mean_higher=int((seed_means>0).sum()),
                seeds_mean_lower=int((seed_means<0).sum()),**quantiles))
            for seed,value,absolute in zip(SEEDS,seed_means,np.abs(delta[:,keep]).mean(axis=1),strict=True):
                seeds.append(meta | dict(scope=scope,stratum=stratum,seed=seed,mean_delta=value,mean_absolute_delta=absolute))
    return pd.concat(structure,ignore_index=True), pd.DataFrame(summary), pd.DataFrame(seeds)


def _page(title, subtitle):
    fig = plt.figure(figsize=(14.2, 8.6), facecolor='white')
    fig.text(.038,.956,title,fontsize=25,weight='bold',va='top')
    fig.text(.038,.908,subtitle,fontsize=11,color=MUTED,va='top')
    return fig


def _outline(fig, page):
    fig.text(.963,.031,f'Prediction shifts  /  {page}',fontsize=9,color=MUTED,ha='right')


def distributions(structure, summary):
    fig = _page('How the four-pH profile changes individual predictions',
                'Shared-recipe ALL4 minus REP4  •  327 structures per model and endpoint, including structures without observed labels')
    fig.text(.275,.855,'Fold ratio of ALL4 to REP4 predictions',fontsize=10,weight='bold',color=INK)
    fig.text(.275,.337,'ALL4 minus REP4 prediction (percentage points)',fontsize=10,weight='bold',color=INK)
    grids = [fig.add_gridspec(5,3,left=.275,right=.963,top=.79,bottom=.423,hspace=.17,wspace=.17),
             fig.add_gridspec(2,3,left=.275,right=.963,top=.314,bottom=.161,hspace=.21,wspace=.17)]
    subset = structure.query("procedure == 'primary' and arm == 'ALL4' and comparator_arm == 'REP4'")
    details = summary.query("procedure == 'primary' and arm == 'ALL4' and comparator_arm == 'REP4' and scope == 'all'")
    rng = np.random.default_rng(851)
    for j,(endpoint,(label,color)) in enumerate(ENDPOINTS.items()):
        is_log = j < 5
        g = grids[0 if is_log else 1]
        row = j if is_log else j-5
        limit = 2 if is_log else 20
        tick_labels = ('¼×','½×','1×','2×','4×') if is_log else ('−20','−10','0','+10','+20')
        for i,(backbone,model) in enumerate(MODELS.items()):
            ax = fig.add_subplot(g[row,i])
            if j == 0:
                ax.set_title(model,fontsize=13,weight='bold',pad=7)
            x = subset.loc[subset.backbone.eq(backbone)&subset.endpoint.eq(endpoint),'mean_delta'].to_numpy()
            interval = details.loc[details.backbone.eq(backbone)&details.endpoint.eq(endpoint)].iloc[0]
            low, high = x < -limit, x > limit
            mid = ~(low|high)
            ax.axvline(0,color=AXES,lw=.85,zorder=0)
            ax.scatter(x[mid],rng.uniform(-.26,.26,int(mid.sum())),s=7,color=color,alpha=.22,lw=0,zorder=2)
            ax.plot([interval.p10,interval.p90],[0,0],lw=1.7,color=color,zorder=3)
            ax.plot([interval.p25,interval.p75],[0,0],lw=5,color=color,solid_capstyle='butt',zorder=4)
            ax.scatter(interval['median'],0,s=20,fc='white',ec=INK,lw=.9,zorder=5)
            for marker,where,n in (('<',-limit+.045*limit,low.sum()),('>',limit-.045*limit,high.sum())):
                if n:
                    ax.scatter(where,.32,s=13,color=INK,marker=marker,zorder=6)
                    ax.text(where+(.055*limit if marker=='<' else -.055*limit),.32,str(n),ha='left' if marker=='<' else 'right',va='center',fontsize=9.5)
            ax.set(xlim=(-limit,limit),ylim=(-.40,.45),yticks=[])
            ax.spines[['left','bottom']].set_visible(False)
            ax.set_xticks(np.linspace(-limit,limit,5))
            ax.set_xticklabels(tick_labels if j in (4,6) else ['']*5)
            ax.tick_params(axis='x',length=0,labelsize=10.5,pad=4)
            ax.grid(axis='x',color=GRID,lw=.7)
            if i==0:
                ax.text(-.05,.5,label,color=color,fontsize=13,weight='bold',ha='right',va='center',transform=ax.transAxes)
    fig.text(.038,.102,'Each faint dot is one structure’s mean paired prediction change across seeds. Circle, thick line and thin line mark the median, middle 50% and middle 80%.',
             fontsize=11.2,color=MUTED)
    fig.text(.038,.073,'Triangles count values beyond the displayed range. Changes compare predictions from separately fitted models.',fontsize=11.2,color=MUTED)
    _outline(fig,1)
    return fig


def _heat(ax, values, norm, cmap, row_labels, col_labels, xlabel, formatter, label_left):
    n_rows, n_cols = values.shape
    image = ax.pcolormesh(np.arange(n_cols+1)-.5, np.arange(n_rows+1)-.5, values,
                          cmap=cmap, norm=norm, shading='flat', edgecolors='white', linewidths=.35)
    ax.set(xlim=(-.5, n_cols-.5), ylim=(n_rows-.5, -.5))
    ax.set_xticks(np.arange(len(col_labels)),col_labels,fontsize=9.8)
    ax.set_yticks(np.arange(len(row_labels)),row_labels if label_left else ['']*len(row_labels),fontsize=13.1)
    ax.tick_params(axis='both',length=0,pad=5)
    ax.spines[:].set_visible(False)
    ax.tick_params(top=xlabel,labeltop=xlabel,bottom=False,labelbottom=False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i,j]
            red,green,blue,_ = cmap(norm(val))
            luminance = sum(weight*((v+.055)/1.055)**2.4 if v>.04045 else weight*v/12.92
                            for weight,v in zip((.2126,.7152,.0722),(red,green,blue),strict=True))
            tone = 'white' if luminance<.20 else INK
            ax.text(j,i,formatter(val),fontsize=13,ha='center',va='center',color=tone)
    return image


def _colors(fig, images, labels, tick_marks):
    for i,(image,label,ticks) in enumerate(zip(images,labels,tick_marks,strict=True)):
        ax = fig.add_axes([.280+.365*i,.082,.270,.015])
        bar = fig.colorbar(image,cax=ax,orientation='horizontal',ticks=ticks)
        bar.outline.set_visible(False)
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor("face")
        bar.ax.tick_params(length=0,labelsize=8.5,pad=3)
        bar.set_label(label,fontsize=10.5,labelpad=1)


def subgroup_shifts(summary):
    fig = _page('Which compounds move up or down?',
                'Shared-recipe ALL4 minus REP4  •  signed mean prediction change within each subgroup')
    grid = [fig.add_gridspec(1,3,left=.233,right=.963,top=.743,bottom=.373,wspace=.18),
            fig.add_gridspec(1,3,left=.233,right=.963,top=.348,bottom=.174,wspace=.18)]
    cmap = LinearSegmentedColormap.from_list('prediction_direction',DIVERGING)
    norms = [TwoSlopeNorm(vmin=-.30,vcenter=0,vmax=.30),TwoSlopeNorm(vmin=-5.5,vcenter=0,vmax=5.5)]
    subgroups = [('class','anionic'),('class','cationic'),('class','neutral'),('class','zwitterionic'),
                 ('switch_status','stable'),('switch_status','switcher')]
    cols = ['Acid\nonly\n65','Base\nonly\n90','Neither\nactive\n119','Dual\nactive\n53','Stable\n\n121','Switched\n\n206']
    frame = summary.query("procedure == 'primary' and arm == 'ALL4' and comparator_arm == 'REP4'")
    images = []
    for i,(backbone,model) in enumerate(MODELS.items()):
        lookup = frame[frame.backbone.eq(backbone)].set_index(['endpoint','scope','stratum']).mean_signed
        for block, eps in enumerate((list(ENDPOINTS)[:5],list(ENDPOINTS)[5:])):
            vals = np.array([[lookup[(e,*g)] for g in subgroups] for e in eps])
            if np.any((vals < norms[block].vmin) | (vals > norms[block].vmax)):
                raise ValueError("Signed shifts exceed the heatmap limits")
            ax = fig.add_subplot(grid[block][0,i])
            image = _heat(ax,vals,norms[block],cmap,[ENDPOINTS[e][0] for e in eps],cols,block==0,
                          (lambda x:f'{x:+.2f}') if block==0 else (lambda x:f'{x:+.1f}'),i==0)
            if i == 0:
                for tick, ep in zip(ax.get_yticklabels(), eps):
                    tick.set_color(ENDPOINT_COLORS[ep])
            ax.axvline(3.5,lw=3,color='white')
            if i==0:
                images.append(image)
            if block==0:
                ax.set_title(model,fontsize=13,weight='bold',pad=15)
    fig.text(.038,.127,'First four columns use audited pH 7.4 categories. All 90 base-only structures are stable and receive identical ALL4 and REP4 vectors.',
             fontsize=11.2,color=MUTED)
    _colors(fig,images,['Decrease  ←  log₂ fold ratio  →  increase','Decrease  ←  percentage points  →  increase'],[[-.3,-.15,0,.15,.3],[-5.5,-2.75,0,2.75,5.5]])
    _outline(fig,2)
    return fig


def shift_magnitudes(summary):
    fig = _page('How much do predictions move across input comparisons?',
                'Median magnitude across 327 held-out structures  •  absolute value of each structure’s nine-seed mean change')
    grid = [fig.add_gridspec(1,3,left=.233,right=.963,top=.785,bottom=.390,wspace=.18),
            fig.add_gridspec(1,3,left=.233,right=.963,top=.364,bottom=.195,wspace=.18)]
    cmap = LinearSegmentedColormap.from_list('prediction_magnitude',['#F8F9F7','#78A69C','#245567'])
    norms = [Normalize(vmin=0,vmax=.70),Normalize(vmin=0,vmax=5.2)]
    cols = ['ALL4\nREP4','ALL4\nREP4\nTuned','ADD2\nREP2','ADD65\nREP2','ADD5\nREP2','REF74\nREF0','REP4\nREF74']
    frame = summary[summary.scope.eq('all')]
    images = []
    for i,(backbone,model) in enumerate(MODELS.items()):
        lookup = frame[frame.backbone.eq(backbone)].set_index(['endpoint','procedure','arm','comparator_arm']).median_absolute
        for block,eps in enumerate((list(ENDPOINTS)[:5],list(ENDPOINTS)[5:])):
            vals = np.array([[lookup[(e,procedure,arm,comp)] for procedure,arm,comp,_ in CONTRASTS] for e in eps])
            if np.any((vals < norms[block].vmin) | (vals > norms[block].vmax)):
                raise ValueError("Signed shifts exceed the heatmap limits")
            ax = fig.add_subplot(grid[block][0,i])
            image = _heat(ax,vals,norms[block],cmap,[ENDPOINTS[e][0] for e in eps],cols,block==0,
                          (lambda x:f'{x:.2f}') if block==0 else (lambda x:f'{x:.1f}'),i==0)
            if i==0:
                images.append(image)
            if block==0:
                ax.set_title(model,fontsize=13,weight='bold',pad=15)
    fig.text(.038,.155,'Top arm in each column is compared with the arm below it. REP2 and REP4 repeat pH 7.4. REF74 supplies it once, and REF0 uses structure alone.',fontsize=11.2,color=MUTED)
    fig.text(.038,.130,'Tuning uses the same structures and folds and reuses matching fits.',fontsize=11.2,color=MUTED)
    _colors(fig,images,['Median absolute log₂ fold change','Median absolute percentage-point change'],[[0,.35,.70],[0,2.6,5.2]])
    _outline(fig,3)
    return fig


def main():
    predictions,ledger,inputs,campaign_id = load_inputs()
    structure, summary, seeds = summarise(predictions,ledger)
    if (len(structure),len(summary),len(seeds)) != (48069,1029,9261):
        raise ValueError('Incomplete descriptive prediction shift output')
    tables = {'structure_shifts.csv':structure,'shift_summary.csv':summary,'seed_shifts.csv':seeds}
    for filename,table in tables.items():
        table.to_csv(OUT/filename,index=False)
    filename = 'prediction_shifts.pdf'
    with PdfPages(OUT/filename,metadata={'Title':'Categorical pH prediction shifts','Author':'PIPK-Net','Subject':'Paired held-out predictions'}) as pdf:
        for page in (distributions(structure,summary),subgroup_shifts(summary),shift_magnitudes(summary)):
            pdf.savefig(page)
            plt.close(page)
    outputs = {name:sha(OUT/name) for name in [*tables,filename,'render.py']}
    if (OUT/'interpretation.txt').is_file():
        outputs['interpretation.txt'] = sha(OUT/'interpretation.txt')
    (OUT/'manifest.json').write_text(json.dumps(dict(
        campaign_id=campaign_id,inputs=inputs,outputs=outputs,rows={k:len(v) for k,v in tables.items()},
        seeds=list(SEEDS), structures=327, groups=238, pages=3,
        estimand='Per structure, mean paired difference across nine independently trained seed trajectories. Positive endpoints use log2 prediction ratio. F and PPB use percentage-point differences.',
        subgroup_definition='ALL4, REF74 and REP4 comparisons use stability across all added pHs. Individual-pH comparisons use stability for the corresponding pH.'),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
