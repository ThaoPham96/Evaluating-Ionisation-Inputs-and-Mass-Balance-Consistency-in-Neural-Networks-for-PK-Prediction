"""Author-review figures for the frozen categorical-pH follow-up; no model fitting."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ionisation_style import AXES, BLUE, CONTROL, FONT, GRID, INK, MUTED, ENDPOINT_COLORS
EXP = HERE.parents[1]
SCORE = EXP / 'outputs/ph_followup/scoring'
MODELS = {'GraphConv': 'GraphConv', 'Chemprop': 'Chemprop',
          'ChemBERTa_head': 'Frozen ChemBERTa + MLP'}
ENDPOINTS = {
    't_half': ('Half-life', ENDPOINT_COLORS["t_half"], 316),
    'Vd': ('Volume of distribution', ENDPOINT_COLORS["Vd"], 259),
    'CL': ('Clearance', ENDPOINT_COLORS["CL"], 246),
    'Cmax': ('Peak concentration', ENDPOINT_COLORS["Cmax"], 190),
    'Tmax': ('Time to peak', ENDPOINT_COLORS["Tmax"], 249),
    'F': ('Bioavailability', ENDPOINT_COLORS["F"], 172),
    'PPB': ('Protein binding', ENDPOINT_COLORS["PPB"], 292),
}
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909]
JITTER = np.array([-.075, .07, -.04, .09, .015, -.06, .045, -.012, .06])
KEY = ['procedure', 'arm', 'comparator_procedure', 'comparator_arm',
       'backbone', 'scope', 'stratum', 'endpoint']
LABELS = {
    'REF0': 'No ionisation input', 'REF74': 'pH 7.4 category',
    'REP2': 'pH 7.4 repeated twice', 'REP4': 'pH 7.4 repeated four times',
    'ADD2': 'pH 7.4 + 2.0', 'ADD65': 'pH 7.4 + 6.5',
    'ADD5': 'pH 7.4 + 5.0', 'ALL4': 'All four pH categories',
    'P2': 'Raw acidic/basic pKa', 'P3': 'Continuous pH 7.4',
    'P4': 'Continuous four-pH profile', 'P5': 'Continuous pH 7.4 repeated four times',
}
plt.rcParams.update({'font.family': FONT, 'font.size': 13, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': INK,
                     'axes.edgecolor': AXES, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'svg.fonttype': 'none'})


def select(table, **criteria):
    mask = pd.Series(True, index=table.index)
    for k, v in criteria.items():
        mask &= table[k].eq(v)
    return table.loc[mask]


def one(table, **criteria):
    frame = select(table, **criteria)
    if len(frame) != 1:
        raise ValueError((criteria, len(frame)))
    return frame.iloc[0]


def settings(arm, comparator, procedure='primary', comparator_procedure=None):
    return dict(arm=arm, comparator_arm=comparator, procedure=procedure,
                comparator_procedure=comparator_procedure or procedure)


def seed_values(row):
    frame = select(SEED_TABLE, **{k: row[k] for k in KEY}).sort_values('seed')
    assert frame.seed.tolist() == SEEDS
    # All means, seeds and CI bounds share the fixed 9-seed comparator mean.
    return 100 * frame.delta.to_numpy() / row.comparator_mae


def limits(rows, include_adjusted=False):
    values = [0.]
    for _, r in rows.iterrows():
        values.extend(seed_values(r))
        values.append(r.relative_delta_pct)
        for k in ['crossed95_low', 'crossed95_high'] + (
                ['crossed_adjusted_low', 'crossed_adjusted_high'] if include_adjusted else []):
            if np.isfinite(r[k]):
                values.append(100 * r[k] / r.comparator_mae)
    max_abs = max(abs(np.array(values)))
    bound = max(2., np.ceil(max_abs * 1.12))
    return (-bound, bound)


def style(ax, xlim, endpoint_labels=True):
    ax.axvline(0, color=AXES, lw=.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(xlim)
    ax.set_ylim(6.65, -.65)
    ax.set_yticks(range(7))
    ax.set_yticklabels([f'{v[0]}  (n={v[2]})' for v in ENDPOINTS.values()]
                       if endpoint_labels else [])
    if endpoint_labels:
        for tick, color in zip(ax.get_yticklabels(), ENDPOINT_COLORS.values()):
            tick.set_color(color)
    ax.grid(axis='x', color=GRID, lw=.8, zorder=0)
    ax.tick_params(axis='y', length=0, pad=8)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', labelsize=12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))


def point(ax, row, y, marker='D', hollow=False, show_seeds=True, primary=False, by_endpoint=False):
    color = CONTROL if row.arm in {'REP2', 'REP4', 'P5'} else BLUE
    if by_endpoint and color == BLUE:
        color = ENDPOINT_COLORS[row.endpoint]
    if show_seeds:
        ax.scatter(seed_values(row), y + JITTER, s=11, color=color, alpha=.5,
                   edgecolor='white', linewidth=.2, zorder=2)
    factor = 100 / row.comparator_mae
    if bool(row.interval_eligible):
        if primary:
            ax.plot(factor * np.array([row.crossed_adjusted_low, row.crossed_adjusted_high]),
                    [y, y], color=color, lw=1.0, zorder=3)
        ax.plot(factor * np.array([row.crossed95_low, row.crossed95_high]),
                [y, y], color=color, lw=2.4 if primary else 1.35, zorder=3)
    ax.plot(row.relative_delta_pct, y, marker=marker, markersize=5.1,
            markerfacecolor='white' if hollow else color, markeredgecolor=color if hollow else INK,
            markeredgewidth=.65, linestyle='none', zorder=4)


def title(fig, heading, subtitle):
    fig.text(.05, .968, heading, fontsize=21, weight='bold', va='top')
    fig.text(.05, .929, subtitle, fontsize=12.5, color=MUTED, va='top')


def footer(fig, lines, y=.057):
    for j, line in enumerate(lines):
        fig.text(.05, y - j*.018, line, fontsize=11, color=MUTED, va='top')


def save(fig, stem, caption, rows=None, pdf=True):
    if pdf:
        fig.savefig(HERE/f'{stem}.pdf', facecolor='white')
    fig.savefig(HERE/f'{stem}.png', dpi=210, facecolor='white')
    fig.savefig(HERE/f'{stem}.svg', facecolor='white')
    (HERE/f'{stem}.txt').write_text(caption, encoding='utf-8')
    if rows is not None:
        rows.to_csv(HERE/f'{stem}_plotted.csv', index=False)


def all_rows(contrast):
    return select(CONTRASTS, **contrast, scope='all', stratum='all')


def forest_grid(contrasts, headings, figure_title, subtitle, stem, pdf=None, save_pdf=True):
    frames = [all_rows(c) for c in contrasts]
    xlim = limits(pd.concat(frames))
    compact = stem in {"reference_controls", "individual_ph_results"}
    height, gap = (10.7, .31) if compact else (11.3, .47)
    fig, axes = plt.subplots(len(contrasts), 3, figsize=(14, height), squeeze=False)
    fig.subplots_adjust(left=.225, right=.965, top=.85, bottom=.11, hspace=gap, wspace=.2)
    title(fig, figure_title, subtitle)
    for i, (frame, heading) in enumerate(zip(frames, headings)):
        top = axes[i, 0].get_position().y1
        fig.text(.05, top+.012, f'{chr(65+i)}   {heading}', fontsize=13, weight='bold')
        for j, (model, model_name) in enumerate(MODELS.items()):
            ax = axes[i, j]
            style(ax, xlim, endpoint_labels=j == 0)
            if i == 0:
                ax.set_title(model_name, fontsize=13, weight='bold', pad=24)
            for y, ep in enumerate(ENDPOINTS):
                point(ax, one(frame, backbone=model, endpoint=ep), y, by_endpoint=stem != 'reference_controls')
    fig.supxlabel('Change in MAE (% of comparator mean MAE)     ← lower error     |     higher error →',
                  x=.59, y=.072, fontsize=13)
    footer(fig, ['Dots: nine paired seeds. Diamonds: mean. Bars: crossed seed/group 95% intervals (exploratory).',
                 'Held-out grouped outer CV; 327 structures, 238 groups. Endpoint counts shown at left. Positive changes mean worse prediction.'], y=.044)
    save(fig, stem, figure_title+'\n'+subtitle+'\n'+
         '\n'.join(headings)+'\n'+COMMON_CAPTION, pd.concat(frames), pdf=save_pdf)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)


def main_figure():
    shared = all_rows(settings('ALL4', 'REP4'))
    tuned = all_rows(settings('ALL4', 'REP4', 'tuned'))
    fig = plt.figure(figsize=(14, 10.8))
    title(fig, 'How does a four-pH profile change prediction accuracy?',
          'All four pHs (7.4, 2.0, 6.5, 5.0) versus pH 7.4 repeated four times: equal input width, new information only in the profile.')
    top = fig.add_gridspec(1, 2, left=.20, right=.95, bottom=.665, top=.84, wspace=.36)
    for j, ep in enumerate(['Vd', 'F']):
        ax = fig.add_subplot(top[0,j])
        frame = select(shared, endpoint=ep)
        ax.axvline(0, color=AXES, lw=.8, ls=(0, (4, 3)))
        ax.grid(axis='x', color=GRID)
        ax.set_xlim(limits(frame, True))
        ax.set_ylim(2.6, -.6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_yticks(range(3), list(MODELS.values()) if j == 0 else [])
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        ax.set_title(f'{chr(65+j)}   {ENDPOINTS[ep][0]} (n={ENDPOINTS[ep][2]})',
                     loc='left', fontsize=13, weight='bold', pad=16, color=ENDPOINT_COLORS[ep])
        for y, model in enumerate(MODELS):
            r = one(frame, backbone=model)
            point(ax, r, y, primary=True, by_endpoint=True)
            ax.text(.97, 1-(y+.5)/3.3, f'{r.relative_delta_pct:+.1f}%', transform=ax.transAxes,
                    ha='right', va='center', fontsize=12.5, color=ENDPOINT_COLORS[ep],
                    bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=1.5))
        ax.set_xlabel('Change in MAE (%)', labelpad=6)
    fig.text(.05, .585,
             'Primary tests: thin bars = adjusted 99.17% intervals across six tests; thick bars = ordinary 95% intervals.',
             fontsize=12, color=MUTED)
    fig.text(.05, .550, 'C   All seven endpoints, including a separate tuning check', fontsize=13, weight='bold')
    fig.text(.05, .518, 'Filled diamond: shared recipe    Open diamond: separate tuning    Small dots: nine paired seeds',
             fontsize=12, color=MUTED)
    bottom = fig.add_gridspec(1, 3, left=.225, right=.965, bottom=.15, top=.47, wspace=.20)
    xlim = limits(pd.concat([shared, tuned]))
    for j, (model, model_name) in enumerate(MODELS.items()):
        ax = fig.add_subplot(bottom[0,j])
        style(ax, xlim, endpoint_labels=j == 0)
        ax.set_title(model_name, fontsize=13, weight='bold', pad=12)
        for y, ep in enumerate(ENDPOINTS):
            point(ax, one(shared, backbone=model, endpoint=ep), y-.16)
            point(ax, one(tuned, backbone=model, endpoint=ep), y+.16, hollow=True)
    fig.text(.59, .102, 'Change in MAE (%)     ← lower error     |     higher error →', ha='center', fontsize=11)
    footer(fig, ['Lower panels: exploratory crossed seed/group 95% intervals. Separately tuned on the same cohort/folds; some fitted predictions reused.',
                 'Five grouped outer folds; nine paired seeds; 327 structures, 238 groups. MAE changes use each model/endpoint/control as denominator.',
                 'MAE: log10 units for positive endpoints; percentage points for F and PPB. No mass-balance penalty in these experiments.'], y=.070)
    save(fig, 'ph_main_results', COMMON_CAPTION+'\nPrimary panels show adjusted 99.1667% intervals for six tests.\n',
         pd.concat([shared, tuned]))
    plt.close(fig)


def contrast_atlas():
    cases = CONTRASTS.loc[CONTRASTS.scope.eq('all'),
                          ['procedure','arm','comparator_procedure','comparator_arm','purpose']].drop_duplicates()
    with PdfPages(HERE/'all_mae_contrasts.pdf') as pdf:
        for start in range(0, len(cases), 3):
            batch = cases.iloc[start:start+3]
            specs, labels = [], []
            for _, c in batch.iterrows():
                specs.append(settings(c.arm, c.comparator_arm, c.procedure, c.comparator_procedure))
                proc_label={'primary':'shared recipe','tuned':'tuning sensitivity','historical':'earlier continuous-input runs'}[c.procedure]
                purpose_label={'headline':'primary on Vd/F only','tuning_sensitivity':'exploratory'}.get(c.purpose,c.purpose)
                labels.append(f'{LABELS[c.arm]} − {LABELS[c.comparator_arm]}  [{proc_label}; {purpose_label}]')
            forest_grid(specs, labels, f'All MAE comparisons | {start//3+1} of 8',
                        'Every prespecified all-cohort comparison, including historical continuous-input controls. See main figure for adjusted primary intervals.',
                        f'contrast_atlas_{start//3+1:02d}', pdf=pdf, save_pdf=False)


def subgroup_atlas():
    with PdfPages(HERE/'subgroups_and_interactions.pdf') as pdf:
        classes = select(CONTRASTS, scope='class')
        assert set(classes.arm) == {'ALL4'} and set(classes.comparator_arm) == {'REP4'}
        fig, axes = plt.subplots(1, 3, figsize=(15, 10.5))
        fig.subplots_adjust(left=.22,right=.97,top=.82,bottom=.15,wspace=.2)
        title(fig, 'pH 7.4 classes | four-pH profile versus repeated pH 7.4',
              'All seven endpoints; subgroup changes use the subgroup-specific comparator MAE.')
        class_names = [('anionic','Acid-only'),('cationic','Base-only'),('neutral','Neither'),('zwitterionic','Dual-active')]
        markers = ['D','o','s','^']
        offsets = [-.27,-.09,.09,.27]
        xlim = limits(classes)
        for j,(model,name) in enumerate(MODELS.items()):
            ax=axes[j]; style(ax,xlim,j==0); ax.set_title(name,fontsize=11,weight='bold')
            for y,ep in enumerate(ENDPOINTS):
                for (cl,_),marker,offset in zip(class_names,markers,offsets):
                    r=one(classes,backbone=model,endpoint=ep,stratum=cl)
                    point(ax,r,y+offset,marker=marker,hollow=True,show_seeds=False)
                    ax.text(.985,y+offset,f'n={int(r.n_structures)}',transform=ax.get_yaxis_transform(),
                            va='center',ha='right',fontsize=7,color=MUTED,
                            bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=.5))
        handles=[Line2D([],[],marker=m,color=INK,markerfacecolor='white',linestyle='none',label=l)
                 for m,(_,l) in zip(markers,class_names)]
        fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.05,.89),frameon=False,ncol=4)
        fig.text(.60,.11,'Change in MAE (%)     ← lower error     |     higher error →',ha='center',fontsize=11)
        footer(fig,['Markers: mean. Bars: exploratory crossed 95% intervals. Subgroup labelled n shown beside each marker row.',
                    'No interval for fewer than 10 labels or 5 groups. Class labels describe pKa-based activation, not verified microspecies.'],y=.068)
        save(fig,'subgroups_01_classes',COMMON_CAPTION,classes,pdf=False);pdf.savefig(fig);plt.close(fig)

        interactions=pd.read_csv(SCORE/'switcher_interactions.csv')
        keys=['procedure','arm','comparator_procedure','comparator_arm','contrast']
        cases=interactions[keys].drop_duplicates()
        for page,(_,c) in enumerate(cases.iterrows(),start=2):
            spec=settings(c.arm,c.comparator_arm,c.procedure,c.comparator_procedure)
            rows=select(CONTRASTS,**spec,scope='switch_status')
            rows=rows.loc[rows.stratum.str.startswith(c.contrast+'/')]
            ir=select(interactions,**spec,contrast=c.contrast)
            fig,axes=plt.subplots(2,3,figsize=(15,11.5))
            fig.subplots_adjust(left=.225,right=.965,top=.83,bottom=.13,hspace=.48,wspace=.2)
            switch_label={'any_added_pH':'Any added pH','pH2':'pH 2.0','pH5':'pH 5.0','pH6.5':'pH 6.5'}[c.contrast]
            status='descriptive' if c.arm=='ADD65' and c.comparator_arm=='REP2' else 'exploratory'
            title(fig,f'Category switching | {LABELS[c.arm]} − {LABELS[c.comparator_arm]}',
                  f'{switch_label}: a switch means a different category from pH 7.4. Shared training recipe; {status} subgroup analysis.')
            fig.text(.05,.875,'A   Subgroup changes: filled diamond = switcher; open circle = stable',fontsize=11,weight='bold')
            xlim=limits(rows)
            ilim=[]
            for _,r in ir.iterrows():
                base=one(all_rows(spec),backbone=r.backbone,endpoint=r.endpoint).comparator_mae
                ilim.extend(100*np.array([r.contrast_of_deltas,r.crossed95_low,r.crossed95_high])/base)
            bound=max(2,np.ceil(np.nanmax(np.abs(ilim))*1.12))
            fig.text(.05,.481,'B   Direct interaction: switcher change minus stable change',fontsize=11,weight='bold')
            for j,(model,name) in enumerate(MODELS.items()):
                ax=axes[0,j]; style(ax,xlim,j==0);ax.set_title(name,fontsize=11,weight='bold',pad=12)
                for y,ep in enumerate(ENDPOINTS):
                    for status,offset,marker,hollow in [('switcher',-.16,'D',False),('stable',.16,'o',True)]:
                        r=one(rows,backbone=model,endpoint=ep,stratum=c.contrast+'/'+status)
                        point(ax,r,y+offset,marker=marker,hollow=hollow,show_seeds=False)
                        ax.text(.985,y+offset,f'n={int(r.n_structures)}',transform=ax.get_yaxis_transform(),
                                va='center',ha='right',fontsize=7,color=MUTED,
                                bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=.5))
                ax.set_xlabel('MAE change (% of subgroup control MAE)',fontsize=9)
                ax=axes[1,j];style(ax,(-bound,bound),j==0)
                for y,ep in enumerate(ENDPOINTS):
                    r=one(ir,backbone=model,endpoint=ep)
                    base=one(all_rows(spec),backbone=model,endpoint=ep).comparator_mae
                    value,lo,hi=100*np.array([r.contrast_of_deltas,r.crossed95_low,r.crossed95_high])/base
                    color=ENDPOINTS[ep][1]
                    ax.plot([lo,hi],[y,y],color=color,lw=1.4)
                    ax.plot(value,y,'D',ms=5,color=color,mec=INK,mew=.6)
                ax.set_xlabel('Interaction (% of overall control MAE)',fontsize=9)
            footer(fig,['Interaction below zero: greater error reduction (or smaller increase) in switchers. This directly tests subgroup differences.',
                        'Bars: crossed seed/group 95% intervals. Upper rows use subgroup denominators; interactions use the overall comparator denominator.',
                        'No organ-specific outcomes are measured. Switching does not establish improved physiology or tissue distribution.'],y=.071)
            save(fig,f'subgroups_{page:02d}_switching',COMMON_CAPTION,rows,pdf=False)
            pdf.savefig(fig);plt.close(fig)


COMMON_CAPTION = '''Categorical-pH follow-up on 327 structures in 238 frozen parent/scaffold groups.
Five grouped outer folds are pooled within each of nine matched seeds; seeds are not independent drugs.
The seven endpoint labelled counts are: t_half 316, Vd 259, CL 246, Cmax 190, Tmax 249, F 172, PPB 292.
All endpoint weights in training are 1, with no mass-balance penalty. Targets are scaled using training data only.
Relative MAE changes and their intervals divide absolute paired MAE differences by a fixed nine-seed mean comparator MAE.
Log10 MAE is used for positive endpoints, and percentage-point MAE for F and PPB. Zero PPB labels remain in MAE.
95% intervals resample whole seeds and whole frozen groups jointly (10,000 draws), conditional on fixed data/folds/selected models.
Only ALL4 minus REP4 on Vd and F across the three models is the six-test primary family (adjusted 99.1667% intervals).
All other intervals are exploratory; ADD65 minus REP2 is descriptive. These many secondary contrasts are not multiplicity-adjusted.
Tuning is performed separately on the same cohort and folds; some final fitted predictions are exactly reused. It is a sensitivity analysis, not independent replication.
ALL4 and REP4 both have 16 inputs divided by 2. ADD2/ADD65/ADD5 and REP2 have 8 inputs divided by sqrt(2).
REF74 is a four-channel pH 7.4 one-hot input. REF0 has no ionisation input. pH values are numeric input conditions, not organ simulations.
Negative MAE changes indicate lower error. Colours identify endpoints; markers/legends identify conditions.
These are large author-review figures, not a final journal-sized layout.
'''


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--main-only',action='store_true')
    parser.add_argument('--review-figures', action='store_true', help='Render the three source pages used by the supplement')
    args=parser.parse_args()
    HERE.mkdir(parents=True,exist_ok=True)
    CONTRASTS=pd.read_csv(SCORE/'contrasts.csv')
    SEED_TABLE=pd.read_csv(SCORE/'contrast_seeds.csv')
    assert not CONTRASTS[KEY].duplicated().any()
    assert CONTRASTS.n_seeds.eq(9).all()
    assert len(CONTRASTS.loc[CONTRASTS.primary_test]) == 6
    manifest=json.loads((SCORE/'scoring_manifest.json').read_text())
    main_figure()
    if args.main_only:
        manifest_path=HERE/'figure_manifest.json'
        existing=json.loads(manifest_path.read_text())
        for p in HERE.glob('ph_main_results.*'):
            if p.suffix in {'.pdf','.png','.svg'}:
                existing['output_sha256'][p.name]=sha256(p.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(existing,indent=2),encoding='utf-8')
        raise SystemExit(0)
    forest_grid([settings('ADD2','REP2'),settings('ADD65','REP2'),settings('ADD5','REP2')],
                ['Add pH 2.0 versus repeated pH 7.4',
                 'Add pH 6.5 versus repeated pH 7.4 (descriptive)',
                 'Add pH 5.0 versus repeated pH 7.4'],
                'Which added pH changes prediction?',
                'Each added-pH input is compared with two repeated pH 7.4 blocks: matched width and normalisation.',
                'individual_ph_results')
    forest_grid([settings('REF74','REF0'),settings('REP4','REF74'),settings('ALL4','REF74')],
                ['pH 7.4 category versus no ionisation input',
                 'Repeat pH 7.4 four times versus one pH 7.4 block',
                 'All four pHs versus one pH 7.4 block'],
                'Information and parameterisation controls',
                'The repeated-input control isolates changes caused by the input representation without adding new pH information.',
                'reference_controls')
    if not args.review_figures:
        contrast_atlas()
        subgroup_atlas()
    (HERE/'CAPTIONS_AND_METHODS.txt').write_text(COMMON_CAPTION,encoding='utf-8')
    sources={str(p.relative_to(EXP)):sha256(p.read_bytes()).hexdigest()
             for p in SCORE.glob('*') if p.is_file()}
    outputs={p.name:sha256(p.read_bytes()).hexdigest()
             for p in HERE.iterdir() if p.is_file() and p.suffix in {'.pdf','.png','.svg','.csv'}}
    (HERE/'figure_manifest.json').write_text(json.dumps({'source_sha256':sources,'output_sha256':outputs,
        'campaign_id':manifest.get('campaign_id'),'intervals':'crossed seed/group bootstrap',
        'n_seeds':9,'n_structures':327,'n_groups':238},indent=2),encoding='utf-8')
    print(json.dumps({'created_pdfs':[p.name for p in HERE.glob('*.pdf')],'figure_pages':len(list(HERE.glob('*.png')))}))
