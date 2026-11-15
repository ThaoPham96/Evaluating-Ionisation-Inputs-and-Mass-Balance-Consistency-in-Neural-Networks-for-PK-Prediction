"""Render frozen ionisation results; no fitting, selection or source-data changes."""
from pathlib import Path
import sys
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ionisation_style import AUDITED_CLASSES, AXES, BASELINE, BLUE, CATEGORIES, CATEGORY_EDGE, CONTROL, FONT, GRID, INK, MODELS as MODEL_NAMES, MUTED, ENDPOINT_COLORS
EXP = HERE.parents[1]
SCORES = EXP / 'outputs' / 'scoring'
MODELS = [(model, name, BLUE) for model, name in MODEL_NAMES.items()]
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909]
CLASSES = [(key, name, colour) for (key, (_, colour, _)), name in zip(CATEGORIES.items(), AUDITED_CLASSES)]
JITTER = np.array([-.13, .11, -.07, .15, .02, -.10, .07, -.02, .13])
plt.rcParams.update({'font.family': FONT,
                     'font.size': 12, 'axes.labelsize': 12, 'xtick.labelsize': 11.8,
                     'ytick.labelsize': 12.2, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': INK, 'axes.edgecolor': AXES,
                     'axes.linewidth': .7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'svg.fonttype': 'none', 'savefig.facecolor': 'white'})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    manifest = json.loads((SCORES / 'scoring_manifest.json').read_text())
    assert manifest['status'] == 'report_ready'
    tables = {}
    for name in ['contrasts', 'contrast_seeds', 'endpoint_metrics', 'endpoint_summary', 'coherence']:
        path = SCORES / f'{name}.csv'
        assert sha(path) == manifest['files'][path.name], f'Score hash changed: {name}'
        tables[name] = pd.read_csv(path)
    return tables, manifest


def subset(df, **criteria):
    keep = pd.Series(True, index=df.index)
    for name, value in criteria.items():
        keep &= df[name].eq(value)
    return df.loc[keep].copy()


def one(df, **criteria):
    rows = subset(df, **criteria)
    assert len(rows) == 1, (criteria, len(rows))
    return rows.iloc[0]


def style(ax, horizontal=False):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x' if horizontal else 'y', color=GRID, lw=.65)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.65)
    if horizontal:
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=9)


def heading(fig, x, y, letter, title, subtitle):
    fig.text(x, y, letter, fontsize=17, fontweight='bold', va='top')
    fig.text(x + .027, y - .002, title, fontsize=12.5, fontweight='bold', va='top')
    fig.text(x + .027, y - .030, subtitle, fontsize=11.8, color=MUTED, va='top')


def header(fig, title, subtitle):
    fig.text(.045, .969, title, fontsize=20, fontweight='bold', va='top')
    fig.text(.045, .934, subtitle, fontsize=12, color=MUTED, va='top')


def save(fig, stem, caption, inputs):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(HERE / f'{stem}.{ext}', dpi=240)
    plt.close(fig)
    (HERE / f'{stem}_caption.txt').write_text(caption + '\n', encoding='utf-8')
    pd.concat(inputs, ignore_index=True, sort=False).to_csv(HERE / f'{stem}_plotted_data.csv', index=False)


def headline(ax, tables, cohort, arm, comparator):
    plotted = []
    color = ENDPOINT_COLORS['Vd']
    for i, (model, label, _) in enumerate(MODELS):
        filt = dict(cohort=cohort, procedure='primary', backbone=model, arm=arm,
                    comparator_procedure='primary', comparator_backbone=model,
                    comparator_arm=comparator, scope='all', stratum='all', endpoint='Vd')
        r = one(tables['contrasts'], **filt)
        seeds = subset(tables['contrast_seeds'], **filt).set_index('seed').loc[SEEDS]
        assert len(seeds) == 9 and seeds.index.is_unique
        scale = 100 / r.comparator_mae
        values = seeds.delta.to_numpy() * scale
        np.testing.assert_allclose(values.mean(), r.relative_delta_pct, atol=1e-10)
        np.testing.assert_allclose(seeds.comparator_mae.mean(), r.comparator_mae, atol=1e-12)
        y = 2 - i
        ax.scatter(values, y + JITTER, s=22, color=color, alpha=.35, linewidths=0, zorder=3)
        lo, hi = np.array([r.crossed_adjusted_low, r.crossed_adjusted_high]) * scale
        lo95, hi95 = np.array([r.crossed95_low, r.crossed95_high]) * scale
        assert min(lo, *values) > -23 and max(hi, *values) < 13, 'Headline range clips data'
        ax.plot([lo, hi], [y, y], color=color, lw=1.1, zorder=4)
        ax.plot([lo, lo], [y-.055, y+.055], color=color, lw=1.1)
        ax.plot([hi, hi], [y-.055, y+.055], color=color, lw=1.1)
        ax.plot([lo95, hi95], [y, y], color=color, lw=3.2, solid_capstyle='butt', zorder=5)
        ax.plot(r.relative_delta_pct, y, 'D', ms=6, color=color, mec='white', mew=.7, zorder=6)
        percentage = f'{r.relative_delta_pct:+.2f}%' if abs(r.relative_delta_pct) < .1 else f'{r.relative_delta_pct:+.1f}%'
        ax.text(.98, y, percentage, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=11, fontweight='bold', color=INK)
        r = r.copy(); r['plot_panel'] = 'headline'; r['plot_scale'] = scale
        r['plot_record'] = 'summary'
        plotted.append(r.to_frame().T)
        exported_seeds = seeds.reset_index()
        exported_seeds['plot_panel'] = 'headline'
        exported_seeds['plot_record'] = 'seed'
        exported_seeds['plot_scale'] = scale
        exported_seeds['plotted_relative_delta_pct'] = values
        plotted.append(exported_seeds)
    style(ax, horizontal=True)
    ax.axvline(0, lw=1, color=AXES, ls=(0,(3,3)))
    ax.set(xlim=(-23, 13), ylim=(-.48, 2.48), yticks=[2,1,0],
           yticklabels=[m[1] for m in MODELS], xticks=[-20,-10,0,10],
           xlabel='Change in Vd MAE (% of comparator MAE)')
    ax.xaxis.label.set_color(ENDPOINT_COLORS['Vd'])
    return plotted


def absolute_encodings(axes, tables, cohort, arms, labels):
    plotted = []
    for mi, (model, label, _) in enumerate(MODELS):
        ax = axes[mi]
        ax.set_title(label, fontsize=13, weight="bold")
        for ai, arm in enumerate(arms):
            r = subset(tables['endpoint_metrics'], cohort=cohort, procedure='primary',
                       backbone=model, arm=arm, endpoint='Vd').set_index('seed').loc[SEEDS]
            assert len(r) == 9 and r.index.is_unique
            summary = one(tables['endpoint_summary'], cohort=cohort, procedure='primary',
                          backbone=model, arm=arm, endpoint='Vd')
            np.testing.assert_allclose(r.mae.mean(), summary.mae_mean, atol=1e-12)
            np.testing.assert_allclose(r.mae.std(ddof=1), summary.mae_seed_sd, atol=1e-12)
            assert min(r.mae.min(), summary.mae_mean-summary.mae_seed_sd) > .365
            assert max(r.mae.max(), summary.mae_mean+summary.mae_seed_sd) < .555
            y = len(arms)-1-ai
            color = BASELINE if arm in {"C0", "P0"} else CONTROL if arm == "P5" else BLUE
            ax.scatter(r.mae, y + JITTER*.47, s=16, color=color, alpha=.38, lw=0, zorder=3)
            ax.errorbar(r.mae.mean(), y, xerr=r.mae.std(ddof=1), fmt='D', ms=5,
                        color=color, mec='white', mew=.6, elinewidth=1.5, capsize=3, zorder=5)
            r = r.reset_index(); r['plot_panel'] = 'encodings'
            plotted.append(r)
        style(ax, horizontal=True)
        ax.set(yticks=range(len(arms))[::-1], yticklabels=labels if mi == 0 else [],
               ylim=(-.55, len(arms)-.45), xlim=(.365,.555),
               xticks=[.40,.45,.50,.55], xlabel="Vd MAE (log10 units)")
        ax.xaxis.label.set_color(ENDPOINT_COLORS["Vd"])
    return plotted


def overview(tables):
    fig = plt.figure(figsize=(12.5, 11.6))
    header(fig, "Ionisation information and Vd prediction",
           "Five grouped outer folds · Nine paired seeds · Mass-balance penalty off")
    heading(fig, .045, .875, "A", "Adding the ionisation category",
            "Audited class cohort · 494 structures · 388 with Vd")
    heading(fig, .540, .875, "B", "Adding a multi-pH profile",
            "Numerical-pKa cohort · 327 structures · 259 with Vd")
    axa = fig.add_axes([.215,.686,.270,.132])
    axb = fig.add_axes([.705,.686,.270,.132])
    plotted = headline(axa,tables,'class','C2','C0') + headline(axb,tables,'numerical','P4','P3')
    fig.text(.045,.625,"A–B  ·  Dots: paired seeds  ·  Diamond: mean  ·  Thick bar: 95% interval  ·  Thin bar: 99.17% interval",
             fontsize=12, color=MUTED)
    heading(fig,.045,.585,"C","Categorical encodings", "Audited class cohort")
    heading(fig,.045,.325,"D","Numerical representations", "Numerical-pKa cohort")
    axc = [fig.add_axes([.215+j*.253,.390,.223,.135]) for j in range(3)]
    axd = [fig.add_axes([.215+j*.253,.100,.223,.166]) for j in range(3)]
    plotted += absolute_encodings(axc,tables,'class',['C0','C1','C2','C3'],
                                  ['Structure only','Charge-sign scalar','One-hot category','Learned embedding'])
    plotted += absolute_encodings(axd,tables,'numerical',['P0','P1','P2','P3','P4','P5'],
                                  ['Structure only','One-hot category','Raw pKas','pH 7.4 pair','Multi-pH profile','Repeated pH 7.4'])
    handles = [Line2D([],[],marker='D',ls='none',color=colour,label=label,ms=5)
               for colour, label in ((BASELINE,"Structure only"), (BLUE,"Ionisation input"),
                                     (CONTROL,"Repeated-input control"))]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.045,.033),
               ncol=3, frameon=False, fontsize=12, borderaxespad=0)
    fig.text(.045,.019,"C–D  ·  Dots: nine seeds  ·  Diamond: mean  ·  Bars: ±1 seed SD", fontsize=12, color=MUTED)
    caption = (
        'Figure 1. Ionisation information and Vd prediction. All points are held-out predictions pooled '
        'across five frozen parent/scaffold-group outer folds, with nine paired final-training seeds '
        'and five inner folds for model selection/calibration. The transformer is frozen and its MLP '
        'head is trained. The physiology penalty is zero; all supervised endpoint weights are one '
        'after training-only target scaling. A and C use the audited 494-structure class cohort '
        '(388 Vd labels); B and D use its 327-structure numerical-pKa subset (259 Vd labels). '
        'A compares fixed four-class one-hot input with no ionisation input. B compares eight '
        'Henderson-Hasselbalch acidic/basic propensity channels at pH 2, 5, 6.5, 7.4 with the two '
        'channels at 7.4. These are descriptor representations of archived predicted pKas, not '
        'validated organ-compartment PK simulations. In A-B, the seed values are 100 times each '
        'paired log10-MAE difference divided by the comparator MAE averaged over all nine seeds. '
        'The same fixed denominator scales the absolute-difference bootstrap interval; it is '
        'not a bootstrap of a random ratio. Thick lines show 95% and thin lines show 99.1667% '
        'equal-tailed crossed seed/group bootstrap intervals (10,000 draws), the latter accounting '
        'for six prespecified Vd comparisons. Intervals are conditional on observed data, folds '
        'and selected settings. The adjusted interval excludes zero for the Chemprop and frozen '
        'ChemBERTa category comparisons, but not GraphConv or any multi-pH comparison. C-D show '
        'absolute Vd MAE; small points are seeds, diamonds are means and bars are sample seed SD '
        '(not confidence intervals). C includes scalar -1/0/+1 input (neutral and zwitterionic '
        'both 0), a four-dimensional one-hot vector and a learned 16-dimensional category '
        'embedding. D includes no ionisation, one-hot, the raw acidic/basic pKa pair, pH-7.4 '
        'propensities, the multi-pH profile and a repeated pH-7.4 control. Both eight-channel '
        'inputs are divided by two after training-only standardisation. Shared-recipe results '
        'are shown; independent tuning sensitivities appear in Figure 3. No between-cohort '
        'performance comparison is implied.')
    save(fig,'ionisation_overview',caption,plotted)


def diagnostics(tables):
    fig = plt.figure(figsize=(12.5,11.3))
    header(fig,'Cohort, baseline and mechanistic checks',
           'Ionisation study  |  Exploratory comparisons on the same held-out grouped folds')
    heading(fig,.045,.884,'A','The two audited cohorts',
            'The numerical-pKa cohort is a subset of the class cohort')
    axa=fig.add_axes([.145,.681,.335,.133]); counts=[]
    for y,cohort in [(1,'class'),(0,'numerical')]:
        d=pd.read_csv(EXP/'data/release'/cohort/'model_dataset.csv')
        assert len(d)=={'class':494,'numerical':327}[cohort]
        start=0
        for key,label,color in CLASSES:
            n=int(d.ionisation_class.eq(key).sum())
            axa.barh(y,n,left=start,height=.43,color=color,edgecolor='white',lw=1)
            axa.text(start+n/2,y,str(n),ha='center',va='center',color='white' if key == 'anionic' else INK,fontsize=12,fontweight='bold')
            counts.append(pd.DataFrame([dict(plot_panel='cohort',cohort=cohort,ionisation_class=key,n=n)]))
            start+=n
        axa.text(start+8,y,str(len(d)),va='center',fontsize=12,fontweight='bold')
    style(axa,horizontal=True)
    axa.set(xlim=(0,540),ylim=(-.55,1.55),yticks=[1,0],yticklabels=['Class','Numerical pKa'],
            xlabel='Structures',xticks=[0,100,200,300,400,500])
    fig.legend(handles=[Patch(facecolor=c, edgecolor=CATEGORY_EDGE, linewidth=.35, label=n) for _,n,c in CLASSES],loc='center',
               bbox_to_anchor=(.276,.616),ncol=2,frameon=False,fontsize=12)
    heading(fig,.550,.884,'B','Simple controls and neural predictors',
            'Class cohort: 388 structures with Vd  |  Lower is better')
    axb=fig.add_axes([.758,.645,.216,.180])
    specs=[('control','CPU','overall_median','Overall median',CONTROL),
           ('control','CPU','class_median','Class median',CONTROL),
           ('control','CPU','morgan_ridge','Morgan ridge',CONTROL),
           ('control','CPU','morgan_class_ridge','Morgan + category',CONTROL)]
    specs += [('primary',m,'C2',n,c) for m,n,c in MODELS]
    for i,(procedure,model,arm,label,color) in enumerate(specs):
        r=one(tables['endpoint_summary'],cohort='class',procedure=procedure,backbone=model,arm=arm,endpoint='Vd')
        y=len(specs)-1-i
        axb.plot(r.mae_mean,y,'D' if procedure=='primary' else 's',ms=5.5,color=color)
        if procedure=='primary':
            vals=subset(tables['endpoint_metrics'],cohort='class',procedure=procedure,backbone=model,arm=arm,endpoint='Vd')
            axb.scatter(vals.mae,y+JITTER,s=13,color=color,alpha=.35,lw=0)
            axb.errorbar(r.mae_mean,y,xerr=r.mae_seed_sd,color=color,lw=1.6,capsize=2)
            vals=vals.copy();vals['plot_panel']='controls';vals['plot_record']='seed'
            counts.append(vals)
        r=r.copy();r['plot_panel']='controls';r['plot_record']='summary';counts.append(r.to_frame().T)
    style(axb,horizontal=True)
    axb.set(xlim=(.37,.61),ylim=(-.6,len(specs)-.3),yticks=list(range(len(specs)-1,-1,-1)),
            yticklabels=[s[3] for s in specs],xlabel='Vd MAE (log10 units)',xticks=[.4,.45,.5,.55,.6])
    axb.xaxis.label.set_color(ENDPOINT_COLORS['Vd'])
    axb.tick_params(axis='y',labelsize=11.5)
    fig.text(.550,.592,'Blue: neural models with category input. Bars: ±1 seed SD.',fontsize=11,color=MUTED)
    heading(fig,.045,.548,'C','Category effects within each audited class',
            'One-hot class versus structure only · Bars: exploratory 95% bootstrap intervals')
    within = [fig.add_axes([.215+j*.253,.342,.223,.133]) for j in range(3)]
    class_ns=[]
    for ci,(key,label,_) in enumerate(CLASSES):
        for mi,(model,model_label,color) in enumerate(MODELS):
            ax = within[mi]
            r=one(tables['contrasts'],cohort='class',procedure='primary',backbone=model,arm='C2',
                  comparator_procedure='primary',comparator_backbone=model,comparator_arm='C0',
                  scope='class',stratum=key,endpoint='Vd')
            k=100/r.comparator_mae
            color=ENDPOINT_COLORS['Vd']
            y=3-ci
            assert r.crossed95_low*k > -40 and r.crossed95_high*k < 15
            ax.plot(np.array([r.crossed95_low,r.crossed95_high])*k,[y,y],color=color,lw=1.6)
            ax.plot(r.relative_delta_pct,y,'D',color=color,mec='white',mew=.6,ms=5)
            r=r.copy();r['plot_panel']='within_class';counts.append(r.to_frame().T)
            if mi==0:class_ns.append(int(r.n_structures))
    for mi, (ax, (_, label, _)) in enumerate(zip(within, MODELS)):
        style(ax,horizontal=True)
        ax.axvline(0,color=AXES,lw=1,ls=(0,(3,3)))
        ax.set(xlim=(-40,15),ylim=(-.55,3.55),yticks=[3,2,1,0],
               yticklabels=[f'{label} (n={n})' for (_,label,_),n in zip(CLASSES,class_ns)] if mi == 0 else [],
               xlabel='Change in Vd MAE (%)',xticks=[-40,-20,0])
        ax.xaxis.label.set_color(ENDPOINT_COLORS['Vd'])
        ax.set_title(label, fontsize=13, weight='bold')
    heading(fig,.045,.285,'D','Predicted-triad consistency',
            'All 494 class-cohort structures · Lines connect paired seeds · Bars: ±1 seed SD')
    for mi,(model,label,_) in enumerate(MODELS):
        ax = fig.add_axes([.215+mi*.253,.085,.223,.121])
        vals=[]
        for ai,(arm,color) in enumerate((('C0',BASELINE),('C2',BLUE))):
            r=subset(tables['coherence'],cohort='class',procedure='primary',backbone=model,
                     arm=arm,scope='all',stratum='all').set_index('seed').loc[SEEDS]
            assert len(r)==9 and r.n_structures.eq(494).all()
            v=r.mean_absolute_log_residual.to_numpy();vals.append(v)
            assert min(v.min(), v.mean()-v.std(ddof=1)) > .09
            assert max(v.max(), v.mean()+v.std(ddof=1)) < .17
            ax.scatter(ai+JITTER*.14,v,s=20,color=color,alpha=.45,lw=0,zorder=4)
            ax.errorbar(ai,v.mean(),yerr=v.std(ddof=1),fmt='D',color=color,mec='white',mew=.6,
                        capsize=3,ms=6,lw=1.5,zorder=5)
            r=r.reset_index();r['plot_panel']='coherence';counts.append(r)
        for a,b in zip(*vals):
            ax.plot([0,1],[a,b],color=CONTROL,alpha=.24,lw=.7,zorder=2)
        style(ax)
        ax.set(xticks=[0,1],xticklabels=['Structure only','Ionisation category'],xlim=(-.4,1.4),ylim=(.09,.17))
        ax.set_title(label, fontsize=13, weight='bold')
        if mi > 0:
            ax.tick_params(labelleft=False)
    fig.text(.045,.15,'Mean absolute\ntriad residual (log10)',fontsize=12,color=INK,va='center')
    fig.text(.045,.020,'Triad residual = log10[half-life / (ln(2) × volume / clearance)]. The mass-balance penalty is off.',fontsize=11,color=MUTED)
    caption=(
        'Figure 4. Cohort, baseline and mechanistic checks for the ionisation study. A shows all '
        '494 supported categorical assignments (101 anionic, 163 cationic, 171 neutral, 59 '
        'zwitterionic) and the 327-structure numerical subset (65, 90, 119, 53). Colours distinguish the four audited classes. B shows '
        'held-out Vd MAE on the same 388 labelled class-cohort structures. Overall and ionisation-class '
        'medians are estimated in each training fold. Morgan ridge uses 2048-bit radius-2 '
        'fingerprints with or without the class one-hot vector; regularisation is selected in '
        'inner CV. These deterministic controls have one prediction trajectory; their lack '
        'of a seed-SD bar does not imply zero population uncertainty. Neural points are nine '
        'seeds, diamonds are means and bars are sample seed SD. No model superiority test is '
        'encoded in this absolute-error panel. C shows the C2-C0 comparison within each class, '
        'using only observed Vd labels: 79 anionic, 139 cationic, 123 neutral and 47 zwitterionic '
        'structures. The absolute paired MAE differences and 95% crossed seed/group bootstrap '
        'intervals are divided by the fixed comparator mean MAE within that model and stratum '
        'and multiplied by 100. These are exploratory intervals with no multiple-comparison '
        'adjustment. D uses all 494 structures because the triad consists of predicted endpoints '
        'and requires no measured triad. The plotted metric is mean absolute '
        'log10[t_half/(ln(2)*Vd/CL)] across structures for each seed. Faint lines connect paired '
        'seeds, with horizontal offsets solely for visibility. Diamonds and bars show the mean '
        'and sample SD of the nine trajectory values. This assesses internal algebraic '
        'agreement, not agreement with measured PK. Neither arm has a mass-balance training '
        'penalty. Frozen ChemBERTa refers to its frozen encoder with a trained MLP head.')
    save(fig,'ionisation_diagnostics',caption,counts)


def main():
    tables,manifest=load()
    overview(tables);diagnostics(tables)
    outputs={p.name:sha(p) for stem in ['ionisation_overview','ionisation_diagnostics']
             for p in HERE.glob(stem+'*') if p.is_file()}
    record={'campaign_id':manifest['campaign_id'],'new_training':False,'seed_count':9,
            'input_score_manifest_sha256':sha(SCORES/'scoring_manifest.json'),
            'scoring_inputs':manifest['files'],'script_sha256':sha(Path(__file__)),
            'cohort_inputs':{cohort:sha(EXP/'data/release'/cohort/'model_dataset.csv')
                             for cohort in ['class','numerical']},
            'bootstrap_display':'Absolute-difference intervals scaled by fixed mean comparator MAE',
            'outputs':outputs}
    (HERE/'manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('Rendered overview and diagnostics; score hashes and paired-seed means verified.')


if __name__=='__main__':
    main()
