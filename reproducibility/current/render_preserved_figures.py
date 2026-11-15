"""Mechanical figure corrections using frozen plotted data; no fitting or rescoring."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'figures_final'

OLD = ROOT / 'Unified_ionisation_combined_20261114/reporting/output'
records, hashes = {}, {}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def module(name, path):
    hashes[str(path)] = sha(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def csv(p):
    hashes[str(p)] = sha(p)
    return pd.read_csv(p, float_precision='round_trip')

def geometry(fig):
    """Record all data coordinates and panel positions, excluding text appearance."""
    h = hashlib.sha256()
    for ax in fig.axes:
        h.update(repr((ax.get_position().bounds, ax.get_xlim(), ax.get_ylim())).encode())
        for line in ax.lines:
            for arr in (line.get_xdata(), line.get_ydata()):
                h.update(np.asarray(arr).tobytes())
        for coll in ax.collections:
            h.update(np.asarray(coll.get_offsets()).tobytes())
            if coll.get_array() is not None:
                h.update(np.asarray(coll.get_array()).tobytes())
            for path in coll.get_paths():
                h.update(path.vertices.tobytes())
    return h.hexdigest()

def save(fig, directory, stem):
    directory.mkdir(exist_ok=True)
    before = geometry(fig)
    changes = []
    for label in fig.findobj(Text):
        old = label.get_text()
        new = old.replace('ChemBERTA', 'ChemBERTa').replace('MAE relative to structure only (%)', 'MAE vs structure only (%)').replace(r'\ln 2', r'\ln\,2')
        if new != old:
            label.set_text(new)
            changes.append(dict(old=old, new=new))
        if stem == 'mass_balance_main_v3' and new == 'Molecular structure':
            x, y = label.get_position()
            label.set_position((x + 2.5, y))
            changes.append(dict(label=new, horizontal_shift=2.5, purpose='font-compatible panel-letter spacing'))
    if stem == 'combined_main':
        for ax in fig.axes:
            if any('%' in t.get_text() for t in ax.get_xticklabels()):
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'.replace('-', '\u2212')))
                changes.append(dict(ticks='plain numbers; percent unit retained in axis title'))
    if stem == 'ionisation_supplementary_selected':
        for ax in fig.axes:
            if len(ax.texts) != 35 or not ax.collections:
                continue
            mesh = ax.collections[0]
            values = np.asarray(mesh.get_array()).reshape(5,7)
            for label in ax.texts:
                x, y = label.get_position()
                value = values[int(y), int(x)]
                rgb = np.asarray(mesh.cmap(mesh.norm(value)))[:3]
                linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
                luminance = linear @ np.array([.2126, .7152, .0722])
                black_contrast = (luminance + .05) / .05
                white_contrast = 1.05 / (luminance + .05)
                colour = 'white' if value <= -10 or white_contrast > black_contrast else 'black'
                label.set_color(colour)
                label.set_path_effects([])
                contrast = white_contrast if colour == 'white' else black_contrast
                assert contrast >= 3
                changes.append(dict(cell=label.get_text(), colour=colour, contrast=float(contrast)))
    after = geometry(fig)
    assert before == after, stem
    fig.canvas.draw()
    for ext in ['pdf','png','svg']:
        metadata = {'CreationDate':None} if ext == 'pdf' else {'Date':None} if ext == 'svg' else None
        fig.savefig(directory / f'{stem}.{ext}', dpi=300, facecolor='white', metadata=metadata)
    records[stem] = dict(geometry_before=before, geometry_after=after, changes=changes,
                         outputs={ext:sha(directory/f'{stem}.{ext}') for ext in ['pdf','png','svg']})
    plt.close(fig)

def mass_fold_panel(fig, curves, mod):
    """Show the saved mean ECDFs and exact 1.5/2/5-fold values in Panel C."""
    mod.fs.letter(fig, 'C', .298, .752)
    thresholds=(1.5,2.0,5.0)
    styles=[('baseline',mod.fs.BASELINE_GREY,mod.fs.BASELINE_DASH,1.2,'No loss'),
            ('selected_positive',mod.fs.LOSS,'-',1.6,'With loss')]
    values=[]
    for column,(model,title) in enumerate(mod.MODELS):
        ax=mod.model_axes(fig,column,.569,.161,title)
        lookup={}
        for policy,colour,linestyle,linewidth,_ in styles:
            d=curves[curves.backbone.eq(model)&curves.policy.eq(policy)].sort_values('fold_disagreement')
            assert d.mean_cumulative_percent.is_monotonic_increasing
            x=d.fold_disagreement.to_numpy();y=d.mean_cumulative_percent.to_numpy()
            # A post-step empirical CDF is evaluated at the last observed x <= cutoff.
            # Interpolation would incorrectly allocate partial observations.
            at=lambda t:float(y[np.searchsorted(x,t,side='right')-1])
            shown=d[d.fold_disagreement.le(5.12)]
            ax.step(np.r_[shown.fold_disagreement,5.12],
                    np.r_[shown.mean_cumulative_percent,at(5.12)],where='post',
                    color=colour,lw=linewidth,ls=linestyle,zorder=3)
            for threshold in thresholds:
                percent=at(threshold);lookup[policy,threshold]=percent
                ax.plot(threshold,percent,'o',ms=2.8,color=colour,mec='white',mew=.35,zorder=5)
                values.append(dict(backbone=model,policy=policy,fold_threshold=threshold,
                                   mean_percent=percent,n_structures_per_seed=952,n_seeds=9))
        for threshold in thresholds:
            ax.plot([threshold,threshold],[0,max(lookup[p,threshold] for p,*_ in styles)],
                    color=mod.fs.REFERENCE,lw=.6,ls=(0,(3,2.5)),zorder=1)
        ax.set(xscale='log',xlim=(1/1.03,5.12),ylim=(-2.5,102.5),yticks=[0,25,50,75,100])
        ax.set_xticks([1,1.5,2,5],['1','1.5','2','5']);ax.minorticks_off()
        ax.grid(axis='y',color=mod.GRID,lw=.5);ax.set_axisbelow(True)
        ax.spines['bottom'].set_bounds(1,5);ax.spines['left'].set_bounds(0,100)
        if column==0:ax.set_ylabel('Predictions within fold (%)',labelpad=3)
        else:ax.tick_params(axis='y',labelleft=False,length=0)
        # The inset occupies only the empty lower-right area below both curves.
        ax.add_patch(Rectangle((.24,.025),.77,.575,transform=ax.transAxes,
                               facecolor='white',edgecolor='none',zorder=6,clip_on=False))
        labels=[]
        def label(x,y,text,**kw):
            obj=ax.text(x,y,text,transform=ax.transAxes,fontsize=mod.SMALL,
                        va='center',zorder=8,clip_on=False,**kw)
            labels.append(obj)
        label(.64,.55,'Within fold (%)',ha='center',fontweight='bold')
        for x,(policy,colour,linestyle,linewidth,name) in zip([.67,1.0],styles):
            ax.plot([x-.20,x],[.47,.47],transform=ax.transAxes,color=colour,
                    ls=linestyle,lw=linewidth,zorder=8,clip_on=False)
            label(x,.41,name,ha='right')
        for y,threshold in zip([.29,.17,.05],thresholds):
            label(.19,y,f'\u2264{threshold:g}\u00d7',ha='left')
            label(.67,y,f'{lookup["baseline",threshold]:.2f}',ha='right')
            label(1.0,y,f'{lookup["selected_positive",threshold]:.2f}',ha='right')
        ax._fold_inset_labels=labels
    fig.text((mod.LEFT+mod.RIGHT)/2,.536,'Predicted mass-balance\nfold disagreement',ha='center',va='top')
    pd.DataFrame(values).to_csv(mod.ROOT/'predicted_fold_thresholds.csv',index=False)


def mass_main():
    source = ROOT / 'Mass_balance_experiment_20261113/supplementary_completion/main_figure_layout_v3'
    mod = module('minor_mass_main', source / 'render_v3.py')
    mod.ROOT = OUT / 'fig1'
    mod.ROOT.mkdir(exist_ok=True)
    with plt.rc_context():
        mod.fs.configure()
        fig = plt.figure(figsize=(mod.WIDTH, mod.BODY_HEIGHT), facecolor='white')
        mod.observed_panel(fig, csv(mod.INPUTS/'same_row_complete_triads.csv'))
        mass_fold_panel(fig, csv(mod.INPUTS/'cdf_values.csv.gz'), mod)
        mod.split_panel(fig, csv(mod.INPUTS/'split_1_5_summary.csv'))
        mod.endpoint_panel(fig, csv(mod.INPUTS/'endpoint_seed_values.csv'), csv(mod.INPUTS/'average_seed_values.csv'),
                           csv(mod.INPUTS/'endpoint_summary.csv'))
        for label in fig.findobj(Text):
            label.set_color('black')
        ratio = mod.BODY_HEIGHT / mod.HEIGHT
        for ax in fig.axes:
            x,y,w,h = ax.get_position().bounds
            ax.set_position([x,y*ratio,w,h*ratio])
        for label in fig.texts:
            x,y = label.get_position()
            label.set_position((x,y*ratio))
        fig.set_size_inches(mod.WIDTH, mod.HEIGHT)
        mod.schematic(fig)
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        for ax in fig.axes:
            labels=getattr(ax,'_fold_inset_labels',[])
            boxes=[label.get_window_extent(renderer) for label in labels]
            for i,a in enumerate(boxes):
                for j,b in enumerate(boxes[i+1:],i+1):
                    assert not a.overlaps(b),(labels[i].get_text(),labels[j].get_text())
        save(fig, mod.ROOT, mod.STEM)

def mass_supp():
    source = ROOT / 'Mass_balance_experiment_20261113/supplementary_completion/supplementary_selected_v1'
    mod = module('minor_mass_supp', source/'render.py')
    data = {name:csv(source/f'{name}.csv') for name in ['encoder_seed_metrics','encoder_contrasts',
            'single_task_contrasts','single_task_average','single_task_ecdf','training_loss_strata']}
    with plt.rc_context():
        mod.fs.configure()
        fig = plt.figure(figsize=mod.SIZE, facecolor='white')
        mod.encoder_panels(fig, data['encoder_seed_metrics'], data['encoder_contrasts'])
        mod.single_task_panels(fig, pd.concat([data['single_task_contrasts'],data['single_task_average']],ignore_index=True),data['single_task_ecdf'])
        mod.strata_panel(fig,data['training_loss_strata'])
        for label in fig.findobj(Text):
            label.set_color('black')
            # The public runtime uses DejaVu Sans when Arial is unavailable.
            # Wrap these adjacent labels without changing any data coordinates.
            label.set_text(label.get_text().replace(
                'Predictions within\n1.25-fold disagreement (%)',
                'Predictions within\n1.25-fold (%)').replace(
                'Change in triad MAE with loss\n(% of baseline MAE), 95% interval',
                'Triad MAE change with loss\n(% of baseline), 95% interval'))
        assert not mod.fs.check_layout(fig), mod.fs.check_layout(fig)
        save(fig, OUT/'figS1', mod.STEM)

def main(output):
    global OUT
    OUT=Path(output)
    OUT.mkdir(exist_ok=True,parents=True)
    mass_main()
    mass_supp()
    adapter=module('current_unified_adapter',ROOT/'Unified_ionisation_combined_20261114/reporting/render_staged.py')
    adapter.save=save
    with plt.rc_context():
        limits=json.loads((OLD/'fig2/scatter_inputs/manifest.json').read_text())['axis_limits_log10']
        adapter.render_s2(OUT/'figS2',ROOT/'Ionisation_experiment_20261113',
            csv(OLD/'figS2/single_ph_estimates.csv'),csv(OLD/'figS2/single_ph_seed_values.csv'),
            csv(OLD/'figS2/scatter_points.csv'),csv(OLD/'figS2/scatter_metrics.csv'),limits,hashes)
    (OUT/'figure_display_manifest.json').write_text(json.dumps(dict(figures=records,input_sha256=hashes),indent=2)+'\n')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
