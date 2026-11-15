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
        new = old.replace('ChemBERTA', 'ChemBERTa').replace('MAE relative to structure only (%)', 'MAE vs structure only (%)')
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

def mass_main():
    source = ROOT / 'Mass_balance_experiment_20261113/supplementary_completion/main_figure_layout_v3'
    mod = module('minor_mass_main', source / 'render_v3.py')
    mod.ROOT = OUT / 'fig1'
    mod.ROOT.mkdir(exist_ok=True)
    with plt.rc_context():
        mod.fs.configure()
        fig = plt.figure(figsize=(mod.WIDTH, mod.BODY_HEIGHT), facecolor='white')
        mod.observed_panel(fig, csv(mod.INPUTS/'same_row_complete_triads.csv'))
        mod.continuous_panel(fig, csv(mod.INPUTS/'cdf_values.csv.gz'))
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
