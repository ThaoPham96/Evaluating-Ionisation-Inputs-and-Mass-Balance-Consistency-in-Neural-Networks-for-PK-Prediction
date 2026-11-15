"""Matched observed-predicted panels: structure only versus pH 7.4 input."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ionisation_style import AXES, BLUE, CATEGORIES, CATEGORY_EDGE, CONTROL, FONT, GRID, INK, MUTED, NEUTRAL_BOX, ENDPOINT_COLORS
SOURCE = HERE.parent
ARMS = {'REF0': 'Structure only', 'REF74': 'Structure + ionisation at pH 7.4'}
MODELS = {'GraphConv': 'GraphConv', 'Chemprop': 'Chemprop',
          'ChemBERTa_head': 'Frozen ChemBERTa + MLP'}
ENDPOINTS = {'Vd': ('Volume of distribution', 'L', 259),
             'CL': ('Clearance', 'L/h', 246)}
plt.rcParams.update({'font.family': FONT, 'font.size': 12, 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': INK,
    'axes.edgecolor': AXES, 'axes.linewidth': .65,
    'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    previous = json.loads((SOURCE / 'manifest.json').read_text())
    for name in ['plotted_points.csv', 'metrics.csv']:
        assert digest(SOURCE / name) == previous['outputs'][name]
    for path, expected in previous['sources'].items():
        assert digest(Path(path)) == expected
    data = pd.read_csv(SOURCE / 'plotted_points.csv', dtype={'structure_id': str})
    data = data.loc[data.arm.isin(ARMS)].copy()
    old_metrics = pd.read_csv(SOURCE / 'metrics.csv')
    metrics = []
    for ep, (_, _, n) in ENDPOINTS.items():
        for model in MODELS:
            paired = []
            for arm in ARMS:
                d = data.loc[data.endpoint.eq(ep) & data.backbone.eq(model) & data.arm.eq(arm)]
                assert len(d) == n and d.structure_id.nunique() == n
                assert d.ionisation_class.isin(CATEGORIES).all()
                y, x = d.observed_log10.to_numpy(), d.predicted_log10.to_numpy()
                assert np.isfinite(x).all() and np.isfinite(y).all()
                r2 = 1 - np.square(y-x).sum() / np.square(y-y.mean()).sum()
                mae = np.abs(y-x).mean()
                source_metric = old_metrics.loc[old_metrics.arm.eq(arm) & old_metrics.backbone.eq(model) & old_metrics.endpoint.eq(ep)].iloc[0]
                assert np.isclose(r2, source_metric.r2_displayed_seed_mean)
                metrics.append({'arm': arm, 'backbone': model, 'endpoint': ep,
                                'n': n, 'r2_displayed_seed_mean': r2,
                                'mae_log10_displayed_seed_mean': mae})
                paired.append(d.set_index('structure_id').sort_index())
            assert paired[0][['observed_log10', 'ionisation_class', 'outer_fold']].equals(
                paired[1][['observed_log10', 'ionisation_class', 'outer_fold']])
    metrics = pd.DataFrame(metrics)
    with PdfPages(HERE / 'baseline_vs_ionisation.pdf') as pdf:
        for ep, (name, unit, n) in ENDPOINTS.items():
            fig, axes = plt.subplots(2, 3, figsize=(12.8, 10.2))
            fig.subplots_adjust(left=.075, right=.975, bottom=.15, top=.79,
                                hspace=.44, wspace=.22)
            fig.text(.045, .975, f'{name}: effect of adding ionisation',
                     fontsize=18, weight='bold', va='top', color=ENDPOINT_COLORS[ep])
            fig.text(.045, .935,
                     'Same drugs, grouped folds and nine training seeds; one mean held-out prediction per drug',
                     fontsize=11.8, color=MUTED)
            handles = [Line2D([], [], marker=marker, mfc=color, mec=CATEGORY_EDGE, mew=.4, linestyle='none',
                              markersize=6, label=label) for label, color, marker in CATEGORIES.values()]
            fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.04, .915),
                       frameon=False, ncol=4, fontsize=11.8,
                       handletextpad=.35, columnspacing=1.5)
            fig.text(.075, .858, 'A  Structure only', fontsize=13, weight='bold', color=CONTROL,
                     bbox=dict(facecolor=NEUTRAL_BOX, edgecolor='none', pad=3))
            fig.text(.075, .462, 'B  Structure + ionisation at pH 7.4',
                     fontsize=13, weight='bold', color=BLUE)
            lo, hi = previous['axis_limits_log10'][ep]
            for i, arm in enumerate(ARMS):
                for j, (model, label) in enumerate(MODELS.items()):
                    ax = axes[i, j]
                    d = data.loc[data.arm.eq(arm) & data.backbone.eq(model) & data.endpoint.eq(ep)]
                    r = metrics.loc[metrics.arm.eq(arm) & metrics.backbone.eq(model) & metrics.endpoint.eq(ep)].iloc[0]
                    ax.plot([lo, hi], [lo, hi], color=AXES, ls=(0, (4, 3)), lw=.85, zorder=1)
                    for category, (_, colour, marker) in CATEGORIES.items():
                        c = d.loc[d.ionisation_class.eq(category)]
                        ax.scatter(c.predicted_log10, c.observed_log10, s=18,
                                   facecolor=colour, marker=marker, alpha=.78, edgecolor=CATEGORY_EDGE, linewidth=.28, zorder=3)
                    ax.set(xlim=(lo, hi), ylim=(lo, hi))
                    ax.set_aspect('equal', adjustable='box')
                    ax.grid(color=GRID, lw=.65, zorder=0)
                    ax.xaxis.set_major_locator(MultipleLocator(1))
                    ax.yaxis.set_major_locator(MultipleLocator(1))
                    ax.set_xlabel(f'Predicted log10 {ep} ({unit})', fontsize=12)
                    ax.set_ylabel(f'Observed log10 {ep} ({unit})' if j == 0 else '', fontsize=12)
                    ax.tick_params(labelsize=11.2)
                    ax.set_title(label + '\n' + r'$R^2$' + f' = {r.r2_displayed_seed_mean:.3f}  |  n = {n}',
                                 fontsize=12.6, weight='bold', pad=7)
            fig.text(.045, .065,
                     'Colours and marker shapes annotate the pH 7.4 category in both rows.',
                     fontsize=11.4, color=MUTED)
            fig.text(.045, .041,
                     'Predictions are averaged in log10 space over nine seeds; R² describes these displayed averages. Dashed line: equality.',
                     fontsize=11.2, color=MUTED)
            fig.text(.975, .022, f'{list(ENDPOINTS).index(ep)+1} / 2', ha='right', fontsize=8.5, color=MUTED)
            pdf.savefig(fig)
            for ext in ['png', 'svg']:
                fig.savefig(HERE / f'{ep.lower()}_baseline_vs_ionisation.{ext}', dpi=240, facecolor='white')
            plt.close(fig)
    metrics.to_csv(HERE / 'metrics.csv', index=False)
    data.to_csv(HERE / 'plotted_points.csv', index=False)
    (HERE / 'caption.txt').write_text(
        'Observed versus predicted Vd and clearance with and without ionisation input.\n'
        'Page 1: Vd, 259 labelled structures. Page 2: clearance, 246 labelled structures. '
        'Each page compares structure-only models (REF0, top row) with the same backbone supplied with '
        'a four-channel one-hot pH 7.4 ionisation category (REF74, bottom row). Columns are GraphConv, '
        'Chemprop and frozen ChemBERTa with a trained MLP head. The cohort, grouped folds and nine '
        'training seeds are matched. Both arms use equal explicit endpoint loss weights and no mass-balance penalty.\n'
        'Each point represents one structure. The horizontal coordinate is the mean of its nine '
        'out-of-fold log10 predictions; the vertical coordinate is its observed log10 endpoint. '
        'The average corresponds to a geometric mean prediction in physical units. R-squared is '
        '1 - sum((observed - predicted)^2)/sum((observed - mean(observed))^2), evaluated on these displayed '
        'averages. It differs from the mean of nine separately scored seeds used in the main comparison. '
        'The dashed line denotes equality. Axes are shared across conditions and models for each endpoint, '
        'and every labelled point is retained.\n'
        'Colours and marker shapes show the original pH 7.4 category in both rows; they annotate the baseline plots '
        'but were not supplied to the baseline models. Acid-only, base-only, neither-active and dual-active '
        'are operational acidic/basic pKa threshold categories. They do not identify resolved microspecies. '
        'This comparison measures the effect of supplying ionisation information and does not isolate '
        'a learned embedding mechanism. No new training was required.\n', encoding='utf-8')
    (HERE / 'manifest.json').write_text(json.dumps({
        'sources': {str(SOURCE / f): digest(SOURCE / f) for f in ['manifest.json', 'plotted_points.csv', 'metrics.csv']},
        'axis_limits_log10': previous['axis_limits_log10'], 'arms': list(ARMS),
        'outputs': {p.name: digest(p) for p in HERE.iterdir() if p.suffix in ['.pdf', '.png', '.svg', '.csv', '.txt']}
    }, indent=2), encoding='utf-8')
    print(metrics.to_string(index=False))
