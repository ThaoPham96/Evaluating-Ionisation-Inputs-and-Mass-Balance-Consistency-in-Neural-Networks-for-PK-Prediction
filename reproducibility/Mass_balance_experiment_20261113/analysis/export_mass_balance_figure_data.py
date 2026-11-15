"""Export figure inputs from verified primary outer predictions; CPU only."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from mass_balance_campaign import PRIMARY, SEEDS, ENDPOINTS, read_csv, sha256, write_json
from mass_balance_controls import ALPHA_GRID, partial_correction, residual_log10
from summarize_mass_balance_campaign import (
    Cohort, Predictions, POLICIES, scoring_units, seed_metrics, summarize_seed_rows, write_table,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', type=Path, required=True)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cohort = Cohort(args.release_dir)
    source = Predictions(args.campaign_dir, cohort)
    ids = list(cohort.index)
    outer = {r['structure_id']: r for r in read_csv(args.release_dir / 'outer.csv')}
    names = ('t_half_log10', 'Vd_log10', 'CL_log10', 'F_percent', 'PPB_percent', 'Cmax_log10', 'Tmax_log10')
    columns = ['backbone', 'policy', 'seed', 'structure_id', 'group_id', 'outer_fold',
               'selected_lambda', 'selected_alpha', 'same_row_triad_eligible']
    columns += ['pred_' + name for name in names]
    columns += ['observed_endpoint_' + name for name in names]
    columns += ['observed_same_row_' + name for name in names[:3]]
    columns += ['signed_residual_log10', 'abs_residual_log10', 'within_1_25',
                'same_row_triad_log10_mae']
    row_count = 0
    alpha_rows = []
    with gzip.open(out / 'primary_outer_predictions.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for backbone in PRIMARY:
            baseline, missing = source.pooled(backbone, 'baseline', SEEDS)
            if missing:
                raise ValueError(missing)
            for policy in POLICIES:
                arrays, missing = source.pooled(backbone, policy, SEEDS)
                if missing:
                    raise ValueError(missing)
                for seed, transformed in arrays.items():
                    if not np.isfinite(transformed).all():
                        raise ValueError('Unexpected nonfinite primary predictions')
                    pred = scoring_units(transformed)
                    residual = residual_log10(pred[:, :3])
                    errors = np.abs(pred[:, :3] - cohort.triad_truth).mean(axis=1)
                    for i, sid in enumerate(ids):
                        fold = int(outer[sid]['outer_fold'])
                        loss = source.selection(backbone, fold, 'lambda')
                        strength = (loss['selected_lambda'] if policy == 'selected_positive' else
                                    loss['selected_zero_eligible_lambda'] if policy == 'zero_eligible' else 0)
                        alpha = ''
                        for method in ('orthogonal', 'half_life'):
                            if policy == 'selected_alpha_' + method:
                                alpha = source.selection(backbone, fold, 'baseline')['alpha_controls'][method]['selected_alpha']
                            elif policy == 'exact_' + method:
                                alpha = 1.0
                        row = [backbone, policy, seed, sid, outer[sid]['group_id'], fold,
                               strength, alpha, int(cohort.triad_eligible[i])]
                        row += list(pred[i]) + list(cohort.endpoint_truth[i]) + list(cohort.triad_truth[i])
                        row += [residual[i], abs(residual[i]), int(abs(residual[i]) <= np.log10(1.25)), errors[i]]
                        writer.writerow(['' if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for v in row])
                        row_count += 1
            # Descriptive fixed-alpha curves use the same first three seeds as fixed-lambda curves.
            for method in ('orthogonal', 'half_life'):
                for alpha in ALPHA_GRID:
                    policy = f'fixed_alpha_{method}_{alpha:g}'
                    for seed in SEEDS[:3]:
                        values = baseline[seed].copy()
                        values[:, :3] = partial_correction(values[:, :3], alpha, method)
                        metrics = seed_metrics(cohort, values, baseline[seed])
                        alpha_rows.extend(dict(backbone=backbone, policy=policy, seed=seed,
                                               domain=domain, metric=metric, value=value)
                                          for domain, entries in metrics.items() for metric, value in entries.items())
            print(f'Exported {backbone}', flush=True)
    expected_rows = len(PRIMARY) * len(POLICIES) * len(SEEDS) * len(ids)
    if row_count != expected_rows:
        raise ValueError((row_count, expected_rows))
    write_table(out / 'fixed_alpha_seed_metrics.csv',
                ('backbone', 'policy', 'seed', 'domain', 'metric', 'value'), alpha_rows)
    alpha_aggregates = summarize_seed_rows(alpha_rows, {r['policy']: SEEDS[:3] for r in alpha_rows})
    write_table(out / 'fixed_alpha_summary.csv',
                ('backbone', 'policy', 'domain', 'metric', 'n_seeds_expected', 'n_seeds_defined', 'mean', 'sd'),
                alpha_aggregates)
    fit_rows = []
    for directory in sorted((args.campaign_dir / 'results').iterdir()):
        request = json.loads((directory / 'request.json').read_text())['request']
        summary = json.loads((directory / 'summary.json').read_text())
        fit_rows.append({**{k: request.get(k) for k in ('logical_id', 'block', 'backbone', 'outer_fold',
                          'inner_fold', 'seed', 'lambda_value', 'endpoint', 'candidate_id')},
                        'depth': summary['config'].get('depth'),
                        'actual_epochs': summary['actual_epochs'], 'best_epoch': summary['best_epoch'],
                        'parameter_count': summary['parameter_count'],
                        'predictions_path': f'campaign/results/{directory.name}/predictions.csv'})
    if len(fit_rows) != 4240:
        raise ValueError('Expected all 4240 published fits')
    write_table(out / 'all_fits_index.csv', tuple(fit_rows[0]), fit_rows)
    write_json(out / 'manifest.json', {
        'campaign_id': source.campaign_id,
        'primary_prediction_rows': row_count,
        'unique_structures_per_model_policy_seed': len(ids),
        'fit_count': len(fit_rows),
        'primary_policies': list(POLICIES),
        'primary_seeds': list(SEEDS),
        'fixed_alpha_curve_seeds': list(SEEDS[:3]),
        'scoring': 'Five endpoints log10; F/PPB percentage units. Same-row triad truth is separate from endpoint truth.',
        'code_sha256': sha256(Path(__file__)),
        'outputs': {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'manifest.json'},
    })
    print(f'{row_count:,} primary prediction rows; {len(fit_rows):,} fit records')


if __name__ == '__main__':
    main()
