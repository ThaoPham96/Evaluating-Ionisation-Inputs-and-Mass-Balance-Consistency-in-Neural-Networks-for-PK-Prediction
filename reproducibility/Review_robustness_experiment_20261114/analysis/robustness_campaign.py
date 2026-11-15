"""Frozen 60-unit T matrix and F/R0/R2 bridges, with verified reference reuse."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import combined_campaign as a
from unified_features import verify_categories
from unified_campaign import INPUT_HASHES

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('P0', 'P1', 'P2', 'P3')
INPUTS = dict(zip(ARMS, ('structure', 'category', 'raw_pka', 'fractions')))
UNITS = {f'{model}-outer{outer}-{arm}': (model, outer, arm)
         for model in a.BACKBONES for outer in a.FOLDS for arm in ARMS}
IMAGE = 'sha256:6a533de192716278129758f72961ff0e635534d20e90ad15f19b6ee20d566487'
COUNTS = dict(units=60, architecture=3000, positive_lambda=1800, T_final=1080,
              R0_calibration_reused=150, R0_final_max=270, R2_final_max=270, total_max=6420)
ADDED_FIELDS = ['protocol', 'input', 'penalty', 'condition', 'source_campaign_id', 'source_fit_id',
                'reference_fit_id', 'config_id', 'fixed_epochs', 'selection_id']


@dataclass
class Campaign:
    root: Path
    plan: dict
    dataset: dict
    triads: dict
    references: dict

    @property
    def release(self): return self.root / 'data/release/numerical'
    @property
    def features(self): return self.root / 'features/numerical'


def code_identity(root):
    return {p.relative_to(root).as_posix(): a.sha(p) for folder in ('analysis', 'tests', 'deployment')
            for p in sorted((root / folder).rglob('*.py'))}


def load_campaign(root=ROOT):
    root = Path(root)
    release, features = root / 'data/release/numerical', root / 'features/numerical'
    inputs = a.input_identity(release, features)
    dataset, splits = a.read_population(release)
    triads = a.by_id(a.rows(root / 'runtime' / a.TRIADS))
    a.validate_triads(triads, dataset, splits)
    recipes = a.read_json(root / 'runtime/p0_recipes.json')
    for name, checksum in INPUT_HASHES.items():
        if a.sha(root / 'runtime' / name) != checksum: raise ValueError('Original recipes/triads identity changed')
    grid = a.read_json(root / 'runtime/search_grid.json')
    if set(grid) != set(a.BACKBONES) or any(len(v) != 10 for v in grid.values()):
        raise ValueError('Exactly ten candidates per model are required')
    if any(grid['Chemprop'][i-1]['hidden_dim'] != 600 for i in (2, 7, 8)):
        raise ValueError('Chemprop candidate IDs 2/7/8 must resolve to width 600')
    for key, recipe in recipes.items():
        model = key.split('-outer')[0]
        if recipe['config'] != grid[model][recipe['candidate_index']]:
            raise ValueError(f'R0 calibration cannot reuse the grid: {key}')
    refs = a.read_json(root / 'runtime/references/reference_manifest.json')
    if a.sha(root / 'runtime/search_grid.json') != refs['grid_sha256']:
        raise ValueError('Original ten-candidate grid changed')
    for filename, checksum in refs['files'].items():
        if a.sha(root / 'runtime/references' / filename) != checksum:
            raise ValueError(f'Frozen reference changed: {filename}')
    gate = a.read_json(root / 'runtime/compatibility.json')
    if gate['passed'] is not True or gate['source_hashes'] != compatibility_sources(root):
        raise ValueError('Compatibility evidence absent or stale; run prepare before launch')
    inventory = verify_categories(list(dataset.values()))
    identity = dict(schema_version=1, inputs=inputs, splits=splits, recipes=recipes, grid=grid,
        reference_manifest_sha256=a.sha(root / 'runtime/references/reference_manifest.json'),
        compatibility_sha256=a.sha(root / 'runtime/compatibility.json'),
        memory_preflight_sha256=a.sha(root / 'runtime/memory_preflight.json') if (root / 'runtime/memory_preflight.json').exists() else None,
        triads_sha256=a.sha(root / 'runtime' / a.TRIADS), code=code_identity(root),
        environment_image=IMAGE, expected_environments=refs['environments'],
        counts=COUNTS, seeds=list(a.SEEDS), search_seed=42, lambdas=list(a.LAMBDAS),
        max_epochs=200, patience=30, min_delta=1e-4, task_weights=[1.] * 7,
        ph_inventory=inventory, feature_arms=list(ARMS),
        architecture_selection='pooled endpoint squared-error sums/counts; equal endpoint mean; candidate ID tie-break',
        lambda_selection='pooled complete-triad log10 MAE; equal endpoint mean; smallest lambda tie-break',
        scaling='train-only population SD for both P2 and P3 and endpoint targets',
        references='F archived immutable; T own input-specific architecture; no F/T denominator mixing',
        deduplication='within unit only, identical complete request; no old/new training cache reuse')
    return Campaign(root, dict(campaign_id=a.digest(identity), identity=identity), dataset, triads, refs)


def compatibility_sources(root):
    paths = [root / 'analysis/combined_train.py', root / 'runtime/reference_combined_train.py',
             root / 'tests/test_robustness.py']
    paths += sorted((root / 'analysis/vendor_ionisation').rglob('*.py'))
    paths += [root / 'analysis/combined_features.py', root / 'analysis/unified_features.py']
    return {p.relative_to(root).as_posix(): a.sha(p) for p in paths}


def prepare(root, archive, original):
    """Local-only: verify archives, freeze R2 calibration and run execution parity."""
    source = original / 'outputs/export/combined'
    grid_source = original.parent / 'Ionisation_experiment_20261113/config/search_grid.json'
    if a.sha(root / 'runtime/search_grid.json') != a.sha(grid_source): raise ValueError('Copied grid differs from archived grid')
    dest = root / 'runtime/references'; dest.mkdir(parents=True, exist_ok=True)
    manifest = a.read_json(source / 'manifest.json')
    for name, checksum in manifest['files'].items():
        if a.sha(source / name) != checksum: raise ValueError(f'Archived export changed: {name}')
    # The only permitted trainer extension is the registration of the new arms.
    previous = (original / 'analysis/combined_train.py').read_text(encoding='utf-8')
    current = (root / 'analysis/combined_train.py').read_text(encoding='utf-8')
    extension = "CELLS.update({(arm, loss): f'{arm}_{int(loss)}' for arm in ('P2', 'P3') for loss in (False, True)})\n"
    if current.replace(extension, '') != previous:
        raise ValueError('Reference execution changed beyond the reviewed arm registration')
    for p in (original / 'analysis/vendor_ionisation').rglob('*.py'):
        if a.sha(p) != a.sha(root / 'analysis/vendor_ionisation' / p.relative_to(original / 'analysis/vendor_ionisation')):
            raise ValueError('Vendored implementation changed')
    for name in ('combined_features.py', 'unified_features.py'):
        if a.sha(original / 'analysis' / name) != a.sha(root / 'analysis' / name):
            raise ValueError('Reference feature execution changed')
    shutil.copyfile(original / 'analysis/combined_train.py', root / 'runtime/reference_combined_train.py')
    shutil.copyfile(source / 'final_predictions.csv', dest / 'final_predictions.csv')
    shutil.copyfile(source / 'selections.json', dest / 'selections.json')
    dataset, splits = a.read_population(root / 'data/release/numerical')
    inner = defaultdict(list)
    for row in a.rows(source / 'inner_grid_predictions.csv'):
        inner[row['backbone'], int(row['outer_fold']), row['feature_arm'], float(row['lambda_value'])].append(row)
    environments, fit_provenance, r2 = {}, {}, {}
    selections = a.read_json(source / 'selections.json')
    for selection in selections:
        model, outer = selection['backbone'], selection['outer_fold']
        unit = f'{model}-outer{outer}'
        expected = set(splits['inner'][str(outer)])
        for arm in ('P0', 'P1'):
            records = []
            for entry in selection['grid']:
                lam = entry['lambda_value']; pool = inner[model, outer, arm, lam]
                if len(pool) != len(expected) or {r['structure_id'] for r in pool} != expected:
                    raise ValueError(f'Archived R2 calibration coverage failure: {unit}/{arm}/{lam}')
                eligible = [r for r in pool if r['same_row_triad_eligible'] == '1']
                errors = {ep: statistics.fmean(abs(float(r['q_'+ep]) - float(r['true_'+ep])) for r in eligible)
                          for ep in a.TRIAD_ENDPOINTS}
                saved = entry['arms'][arm]
                if any(abs(errors[ep] - saved['endpoint_log10_mae'][ep]) > 1e-10 for ep in errors):
                    raise ValueError('Archived R2 score does not replay from inner predictions')
                for ep in saved['epochs']:
                    path = archive / unit / 'fits' / ep['fit_id']
                    marker = a.read_json(path / 'complete.json')
                    for name in ('summary.json', 'request.json', 'predictions.csv', 'scaler.json'):
                        if a.sha(path / name) != marker['files'][name]: raise ValueError(f'Changed archive: {path}/{name}')
                    summary = a.read_json(path / 'summary.json')
                    request = a.read_json(path / 'request.json')['request']
                    if summary['best_epoch'] != ep['best_epoch'] or request['feature_arm'] != arm or request['lambda_value'] != lam:
                        raise ValueError('Archived selection epochs/request differ')
                    if set(request['val_ids']) != {r['structure_id'] for r in pool if r['fit_id'] == ep['fit_id']}:
                        raise ValueError('Archived inner fit IDs mismatch')
                    env = summary['environment']
                    if model in environments and environments[model] != env: raise ValueError('Mixed archived environments')
                    environments[model] = env
                    fit_provenance[ep['fit_id']] = dict(complete_sha256=a.sha(path / 'complete.json'),
                        summary_sha256=a.sha(path / 'summary.json'), request_sha256=a.sha(path / 'request.json'),
                        predictions_sha256=a.sha(path / 'predictions.csv'), best_epoch=summary['best_epoch'],
                        environment_hash=summary['environment_hash'])
                records.append(dict(lambda_value=lam, score=statistics.fmean(errors.values()),
                    endpoint_log10_mae=errors, epochs=saved['epochs'], median_epoch=saved['median_epoch']))
            selected = min((r for r in records if r['lambda_value'] > 0), key=lambda r: (r['score'], r['lambda_value']))
            zero_eligible = min(records, key=lambda r: (r['score'], r['lambda_value']))
            r2[f'{unit}-{arm}'] = dict(grid=records, selected_lambda=selected['lambda_value'],
                zero_eligible_lambda=zero_eligible['lambda_value'], median_epoch=selected['median_epoch'],
                shared_lambda=selection['selected_lambda'], n_complete_triads=len(eligible))
    # Bind all reused final fit identities and stopping epochs as well.
    for row in a.rows(dest / 'final_predictions.csv'):
        fid = row['fit_id']
        if fid in fit_provenance: continue
        path = archive / f"{row['backbone']}-outer{row['outer_fold']}" / 'fits' / fid
        marker = a.read_json(path / 'complete.json')
        summary = a.read_json(path / 'summary.json')
        for name in ('summary.json', 'request.json', 'predictions.csv'):
            if a.sha(path / name) != marker['files'][name]: raise ValueError(f'Changed archived final: {path}')
        if summary['environment'] != environments[row['backbone']]: raise ValueError('Final reference environment differs')
        fit_provenance[fid] = dict(complete_sha256=a.sha(path / 'complete.json'),
            summary_sha256=a.sha(path / 'summary.json'), request_sha256=a.sha(path / 'request.json'),
            predictions_sha256=a.sha(path / 'predictions.csv'), best_epoch=summary['best_epoch'],
            environment_hash=summary['environment_hash'])
    a.write_json(dest / 'r2_calibration.json', r2)
    a.write_json(dest / 'fit_provenance.json', fit_provenance)
    reference = dict(campaign_id=manifest['campaign_id'], original_directory=str(original), archive_directory=str(archive),
        source_export_manifest=manifest, source_export_manifest_sha256=a.sha(source / 'manifest.json'),
        source_inner_predictions_sha256=a.sha(source / 'inner_grid_predictions.csv'), environments=environments,
        original_trainer_sha256=a.sha(original / 'analysis/combined_train.py'), image=IMAGE,
        grid_sha256=a.sha(grid_source),
        files={name: a.sha(dest / name) for name in ('final_predictions.csv', 'selections.json', 'r2_calibration.json', 'fit_provenance.json')})
    a.write_json(dest / 'reference_manifest.json', reference)
    command = [sys.executable, '-m', 'pytest', str(root / 'tests/test_robustness.py'), '-q']
    result = subprocess.run(command, text=True, capture_output=True)
    (root / 'runtime/compatibility_test_output.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode: raise RuntimeError('Compatibility tests failed; see runtime/compatibility_test_output.txt')
    a.write_json(root / 'runtime/compatibility.json', dict(passed=True, source_hashes=compatibility_sources(root),
        command=command, test_output_sha256=a.sha(root / 'runtime/compatibility_test_output.txt'),
        local_python=sys.version, evidence='Actual previous/new trainer equality across all three backbones P0/P1 +/-lambda; checkpoint states, curves, predictions and scalers exact',
        limitation='CPU execution parity is not a GPU numerical-equivalence experiment; pinned original T4 image/environment is checked on every cloud unit before fitting'))
    return dict(compatibility_passed=True, references_frozen=True, R2_new_final_before_exact_dedup=
                sum(v['selected_lambda'] != v['shared_lambda'] for v in r2.values())*9,
                next_step='Run memory_preflight.py, then robustness_campaign.py preflight before packaging')


def request_for(c, unit, config, strength=0., inner=None, seed=42, epochs=None):
    model, outer, arm = UNITS[unit]
    split = c.plan['identity']['splits']; part = split['inner'][str(outer)]
    validation = sorted(sid for sid, fold in part.items() if fold == inner)
    cell = f'{int(arm == "P1")}{int(strength > 0)}' if arm in ('P0', 'P1') else f'{arm}_{int(strength > 0)}'
    request = dict(campaign_id=c.plan['campaign_id'], cohort='numerical', backbone=model,
        feature_arm=arm, cell=cell, outer_fold=outer, inner_fold=inner, config=config,
        seed=seed, max_epochs=200, fixed_epochs=epochs, lambda_value=strength, task_weights=[1.] * 7,
        train_ids=sorted(sid for sid, fold in part.items() if fold != inner), val_ids=validation,
        predict_ids=validation if inner else sorted(sid for sid, fold in split['outer'].items() if fold == outer))
    request['fit_id'] = 'ROB-' + a.digest(request)[:24]
    return request


def architecture_requests(c, unit):
    model, _, _ = UNITS[unit]
    return [[request_for(c, unit, cfg, inner=inner) for inner in a.FOLDS]
            for cfg in c.plan['identity']['grid'][model]]


def pool_score(c, requests, fits):
    sums = {ep: 0. for ep in a.ENDPOINTS}; counts = {ep: 0 for ep in a.ENDPOINTS}
    pool, epochs = {}, []
    for req in requests:
        summary, predictions = fits[req['fit_id']]
        for ep in sums:
            sums[ep] += summary['val_squared_error_sums'][ep]
            counts[ep] += summary['val_observed_counts'][ep]
        for row in predictions:
            sid = row['structure_id']
            if sid in pool: raise ValueError('Repeated inner prediction')
            pool[sid] = row
        epochs.append(dict(inner_fold=req['inner_fold'], fit_id=req['fit_id'], best_epoch=summary['best_epoch']))
    outer = requests[0]['outer_fold']
    if set(pool) != set(c.plan['identity']['splits']['inner'][str(outer)]) or len(epochs) != 5:
        raise ValueError('Incomplete pooled inner coverage')
    eligible = sorted(pool.keys() & c.triads.keys())
    errors = {ep: statistics.fmean(abs(float(pool[sid]['pred_'+ep]) - math.log10(float(c.triads[sid][a.ENDPOINTS[ep]])))
                                  for sid in eligible) for ep in a.TRIAD_ENDPOINTS}
    if any(counts[ep] <= 0 for ep in counts): raise ValueError('No labels for architecture selection')
    return dict(val_squared_error_sums=sums, val_observed_counts=counts,
        val_equal_mse=statistics.fmean(sums[ep] / counts[ep] for ep in sums),
        score=statistics.fmean(errors.values()), endpoint_log10_mae=errors,
        epochs=epochs, median_epoch=int(statistics.median(e['best_epoch'] for e in epochs)),
        n_complete_triads=len(eligible))


def architecture_selection(c, unit, fits):
    grid = [dict(config_id=i+1, config=reqs[0]['config'], **pool_score(c, reqs, fits))
            for i, reqs in enumerate(architecture_requests(c, unit))]
    selected = min(grid, key=lambda r: (r['val_equal_mse'], r['config_id']))
    return dict(grid=grid, selected_config_id=selected['config_id'], config=selected['config'])


def lambda_requests(c, unit, selection):
    return [[request_for(c, unit, selection['config'], strength=lam, inner=inner) for inner in a.FOLDS]
            for lam in a.LAMBDAS[1:]]


def selection_record(c, unit, fits):
    architecture = architecture_selection(c, unit, fits)
    zero = [request_for(c, unit, architecture['config'], inner=inner) for inner in a.FOLDS]
    grids = [zero] + lambda_requests(c, unit, architecture)
    grid = [dict(lambda_value=reqs[0]['lambda_value'], **pool_score(c, reqs, fits)) for reqs in grids]
    selected = min(grid[1:], key=lambda r: (r['score'], r['lambda_value']))
    including_zero = min(grid, key=lambda r: (r['score'], r['lambda_value']))
    model, outer, arm = UNITS[unit]
    recipe = c.plan['identity']['recipes'][f'{model}-outer{outer}']
    result = dict(campaign_id=c.plan['campaign_id'], unit=unit, architecture=architecture,
        grid=grid, selected_lambda=selected['lambda_value'], zero_eligible_lambda=including_zero['lambda_value'])
    if arm in ('P2', 'P3'):
        result['R0'] = dict(architecture['grid'][recipe['candidate_index']])
    else:
        result['R2'] = a.read_json(c.root / 'runtime/references/r2_calibration.json')[unit]
    return result


def final_conditions(c, unit, selection):
    """Logical conditions point to complete requests; exact identical requests alias."""
    model, outer, arm = UNITS[unit]
    arch = selection['architecture']; cfg = arch['config']; config_id = arch['selected_config_id']
    grid = {row['lambda_value']: row for row in selection['grid']}
    result = []
    for seed in a.SEEDS:
        for strength in (0., selection['selected_lambda']):
            request = request_for(c, unit, cfg, strength, seed=seed, epochs=grid[strength]['median_epoch'])
            result.append(dict(protocol='T', penalty='positive' if strength else 'off', config_id=config_id, request=request))
        if arm in ('P2', 'P3'):
            r0 = selection['R0']
            request = request_for(c, unit, r0['config'], seed=seed, epochs=r0['median_epoch'])
            result.append(dict(protocol='F', penalty='off', config_id=r0['config_id'], request=request))
        else:
            r2 = selection['R2']
            if r2['selected_lambda'] != r2['shared_lambda']:
                recipe = c.plan['identity']['recipes'][f'{model}-outer{outer}']
                request = request_for(c, unit, recipe['config'], r2['selected_lambda'], seed=seed, epochs=r2['median_epoch'])
                result.append(dict(protocol='F', penalty='separate', config_id=recipe['candidate_index']+1, request=request))
    return result


def verify_pairing(directory, requests, fits):
    pairs = defaultdict(list)
    for req in requests:
        pairs[a.digest(req['config']), req['seed'], req['inner_fold']].append(req)
    for pair in pairs.values():
        summaries = [fits[req['fit_id']][0] for req in pair]
        if len({s['initial_state_hash'] for s in summaries}) != 1: raise ValueError('Unpaired initialization')
        streams = {}
        for req in pair:
            for row in a.rows(directory / req['fit_id'] / 'learning_curve.csv'):
                key = row['epoch']; value = tuple(row[k] for k in ('batch_order_seed', 'dropout_seed', 'batch_order_sha256'))
                if key in streams and streams[key] != value: raise ValueError('Unpaired batching/dropout')
                streams[key] = value


def execute_unit(c, unit, output, device='cuda', read_only=False):
    from combined_train import run_fit, base, _hashes
    import torch
    path = output / unit; directory = path / 'fits'
    if not read_only: path.mkdir(parents=True, exist_ok=True)
    first = architecture_requests(c, unit)[0][0]
    if not read_only:
        _, env = _hashes(first, c.release, c.features, torch.device(device))
        if env != c.references['environments'][first['backbone']]:
            raise ValueError('Production runtime differs from archived reference; no fit started')
    fits, requests, markers = {}, {}, {}
    def collect(groups):
        for req in groups:
            fid = req['fit_id']
            if fid in requests:
                if requests[fid] != req: raise ValueError('Fit-ID collision')
                continue
            if not read_only: run_fit(req, c.release, c.features, directory / fid, device)
            fits[fid] = a.validate_fit(directory / fid, req)
            if fits[fid][0]['environment'] != c.references['environments'][req['backbone']]:
                raise ValueError('Fit environment differs from original references')
            requests[fid] = req; markers[fid] = a.sha(directory / fid / 'complete.json')
    def work():
        collect(req for group in architecture_requests(c, unit) for req in group)
        arch = architecture_selection(c, unit, fits)
        collect(req for group in lambda_requests(c, unit, arch) for req in group)
        selection = selection_record(c, unit, fits)
        conditions = final_conditions(c, unit, selection)
        collect(row['request'] for row in conditions)
        verify_pairing(directory, list(requests.values()), fits)
        if read_only:
            if a.read_json(path / 'selection.json') != selection: raise ValueError('Selection replay mismatch')
        else: a.write_json(path / 'selection.json', selection)
        marker = dict(campaign_id=c.plan['campaign_id'], unit_id=unit,
            selection_sha256=a.sha(path / 'selection.json'), fit_markers=markers,
            inner_fits=sum(r['inner_fold'] is not None for r in requests.values()),
            final_fits=sum(r['inner_fold'] is None for r in requests.values()), unique_fits=len(requests),
            conditions=[dict(protocol=row['protocol'], penalty=row['penalty'], config_id=row['config_id'],
                             fit_id=row['request']['fit_id']) for row in conditions])
        if read_only:
            if a.read_json(path / 'complete_unit.json') != marker: raise ValueError('Invalid unit marker')
        else: a.write_json(path / 'complete_unit.json', marker)
        return marker, selection, conditions, fits
    if read_only: return work()
    with base._fit_lock(path):
        identity = dict(campaign_id=c.plan['campaign_id'], unit=unit)
        if (path / 'unit.json').exists() and a.read_json(path / 'unit.json') != identity:
            raise ValueError('Output is bound to a different campaign')
        a.write_json(path / 'unit.json', identity)
        return work()


def preflight(c):
    memory = a.read_json(c.root / 'runtime/memory_preflight.json')
    if (memory.get('passed') is not True or memory.get('script_sha256') != a.sha(c.root / 'analysis/memory_preflight.py')
            or memory.get('trainer_sha256') != a.sha(c.root / 'analysis/combined_train.py')):
        raise ValueError('Two-process memory preflight missing or stale')
    ids = []
    from combined_train import _check_request, _check_splits
    for unit in UNITS:
        for group in architecture_requests(c, unit):
            for req in group:
                split_ids, _, _, _ = _check_request(req)
                _check_splits(req, split_ids, c.release)
                ids.append(req['fit_id'])
    if len(ids) != 3000 or len(set(ids)) != 3000: raise ValueError('Architecture request coverage/collision')
    r2 = a.read_json(c.root / 'runtime/references/r2_calibration.json')
    additional = sum(v['selected_lambda'] != v['shared_lambda'] for v in r2.values()) * 9
    return dict(campaign_id=c.plan['campaign_id'], counts=COUNTS, R2_new_final_before_exact_dedup=additional,
        strict_upper_bound=5880+270+additional, units=list(UNITS), compatibility='passed',
        runtime_gate='must match archived per-backbone environment and pinned T4 image before any production fit')


def export(c, output, destination):
    records, selections, markers = [], {}, {}
    provenance = a.read_json(c.root / 'runtime/references/fit_provenance.json')
    r2 = a.read_json(c.root / 'runtime/references/r2_calibration.json')
    for row in a.rows(c.root / 'runtime/references/final_predictions.csv'):
        arm, model, outer = row['feature_arm'], row['backbone'], int(row['outer_fold'])
        unit = f'{model}-outer{outer}-{arm}'; recipe = c.plan['identity']['recipes'][f'{model}-outer{outer}']
        row.update(protocol='F', input=INPUTS[arm], penalty='off' if float(row['lambda_value']) == 0 else 'shared',
            source_campaign_id=c.references['campaign_id'], source_fit_id=row['fit_id'], reference_fit_id='',
            config_id=recipe['candidate_index']+1, fixed_epochs=provenance[row['fit_id']]['best_epoch'], selection_id=unit)
        row['condition'] = arm+'_'+row['penalty']; records.append(row)
        if row['penalty'] == 'shared' and r2[unit]['selected_lambda'] == r2[unit]['shared_lambda']:
            records.append(dict(row, penalty='separate', condition=arm+'_separate'))
    for unit in UNITS:
        marker, selection, conditions, fits = execute_unit(c, unit, output, read_only=True)
        selections[unit] = selection; markers[unit] = marker
        for condition in conditions:
            req = condition['request']
            for pred in fits[req['fit_id']][1]:
                row = a.export_row(req, pred, c.dataset, c.triads)
                row.update(protocol=condition['protocol'], input=INPUTS[req['feature_arm']], penalty=condition['penalty'],
                    condition=req['feature_arm']+'_'+condition['penalty'], source_campaign_id=c.plan['campaign_id'],
                    source_fit_id=req['fit_id'], reference_fit_id='', config_id=condition['config_id'],
                    fixed_epochs=req['fixed_epochs'], selection_id=unit)
                records.append(row)
    reference = {(r['protocol'], r['feature_arm'], r['backbone'], int(r['outer_fold']), int(r['seed'])): r['fit_id']
                 for r in records if r['penalty'] == 'off'}
    for row in records:
        row['reference_fit_id'] = reference[row['protocol'], row['feature_arm'], row['backbone'], int(row['outer_fold']), int(row['seed'])]
    expected_conditions = [('T', arm, penalty) for arm in ARMS for penalty in ('off', 'positive')]
    expected_conditions += [('F', arm, penalty) for arm in ('P0', 'P1') for penalty in ('off', 'shared', 'separate')]
    expected_conditions += [('F', arm, 'off') for arm in ('P2', 'P3')]
    expected = {(protocol, arm, penalty, model, seed, sid) for protocol, arm, penalty in expected_conditions
                for model in a.BACKBONES for seed in a.SEEDS for sid in c.dataset}
    keys = [(r['protocol'], r['feature_arm'], r['penalty'], r['backbone'], int(r['seed']), r['structure_id']) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected: raise ValueError('Final export coverage/duplicates')
    for row in records:
        if int(row['outer_fold']) != int(c.dataset[row['structure_id']]['outer_fold']): raise ValueError('Outer leakage')
        if any(not math.isfinite(float(row['q_'+ep])) for ep in a.ENDPOINTS): raise ValueError('Nonfinite export')
    unique_new = set(fid for m in markers.values() for fid in m['fit_markers'])
    if len(unique_new) > COUNTS['total_max']: raise ValueError('Approved fit count exceeded')
    a.write_csv(destination / 'final_predictions.csv', a.EXPORT_FIELDS + ADDED_FIELDS, records)
    a.write_json(destination / 'selections.json', selections)
    a.write_json(destination / 'plan.json', c.plan)
    a.write_json(destination / 'unit_markers.json', markers)
    manifest = dict(campaign_id=c.plan['campaign_id'], final_rows=len(records), unique_new_fits=len(unique_new),
        conditions=[list(x) for x in expected_conditions], references_manifest=c.plan['identity']['reference_manifest_sha256'],
        files={name:a.sha(destination/name) for name in ('final_predictions.csv', 'selections.json', 'plan.json', 'unit_markers.json')})
    a.write_json(destination / 'manifest.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'preflight', 'units', 'run-unit', 'export'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--archive', type=Path, help='Original author archive, only for rebuilding reference evidence')
    p.add_argument('--original', type=Path, default=ROOT.parent/'Unified_ionisation_combined_20261114')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--destination', type=Path)
    p.add_argument('--unit', choices=UNITS)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    if args.command == 'units': result = list(UNITS)
    elif args.command == 'prepare':
        if args.archive is None: p.error('prepare requires an explicit --archive; shipped reference evidence needs no prepare step')
        result = prepare(args.root.resolve(), args.archive, args.original)
    else:
        c = load_campaign(args.root.resolve())
        if args.command == 'preflight': result = preflight(c)
        elif args.command == 'run-unit':
            if not args.unit or not args.output_dir: p.error('run-unit needs --unit and --output-dir')
            result = execute_unit(c, args.unit, args.output_dir, args.device)[0]
            result = {k:v for k,v in result.items() if k not in ('fit_markers', 'conditions')}
        else:
            if not args.output_dir or not args.destination: p.error('export needs --output-dir and --destination')
            result = export(c, args.output_dir, args.destination)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
