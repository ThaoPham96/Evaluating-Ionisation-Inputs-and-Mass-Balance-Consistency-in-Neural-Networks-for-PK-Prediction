"""Verify the checkpoint-free robustness archive and replay every selection."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / 'reproducibility/Review_robustness_experiment_20261114'
sys.path.insert(0, str(ROOT / 'analysis'))
import robustness_campaign as campaign

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path):
    return json.loads(path.read_text())

def verify(output):
    export = ROOT / 'outputs/export'
    manifest = read(export / 'manifest.json')
    for name, expected in manifest['files'].items():
        assert digest((export / name).read_bytes()) == expected, name
    plan = read(export / 'plan.json')
    identity = read(ROOT / 'PUBLIC_IDENTITY.json')
    assert plan['campaign_id'] == manifest['campaign_id'] == identity['archived_campaign_id']
    assert campaign.a.digest(plan['identity']) == manifest['public_packaging']['public_plan_identity_sha256']
    # This context is explicitly the archived scientific plan. It is used only
    # for saved-request/selection replay, never for fresh training.
    dataset, _ = campaign.a.read_population(ROOT / 'data/release/numerical')
    triads = campaign.a.by_id(campaign.a.rows(ROOT / 'runtime/same_row_complete_triads.csv'))
    c = campaign.Campaign(ROOT, plan, dataset, triads, read(ROOT / 'runtime/references/reference_manifest.json'))
    selections = read(export / 'selections.json')
    markers = read(export / 'unit_markers.json')
    rows = campaign.a.rows(export / 'final_predictions.csv')
    assert len(rows) == manifest['final_rows'] == 141264
    expected = {(protocol, arm, penalty, model, seed, sid)
                for protocol, arm, penalty in manifest['conditions']
                for model in campaign.a.BACKBONES for seed in campaign.a.SEEDS for sid in dataset}
    observed = [(r['protocol'], r['feature_arm'], r['penalty'], r['backbone'], int(r['seed']), r['structure_id']) for r in rows]
    assert len(observed) == len(set(observed)) and set(observed) == expected
    by_fit = defaultdict(list)
    reference = {}
    for row in rows:
        by_fit[row['fit_id']].append(row)
        if row['penalty'] == 'off':
            key = (row['protocol'], row['feature_arm'], row['backbone'], row['outer_fold'], row['seed'])
            if key in reference:
                assert reference[key] == row['fit_id']
            reference[key] = row['fit_id']
    for row in rows:
        key = (row['protocol'], row['feature_arm'], row['backbone'], row['outer_fold'], row['seed'])
        assert row['reference_fit_id'] == reference[key]
        assert int(row['outer_fold']) == int(dataset[row['structure_id']]['outer_fold'])

    all_fits = set()
    verified_units = []
    artifacts = 0
    compared_rows = 0
    def compare_predictions(fid, predictions):
        nonlocal compared_rows
        pred = {r['structure_id']: r for r in predictions}
        for row in by_fit.get(fid, []):
            saved = pred[row['structure_id']]
            for endpoint in campaign.a.ENDPOINTS:
                key = ('physical_' if endpoint in campaign.a.BOUNDED else 'pred_') + endpoint
                assert float(row['q_' + endpoint]) == float(saved[key])
            compared_rows += 1

    def unit_check(unit, fits, requests):
        reconstructed = campaign.selection_record(c, unit, fits)
        assert reconstructed == selections[unit], ('Selection replay mismatch', unit)
        expected_requests = {}
        arch = campaign.architecture_selection(c, unit, fits)
        groups = campaign.architecture_requests(c, unit) + campaign.lambda_requests(c, unit, arch)
        for group in groups:
            for req in group:
                expected_requests[req['fit_id']] = req
        for condition in campaign.final_conditions(c, unit, reconstructed):
            req = condition['request']
            expected_requests[req['fit_id']] = req
        assert requests == expected_requests, ('Request coverage or assignment mismatch', unit)
        assert set(requests) == set(markers[unit]['fit_markers'])
        assert len(requests) == markers[unit]['unique_fits']
        verified_units.append(unit)

    inventory = read(ROOT / 'fit_provenance/manifest.json')
    for item in inventory['files']:
        path = ROOT / item['path']
        assert digest(path.read_bytes()) == item['sha256']
        current = None
        fits, requests = {}, {}
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                unit, fid, raw = record['unit'], record['fit_id'], record['files']
                if current is not None and unit != current:
                    unit_check(current, fits, requests)
                    fits, requests = {}, {}
                current = unit
                assert fid not in all_fits
                all_fits.add(fid)
                assert digest(raw['complete.json'].encode()) == markers[unit]['fit_markers'][fid]
                complete = json.loads(raw['complete.json'])
                assert set(raw) == {'complete.json', 'request.json', 'summary.json', 'scaler.json', 'predictions.csv'}
                for name, text in raw.items():
                    if name != 'complete.json':
                        assert digest(text.encode()) == complete['files'][name], (fid, name)
                        artifacts += 1
                saved = json.loads(raw['request.json'])
                req = saved['request']
                assert req['fit_id'] == fid == 'ROB-' + campaign.a.digest({k:v for k,v in req.items() if k != 'fit_id'})[:24]
                ids = {k:req[k] for k in ('train_ids', 'val_ids', 'predict_ids')}
                assert saved['effective_ids'] == ids
                assert saved['resolved_hash'] == complete['resolved_hash']
                summary = json.loads(raw['summary.json'])
                for key in ('fit_id', 'campaign_id', 'backbone', 'feature_arm', 'cell', 'seed', 'outer_fold', 'inner_fold', 'lambda_value', 'config', 'task_weights'):
                    assert summary[key] == req[key], (fid, key)
                assert summary['resolved_hash'] == complete['resolved_hash']
                assert summary['checkpoint_reload_verified'] is True
                assert 1 <= summary['best_epoch'] <= summary['actual_epochs'] <= req['max_epochs']
                if req['fixed_epochs'] is not None:
                    assert req['fixed_epochs'] == summary['best_epoch'] == summary['actual_epochs']
                else:
                    assert math.isfinite(summary['val_equal_mse'])
                assert summary['environment'] == c.references['environments'][req['backbone']]
                predictions = list(csv.DictReader(io.StringIO(raw['predictions.csv'])))
                assert [r['structure_id'] for r in predictions] == req['predict_ids']
                for row in predictions:
                    for ep in campaign.a.ENDPOINTS:
                        transformed, physical = float(row['pred_' + ep]), float(row['physical_' + ep])
                        assert math.isfinite(transformed) and math.isfinite(physical)
                        if ep in campaign.a.BOUNDED:
                            assert 0 <= physical <= 100
                        else:
                            assert physical > 0 and math.isclose(math.log10(physical), transformed, abs_tol=1e-12)
                fits[fid] = (summary, predictions)
                requests[fid] = req
                if req['inner_fold'] is None:
                    compare_predictions(fid, predictions)
        if current is not None:
            unit_check(current, fits, requests)
    assert len(all_fits) == manifest['unique_new_fits'] == 6105
    assert set(verified_units) == set(campaign.UNITS) and len(verified_units) == 60
    historical = defaultdict(dict)
    for row in campaign.a.rows(ROOT / 'runtime/references/final_predictions.csv'):
        historical[row['fit_id']][row['structure_id']] = row
    for fid in set(by_fit) - all_fits:
        assert fid in historical
        for row in by_fit[fid]:
            old = historical[fid][row['structure_id']]
            for key in ('group_id', 'outer_fold', 'seed', 'backbone', 'fit_id', 'feature_arm', 'lambda_value'):
                assert row[key] == old[key]
            for ep in campaign.a.ENDPOINTS:
                assert float(row['q_' + ep]) == float(old['q_' + ep])
            compared_rows += 1
    assert compared_rows == len(rows)
    result = dict(status='PASS', archived_campaign_id=plan['campaign_id'], fits=len(all_fits), units=len(verified_units),
                  exact_raw_artifacts_verified=artifacts, final_rows_compared=compared_rows, logical_conditions=len(manifest['conditions']),
                  architecture_lambda_and_epoch_selection_replay='exact for all60units', reference_fit_id='same-input loss-off identity verified',
                  scope='Saved requests, completion markers, summaries, scalers and predictions verified. Checkpoint and learning-curve hashes retained but those omitted artifacts are not revalidated by this public command.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    verify(parser.parse_args().output)
