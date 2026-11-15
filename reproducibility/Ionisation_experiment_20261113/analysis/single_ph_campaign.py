"""Plan, execute and export the single-pH ionisation experiment locally."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
import multiprocessing
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
import time

from ionisation_campaign import BACKBONES, ENDPOINTS, SEARCH_SEED, SEEDS, digest, fit_request, read, rows, sha, write
import ph_ionisation_campaign as previous
from ph_ionisation_features import FINGERPRINT, load
from single_ph_features import ARM_PH, validate_ledger_codes


ROOT = previous.ROOT
PRIOR_PLAN = Path('runtime/ph_plan.json')
PRIOR_EXPORT = Path('outputs/ph_followup/outer_predictions.csv')
PROTOCOL = Path('planning/single_ph_v1/SINGLE_PH_PROTOCOL.txt')
FOLLOWUP_CAMPAIGN = '27b0bc93f682a3ac04f773b3240f495e2d106c278b42722381328bb5daf67d58'
COHORT = 'numerical'
REFERENCES = ('REF0', 'REF74')
OUTER_FOLDS = tuple(range(1, 6))
INNER_FOLDS = tuple(range(1, 6))
FIT_COUNT = len(ARM_PH) * (len(INNER_FOLDS) + len(SEEDS))


def unit_spec():
    return [dict(unit_id=f'single-{backbone}-outer{fold}', cohort=COHORT, backbone=backbone,
                 outer_fold=fold, new_fits_upper_bound=FIT_COUNT)
            for backbone in ('Chemprop', 'GraphConv', 'ChemBERTa_head') for fold in OUTER_FOLDS]


def _previous_inputs(root):
    path = root / PRIOR_PLAN
    prior = read(path)
    if prior['campaign_id'] != FOLLOWUP_CAMPAIGN:
        raise ValueError('Expected the completed categorical pH follow-up plan')
    outer, inner = previous.verify_plan(root, prior)
    feature, codes = load(root / previous.LEDGER, list(outer), FINGERPRINT)
    source_rows = {row['structure_id']: row for row in rows(root/'data/release'/COHORT/'model_dataset.csv')}
    if len(source_rows) != 327 or len({record['group_id'] for record in source_rows.values()}) != 238:
        raise ValueError('Expected the original 327 structures and 238 frozen groups')
    validate_ledger_codes((source_rows[sid] for sid in outer), codes)
    return prior, outer, inner, feature


def _reference_records(prior):
    old = prior['identity']
    references = [r for r in old['reference_fits'] if r['arm'] in REFERENCES]
    keys = {(r['arm'], r['backbone'], r['outer_fold'], r['seed']) for r in references}
    expected = {(arm, bb, fold, seed) for arm in REFERENCES for bb in BACKBONES for fold in OUTER_FOLDS for seed in SEEDS}
    if len(references) != len(expected) or keys != expected:
        raise ValueError('Expected 270 original REF0 and REF74 final source fits')
    for record in references:
        anchor = old['anchor_recipes'][f"{record['backbone']}-outer{record['outer_fold']}"]
        if (record['source_campaign_id'] != previous.OLD_CAMPAIGN or record['procedure'] != 'primary' or
                record['source_arm'] != {'REF0': 'P0', 'REF74': 'P1'}[record['arm']] or
                record['configuration_index'] != anchor['candidate_index'] or
                len(record['calibration']) != len(INNER_FOLDS) or
                record['epochs'] != int(statistics.median(c[1] for c in record['calibration']))):
            raise ValueError('Historical reference does not use the frozen P0 recipe and arm calibration')
    return references


def _reference_export(root, references):
    """Validate published reference trajectories and compare their exact source fits to the frozen plan."""
    import pandas as pd
    import ionisation_score as common
    import ph_ionisation_score as scorer

    path = root / PRIOR_EXPORT
    manifest_path = path.with_suffix('.manifest.json')
    provenance_path = path.with_suffix('.provenance.json')
    prediction_hash = sha(path)
    manifest = scorer.checked_campaign(path, prediction_hash, FINGERPRINT)
    if (manifest is None or manifest['campaign_id'] != FOLLOWUP_CAMPAIGN or
            manifest['prior_campaign_id'] != previous.OLD_CAMPAIGN):
        raise ValueError('Reference prediction export does not match the verified categorical pH follow-up')
    frame = common.read_csv(path)
    if len(frame) != manifest['rows']:
        raise ValueError('Reference export row count changed')
    data = common.load_cohort(root/'data/release', COHORT)
    scorer.validate_predictions(frame, data, FOLLOWUP_CAMPAIGN)
    sources = scorer.prediction_sources(frame, FOLLOWUP_CAMPAIGN)
    if scorer.check_provenance(path, manifest, sources) is None:
        raise ValueError('Reference prediction export requires verified source-fit provenance')
    ledger = read(provenance_path)
    recorded = {(r['arm'], r['backbone'], r['outer_fold'], r['seed']):r
                for r in ledger['fits'] if r['arm'] in REFERENCES}
    expected = {(r['arm'], r['backbone'], r['outer_fold'], r['seed']):r for r in references}
    # Source metadata are carried forward from the verified original fits.
    if recorded.keys() != expected.keys() or any(
            any(recorded[key].get(field) != value for field, value in entry.items())
            for key, entry in expected.items()):
        raise ValueError('REF0 or REF74 source-fit provenance differs from the frozen plan')
    chosen = frame.loc[frame.arm.isin(REFERENCES)].copy()
    if len(chosen) != len(REFERENCES) * len(BACKBONES) * len(SEEDS) * len(data):
        raise ValueError('REF0 or REF74 trajectory coverage is incomplete')
    if sha(path) != prediction_hash:
        raise ValueError('Reference predictions changed during verification')
    bindings = {str(p.relative_to(root).as_posix()): sha(p) for p in (path, manifest_path, provenance_path)}
    return bindings, pd.DataFrame(chosen)


def _code_bindings(root, prior):
    paths = ('single_ph_campaign.py', 'single_ph_train.py', 'single_ph_features.py',
             'ionisation_score.py', 'ph_ionisation_score.py')
    return prior['identity']['code'] | {f'analysis/{name}': sha(root/'analysis'/name) for name in paths}


def _identity(root, prior, feature, references, prediction_hashes):
    old = prior['identity']
    return dict(protocol='single_ph_v1', protocol_sha256=sha(root/PROTOCOL), arm_ph=ARM_PH,
                code=_code_bindings(root, prior), datasets=old['datasets'], units=unit_spec(),
                seeds=list(SEEDS), search_seed=SEARCH_SEED, max_epochs=200, patience=30,
                endpoint_weights=[1]*len(ENDPOINTS), lambda_value=0,
                source_followup_campaign=FOLLOWUP_CAMPAIGN, source_followup_plan_sha256=sha(root/PRIOR_PLAN),
                prior_campaign_id=previous.OLD_CAMPAIGN, prior_predictions=prediction_hashes,
                grid=old['grid'], anchor_recipes=old['anchor_recipes'], reference_fits=references,
                categorical_features_fingerprint=feature['fingerprint'],
                categorical_features_sha256=sha(root/previous.LEDGER/'manifest.json'))


def build_plan(root):
    root = Path(root).resolve()
    prior, _, _, feature = _previous_inputs(root)
    references = _reference_records(prior)
    prediction_hashes, _ = _reference_export(root, references)
    identity = _identity(root, prior, feature, references, prediction_hashes)
    return dict(campaign_id=digest(identity), identity=identity, training_ready=True,
                units=identity['units'], new_fits_upper_bound=FIT_COUNT * len(identity['units']))


def verify_plan(root, plan):
    root = Path(root).resolve()
    identity = plan['identity']
    if (plan['training_ready'] is not True or plan['campaign_id'] != digest(identity) or
            plan['units'] != unit_spec() or plan['new_fits_upper_bound'] != FIT_COUNT * len(unit_spec())):
        raise ValueError('Invalid single-pH campaign identity or fit count')
    prior, outer, inner, feature = _previous_inputs(root)
    references = _reference_records(prior)
    # Check the immutable export bytes before training. Revalidate their rows at export.
    paths = (PRIOR_EXPORT, PRIOR_EXPORT.with_suffix('.manifest.json'), PRIOR_EXPORT.with_suffix('.provenance.json'))
    expected_paths = {p.as_posix() for p in paths}
    if set(identity['prior_predictions']) != expected_paths:
        raise ValueError('Reference export paths differ from the frozen follow-up')
    previous._hash_files(root, identity['prior_predictions'])
    expected = _identity(root, prior, feature, references, identity['prior_predictions'])
    if identity != expected:
        raise ValueError('Single-pH protocol, code, features or anchor recipes changed after planning')
    return outer, inner


def _execute(args):
    request, release, features, ledger, output, device = args
    from single_ph_train import run_fit
    return run_fit(request, release, features, output, device, ledger_dir=ledger)


def _calibration_requests(plan, unit, outer, inner):
    key = f"{unit['backbone']}-outer{unit['outer_fold']}"
    config = plan['identity']['anchor_recipes'][key]['config']
    return [fit_request(plan, unit, arm, config, SEARCH_SEED, outer, inner, inner=subfold)
            for arm in ARM_PH for subfold in INNER_FOLDS]


def _final_requests(plan, unit, outer, inner, calibration, summaries):
    key = f"{unit['backbone']}-outer{unit['outer_fold']}"
    recipe = plan['identity']['anchor_recipes'][key]
    selected, requests, records = {}, [], []
    for arm in ARM_PH:
        best = [summaries[req['fit_id']]['best_epoch'] for req in calibration if req['arm'] == arm]
        if len(best) != len(INNER_FOLDS) or any(type(e) is not int or not 1 <= e <= 200 for e in best):
            raise ValueError('Expected five valid calibration epochs per single-pH arm')
        epochs = int(statistics.median(best))
        selected[arm] = dict(candidate_index=recipe['candidate_index'], epochs=epochs)
        for seed in SEEDS:
            req = fit_request(plan, unit, arm, recipe['config'], seed, outer, inner, epochs=epochs)
            requests.append(req)
            records.append(dict(procedure='primary', arm=arm, seed=seed, fit_id=req['fit_id'],
                                candidate_index=recipe['candidate_index'], epochs=epochs, source_unit_id=unit['unit_id']))
    return requests, selected, records


def _selection(plan, unit, calibration, selected, records):
    return dict(campaign_id=plan['campaign_id'], unit=unit, selected=selected,
                calibration_fit_ids={arm:[req['fit_id'] for req in calibration if req['arm']==arm] for arm in ARM_PH},
                final_requests=records)


def _check_fit(plan, folder, request, marker_hash=None, *, full):
    path = folder/'fits'/request['fit_id']
    summary, scaler, _ = previous._verify_fit(path, request, marker_hash, full=full)
    descriptor = scaler['descriptor']
    if (descriptor['ph_blocks'] != [ARM_PH[request['arm']]] or descriptor['input_dim'] != 4 or
            descriptor['width_divisor'] != 1 or
            descriptor['feature_ledger_fingerprint'] != plan['identity']['categorical_features_fingerprint']):
        raise ValueError(f'Unexpected single-pH feature specification: {path}')
    return summary


def _verify_complete(plan, unit, outer, inner, folder, *, full):
    marker, selection = previous._unit_marker(plan, unit, folder)
    calibration = _calibration_requests(plan, unit, outer, inner)
    hashes = marker['fit_markers']
    summaries = {req['fit_id']:_check_fit(plan, folder, req, hashes[req['fit_id']], full=full) for req in calibration}
    finals, selected, records = _final_requests(plan, unit, outer, inner, calibration, summaries)
    if (selection != _selection(plan, unit, calibration, selected, records) or
            marker['unique_fits'] != FIT_COUNT or set(hashes) != {r['fit_id'] for r in calibration+finals}):
        raise ValueError('Completed single-pH unit differs from the frozen requests')
    for req in finals:
        _check_fit(plan, folder, req, hashes[req['fit_id']], full=full)
    return marker, finals


def _run_unit(root, plan, unit, outer, inner, output, workers, device, pool):
    from _vendor import ensure_vendor
    ensure_vendor()
    from mass_balance_train import _fit_lock

    folder = output / unit['unit_id']
    folder.mkdir(parents=True, exist_ok=True)
    with _fit_lock(folder):
        if (folder/'complete_unit.json').is_file():
            return _verify_complete(plan, unit, outer, inner, folder, full=True)[0]
        start = time.monotonic()

        def execute(requests):
            args = [(req, root/'data/release'/COHORT, root/'features'/COHORT, root/previous.LEDGER,
                     folder/'fits'/req['fit_id'], device) for req in requests]
            results = map(_execute, args) if pool is None else pool.map(_execute, args, chunksize=1)
            return {req['fit_id']:summary for req, summary in zip(requests, results, strict=True)}

        calibration = _calibration_requests(plan, unit, outer, inner)
        summaries = execute(calibration)
        finals, selected, records = _final_requests(plan, unit, outer, inner, calibration, summaries)
        write(folder/'selection.json', _selection(plan, unit, calibration, selected, records))
        execute(finals)
        markers = {}
        for req in calibration+finals:
            _check_fit(plan, folder, req, full=True)
            fid = req['fit_id']
            markers[fid] = sha(folder/'fits'/fid/'complete.json')
        if len(markers) != FIT_COUNT:
            raise ValueError('Single-pH unit does not contain exactly 42 fits')
        marker = dict(campaign_id=plan['campaign_id'], unit=unit, final_records=records,
                      fit_markers=markers, unique_fits=len(markers), selection_sha256=sha(folder/'selection.json'),
                      elapsed_seconds=time.monotonic()-start, workers=workers)
        write(folder/'complete_unit.json', marker)
        return marker


def run(root, plan, output, workers=1, device='cuda', unit_id=None):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('Workers must be an integer from 1 to 8')
    root, output = Path(root).resolve(), Path(output).resolve()
    if any(output.is_relative_to(root/p) for p in ('data', 'features', 'runtime', 'outputs/ph_followup')):
        raise ValueError('Single-pH fit artifacts must be outside frozen input directories')
    outer, inner = verify_plan(root, plan)
    units = [unit for unit in plan['units'] if unit_id is None or unit['unit_id'] == unit_id]
    if not units:
        raise ValueError('Unknown single-pH work unit')
    executor = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
                if workers > 1 else nullcontext())
    results = []
    with executor as pool:
        for unit in units:
            marker = _run_unit(root, plan, unit, outer, inner, output, workers, device, pool)
            results.append(marker)
            print(json.dumps(dict(unit_id=unit['unit_id'], fits=marker['unique_fits'],
                                  elapsed_seconds=marker['elapsed_seconds'])), flush=True)
    return results


def export(root, plan, output, destination):
    import numpy as np
    import pandas as pd
    import ionisation_score as common

    root, output, destination = (Path(path).resolve() for path in (root, output, destination))
    if destination.exists() or any(destination.is_relative_to(root/p) for p in
            ('data', 'features', 'runtime', 'outputs/ph_followup', 'outputs/scoring')):
        raise ValueError('Write single-pH predictions to a new destination outside existing campaign data')
    outer, inner = verify_plan(root, plan)
    references = plan['identity']['reference_fits']
    bindings, historical = _reference_export(root, references)
    if bindings != plan['identity']['prior_predictions']:
        raise ValueError('Historical reference export changed during collection')
    frames, provenance = [historical], list(references)
    value_columns = [*common.PRED_COLUMNS, *common.PHYSICAL_COLUMNS]
    for unit in plan['units']:
        folder = output/unit['unit_id']
        marker, finals = _verify_complete(plan, unit, outer, inner, folder, full=True)
        for req in finals:
            fid = req['fit_id']
            frame = common.read_csv(folder/'fits'/fid/'predictions.csv')
            if frame.columns.tolist() != ['structure_id', *value_columns] or frame.structure_id.tolist() != req['predict_ids']:
                raise ValueError(f'Unexpected prediction columns or frozen held-out IDs for {fid}')
            transformed = frame[list(common.PRED_COLUMNS)].to_numpy(dtype=float)
            physical = frame[list(common.PHYSICAL_COLUMNS)].to_numpy(dtype=float)
            if (not np.isfinite(transformed).all() or not np.isfinite(physical).all() or
                    not np.allclose(common.inverse_physical(transformed), physical, rtol=1e-6, atol=0)):
                raise ValueError(f'Invalid single-pH predictions for {fid}')
            fields = {k:req[k] for k in ('cohort','backbone','outer_fold','arm','seed')}
            fields.update(procedure='primary', source_campaign_id=plan['campaign_id'], source_fit_id=fid)
            frames.append(frame.assign(**fields))
            provenance.append(dict(**fields, source_unit_id=unit['unit_id'], source_request_hash=digest(req),
                                   source_fit_marker_sha256=marker['fit_markers'][fid]))

    combined = pd.concat(frames, ignore_index=True)
    if (len(combined) != 44145 or len(provenance) != 675 or
            combined.duplicated(['cohort','procedure','backbone','arm','seed','structure_id']).any()):
        raise ValueError('Single-pH export does not have complete unique trajectory coverage')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.single-ph-export-', dir=destination.parent) as directory:
        staging = Path(directory)/destination.name
        combined.to_csv(staging, index=False)
        ledger = staging.with_suffix('.provenance.json')
        write(ledger, dict(campaign_id=plan['campaign_id'], prior_campaign_id=previous.OLD_CAMPAIGN, fits=provenance))
        manifest = staging.with_suffix('.manifest.json')
        write(manifest, dict(campaign_id=plan['campaign_id'], sha256=sha(staging), rows=len(combined),
                             units=len(plan['units']), release_fingerprints={COHORT:FINGERPRINT},
                             provenance_sha256=sha(ledger), prior_campaign_id=previous.OLD_CAMPAIGN,
                             source_followup_campaign=FOLLOWUP_CAMPAIGN,
                             categorical_feature_fingerprint=plan['identity']['categorical_features_fingerprint']))
        for path in (staging, ledger, manifest):
            path.replace(destination.parent/path.name)
    return len(combined)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan')
    plan.add_argument('--out', type=Path, required=True)
    for name in ('run-unit', 'run-all'):
        runner = commands.add_parser(name)
        runner.add_argument('--plan', type=Path, required=True)
        runner.add_argument('--output', type=Path, required=True)
        runner.add_argument('--workers', type=int, default=1)
        runner.add_argument('--device', default='cuda')
        if name == 'run-unit':
            runner.add_argument('--unit', required=True)
    collector = commands.add_parser('export')
    collector.add_argument('--plan', type=Path, required=True)
    collector.add_argument('--output', type=Path, required=True)
    collector.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'plan':
        if args.out.exists():
            raise ValueError('Write the single-pH plan to a new file')
        result = build_plan(args.root)
        write(args.out, result)
        print(json.dumps(dict(campaign_id=result['campaign_id'], units=len(result['units']),
                              fits=result['new_fits_upper_bound'])))
    elif args.command == 'export':
        print(f'Exported rows: {export(args.root, read(args.plan), args.output, args.out)}')
    else:
        run(args.root, read(args.plan), args.output, args.workers, args.device, getattr(args, 'unit', None))


if __name__ == '__main__':
    main()
