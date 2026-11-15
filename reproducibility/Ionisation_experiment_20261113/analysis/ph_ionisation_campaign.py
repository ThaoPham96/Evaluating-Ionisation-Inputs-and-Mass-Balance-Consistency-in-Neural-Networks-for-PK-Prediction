"""Freeze, execute and export the categorical-pH follow-up campaign."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import statistics
import time

from ionisation_campaign import BACKBONES, ENDPOINTS, SEARCH_SEED, SEEDS, choose, digest, fit_request, read, rows, sha, verify_release, write
from ph_ionisation_features import BLOCKS, FINGERPRINT, load, prepare

ROOT = Path(__file__).resolve().parents[1]
LEDGER = Path('runtime/ph_category_features')
PROTOCOL = Path('planning/gastric_ph_review_v1/IONISATION_PH_PROTOCOL_v2_FOR_REVIEW.txt')
OLD_CAMPAIGN = '3ffed2efa5ca4a2bdfa3584cb0d1d3d2869d6edc17e1697e6476fa5a356b5e28'
TUNED = ('ALL4', 'REP4')
REFERENCES = {'P0': ('primary', 'REF0'), 'P1': ('primary', 'REF74'),
              **{arm: ('historical', arm) for arm in ('P2', 'P3', 'P4', 'P5')}}
ARTIFACTS = ('summary.json', 'predictions.csv', 'checkpoint.pt', 'scaler.json', 'request.json', 'learning_curve.csv')


def unit_spec():
    units = []
    for backbone in ('Chemprop', 'GraphConv', 'ChemBERTa_head'):
        for fold in range(1, 6):
            common = dict(cohort='numerical', backbone=backbone, outer_fold=fold)
            stem = f'{backbone}-outer{fold}'
            units.append(dict(**common, unit_id=f'core-{stem}', stage='core', depends_on=None, new_fits_upper_bound=84))
            units.append(dict(**common, unit_id=f'tune-{stem}', stage='tune', depends_on=f'core-{stem}', new_fits_upper_bound=108))
    return units


def _hash_files(root, bindings):
    for name, expected in bindings.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError(f'Code or protocol checksum mismatch: {name}')


def _verify_fit(path, request, marker_hash=None, *, full=False):
    marker_path = path/'complete.json'
    if marker_hash is not None and sha(marker_path) != marker_hash:
        raise ValueError(f'Fit completion marker changed: {path}')
    marker = read(marker_path)
    if marker['fit_id'] != request['fit_id'] or set(marker['files']) != set(ARTIFACTS):
        raise ValueError(f'Invalid fit completion marker: {path}')
    checked = ARTIFACTS if full else ('summary.json', 'predictions.csv', 'scaler.json', 'request.json')
    if any(sha(path/name) != marker['files'][name] for name in checked):
        raise ValueError(f'Fit artifact checksum mismatch: {path}')
    recorded = read(path/'request.json')
    if recorded['request'] != request or recorded['request_hash'] != digest(request):
        raise ValueError(f'Fit request changed: {path}')
    if recorded['effective_ids'] != {key: request[key] for key in ('train_ids','val_ids','predict_ids')}:
        raise ValueError(f'Effective fit IDs changed: {path}')
    scaler, summary = read(path/'scaler.json'), read(path/'summary.json')
    if (summary['resolved_hash'] != marker['resolved_hash'] or recorded['resolved_hash'] != marker['resolved_hash'] or
            summary['scaler_hash'] != digest(scaler) or recorded['scaler_hash'] != digest(scaler) or
            summary['descriptor_spec'] != scaler['descriptor'] or not summary['checkpoint_reload_verified']):
        raise ValueError(f'Fit hash, scaler or checkpoint verification mismatch: {path}')
    for key in ('fit_id','arm','cohort','backbone','seed','config','campaign_id','lambda_value','task_weights'):
        if summary[key] != request[key]:
            raise ValueError(f'Fit summary disagrees with request ({key}): {path}')
    if request['fixed_epochs'] is not None and (summary['actual_epochs'] != request['fixed_epochs'] or
                                               summary['best_epoch'] != request['fixed_epochs']):
        raise ValueError(f'Fixed fit epoch mismatch: {path}')
    return summary, scaler, marker


def _prior_evidence(root, prior, prior_output, outer, inner):
    """Prove reused anchor search, calibrations and predictions against their recorded fits."""
    import numpy as np
    import pandas as pd
    from ionisation_train import _prepare_targets, fit_descriptors

    data = pd.read_csv(root/'data/release/numerical/model_dataset.csv', dtype={'structure_id':str}).set_index('structure_id')
    grid = prior['identity']['grid']
    recipes, reused = {}, []
    old_units = {(u['backbone'],u['outer_fold']):u for u in prior['units'] if u['cohort'] == 'numerical'}
    if set(old_units) != {(b,f) for b in BACKBONES for f in range(1,6)}:
        raise ValueError('Original numerical campaign has incomplete unit coverage')

    def check(path, request, done, *, full=False):
        summary, scaler, marker = _verify_fit(path, request, done['fit_markers'][request['fit_id']], full=full)
        effective = {key:request[key] for key in ('train_ids','val_ids','predict_ids')}
        _, _, target = _prepare_targets(data, effective, True)
        _, descriptor = fit_descriptors(request['arm'], data, request['train_ids'])
        for key in ('mean','sd'):
            np.testing.assert_allclose(scaler[key], target[key], rtol=1e-13, atol=1e-13)
        if (any(scaler[k] != target[k] for k in ('endpoints','counts','method')) or
                scaler['descriptor']['method'] != descriptor['method'] or
                scaler['descriptor']['input_dim'] != descriptor['input_dim'] or
                scaler['descriptor'].get('class_order') != descriptor.get('class_order')):
            raise ValueError(f'Historical feature/target semantics mismatch: {path}')
        return summary, marker

    for (backbone, fold), unit in sorted(old_units.items()):
        folder = prior_output/unit['unit_id']
        done, selection = read(folder/'complete_unit.json'), read(folder/'selection.json')
        if (done['campaign_id'] != prior['campaign_id'] or selection['campaign_id'] != prior['campaign_id'] or
                done['unit'] != unit or selection['unit'] != unit or
                done['selection_sha256'] != sha(folder/'selection.json') or done['final_records'] != selection['final_requests']):
            raise ValueError(f'Historical unit selection/completion mismatch: {folder}')
        search, anchor_markers = {}, {}
        for index, config in enumerate(grid[backbone]):
            search[index] = []
            ids = []
            for sub in range(1,6):
                req = fit_request(prior, unit, 'P0', config, SEARCH_SEED, outer, inner, inner=sub)
                fid = req['fit_id']
                summary, _ = check(folder/'fits'/fid, req, done)
                search[index].append(summary)
                anchor_markers[fid] = done['fit_markers'][fid]
                ids.append(fid)
            if ids != selection['search_fit_ids']['P0'][str(index)]:
                raise ValueError(f'Historical P0 search request mismatch: {folder}')
        selected, epochs, scores = choose(search)
        published = selection['selected']['P0']
        if (selected != selection['primary_anchor_index'] or selected != published['candidate_index'] or
                epochs != published['epochs'] or any(scores[i] != published['scores'][str(i)] for i in scores)):
            raise ValueError(f'Historical P0 recipe does not reproduce: {folder}')
        config = grid[backbone][selected]
        recipes[f'{backbone}-outer{fold}'] = dict(candidate_index=selected, config=config,
            prior_unit_id=unit['unit_id'], prior_selection_sha256=done['selection_sha256'],
            prior_complete_unit_sha256=sha(folder/'complete_unit.json'), prior_anchor_search_markers=anchor_markers)

        for source_arm, (procedure, arm) in REFERENCES.items():
            calibration = []
            for sub in range(1,6):
                req = fit_request(prior, unit, source_arm, config, SEARCH_SEED, outer, inner, inner=sub)
                fid = req['fit_id']
                summary, _ = check(folder/'fits'/fid, req, done)
                calibration.append((fid, summary['best_epoch'], done['fit_markers'][fid]))
            if [r[0] for r in calibration] != selection['calibration_fit_ids'][source_arm]:
                raise ValueError(f'Historical arm calibration request mismatch: {folder}/{source_arm}')
            final_epochs = int(statistics.median(r[1] for r in calibration))
            matching = [r for r in done['final_records'] if r['procedure']=='primary' and r['arm']==source_arm]
            if len(matching) != len(SEEDS) or {r['seed'] for r in matching} != set(SEEDS):
                raise ValueError(f'Missing historical final seeds: {folder}/{source_arm}')
            for record in matching:
                req = fit_request(prior, unit, source_arm, config, record['seed'], outer, inner, epochs=final_epochs)
                fid = req['fit_id']
                if (record['fit_id'], record['candidate_index'], record['epochs']) != (fid, selected, final_epochs):
                    raise ValueError(f'Historical final fit request mismatch: {folder}/{source_arm}')
                summary, marker = check(folder/'fits'/fid, req, done, full=source_arm in {'P0','P1'})
                reused.append(dict(procedure=procedure, arm=arm, source_arm=source_arm, backbone=backbone,
                    outer_fold=fold, seed=record['seed'], source_campaign_id=prior['campaign_id'],
                    source_fit_id=fid, source_fit_marker_sha256=done['fit_markers'][fid],
                    source_unit_id=unit['unit_id'], source_request_hash=digest(req), epochs=final_epochs,
                    configuration_index=selected, source_scaler_hash=summary['scaler_hash'],
                    source_predictions_sha256=marker['files']['predictions.csv'], calibration=calibration))
    return recipes, reused


def _verify_prior_predictions(path, numerical_fingerprint, prior_output, mappings):
    import numpy as np
    import pandas as pd

    manifest = read(path.with_suffix('.manifest.json'))
    if (manifest['campaign_id'] != OLD_CAMPAIGN or manifest['sha256'] != sha(path) or
            manifest['release_fingerprints']['numerical'] != numerical_fingerprint):
        raise ValueError('Original outer predictions do not match frozen campaign and numerical cohort')
    table = pd.read_csv(path, dtype={'structure_id':str})
    table = table[(table.cohort=='numerical') & (table.procedure=='primary') & table.arm.isin(REFERENCES)]
    groups = table.groupby(['backbone','outer_fold','arm','seed'], sort=False)
    if len(groups) != len(mappings):
        raise ValueError('Original outer prediction fit coverage differs from provenance mapping')
    for record in mappings:
        key = tuple(record[k] for k in ('backbone','outer_fold','source_arm','seed'))
        source = prior_output/record['source_unit_id']/'fits'/record['source_fit_id']/'predictions.csv'
        current = groups.get_group(key).sort_values('structure_id').reset_index(drop=True)
        old = pd.read_csv(source, dtype={'structure_id':str}).sort_values('structure_id').reset_index(drop=True)
        if current.structure_id.tolist() != old.structure_id.tolist():
            raise ValueError(f'Old exported IDs differ from source fit: {key}')
        np.testing.assert_allclose(current[old.columns[1:]], old.iloc[:,1:], rtol=1e-13, atol=1e-13)
    return {'sha256': manifest['sha256'], 'manifest_sha256': sha(path.with_suffix('.manifest.json'))}


def build_plan(root, prior_plan_path, prior_output, feature_ledger_dir=None, prior_predictions=None):
    root = Path(root).resolve()
    prior_path, prior_output = Path(prior_plan_path), Path(prior_output)
    prior = read(prior_path)
    if (prior['campaign_id'] != OLD_CAMPAIGN or digest(prior['identity']) != OLD_CAMPAIGN or
            prior['training_ready'] is not True or prior['units'] != prior['identity']['units']):
        raise ValueError('Expected the completed original ionisation campaign plan')
    _hash_files(root, prior['identity']['code'])
    dataset, outer, inner = verify_release(root, 'numerical')
    if dataset != prior['identity']['datasets']['numerical'] or dataset['fingerprint'] != FINGERPRINT:
        raise ValueError('Numerical dataset/features differ from the original campaign')
    ledger_path = Path(feature_ledger_dir or root/LEDGER).resolve()
    if ledger_path != root/LEDGER:
        raise ValueError(f'Categorical pH ledger must be at {root/LEDGER}')
    feature = prepare(root, ledger_path)
    recipes, reused = _prior_evidence(root, prior, prior_output, outer, inner)
    pred_path = Path(prior_predictions or root/'outputs/outer_predictions.csv')
    predictions = _verify_prior_predictions(pred_path, FINGERPRINT, prior_output, reused)
    units = unit_spec()
    code = dict(prior['identity']['code'])
    for name in ('ph_ionisation_campaign.py','ph_ionisation_train.py','ph_ionisation_features.py'):
        rel = f'analysis/{name}'
        code[rel] = sha(root/rel)
    identity = dict(protocol='categorical_ph_v2', protocol_sha256=sha(root/PROTOCOL), datasets={'numerical':dataset},
        code=code, categorical_features_sha256=sha(ledger_path/'manifest.json'), categorical_features_fingerprint=feature['fingerprint'],
        grid=prior['identity']['grid'], seeds=list(SEEDS), search_seed=SEARCH_SEED, max_epochs=200, patience=30,
        endpoint_weights=[1]*7, lambda_value=0, units=units, prior_campaign_id=OLD_CAMPAIGN,
        prior_plan_sha256=sha(prior_path), prior_predictions=predictions, anchor_recipes=recipes, reference_fits=reused)
    return dict(campaign_id=digest(identity), identity=identity, training_ready=True,
                new_fits_upper_bound=sum(unit['new_fits_upper_bound'] for unit in units), units=units)


def verify_plan(root, plan):
    root = Path(root).resolve()
    identity = plan['identity']
    if (plan['training_ready'] is not True or plan['campaign_id'] != digest(identity) or identity['protocol'] != 'categorical_ph_v2' or
            plan['units'] != identity['units'] or plan['units'] != unit_spec() or plan['new_fits_upper_bound'] != 2880 or
            identity['prior_campaign_id'] != OLD_CAMPAIGN or identity['seeds'] != list(SEEDS) or
            identity['search_seed'] != SEARCH_SEED or identity['max_epochs'] != 200 or identity['patience'] != 30 or
            identity['lambda_value'] != 0 or identity['endpoint_weights'] != [1]*7):
        raise ValueError('Invalid categorical pH campaign identity')
    _hash_files(root, identity['code'] | {str(PROTOCOL.as_posix()):identity['protocol_sha256']})
    dataset, outer, inner = verify_release(root, 'numerical')
    if identity['datasets'] != {'numerical':dataset} or dataset['fingerprint'] != FINGERPRINT:
        raise ValueError('Numerical dataset/features changed after pH planning')
    if sha(root/LEDGER/'manifest.json') != identity['categorical_features_sha256']:
        raise ValueError('Categorical feature manifest changed after planning')
    feature, _ = load(root/LEDGER, list(outer), FINGERPRINT)
    if feature['fingerprint'] != identity['categorical_features_fingerprint']:
        raise ValueError('Categorical feature fingerprint mismatch')
    return outer, inner


def _execute(args):
    request, release, features, ledger, destination, device = args
    from ph_ionisation_train import run_fit
    return run_fit(request, release, features, destination, device, ledger_dir=ledger)


def _unit_marker(plan, unit, folder):
    marker = read(folder/'complete_unit.json')
    selection = read(folder/'selection.json')
    if (marker['campaign_id'] != plan['campaign_id'] or marker['unit'] != unit or
            selection['campaign_id'] != plan['campaign_id'] or selection['unit'] != unit or
            marker['selection_sha256'] != sha(folder/'selection.json') or
            marker['final_records'] != selection['final_requests']):
        raise ValueError(f'Unit completion does not bind its selection: {folder}')
    return marker, selection


def run_unit(root, plan, unit_id, output, workers=1, device='cuda'):
    from _vendor import ensure_vendor
    ensure_vendor()
    from mass_balance_train import _fit_lock

    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('Workers must be an integer from 1 to 8')
    root, output = Path(root).resolve(), Path(output).resolve()
    outer, inner = verify_plan(root, plan)
    units = {unit['unit_id']:unit for unit in plan['units']}
    if unit_id not in units:
        raise ValueError('Unknown categorical pH unit')
    unit = units[unit_id]
    folder = output/unit_id
    folder.mkdir(parents=True, exist_ok=True)
    identity = plan['identity']
    backbone, fold = unit['backbone'], unit['outer_fold']
    anchor = identity['anchor_recipes'][f'{backbone}-outer{fold}']['candidate_index']
    grid = identity['grid'][backbone]
    procedure = 'tuned' if unit['stage']=='tune' else 'primary'
    arms = TUNED if unit['stage']=='tune' else tuple(BLOCKS)

    with _fit_lock(folder):
        started = time.monotonic()
        summaries, executed, reused = {}, {}, {}
        calibrations, search, selected, final_records = {}, {}, {}, []
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) if workers>1 else None

        def execute(requests):
            args = [(req, root/'data/release/numerical', root/'features/numerical', root/LEDGER,
                     folder/'fits'/req['fit_id'], device) for req in requests]
            results = map(_execute,args) if pool is None else pool.map(_execute,args,chunksize=1)
            for req, summary in zip(requests,results,strict=True):
                fid = req['fit_id']
                summaries[fid], executed[fid] = summary, req

        def reuse(req, dependency, done):
            fid = req['fit_id']
            path = output/dependency/'fits'/fid
            marker_hash = done['fit_markers'][fid]
            summary, _, _ = _verify_fit(path, req, marker_hash, full=True)
            summaries[fid] = summary
            reused[fid] = dict(unit_id=dependency, fit_marker_sha256=marker_hash)

        try:
            pending = []
            if unit['stage']=='tune':
                dependency = unit['depends_on']
                done, previous = _unit_marker(plan, units[dependency], output/dependency)
                for arm in arms:
                    search[arm] = {}
                    for index, config in enumerate(grid):
                        reqs = [fit_request(plan, unit, arm, config, SEARCH_SEED, outer, inner, inner=sub) for sub in range(1,6)]
                        search[arm][index] = [req['fit_id'] for req in reqs]
                        if index==anchor:
                            if search[arm][index] != previous['calibration_fit_ids'][arm]:
                                raise ValueError('Tuning did not reproduce the exact core calibration requests')
                            for req in reqs:
                                reuse(req, dependency, done)
                        else:
                            pending.extend(reqs)
            else:
                for arm in arms:
                    reqs = [fit_request(plan, unit, arm, grid[anchor], SEARCH_SEED, outer, inner, inner=sub) for sub in range(1,6)]
                    calibrations[arm] = [req['fit_id'] for req in reqs]
                    pending.extend(reqs)
            execute(pending)
            for arm in arms:
                if unit['stage']=='tune':
                    candidate, epochs, scores = choose({i:[summaries[f] for f in ids] for i,ids in search[arm].items()})
                    selected[arm] = dict(candidate_index=candidate, epochs=epochs, scores=scores)
                else:
                    candidate = anchor
                    epochs = int(statistics.median(summaries[f]['best_epoch'] for f in calibrations[arm]))
                    selected[arm] = dict(candidate_index=candidate, epochs=epochs)
            pending = []
            for arm in arms:
                candidate, epochs = (selected[arm][k] for k in ('candidate_index','epochs'))
                for seed in SEEDS:
                    req = fit_request(plan, unit, arm, grid[candidate], seed, outer, inner, epochs=epochs)
                    fid = req['fit_id']
                    source_unit_id = unit_id
                    if unit['stage']=='tune' and fid in done['fit_markers']:
                        reuse(req, dependency, done)
                        source_unit_id = dependency
                    else:
                        pending.append(req)
                    final_records.append(dict(procedure=procedure, arm=arm, seed=seed, fit_id=fid,
                                              candidate_index=candidate, epochs=epochs, source_unit_id=source_unit_id))
            write(folder/'selection.json', dict(campaign_id=plan['campaign_id'], unit=unit,
                  primary_anchor_index=anchor, selected=selected, calibration_fit_ids=calibrations,
                  search_fit_ids=search, final_requests=final_records, reused_fits=reused))
            execute(pending)
        finally:
            if pool is not None:
                pool.shutdown(wait=True,cancel_futures=True)

        markers = {}
        for fid, request in executed.items():
            path = folder/'fits'/fid
            _verify_fit(path, request, full=True)
            markers[fid] = sha(path/'complete.json')
        marker = dict(campaign_id=plan['campaign_id'], unit=unit, final_records=final_records, reused_fits=reused,
                      fit_markers=markers, unique_fits=len(markers), selection_sha256=sha(folder/'selection.json'),
                      elapsed_seconds=time.monotonic()-started, workers=workers)
        if len(markers) > unit['new_fits_upper_bound']:
            raise ValueError('Unit exceeded its frozen fit limit')
        write(folder/'complete_unit.json', marker)
        return marker


def export(root, plan, output, destination, prior_predictions=None):
    """Export all new and provenance-bound historical numerical predictions."""
    import pandas as pd

    root, output, destination = map(Path, (root,output,destination))
    outer, inner = verify_plan(root,plan)
    units = {unit['unit_id']:unit for unit in plan['units']}
    markers = {uid:_unit_marker(plan,unit,output/uid) for uid,unit in units.items()}
    frames = []
    provenance = []
    for unit_id, unit in units.items():
        done, _ = markers[unit_id]
        fold = unit['outer_fold']
        procedure = 'tuned' if unit['stage']=='tune' else 'primary'
        arms = TUNED if unit['stage']=='tune' else tuple(BLOCKS)
        records = done['final_records']
        if (len(records) != len(arms)*len(SEEDS) or
                {(r['procedure'],r['arm'],r['seed']) for r in records} != {(procedure,a,s) for a in arms for s in SEEDS}):
            raise ValueError(f'Incomplete categorical pH final coverage: {unit_id}')
        for record in records:
            source_unit_id = record['source_unit_id']
            if source_unit_id not in (unit_id, unit['depends_on']):
                raise ValueError('Final fit points outside its unit/dependency')
            source, _ = markers[source_unit_id]
            config = plan['identity']['grid'][unit['backbone']][record['candidate_index']]
            req = fit_request(plan,unit,record['arm'],config,record['seed'],outer,inner,epochs=record['epochs'])
            fid = req['fit_id']
            if fid != record['fit_id']:
                raise ValueError('Final fit request differs from selection')
            path = output/source_unit_id/'fits'/fid
            _verify_fit(path, req, source['fit_markers'][fid])
            frame = pd.read_csv(path/'predictions.csv',dtype={'structure_id':str})
            if frame.structure_id.tolist() != req['predict_ids']:
                raise ValueError(f'Prediction ID coverage differs for {fid}')
            fields = {k:unit[k] for k in ('cohort','backbone','outer_fold')} | {k:record[k] for k in ('procedure','arm','seed')}
            frame = frame.assign(**fields, source_campaign_id=plan['campaign_id'], source_fit_id=fid)
            frames.append(frame)
            provenance.append(dict(**fields, source_campaign_id=plan['campaign_id'], source_fit_id=fid,
                                   source_unit_id=source_unit_id, source_fit_marker_sha256=source['fit_markers'][fid]))

    previous = Path(prior_predictions or root/'outputs/outer_predictions.csv')
    hashes = plan['identity']['prior_predictions']
    if sha(previous)!=hashes['sha256'] or sha(previous.with_suffix('.manifest.json'))!=hashes['manifest_sha256']:
        raise ValueError('Original predictions changed after categorical pH planning')
    historical = pd.read_csv(previous,dtype={'structure_id':str})
    historical = historical[(historical.cohort=='numerical') & (historical.procedure=='primary') & historical.arm.isin(REFERENCES)]
    by_fit = historical.groupby(['backbone','outer_fold','arm','seed'],sort=False)
    for record in plan['identity']['reference_fits']:
        key = tuple(record[k] for k in ('backbone','outer_fold','source_arm','seed'))
        frame = by_fit.get_group(key)
        frame = frame.assign(**{k:record[k] for k in ('arm','procedure','source_campaign_id','source_fit_id')})
        frames.append(frame)
        provenance.append(record)
    combined = pd.concat(frames,ignore_index=True)
    if combined.duplicated(['cohort','backbone','procedure','arm','seed','structure_id']).any():
        raise ValueError('Duplicate categorical pH outer predictions')
    if destination.resolve() == previous.resolve():
        raise ValueError('Categorical pH export cannot overwrite the prior campaign')
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging = destination.with_name('.'+destination.name+'.tmp')
    combined.to_csv(staging,index=False)
    staging.replace(destination)
    sidecar = destination.with_suffix('.provenance.json')
    write(sidecar, dict(campaign_id=plan['campaign_id'], prior_campaign_id=OLD_CAMPAIGN, fits=provenance))
    write(destination.with_suffix('.manifest.json'), dict(campaign_id=plan['campaign_id'], sha256=sha(destination),
          rows=len(combined), units=len(units), release_fingerprints={'numerical':FINGERPRINT},
          provenance_sha256=sha(sidecar), prior_campaign_id=OLD_CAMPAIGN,
          categorical_feature_fingerprint=plan['identity']['categorical_features_fingerprint']))
    return len(combined)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    subparsers = parser.add_subparsers(dest='command',required=True)
    plan = subparsers.add_parser('plan')
    plan.add_argument('--prior-plan',type=Path,required=True)
    plan.add_argument('--prior-output',type=Path,required=True)
    plan.add_argument('--prior-predictions',type=Path)
    plan.add_argument('--out',type=Path,required=True)
    run = subparsers.add_parser('run-unit')
    run.add_argument('--plan',type=Path,required=True)
    run.add_argument('--unit',required=True)
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--workers',type=int,default=1)
    run.add_argument('--device',default='cuda')
    out = subparsers.add_parser('export')
    out.add_argument('--plan',type=Path,required=True)
    out.add_argument('--output',type=Path,required=True)
    out.add_argument('--out',type=Path,required=True)
    out.add_argument('--prior-predictions',type=Path)
    args = parser.parse_args(argv)
    if args.command=='plan':
        if args.out.resolve()==args.prior_plan.resolve():
            raise ValueError('Write the follow-up plan separately from the prior plan')
        result = build_plan(args.root,args.prior_plan,args.prior_output,prior_predictions=args.prior_predictions)
        write(args.out,result)
        print(json.dumps(dict(campaign_id=result['campaign_id'],units=len(result['units']),fits=result['new_fits_upper_bound'])))
    elif args.command=='run-unit':
        print(json.dumps(run_unit(args.root,read(args.plan),args.unit,args.output,args.workers,args.device)))
    else:
        print('Exported rows:',export(args.root,read(args.plan),args.output,args.out,args.prior_predictions))


if __name__=='__main__':
    main()
