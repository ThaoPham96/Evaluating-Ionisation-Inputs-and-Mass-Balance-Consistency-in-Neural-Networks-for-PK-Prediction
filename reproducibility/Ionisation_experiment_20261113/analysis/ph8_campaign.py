"""Plan, execute and export the frozen 210-fit pH 8.0 extension."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FunctionType

from ionisation_campaign import BACKBONES, SEEDS, digest, read, sha, write
from ph8_features import ARM, FINGERPRINT, LEDGER, PH, prepare
import single_ph_campaign as single


ROOT = single.ROOT
PROTOCOL = Path('planning/ph8_v1/PH8_PROTOCOL.txt')
SINGLE_PLAN = Path('runtime/single_ph_plan.json')
SINGLE_EXPORT = Path('outputs/single_ph/outer_predictions.csv')
SINGLE_CAMPAIGN = '5935d646a2f6dd49b0b03be53cabcd1dd5b0b09fc3e2a3ea5c93ea43bafe55fd'
COHORT = single.COHORT
ARMS = (*single.REFERENCES, ARM)
FIT_COUNT = len(single.INNER_FOLDS) + len(SEEDS)
N_PREDICTIONS = len(ARMS) * len(BACKBONES) * len(SEEDS) * 327
N_FINAL_FITS = len(ARMS) * len(BACKBONES) * len(SEEDS) * len(single.OUTER_FOLDS)


def _bind(function, namespace):
    bound = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


def unit_spec():
    return [dict(unit, unit_id=unit['unit_id'].replace('single-', 'ph8-', 1), new_fits_upper_bound=FIT_COUNT)
            for unit in single.unit_spec()]


def _source_inputs(root):
    prior, outer, inner, _ = single._previous_inputs(root)
    source_plan = read(root/SINGLE_PLAN)
    source_identity = source_plan['identity']
    if (source_plan['campaign_id'] != SINGLE_CAMPAIGN or digest(source_identity) != SINGLE_CAMPAIGN or
            source_identity['anchor_recipes'] != prior['identity']['anchor_recipes'] or
            source_identity['datasets'] != prior['identity']['datasets']):
        raise ValueError('Expected the original completed single-pH study and identical P0 recipes')
    single.previous._hash_files(root, source_identity['code'])
    references = single._reference_records(prior)
    if source_identity['reference_fits'] != references:
        raise ValueError('The single-pH reference fits differ from the original verified fits')
    feature = prepare(root/'data/release'/COHORT/'model_dataset.csv', root/LEDGER)
    return source_identity, outer, inner, references, feature


def _reference_export(root, references):
    """Reproduce REF0 and REF74 exactly in both frozen exports before reuse."""
    import pandas as pd
    import ionisation_score as common
    import ph_ionisation_score as categorical
    import single_ph_score as scorer

    hashes, expected = single._reference_export(root, references)
    source = root/SINGLE_EXPORT
    manifest_path, ledger_path = source.with_suffix('.manifest.json'), source.with_suffix('.provenance.json')
    bindings = {path.relative_to(root).as_posix(): sha(path) for path in (source, manifest_path, ledger_path)}
    manifest = categorical.checked_campaign(source, bindings[SINGLE_EXPORT.as_posix()], FINGERPRINT)
    if (manifest is None or manifest['campaign_id'] != SINGLE_CAMPAIGN or
            manifest['prior_campaign_id'] != single.previous.OLD_CAMPAIGN):
        raise ValueError('Unexpected original single-pH prediction export')
    data = common.load_cohort(root/'data/release', COHORT)
    frame = common.read_csv(source)
    if len(frame) != scorer.N_PREDICTIONS or len(frame) != manifest['rows']:
        raise ValueError('Unexpected original single-pH prediction coverage')
    _, sources = scorer.validate_predictions(frame, data, SINGLE_CAMPAIGN)
    if categorical.check_provenance(source, manifest, sources) is None:
        raise ValueError('The prior single-pH export requires a hashed source-fit ledger')
    old_ledger = [row for row in read(ledger_path)['fits'] if row['arm'] in single.REFERENCES]
    if {digest(row) for row in old_ledger} != {digest(row) for row in references}:
        raise ValueError('The original reference source-fit provenance changed')
    current = frame.loc[frame.arm.isin(single.REFERENCES)]
    keys = ['arm','backbone','outer_fold','seed','structure_id']
    pd.testing.assert_frame_equal(current.sort_values(keys).reset_index(drop=True),
                                  expected[current.columns].sort_values(keys).reset_index(drop=True), check_exact=True)
    single.previous._hash_files(root, bindings)
    return hashes | bindings, current.copy()


def _identity(root, source, references, feature, prediction_hashes):
    paths = ('ph8_campaign.py', 'ph8_train.py', 'ph8_features.py', 'ph8_score.py', 'single_ph_score.py')
    code = source['code'] | {f'analysis/{name}':sha(root/'analysis'/name) for name in paths}
    return dict(protocol='ph8_v1', protocol_sha256=sha(root/PROTOCOL), arm_ph={ARM:PH}, code=code,
                units=unit_spec(), datasets=source['datasets'], seeds=source['seeds'],
                search_seed=source['search_seed'], max_epochs=source['max_epochs'], patience=source['patience'],
                endpoint_weights=source['endpoint_weights'], lambda_value=source['lambda_value'],
                grid=source['grid'], anchor_recipes=source['anchor_recipes'], reference_fits=references,
                prior_campaign_id=single.previous.OLD_CAMPAIGN, source_single_ph_campaign=SINGLE_CAMPAIGN,
                source_single_ph_plan_sha256=sha(root/SINGLE_PLAN), prior_predictions=prediction_hashes,
                categorical_features_fingerprint=feature['fingerprint'],
                categorical_features_sha256=sha(root/LEDGER/'manifest.json'))


def build_plan(root):
    root = Path(root).resolve()
    source, _, _, references, feature = _source_inputs(root)
    hashes, _ = _reference_export(root, references)
    identity = _identity(root, source, references, feature, hashes)
    return dict(campaign_id=digest(identity), identity=identity, training_ready=True,
                units=identity['units'], new_fits_upper_bound=FIT_COUNT*len(identity['units']))


def verify_plan(root, plan):
    root = Path(root).resolve()
    identity = plan['identity']
    if (plan['training_ready'] is not True or plan['campaign_id'] != digest(identity) or
            plan['units'] != unit_spec() or plan['new_fits_upper_bound'] != FIT_COUNT*len(unit_spec())):
        raise ValueError('Invalid pH 8.0 campaign identity or fit count')
    paths = {path.as_posix() for export in (single.PRIOR_EXPORT, SINGLE_EXPORT)
             for path in (export, export.with_suffix('.manifest.json'), export.with_suffix('.provenance.json'))}
    if identity['prior_predictions'].keys() != paths:
        raise ValueError('pH 8.0 plan must bind both original reference exports')
    single.previous._hash_files(root, identity['prior_predictions'])
    source, outer, inner, references, feature = _source_inputs(root)
    if identity != _identity(root, source, references, feature, identity['prior_predictions']):
        raise ValueError('pH 8.0 code, protocol, features or P0 recipe changed after planning')
    return outer, inner


def _execute(args):
    request, release, features, ledger, output, device = args
    from ph8_train import run_fit
    return run_fit(request, release, features, output, device, ledger_dir=ledger)


# Reuse fit request generation, execution, completion verification and resume.
# Frozen module globals stay intact and the submitted worker is importable by spawn.
_namespace = dict(vars(single), ARM_PH={ARM:PH}, FIT_COUNT=FIT_COUNT, verify_plan=verify_plan,
                  _execute=_execute)


def _ph8_unit(root, plan, unit, outer, inner, output, workers, device, pool):
    from types import SimpleNamespace
    namespace = dict(_namespace, previous=SimpleNamespace(LEDGER=LEDGER))
    return _bind(single._run_unit, namespace)(root, plan, unit, outer, inner, output, workers, device, pool)


for _name in ('_calibration_requests', '_final_requests', '_selection', '_check_fit', '_verify_complete'):
    _namespace[_name] = _bind(getattr(single, _name), _namespace)
_namespace['_run_unit'] = _ph8_unit
_calibration_requests = _namespace['_calibration_requests']
_final_requests = _namespace['_final_requests']
_verify_complete = _namespace['_verify_complete']
run = _bind(single.run, _namespace)


def export(root, plan, output, destination):
    import numpy as np
    import pandas as pd
    import ionisation_score as common
    from ph8_score import validate_predictions

    root, output, destination = (Path(path).resolve() for path in (root, output, destination))
    protected = ('data','features','runtime','outputs/ph_followup','outputs/single_ph','outputs/scoring')
    sidecars = (destination.with_suffix('.manifest.json'), destination.with_suffix('.provenance.json'))
    if any(path.exists() for path in (destination, *sidecars)) or any(destination.is_relative_to(root/p) for p in protected):
        raise ValueError('Write pH 8.0 predictions to a new destination outside existing campaign data')
    outer, inner = verify_plan(root, plan)
    references = plan['identity']['reference_fits']
    bindings, historical = _reference_export(root, references)
    if bindings != plan['identity']['prior_predictions']:
        raise ValueError('Historical predictions changed during collection')
    frames, provenance = [historical], list(references)
    prediction_columns = ['structure_id', *common.PRED_COLUMNS, *common.PHYSICAL_COLUMNS]
    for unit in plan['units']:
        folder = output/unit['unit_id']
        marker, finals = _verify_complete(plan, unit, outer, inner, folder, full=True)
        for req in finals:
            fid = req['fit_id']
            frame = common.read_csv(folder/'fits'/fid/'predictions.csv')
            if frame.columns.tolist() != prediction_columns or frame.structure_id.tolist() != req['predict_ids']:
                raise ValueError(f'Unexpected prediction columns or frozen held-out IDs for {fid}')
            values = frame[prediction_columns[1:]].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite pH 8.0 predictions for {fid}')
            fields = {key:req[key] for key in ('cohort','backbone','outer_fold','arm','seed')}
            fields.update(procedure='primary', source_campaign_id=plan['campaign_id'], source_fit_id=fid)
            frames.append(frame.assign(**fields))
            provenance.append(dict(**fields, source_unit_id=unit['unit_id'], source_request_hash=digest(req),
                                   source_fit_marker_sha256=marker['fit_markers'][fid]))
    combined = pd.concat(frames, ignore_index=True)
    data = common.load_cohort(root/'data/release', COHORT)
    _, sources = validate_predictions(combined, data, plan['campaign_id'])
    if len(combined) != N_PREDICTIONS or len(provenance) != N_FINAL_FITS or len(sources) != N_FINAL_FITS:
        raise ValueError('Unexpected pH 8.0 trajectory or source-fit coverage')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.ph8-export-', dir=destination.parent) as directory:
        staging = Path(directory)/destination.name
        combined.to_csv(staging, index=False)
        ledger = staging.with_suffix('.provenance.json')
        write(ledger, dict(campaign_id=plan['campaign_id'], prior_campaign_id=single.previous.OLD_CAMPAIGN, fits=provenance))
        manifest = staging.with_suffix('.manifest.json')
        write(manifest, dict(campaign_id=plan['campaign_id'], sha256=sha(staging), rows=len(combined),
                             units=len(plan['units']), release_fingerprints={COHORT:FINGERPRINT},
                             provenance_sha256=sha(ledger), prior_campaign_id=single.previous.OLD_CAMPAIGN,
                             source_single_ph_campaign=SINGLE_CAMPAIGN,
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
            raise ValueError('Write the pH 8.0 plan to a new file')
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
