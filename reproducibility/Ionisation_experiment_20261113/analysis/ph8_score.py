"""Score the pH 8.0 extension on the existing numerical-cohort endpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import ionisation_score as common
from ionisation_campaign import read, sha, write
from ph8_campaign import ARMS, COHORT, N_FINAL_FITS, N_PREDICTIONS, SINGLE_CAMPAIGN, SINGLE_EXPORT, _bind
from ph8_features import ARM
import single_ph_score as single


# The existing scorer supplies the same masks, paired bootstrap draws and
# endpoint statistics. Only the isolated arm registry and coverage differ.
_namespace = dict(vars(single), NEW_ARMS=(ARM,), ARMS=ARMS, N_PREDICTIONS=N_PREDICTIONS,
                  N_FITS=N_FINAL_FITS)
for _name in ('expected_runs', 'validate_predictions', 'run_contrasts', 'score', 'score_export'):
    _namespace[_name] = _bind(getattr(single, _name), _namespace)
expected_runs = _namespace['expected_runs']
validate_predictions = _namespace['validate_predictions']
score = _namespace['score']


def _check_reference_scores(root, staging):
    original = root/SINGLE_EXPORT.parent/'scoring'
    manifest_path = original/'scoring_manifest.json'
    manifest_hash = sha(manifest_path)
    metadata = read(manifest_path)
    if (metadata['status'] != 'report_ready' or metadata['campaign_id'] != SINGLE_CAMPAIGN or
            metadata['predictions_sha256'] != sha(root/SINGLE_EXPORT) or
            metadata['bootstrap_seed'] != common.BOOTSTRAP_SEED or metadata['bootstrap_draws'] != common.N_BOOTSTRAP):
        raise ValueError('Original single-pH score provenance differs from the frozen comparison')
    row_counts = {}
    for name, expected in (('contrasts.csv',21), ('contrast_seeds.csv',189)):
        source = original/name
        if sha(source) != metadata['files'][name]:
            raise ValueError(f'Original single-pH scores changed: {name}')

        def baseline(path):
            frame = common.read_csv(path)
            keys = ['backbone','endpoint'] + (['seed'] if 'seed' in frame else [])
            return frame.query("arm == 'REF74' and comparator_arm == 'REF0'").sort_values(keys).reset_index(drop=True)

        old, new = baseline(source), baseline(staging/name)
        if len(old) != expected or len(new) != expected:
            raise ValueError(f'Unexpected REF74 versus REF0 score coverage: {name}')
        pd.testing.assert_frame_equal(old, new, check_exact=True)
        row_counts[name] = expected
    if sha(manifest_path) != manifest_hash:
        raise ValueError('Original scoring manifest changed during validation')
    return manifest_hash, row_counts


def score_export(release, predictions, out):
    release, predictions, out = (Path(path).resolve() for path in (release, predictions, out))
    root = release.parents[1]
    if any(out.is_relative_to(root/path) for path in
           ('data','features','runtime','outputs/single_ph','outputs/ph_followup','outputs/scoring')):
        raise ValueError('pH 8.0 scores require a separate output directory')
    collector = read(predictions.with_suffix('.manifest.json'))
    if collector.get('source_single_ph_campaign') != SINGLE_CAMPAIGN:
        raise ValueError('pH 8.0 export does not identify the original single-pH comparison')
    target_manifest = out/'scoring_manifest.json'
    if target_manifest.is_file() and read(target_manifest)['campaign_id'] != collector['campaign_id']:
        raise ValueError('Scoring output directory belongs to a different campaign')
    if out in (predictions, predictions.parent):
        raise ValueError('Scoring requires a directory separate from the prediction export')
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.ph8-score-', dir=out) as temporary:
        staging = Path(temporary)
        metadata = _namespace['score_export'](release, predictions, staging)
        manifest_hash, verified = _check_reference_scores(root, staging)
        counts = {'contrasts.csv':63, 'contrast_seeds.csv':567}
        for name, expected in counts.items():
            if len(common.read_csv(staging/name)) != expected:
                raise ValueError(f'Unexpected pH 8.0 score coverage: {name}')
        metadata.update(source_single_ph_campaign=SINGLE_CAMPAIGN, source_scoring_manifest_sha256=manifest_hash,
                        verified_exact_reference_score_rows=verified, categorical_feature_fingerprint=collector['categorical_feature_fingerprint'],
                        n_contrasts=counts['contrasts.csv'], n_seed_rows=counts['contrast_seeds.csv'])
        write(staging/target_manifest.name, metadata)
        for name in (*metadata['files'], target_manifest.name):
            (staging/name).replace(out/name)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    score_export(args.release, args.predictions, args.out)


if __name__ == '__main__':
    main()
