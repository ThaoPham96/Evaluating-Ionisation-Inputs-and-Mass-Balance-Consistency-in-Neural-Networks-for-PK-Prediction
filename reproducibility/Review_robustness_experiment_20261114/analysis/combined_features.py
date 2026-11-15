"""Verify the original numerical release and its exact P0/P1 feature inputs."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from vendor_ionisation import ionisation_data as frozen


COHORT = 'numerical'
FINGERPRINT = '99e83f5ad3b3802ab2374777eff38f8d9d66e2178237c0e1ea5f23d143948cb8'
RELEASE_SHA256 = '5335be3b4b3cb54232396d896aac49bcb7f376fdefbf3cfa20565ca74708b8a3'
FEATURE_SHA256 = '36188f3353482d1843f1764805b46e23387c71832b56e53f197663d305d6ae5d'
FOLDS = (1, 2, 3, 4, 5)


def verify_manifests(release_dir, feature_dir):
    for directory, expected in ((Path(release_dir), RELEASE_SHA256), (Path(feature_dir), FEATURE_SHA256)):
        frozen.check_hash(directory/'manifest.json', expected)


def load_splits(release_dir):
    """Read checksum-verified ordered split maps from the frozen release."""
    release_dir = Path(release_dir)
    path = release_dir/'manifest.json'
    frozen.check_hash(path, RELEASE_SHA256)
    manifest = frozen.read_json(path)
    frozen.checked_files(release_dir, ('outer.csv', 'inner.csv'), manifest['files'])
    outer_rows = frozen.read_csv(release_dir/'outer.csv')
    outer = {row['structure_id']:int(row['outer_fold']) for row in outer_rows}
    if len(outer) != len(outer_rows) or set(outer.values()) != set(FOLDS):
        raise ValueError('Invalid frozen outer split coverage')
    group = {row['structure_id']:row['group_id'] for row in outer_rows}
    group_outer = {}
    for sid, fold in outer.items():
        if group_outer.setdefault(group[sid], fold) != fold:
            raise ValueError('Frozen group appears in multiple outer folds')

    inner_rows = frozen.read_csv(release_dir/'inner.csv')
    if {int(row['outer_fold']) for row in inner_rows} != set(FOLDS):
        raise ValueError('Invalid frozen inner outer-fold indices')
    inner = {}
    for fold in FOLDS:
        part = [row for row in inner_rows if int(row['outer_fold']) == fold]
        assignment = {row['structure_id']:int(row['inner_fold']) for row in part}
        if (len(part) != len(assignment) or set(assignment) != {sid for sid in outer if outer[sid] != fold} or
                set(assignment.values()) != set(FOLDS)):
            raise ValueError('Invalid frozen inner split coverage')
        group_inner = {}
        for sid, subfold in assignment.items():
            if group_inner.setdefault(group[sid], subfold) != subfold:
                raise ValueError('Frozen group appears in multiple inner folds')
        inner[fold] = assignment
    return outer, inner, group


def verify_inputs(release_dir, feature_dir):
    release_dir, feature_dir = Path(release_dir), Path(feature_dir)
    verify_manifests(release_dir, feature_dir)
    data, release = frozen.read_release(release_dir, COHORT)
    outer, _, groups = load_splits(release_dir)
    feature = frozen.read_json(feature_dir/'manifest.json')
    frozen.checked_files(feature_dir, ('structures.json', 'graphs.pt', 'chemberta.npy'), feature['files'])
    if (release['fingerprint'] != FINGERPRINT or len(data) != 327 or len(set(groups.values())) != 238 or
            data.keys() != outer.keys() or frozen.read_json(feature_dir/'structures.json') != list(data) or
            feature['data']['fingerprint'] != FINGERPRINT or feature['data']['manifest.json'] != RELEASE_SHA256 or
            feature['data']['model_dataset.csv'] != release['files']['model_dataset.csv']):
        raise ValueError('Expected the original numerical cohort, splits and feature order')
    threshold = Decimal('7.4')
    classes = {(True, False):'anionic', (False, True):'cationic',
               (False, False):'neutral', (True, True):'zwitterionic'}
    for sid, row in data.items():
        acid, base = Decimal(row['pka_acid']), Decimal(row['pka_base'])
        if (row['group_id'] != groups[sid] or int(row['outer_fold']) != outer[sid] or
                not acid.is_finite() or not base.is_finite() or
                row['ionisation_class'] != classes[acid < threshold, base > threshold]):
            raise ValueError(f'Frozen split or pH 7.4 category mismatch for {sid}')
    return dict(release=RELEASE_SHA256, features=FEATURE_SHA256, fingerprint=FINGERPRINT,
                structures=len(data), groups=len(set(groups.values())))
