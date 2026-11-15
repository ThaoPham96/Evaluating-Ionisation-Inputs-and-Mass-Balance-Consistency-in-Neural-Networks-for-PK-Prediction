"""Derive and freeze the single four-channel pH 8.0 block."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ionisation_campaign import digest, read, rows, sha, write
from ph_ionisation_features import AUDITED_CLASSES, CATEGORIES, FINGERPRINT

ARM = 'PH8'
PH = '8.0'
WIDTH = len(CATEGORIES)
COUNTS = (82, 75, 128, 42)
SWITCHERS = 32
LEDGER = Path('runtime/ph8_features')


def threshold_code(acid, base):
    acid, base, ph = Decimal(str(acid)), Decimal(str(base)), Decimal(PH)
    if not acid.is_finite() or not base.is_finite():
        raise ValueError('pH 8.0 categories require two finite pKas')
    return {(True, False): 0, (False, True): 1, (False, False): 2, (True, True): 3}[acid < ph, base > ph]


def feature_matrix(codes):
    values = np.asarray(codes)
    if values.ndim != 1 or values.dtype.kind not in 'iu' or np.any((values < 0) | (values >= WIDTH)):
        raise ValueError('Expected one integer pH 8.0 category per structure')
    return np.eye(WIDTH, dtype=np.float32)[values]


def derive(records):
    codes, ref74 = {}, {}
    for record in records:
        sid = record['structure_id']
        if sid in codes:
            raise ValueError('Duplicate structure in pH 8.0 feature source')
        if (record['pka_acid_kind'] != 'predicted' or record['pka_base_kind'] != 'predicted' or
                record['permanent_charge'] != 'none' or
                not all(record[field] for field in ('pka_acid_source','pka_base_source'))):
            raise ValueError(f'Unexpected archived pKa provenance for {sid}')
        codes[sid] = threshold_code(record['pka_acid'], record['pka_base'])
        ref74[sid] = AUDITED_CLASSES.index(record['ionisation_class'])
    counts = tuple(sum(value == code for value in codes.values()) for code in range(WIDTH))
    switchers = sum(codes[sid] != ref74[sid] for sid in codes)
    if len(codes) != 327 or counts != COUNTS or switchers != SWITCHERS:
        raise ValueError('pH 8.0 category inventory differs from the frozen numerical cohort')
    return codes


def load(directory, ids, dataset_path):
    directory, dataset_path = Path(directory), Path(dataset_path)
    manifest = read(directory/'manifest.json')
    identity = {key:value for key,value in manifest.items() if key != 'fingerprint'}
    expected = dict(release_fingerprint=FINGERPRINT, dataset_sha256=sha(dataset_path), arm=ARM,
                    ph=PH, category_order=list(CATEGORIES), class_counts=list(COUNTS),
                    switchers_vs_74=SWITCHERS, structure_count=327, codes_sha256=sha(directory/'codes.json'))
    if identity != expected or manifest['fingerprint'] != digest(expected):
        raise ValueError('pH 8.0 feature ledger or archived pKa data changed')
    codes = read(directory/'codes.json')
    if len(ids) != len(codes) or set(ids) != codes.keys():
        raise ValueError('pH 8.0 feature IDs do not match the numerical cohort')
    values = [codes[sid] for sid in ids]
    feature_matrix(values)
    return manifest, values


def prepare(dataset_path, directory):
    dataset_path, directory = Path(dataset_path), Path(directory)
    dataset_hash = sha(dataset_path)
    codes = derive(rows(dataset_path))
    if sha(dataset_path) != dataset_hash:
        raise ValueError('Archived pKa data changed while deriving pH 8.0 categories')
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.ph8-features-', dir=directory.parent) as temporary:
            stage = Path(temporary)/'ledger'
            stage.mkdir()
            write(stage/'codes.json', codes)
            identity = dict(release_fingerprint=FINGERPRINT, dataset_sha256=dataset_hash, arm=ARM,
                            ph=PH, category_order=list(CATEGORIES), class_counts=list(COUNTS),
                            switchers_vs_74=SWITCHERS, structure_count=len(codes), codes_sha256=sha(stage/'codes.json'))
            write(stage/'manifest.json', identity | {'fingerprint': digest(identity)})
            stage.rename(directory)
    manifest, loaded = load(directory, list(codes), dataset_path)
    if loaded != list(codes.values()):
        raise ValueError('Frozen pH 8.0 codes disagree with strict archived pKa thresholds')
    return manifest
