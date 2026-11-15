"""Outcome-blind categorical pH features for the frozen numerical cohort."""
from __future__ import annotations

import csv
from collections import Counter
from decimal import Decimal
from itertools import product
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ionisation_campaign import digest, read, rows, sha, verify_release, write

PH = ('7.4', '2.0', '6.5', '5.0')
CATEGORIES = ('anionic-like', 'cationic-like', 'neither-active', 'dual-active')
AUDITED_CLASSES = ('anionic', 'cationic', 'neutral', 'zwitterionic')
BLOCKS = {'ADD2': (0, 1), 'ADD65': (0, 2), 'ADD5': (0, 3), 'ALL4': (0, 1, 2, 3),
          'REP2': (0, 0), 'REP4': (0, 0, 0, 0)}
WIDTHS = {arm: 4 * len(blocks) for arm, blocks in BLOCKS.items()}
FINGERPRINT = '99e83f5ad3b3802ab2374777eff38f8d9d66e2178237c0e1ea5f23d143948cb8'
EXPECTED_COUNTS = ((65, 90, 119, 53), (0, 280, 37, 10), (49, 112, 108, 58), (31, 144, 91, 61))
FILES = ('ledger.csv', 'fold_counts.csv', 'codes.json')
SOURCE_FIELDS = ('pka_acid', 'pka_base', 'pka_acid_kind', 'pka_base_kind', 'pka_acid_source',
                 'pka_base_source', 'ionisation_class', 'polyprotic', 'permanent_charge',
                 'mapping_uncertainty', 'evidence_reference', 'changed_graph_evidence')
SHIFTS = tuple(map(Decimal, ('-1', '-0.5', '0', '0.5', '1')))


def threshold_codes(acid, base):
    acid, base = Decimal(str(acid)), Decimal(str(base))
    if not acid.is_finite() or not base.is_finite():
        raise ValueError('Categorical pH descriptors require finite acid and base pKas')
    result = []
    for ph in map(Decimal, PH):
        a, b = acid < ph, base > ph
        code = {(True, False): 0, (False, True): 1, (False, False): 2, (True, True): 3}[a, b]
        result.append((a, b, code, abs(acid-ph), abs(base-ph)))
    return result


def feature_matrix(arm, code_rows):
    blocks = BLOCKS[arm]
    codes = np.asarray(code_rows)
    if (codes.ndim != 2 or codes.shape[1] != len(PH) or codes.dtype.kind not in 'iu' or
            np.any((codes < 0) | (codes > 3))):
        raise ValueError('Expected four integer codes in [0, 3] per structure')
    result = np.eye(4, dtype=np.float32)[codes[:, blocks]].reshape(len(codes), WIDTHS[arm])
    result /= math.sqrt(len(blocks))
    return result


def perturbations(acid, base, original):
    acid, base = Decimal(str(acid)), Decimal(str(base))
    profiles = {tuple(block[2] for block in original)}
    changes = Counter()
    rows = []
    for acid_shift, base_shift in product(SHIFTS, repeat=2):
        if acid_shift == base_shift == 0:
            continue
        shifted = threshold_codes(acid+acid_shift, base+base_shift)
        profile = tuple(block[2] for block in shifted)
        profiles.add(profile)
        row = dict(acid_shift=str(acid_shift), base_shift=str(base_shift), profile='>'.join(map(str,profile)))
        for ph, initial, new in zip(PH, original, shifted, strict=True):
            for side, index in (('acid',0),('base',1)):
                changed = int(initial[index] != new[index])
                key = f'{side}_changes_{ph}'
                changes[key] += changed
                row[key] = changed
        rows.append(row)
    summary = dict(perturbed_profile_count=len(profiles), original_profile_can_change=int(len(profiles)>1), **changes)
    return summary, rows


def _write_rows(path, records):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _fold_counts(ledger, outer, inner, groups):
    def partition(fold, subfold, role, members):
        # Count absent categories as zeros in all partitions.
        common = dict(outer_fold=fold, inner_fold=subfold, partition=role)
        for j, ph in enumerate(PH):
            for code, name in enumerate(CATEGORIES):
                ids = [sid for sid in members if ledger[sid][j] == code]
                yield dict(**common, scope='category', ph=ph, category_or_transition=name,
                           structures=len(ids), groups=len({groups[sid] for sid in ids}))
        for profile in sorted(set(map(tuple, ledger.values()))):
            ids = [sid for sid in members if tuple(ledger[sid]) == profile]
            yield dict(**common, scope='transition', ph='', category_or_transition='>'.join(map(str, profile)),
                       structures=len(ids), groups=len({groups[sid] for sid in ids}))

    records = []
    for fold, assignments in inner.items():
        records.extend(partition(fold, None, 'outer_train', assignments))
        records.extend(partition(fold, None, 'outer_holdout', [sid for sid in outer if outer[sid] == fold]))
        for subfold in range(1, 6):
            records.extend(partition(fold, subfold, 'inner_train', [sid for sid in assignments if assignments[sid] != subfold]))
            records.extend(partition(fold, subfold, 'inner_validation', [sid for sid in assignments if assignments[sid] == subfold]))
    return records


def prepare(root, output):
    """Freeze the pH ledger and its fold coverage beside the campaign plan."""
    root, output = Path(root), Path(output)
    dataset, outer, inner = verify_release(root, 'numerical')
    if dataset['fingerprint'] != FINGERPRINT or dataset['structures'] != 327:
        raise ValueError('Expected the original 327-structure numerical release')
    records = rows(root/'data/release/numerical/model_dataset.csv')
    groups = {row['structure_id']: row['group_id'] for row in records}
    if len(set(groups.values())) != 238:
        raise ValueError('Expected the original 238 frozen groups')
    ledger, code_rows = [], {}
    for record in records:
        sid = record['structure_id']
        if (record['permanent_charge'] != 'none' or record['pka_acid_kind'] != 'predicted' or
                record['pka_base_kind'] != 'predicted' or not all(record[k] for k in ('pka_acid_source','pka_base_source'))):
            raise ValueError(f'Unexpected numerical descriptor provenance for {sid}')
        blocks = threshold_codes(record['pka_acid'], record['pka_base'])
        codes = [block[2] for block in blocks]
        if AUDITED_CLASSES[codes[0]] != record['ionisation_class']:
            raise ValueError(f'Derived pH 7.4 category disagrees with audited class for {sid}')
        code_rows[sid] = codes
        entry = {'structure_id': sid, 'group_id': groups[sid], **{field: record[field] for field in SOURCE_FIELDS}}
        for ph, (acid, base, code, acid_distance, base_distance) in zip(PH, blocks):
            entry.update({f'acid_active_{ph}': int(acid), f'base_active_{ph}': int(base),
                          f'code_{ph}': code, f'category_{ph}': CATEGORIES[code],
                          f'acid_distance_{ph}': str(acid_distance), f'base_distance_{ph}': str(base_distance),
                          f'exact_cutoff_{ph}': int(min(acid_distance, base_distance) == 0),
                          f'near_cutoff_{ph}': int(min(acid_distance, base_distance) <= Decimal('0.5'))})
        entry['transition_pattern'] = '>'.join(map(str, codes))
        stress, _ = perturbations(record['pka_acid'], record['pka_base'], blocks)
        entry.update(stress)
        ledger.append(entry)
    counts = tuple(tuple(sum(c[j] == k for c in code_rows.values()) for k in range(4)) for j in range(4))
    profiles = len(set(map(tuple, code_rows.values())))
    switchers = [sum(c[0] != c[j] for c in code_rows.values()) for j in range(1, 4)]
    if (counts != EXPECTED_COUNTS or profiles != 20 or switchers != [206, 38, 85] or
            [sum(r[f'near_cutoff_{ph}'] for r in ledger) for ph in PH] != [45, 37, 32, 35]):
        raise ValueError('Derived inventory differs from the reviewed protocol')
    if output.exists():
        manifest, _ = load(output, list(code_rows), dataset['fingerprint'])
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/'ledger'
        stage.mkdir()
        _write_rows(stage/'ledger.csv', ledger)
        _write_rows(stage/'fold_counts.csv', _fold_counts(code_rows, outer, inner, groups))
        write(stage/'codes.json', code_rows)
        identity = {'release_fingerprint': FINGERPRINT, 'release': dataset, 'ph_order': list(PH),
                    'category_order': list(CATEGORIES), 'blocks': {a: list(b) for a,b in BLOCKS.items()},
                    'structure_count': len(records), 'group_count': len(set(groups.values())),
                    'class_counts': [list(c) for c in counts], 'transition_profiles': profiles, 'switchers': switchers,
                    'perturbation_shifts': list(map(str,SHIFTS)), 'perturbation_scenarios': 24,
                    'perturbed_structures': sum(r['original_profile_can_change'] for r in ledger),
                    'polyprotic_counts': dict(Counter(r['polyprotic'] for r in records)),
                    'files': {name: sha(stage/name) for name in FILES}}
        manifest = identity | {'fingerprint': digest(identity)}
        write(stage/'manifest.json', manifest)
        stage.rename(output)
    return manifest


def load(directory, ids, release_fingerprint):
    directory = Path(directory)
    manifest = read(directory/'manifest.json')
    identity = {k:v for k,v in manifest.items() if k != 'fingerprint'}
    if (digest(identity) != manifest['fingerprint'] or manifest['release_fingerprint'] != release_fingerprint or
            manifest['ph_order'] != list(PH) or manifest['category_order'] != list(CATEGORIES) or
            manifest['blocks'] != {a:list(b) for a,b in BLOCKS.items()} or set(manifest['files']) != set(FILES)):
        raise ValueError('Invalid categorical pH feature manifest')
    if any(sha(directory/name) != value for name,value in manifest['files'].items()):
        raise ValueError('Categorical pH feature file checksum mismatch')
    codes = read(directory/'codes.json')
    if len(codes) != len(ids) or set(codes) != set(ids):
        raise ValueError('Categorical pH feature IDs do not cover the modelling cohort')
    return manifest, [codes[sid] for sid in ids]
