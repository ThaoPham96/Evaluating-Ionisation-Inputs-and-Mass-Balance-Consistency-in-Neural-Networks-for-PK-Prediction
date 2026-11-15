"""Freeze reviewed identities, endpoint masks and nested group memberships.

CPU-only; no training, remote storage or infrastructure actions. A new content-
addressed directory is created. Existing releases are verified, never overwritten.
"""
from __future__ import annotations

import json
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase

import mass_balance_splits as split
from mass_balance_controls import residual_log10, coherent_label_error_floor

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
PREVIOUS = WORK / 'review_20261112/mass_balance_identity_resolution_20261112'
REVIEW = WORK / 'review_20261112/mass_balance_freeze_20261112'
BATCH = REVIEW / 'identity_batch_review'
TASKS = split.TASKS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def rebuilt_units(records, masks):
    valid = records.loc[records.canonical_smiles.notna() & records.canonical_smiles.ne('') &
                        ~records.identity_exclusion & ~records.provenance_alias_only].copy()
    for mask in masks.itertuples(index=False):
        if valid.loc[mask.row_idx, mask.endpoint] != mask.archived_value:
            raise ValueError('Mask no longer matches archived target')
        valid.loc[mask.row_idx, mask.endpoint] = np.nan
    units, mapping, triads, provenance = [], [], [], []
    for smi, group in valid.groupby('canonical_smiles', sort=True):
        sid = hashlib.sha256(smi.encode()).hexdigest()[:16]
        row = dict(structure_id=sid, canonical_smiles=smi,
                   source_rows=json.dumps(group.index.astype(int).tolist()),
                   Name=' | '.join(group.Name),
                   historical_partition=' | '.join(sorted(group.partition.unique())))
        for col in ['volume_definition', 'study_compatibility', 'modelled_analyte_name',
                    'modelled_analyte_CAS', 'administration_context', 'masked_model_endpoints']:
            row[col] = ' | '.join(sorted(group[col].astype(str).unique()))
        for endpoint in TASKS:
            observations = group[endpoint].dropna()
            if observations.nunique() > 1:
                raise ValueError(f'Conflicting {endpoint} for {row["Name"]}')
            row[endpoint] = observations.iloc[0] if len(observations) else np.nan
            for rid, value in observations.items():
                provenance.append(dict(structure_id=sid, endpoint=endpoint,
                                       row_idx=int(rid), value=float(value)))
        labelled = any(pd.notna(row[t]) for t in TASKS)
        if labelled and group.pending_identity.any():
            raise ValueError(f'Unresolved labelled identity: {row["Name"]}')
        row['has_PK_label'] = labelled
        units.append(row)
        for rid, source in group.iterrows():
            mapping.append(dict(row_idx=int(rid), structure_id=sid, role='source_record'))
            if labelled and source[TASKS[:3]].notna().all():
                triads.append(dict(structure_id=sid, row_idx=int(rid), Name=source.Name,
                                   **{c:float(source[c]) for c in TASKS[:3]}))
    u = pd.DataFrame(units)
    t = pd.DataFrame(triads).drop_duplicates(['structure_id', *TASKS[:3]])
    if t.structure_id.duplicated().any():
        raise ValueError('Multiple distinct complete triads for one structure')
    return u, t, pd.DataFrame(mapping), pd.DataFrame(provenance)


def run():
    approval = json.loads((BATCH / 'review_summary.json').read_text(encoding='utf-8'))
    corrections = pd.read_csv(BATCH / 'batch_review.csv')
    if len(corrections) != 59 or not corrections.row_idx.is_unique:
        raise ValueError('Expected a unique disposition for all 59 staged corrections')
    if not corrections.disposition.isin(['accept', 'modify']).all():
        raise ValueError('Unresolved batch identity decision')
    expected_modifications = sorted(corrections.loc[corrections.disposition.eq('modify'), 'row_idx'].astype(int))
    if str(approval['status']).lower() != 'pass' and not (
        approval.get('review_complete') is True and
        sorted(approval.get('pending_freeze_replacements', [])) == expected_modifications
    ):
        raise ValueError('Batch review has unresolved items beyond listed replacements')
    records = pd.read_csv(PREVIOUS / 'all_records_after_identity_review.csv').set_index('ORIG_INDEX', drop=False)
    archived = records.copy(deep=True)
    masks = pd.read_csv(PREVIOUS / 'endpoint_identity_masks.csv')
    changes = []
    for r in corrections.itertuples(index=False):
        current = records.loc[r.row_idx, 'canonical_smiles']
        canonical = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
        if canonical(current) != canonical(r.staged_smiles):
            raise ValueError(f'Staged structure changed since review: {r.row_idx}')
        new = canonical(r.recommended_smiles)
        if Chem.MolToInchiKey(Chem.MolFromSmiles(new)) != r.recommended_inchikey:
            raise ValueError(f'Recommended graph/key mismatch: {r.row_idx}')
        if new != current:
            changes.append(dict(row_idx=int(r.row_idx), Name=r.Name,
                                before=current, after=new, reason=r.reason, references=r.references))
        records.loc[r.row_idx, 'canonical_smiles'] = new
        records.loc[r.row_idx, 'identity_status'] = 'batch_review_' + r.disposition
    pd.testing.assert_frame_equal(records[TASKS], archived[TASKS])
    pd.testing.assert_series_equal(records.partition, archived.partition)
    units, triads, mapping, provenance = rebuilt_units(records, masks)
    labelled = units.loc[units.has_PK_label].drop(columns='has_PK_label').copy()
    old = pd.read_csv(PREVIOUS / 'mass_balance_labelled_structures.csv')
    # Reconcile observation units by original rows, independently of corrected IDs.
    pd.testing.assert_frame_equal(labelled.set_index('source_rows')[TASKS].sort_index(),
                                  old.set_index('source_rows')[TASKS].sort_index())
    if len(labelled) != 952 or len(triads) != 674:
        raise ValueError('Unexpected cohort change: explicit review needed')
    alias = pd.read_csv(PREVIOUS / 'tramadol_source_aliases.csv')
    row_sid = mapping.set_index('row_idx').structure_id
    for r in alias.itertuples(index=False):
        mapping.loc[len(mapping)] = [r.row_idx, row_sid[r.modelled_representative_row], 'provenance_alias_only']

    x = split.build_splits(labelled, set(triads.structure_id))
    labelled = labelled.merge(x['outer'], on='structure_id', validate='one_to_one')
    comp_n = x['components'].groupby('structure_id').distinct_grouping_parents.first()
    labelled['distinct_grouping_parents'] = labelled.structure_id.map(comp_n)
    labelled['source_definition_sensitivity_exclude'] = (
        labelled.volume_definition.str.contains('source_confirmed_central') |
        labelled.study_compatibility.str.contains('known_dose_regimen_mismatch') |
        labelled.distinct_grouping_parents.gt(1))
    triads = triads.merge(labelled[['structure_id','group_id','outer_fold','volume_definition',
                                    'study_compatibility','source_definition_sensitivity_exclude']],
                         on='structure_id', validate='one_to_one')
    q = np.log10(triads[TASKS[:3]].to_numpy(float))
    triads['signed_residual_log10'] = residual_log10(q)
    triads['disagreement_fold'] = 10 ** np.abs(triads.signed_residual_log10)
    versions = dict(python=platform.python_version(), rdkit=rdBase.rdkitVersion,
                    pandas=pd.__version__, numpy=np.__version__)
    inputs = [PREVIOUS / 'all_records_after_identity_review.csv',
              PREVIOUS / 'mass_balance_labelled_structures.csv',
              PREVIOUS / 'endpoint_identity_masks.csv', PREVIOUS / 'tramadol_source_aliases.csv',
              BATCH / 'batch_review.csv', BATCH / 'review_summary.json',
              REVIEW / 'grouping_reference_index.json',
              Path(__file__), Path(split.__file__), ROOT / 'mass_balance_controls.py']
    inputs.extend(sorted(p for p in (REVIEW/'grouping_sources').iterdir() if p.is_file()))
    fingerprint_inputs = {str(p.relative_to(WORK)): sha(p) for p in inputs}
    fingerprint = hashlib.sha256(json.dumps(dict(inputs=fingerprint_inputs, versions=versions,
                                                 seed=42, folds=5), sort_keys=True).encode()).hexdigest()
    target = ROOT / 'data/mass_balance' / fingerprint[:16]
    if target.exists():
        manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
        for relative, checksum in manifest['files'].items():
            if sha(target / relative) != checksum:
                raise ValueError('Frozen release was modified: ' + relative)
        print(json.dumps({'status':'existing_release_verified','path':str(target)}, indent=2))
        return
    out = target.with_name('_building_' + fingerprint[:16])
    out.mkdir(parents=True)
    records.to_csv(out / 'reviewed_source_records.csv', index=False)
    labelled.sort_values('structure_id').to_csv(out / 'model_dataset.csv', index=False)
    units.loc[~units.has_PK_label].to_csv(out / 'unlabelled_structures_not_used.csv', index=False)
    triads.sort_values('structure_id').to_csv(out / 'same_row_complete_triads.csv', index=False)
    mapping.sort_values('row_idx').to_csv(out / 'source_row_mapping.csv', index=False)
    provenance.sort_values(['structure_id','endpoint','row_idx']).to_csv(out / 'endpoint_provenance.csv', index=False)
    masks.to_csv(out / 'endpoint_identity_masks.csv', index=False)
    pd.DataFrame(changes).to_csv(out / 'final_identity_amendments.csv', index=False)
    for name in ['components','links','outer','inner','coverage','checks']:
        x[name].to_csv(out / (name + '.csv'), index=False)
    write_json(out / 'group_keys.json', x['keys'])
    write_json(out / 'grouping_policy.json', dict(
        seed=42, outer_folds=5, inner_folds=5,
        spectators=split.SPECTATOR_STRUCTURES,
        model_graphs='full reviewed structures; spectators retained',
        parent_normalization='remove stereo/isotope for grouping only; RDKit Cleanup, Uncharger and default tautomer enumeration; preserve separate source-parent and standard InChI connectivity keys as additional links',
        enumeration_limits='RDKit default caps; capped enumerations recorded per component, supplemented by source and InChI parent keys; no exhaustive-tautomer claim',
        scaffold='GetScaffoldForMol then MakeScaffoldGeneric on each non-spectator normalized component',
        multi_parent='all represented active components provide grouping links; repeated identical components contribute once',
        union='connected components through shared parent/source-parent/InChI or generic API scaffold',
        allocation='descending group size then SHA256(seed,stage,group) ties; assign to lightest fold with deterministic hash ties; no endpoint values or missingness in allocation',
        evaluation='fresh nested grouped CV; no untouched hold-out claim'))
    counts = {c:int(labelled[c].notna().sum()) for c in TASKS}
    outer_summary = x['coverage'].query('role == "outer_test"').drop_duplicates('outer_fold')[
        ['outer_fold','n_structures','same_row_complete_triads']].to_dict('records')
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), fingerprint=fingerprint,
                   data_frozen=True, splits_frozen=True, training_ready=False,
                   reason_not_training_ready='Trainer/feature bundle correctness and throughput preflight remain',
                   labelled_structures=len(labelled), unlabelled_excluded=200,
                   complete_same_row_triads=len(triads), independent_groups=labelled.group_id.nunique(),
                   largest_group=int(labelled.groupby('group_id').size().max()),
                   endpoint_counts=counts, outer_folds=outer_summary,
                   inner_validation_min_endpoint_n=int(x['coverage'].query('role == "inner_validation"').observed.min()),
                   inner_validation_min_triads=int(x['coverage'].query('role == "inner_validation"').same_row_complete_triads.min()),
                   overlap_checks_passed=len(x['checks']),
                   observed_median_disagreement_fold=float(triads.disagreement_fold.median()),
                   observed_within_1_25_fraction=float(triads.disagreement_fold.le(1.25).mean()),
                   coherent_prediction_logMAE_floor=coherent_label_error_floor(q),
                   definition_caveat='Same-row triads are not proven same-study or physiologically matched Vz/CL/terminal-half-life observations. Study estimates a soft consistency penalty.',
                   source_sensitivity_excluded_names=labelled.loc[labelled.source_definition_sensitivity_exclude,'Name'].tolist(),
                   wholly_excluded_names=['COLISTIN'], identity_amendments=[r['row_idx'] for r in changes],
                   batch_review_status='57 accepted; 2 recommended amendments applied and graph keys verified',
                   all_archived_PK_values_unchanged_in_this_freeze=True, model_identity_masks=2,
                   old_historical_partitions_retained_as_metadata=True, versions=versions)
    write_json(out / 'summary.json', summary)
    for p in [BATCH/'batch_review.csv', BATCH/'review_summary.json',
              PREVIOUS/'identity_resolution_65.csv', PREVIOUS/'identity_exclusions.csv',
              PREVIOUS/'tramadol_source_aliases.csv']:
        (out / p.name).write_bytes(p.read_bytes())
    (out/'grouping_sources').mkdir()
    for p in (REVIEW/'grouping_sources').iterdir():
        if p.is_file():
            (out/'grouping_sources'/p.name).write_bytes(p.read_bytes())
    (out/'grouping_reference_index.json').write_bytes((REVIEW/'grouping_reference_index.json').read_bytes())
    write_json(out / 'manifest.json', dict(fingerprint=fingerprint, inputs=fingerprint_inputs,
                                         files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
                                         data_frozen=True, splits_frozen=True, training_ready=False))
    assert out.resolve().parent == target.resolve().parent == (ROOT/'data/mass_balance').resolve()
    out.rename(target)
    write_json(REVIEW / 'current_release.json', dict(path=str(target), fingerprint=fingerprint,
                                                   summary=str(target/'summary.json')))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')
    run()
