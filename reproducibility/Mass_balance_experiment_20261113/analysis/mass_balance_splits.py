"""Deterministic parent/scaffold grouped nested CV for the reviewed PK cohort.

Grouping uses structures only. Full model graphs and endpoint values are not
changed. Disconnected components are classified by an explicit spectator list,
never by fragment size. All non-spectator parents contribute grouping links.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize as MS
from rdkit.Chem.Scaffolds import MurckoScaffold

TASKS = ['t1/2(hour)', 'VD(liter)', 'Cl(liter/hour)', 'F(percentage)',
         'PPB(percentage)', 'Cmax_uM', 'Tmax']
SPLIT_SEED = 42
N_FOLDS = 5

# Exact normalized fragment identities, used only in multi-component records.
# These identify grouping spectators; they are retained in each model graph.
SPECTATOR_STRUCTURES = {
    'Cl': 'chloride/HCl', 'Br': 'bromide/HBr', 'O': 'water',
    '[Na+]': 'sodium', '[K+]': 'potassium', '[Ca+2]': 'calcium',
    '[Ag+]': 'silver salt component',
    'CC(=O)O': 'acetate', 'O=[N+]([O-])O': 'nitrate',
    'CS(=O)(=O)O': 'mesylate', 'CCS(=O)(=O)O': 'esylate',
    'O=P(O)(O)O': 'phosphate', 'O=S(=O)(O)O': 'sulfate',
    'O=C(O)C(=O)O': 'oxalate', 'O=C(O)C=CC(=O)O': 'maleate/fumarate',
    'O=C(O)CCC(=O)O': 'succinate', 'O=C(O)CC(O)C(=O)O': 'malate',
    'O=C(O)c1ccccc1': 'benzoate', 'O=C(O)C(O)C(O)C(=O)O': 'tartrate',
    'Cc1ccc(S(=O)(=O)O)cc1': 'tosylate',
    'O=C(O)CC(O)(CC(=O)O)C(=O)O': 'citrate',
    'O=C(O)C(O)C(O)C(O)C(O)CO': 'gluconate',
    'O=C(O)C(O)C(O)C(OC1OC(CO)C(O)C(O)C1O)C(O)CO': 'lactobionate',
    'O=C(O)c1cc2ccccc2c(Cc2c(O)c(C(=O)O)cc3ccccc23)c1O': 'pamoate',
    'NCCO': 'ethanolamine/olamine', 'OCCNCCO': 'diethanolamine/diolamine',
    'C[N+](C)(C)CCO': 'choline',
    'c1ccc(CNCCNCc2ccccc2)cc1': 'benzathine',
}


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@lru_cache(maxsize=4096)
def component_identity(smiles: str):
    """Conservative grouping identity; no changes to model input chemistry.

    Protonation, tautomer, isotope and stereoisomer variants stay in one group.
    Covalent esters/prodrugs are not cleaved; ring analogues link by scaffold.
    """
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    Chem.RemoveStereochemistry(mol)
    mol = MS.Uncharger().uncharge(MS.Cleanup(mol))
    source_parent = Chem.MolToSmiles(mol, isomericSmiles=False)
    inchi_connectivity = Chem.MolToInchiKey(mol).split('-')[0]
    if not inchi_connectivity:
        raise ValueError('No standard InChI grouping key for ' + smiles)
    enumerator = MS.TautomerEnumerator()
    enumeration = enumerator.Enumerate(mol)
    mol = enumerator.PickCanonical(enumeration)
    Chem.RemoveStereochemistry(mol)
    return (Chem.MolToSmiles(mol, isomericSmiles=False), source_parent,
            inchi_connectivity, str(enumeration.status))


def normalize_component(mol) -> str:
    return component_identity(Chem.MolToSmiles(mol))[0]


def spectator_map() -> dict[str, str]:
    return {normalize_component(Chem.MolFromSmiles(s)): name
            for s, name in SPECTATOR_STRUCTURES.items()}


def describe_components(structures: pd.DataFrame):
    """Return a component ledger and exact parent/scaffold grouping keys."""
    spectators = spectator_map()
    ledger, keys = [], {}
    for row in structures.sort_values('structure_id').itertuples(index=False):
        mol = Chem.MolFromSmiles(row.canonical_smiles)
        if mol is None:
            raise ValueError(f'Invalid frozen structure: {row.structure_id}')
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        identities = [component_identity(Chem.MolToSmiles(f)) for f in frags]
        normalized = [x[0] for x in identities]
        active = {s for s in normalized if len(frags) == 1 or s not in spectators}
        if not active:
            raise ValueError(f'All components marked spectators: {row.Name}')
        row_keys = set()
        for index, (frag, parent) in enumerate(zip(frags, normalized)):
            _, source_parent, inchi, tautomer_status = identities[index]
            role = 'grouping_parent' if parent in active else 'salt_or_coformer'
            scaffold = ''
            if role == 'grouping_parent':
                parent_mol = Chem.MolFromSmiles(parent)
                scaf = MurckoScaffold.GetScaffoldForMol(parent_mol)
                if scaf.GetNumAtoms():
                    generic = MurckoScaffold.MakeScaffoldGeneric(scaf)
                    Chem.RemoveStereochemistry(generic)
                    scaffold = Chem.MolToSmiles(generic, isomericSmiles=False)
                    row_keys.add('scaffold:' + scaffold)
                row_keys.add('parent:' + parent)
                row_keys.add('parent_source:' + source_parent)
                row_keys.add('parent_inchi:' + inchi)
            ledger.append({
                'structure_id': row.structure_id, 'Name': row.Name,
                'component_index': index, 'component_smiles': Chem.MolToSmiles(frag),
                'normalized_component': parent, 'role': role,
                'normalized_source_component': source_parent,
                'parent_inchi_connectivity': inchi,
                'tautomer_enumeration_status': tautomer_status,
                'spectator_name': spectators.get(parent, '') if role != 'grouping_parent' else '',
                'generic_api_scaffold': scaffold,
                'distinct_grouping_parents': len(active),
                'basis': 'single connected structure' if len(frags) == 1 else
                         'explicit spectator identity' if role != 'grouping_parent' else
                         'remaining component after explicit spectator classification',
            })
        keys[row.structure_id] = sorted(row_keys)
    return pd.DataFrame(ledger), keys


def connected_groups(keys: dict[str, list[str]]):
    """Connect exact parent OR generic API-scaffold matches transitively."""
    parents = {sid: sid for sid in keys}

    def root(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(x, y):
        a, b = sorted([root(x), root(y)])
        parents[b] = a

    index = defaultdict(list)
    for sid in sorted(keys):
        for key in keys[sid]:
            index[key].append(sid)
    links = []
    for key, members in sorted(index.items()):
        for sid in members[1:]:
            union(members[0], sid)
            links.append({'left_structure_id': members[0], 'right_structure_id': sid,
                          'link_type': key.split(':', 1)[0], 'link_identity': key.split(':', 1)[1]})
    groups = defaultdict(list)
    for sid in sorted(keys):
        groups[root(sid)].append(sid)
    assignment = {}
    for members in groups.values():
        group_id = digest_text('\n'.join(members))[:20]
        assignment.update({sid: group_id for sid in members})
    return assignment, pd.DataFrame(links, columns=[
        'left_structure_id', 'right_structure_id', 'link_type', 'link_identity'])


def allocate_folds(group_sizes: dict[str, int], salt: str, n_folds=N_FOLDS):
    """Balance structure counts; deterministic seed-hash ties, no labels."""
    if len(group_sizes) < n_folds:
        raise ValueError('Insufficient independent groups for requested folds')
    ordered = sorted(group_sizes, key=lambda g: (
        -group_sizes[g], digest_text(f'{SPLIT_SEED}|{salt}|group|{g}')))
    loads = [0] * n_folds
    membership = {}
    for group in ordered:
        fold = min(range(n_folds), key=lambda f: (
            loads[f], digest_text(f'{SPLIT_SEED}|{salt}|{group}|fold|{f}')))
        membership[group] = fold + 1
        loads[fold] += group_sizes[group]
    return membership


def nested_memberships(structures, group_ids):
    cohort = structures[['structure_id']].copy()
    cohort['group_id'] = cohort.structure_id.map(group_ids)
    if cohort.group_id.isna().any() or not cohort.structure_id.is_unique:
        raise ValueError('Missing grouping or duplicate modelling units')
    sizes = cohort.groupby('group_id').size().to_dict()
    outer_map = allocate_folds(sizes, 'outer')
    cohort['outer_fold'] = cohort.group_id.map(outer_map)
    inner = []
    for outer in range(1, N_FOLDS + 1):
        train = cohort[cohort.outer_fold.ne(outer)]
        assignment = allocate_folds(train.groupby('group_id').size().to_dict(), f'outer-{outer}-inner')
        for row in train.itertuples(index=False):
            inner.append({'outer_fold': outer, 'structure_id': row.structure_id,
                          'group_id': row.group_id, 'inner_fold': assignment[row.group_id]})
    return cohort, pd.DataFrame(inner)


def validate_and_count(structures, outer, inner, keys, same_row_triad_ids):
    """Validate graph-key disjointness, partition nesting and label coverage."""
    data = structures.set_index('structure_id')
    all_ids = set(data.index)
    if set(outer.structure_id) != all_ids or outer.structure_id.duplicated().any():
        raise ValueError('Outer membership does not cover each structure once')
    coverage, fold_checks = [], []

    def check(ids, context):
        frame = data.loc[sorted(ids)]
        triad_n = len(set(ids) & same_row_triad_ids)
        if triad_n < 2:
            raise ValueError(f'{context}: fewer than two same-row triads')
        for endpoint in TASKS:
            values = frame[endpoint].dropna()
            if len(values) < 2 or values.nunique() < 2:
                raise ValueError(f'{context}: endpoint coverage/variance failure: {endpoint}')
            coverage.append({**context, 'n_structures': len(frame),
                             'same_row_complete_triads': triad_n,
                             'endpoint': endpoint, 'observed': len(values),
                             'positive': int(values.gt(0).sum()), 'zero': int(values.eq(0).sum())})

    def assert_disjoint(train_ids, test_ids, context):
        train_keys = {k for sid in train_ids for k in keys[sid]}
        test_keys = {k for sid in test_ids for k in keys[sid]}
        if train_ids & test_ids or train_keys & test_keys:
            raise ValueError(f'Structure/parent/scaffold overlap: {context}')
        fold_checks.append({**context, 'shared_structures': 0,
                            'shared_parents_or_scaffolds': 0})

    for o in range(1, N_FOLDS + 1):
        test = set(outer.loc[outer.outer_fold.eq(o), 'structure_id'])
        train = all_ids - test
        assert_disjoint(train, test, {'outer_fold': o, 'inner_fold': 0})
        check(test, {'outer_fold': o, 'inner_fold': 0, 'role': 'outer_test'})
        check(train, {'outer_fold': o, 'inner_fold': 0, 'role': 'outer_train'})
        entries = inner[inner.outer_fold.eq(o)]
        if set(entries.structure_id) != train or entries.structure_id.duplicated().any():
            raise ValueError(f'Inner membership mismatch for outer {o}')
        for i in range(1, N_FOLDS + 1):
            val = set(entries.loc[entries.inner_fold.eq(i), 'structure_id'])
            fit = train - val
            assert_disjoint(fit, val, {'outer_fold': o, 'inner_fold': i})
            check(val, {'outer_fold': o, 'inner_fold': i, 'role': 'inner_validation'})
            check(fit, {'outer_fold': o, 'inner_fold': i, 'role': 'inner_train'})
    return pd.DataFrame(coverage), pd.DataFrame(fold_checks)


def build_splits(structures, same_row_triad_ids):
    components, keys = describe_components(structures)
    group_ids, links = connected_groups(keys)
    outer, inner = nested_memberships(structures, group_ids)
    coverage, checks = validate_and_count(structures, outer, inner, keys, same_row_triad_ids)
    return dict(components=components, keys=keys, group_ids=group_ids, links=links,
                outer=outer, inner=inner, coverage=coverage, checks=checks)
