import hashlib

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from mass_balance_splits import (TASKS, build_splits, connected_groups,
                                describe_components, nested_memberships,
                                normalize_component, validate_and_count)


def cohort(smiles):
    rows = []
    for i, smi in enumerate(smiles):
        rows.append(dict(structure_id=hashlib.sha256(smi.encode()).hexdigest()[:16],
                         canonical_smiles=smi, Name=f'drug{i}',
                         **{task: float(i + 1) for task in TASKS}))
    return pd.DataFrame(rows)


def test_shared_large_counterion_does_not_link_unrelated_apis():
    pamoate = 'O=C(O)c1cc2ccccc2c(Cc2c(O)c(C(=O)O)cc3ccccc23)c1O'
    d = cohort(['Nc1ccccc1.' + pamoate, 'CCN.' + pamoate, 'Nc1ccccc1.Cl'])
    ledger, keys = describe_components(d)
    groups, _ = connected_groups(keys)
    assert groups[d.structure_id[0]] == groups[d.structure_id[2]]
    assert groups[d.structure_id[0]] != groups[d.structure_id[1]]
    assert len(ledger[ledger.spectator_name.eq('pamoate')]) == 2


def test_stoichiometry_and_multiple_active_components_are_not_dropped():
    d = cohort(['Nc1ccccc1', 'CCN', 'Nc1ccccc1.CCN', 'CCN.CCN.Cl'])
    ledger, keys = describe_components(d)
    groups, _ = connected_groups(keys)
    assert len(set(groups.values())) == 1  # transitive union through mixture
    assert ledger[ledger.structure_id.eq(d.structure_id[2])].distinct_grouping_parents.eq(2).all()
    assert ledger[ledger.structure_id.eq(d.structure_id[3])].distinct_grouping_parents.eq(1).all()
    assert d.canonical_smiles.iloc[3] == 'CCN.CCN.Cl'


def test_tautomer_and_stereoisomer_parent_grouping():
    assert normalize_component(Chem.MolFromSmiles('CC(C)=O')) == normalize_component(Chem.MolFromSmiles('C=C(C)O'))
    assert normalize_component(Chem.MolFromSmiles('C[C@H](O)C(=O)O')) == normalize_component(Chem.MolFromSmiles('C[C@@H](O)C(=O)O'))


def test_fold_membership_ignores_order_and_labels():
    d = cohort(['C' * i + 'N' for i in range(1, 31)])
    x = build_splits(d, set(d.structure_id))
    shuffled = d.sample(frac=1, random_state=13).copy()
    shuffled[TASKS] = np.random.default_rng(29).uniform(1, 100, (len(d), len(TASKS)))
    y = build_splits(shuffled, set(shuffled.structure_id))
    sort = lambda t: t.sort_values(list(t.columns)).reset_index(drop=True)
    pd.testing.assert_frame_equal(sort(x['outer']), sort(y['outer']))
    pd.testing.assert_frame_equal(sort(x['inner']), sort(y['inner']))
    assert len(x['checks']) == 30
    assert x['outer'].outer_fold.value_counts().eq(6).all()
    assert len(x['inner']) == 4 * len(d)


def test_outer_test_cannot_enter_inner_folds():
    d = cohort(['C' * i + 'N' for i in range(1, 31)])
    x = build_splits(d, set(d.structure_id))
    bad = x['inner'].copy()
    row = bad.index[bad.outer_fold.eq(1)][0]
    bad.loc[row, 'structure_id'] = x['outer'].loc[x['outer'].outer_fold.eq(1), 'structure_id'].iloc[0]
    with pytest.raises(ValueError, match='Inner membership mismatch'):
        validate_and_count(d, x['outer'], bad, x['keys'], set(d.structure_id))


def test_unusable_endpoint_blocks_freeze_without_changing_split():
    d = cohort(['C' * i + 'N' for i in range(1, 31)])
    x = build_splits(d, set(d.structure_id))
    d['Cl(liter/hour)'] = np.nan
    with pytest.raises(ValueError, match='coverage/variance failure'):
        validate_and_count(d, x['outer'], x['inner'], x['keys'], set(d.structure_id))


def test_group_overlap_is_rejected():
    d = cohort(['C' * i + 'N' for i in range(1, 31)])
    x = build_splits(d, set(d.structure_id))
    a, b = [x['outer'].loc[x['outer'].outer_fold.eq(i), 'structure_id'].iloc[0] for i in (1, 2)]
    bad_keys = {sid: list(keys) for sid, keys in x['keys'].items()}
    bad_keys[a].append('parent:artificial-shared-parent')
    bad_keys[b].append('parent:artificial-shared-parent')
    with pytest.raises(ValueError, match='overlap'):
        validate_and_count(d, x['outer'], x['inner'], bad_keys, set(d.structure_id))


def test_all_spectator_product_requires_review():
    with pytest.raises(ValueError, match='All components marked spectators'):
        describe_components(cohort(['[Na+].[Cl-]']))
