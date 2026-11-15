"""Single-pH feature semantics and width-matched fit initialization."""
import json

import numpy as np
import pandas as pd
import pytest

import ionisation_train as original
from ph_ionisation_features import PH, threshold_codes
import single_ph_features as features
import single_ph_train as trainer


def test_single_block_uses_only_its_ph_and_keeps_four_channels():
    acid_base = [('2.0','2.0'), ('5.0','5.0'), ('6.5','6.5'), ('1.0','3.0'), ('8.0','1.0')]
    codes = np.asarray([[part[2] for part in threshold_codes(a, b)] for a, b in acid_base])
    for index, (arm, ph) in enumerate(features.ARM_PH.items()):
        column = PH.index(ph)
        matrix = features.feature_matrix(arm, codes)
        assert matrix.shape == (5, 4) and matrix.dtype == np.float32
        np.testing.assert_array_equal(matrix.sum(axis=1), np.ones(5))
        np.testing.assert_array_equal(matrix.argmax(axis=1), codes[:, column])
        assert matrix[index].argmax() == 2
        edited = (codes + 1) % 4
        edited[:, column] = codes[:, column]
        np.testing.assert_array_equal(features.feature_matrix(arm, edited), matrix)
    assert set(features.WIDTHS.values()) == {original.WIDTHS['P1']}
    assert not set(features.WIDTHS).intersection(original.WIDTHS)
    with pytest.raises(ValueError, match='integer pH codes'):
        features.feature_matrix('PH2', codes.astype(float))


def test_feature_ledger_validation_rejects_wrong_category_even_when_74_is_unchanged():
    records = [dict(structure_id='one', pka_acid='3.5', pka_base='7.4', ionisation_class='anionic')]
    good = [part[2] for part in threshold_codes('3.5', '7.4')]
    features.validate_ledger_codes(records, [good])
    changed = good.copy()
    changed[1] = (changed[1] + 1) % 4
    with pytest.raises(ValueError, match='disagrees with pKa data'):
        features.validate_ledger_codes(records, [changed])


def test_cpu_ref74_and_single_ph_pair_initialization_reload_and_resume(tmp_path):
    release, structure = tmp_path/'release', tmp_path/'features'
    release.mkdir()
    structure.mkdir()
    ids = [f's{i}' for i in range(12)]
    acid = [4, 8, 4] * 4
    base = [6, 6, 9] * 4
    frame = pd.DataFrame(dict(structure_id=ids, canonical_smiles=['CCO','CCN','CC']*4,
                              ionisation_class=['anionic','neutral','zwitterionic']*4, pka_acid=acid, pka_base=base))
    for endpoint in original.RAW_ENDPOINTS:
        frame[endpoint] = np.arange(1,13)
    frame.to_csv(release/'model_dataset.csv', index=False)
    (structure/'structures.json').write_text(json.dumps(ids), encoding='utf-8')
    np.save(structure/'chemberta.npy', np.ones((len(ids),768),dtype=np.float32))
    ref_request = dict(fit_id='ref', arm='P1', cohort='numerical', backbone='ChemBERTa_head', seed=42,
                       config=dict(hidden_dim=8,ffn_layers=2,dropout=.1,lr=.001,weight_decay=.001),
                       train_ids=ids[:7],val_ids=ids[7:10],predict_ids=ids[7:10],max_epochs=1,diagnostic=True,
                       fixed_epochs=None,task_weights=[1.]*7,lambda_value=0.)
    reference = original.run_fit(ref_request, release, structure, tmp_path/'ref', 'cpu')
    request = dict(ref_request, fit_id='single', arm='PH5')
    outcome = trainer.run_fit(request, release, structure, tmp_path/'single', 'cpu')
    assert outcome['initial_state_hash'] == reference['initial_state_hash']
    assert outcome['common_state_hash'] == reference['common_state_hash']
    assert outcome['descriptor_input_dim'] == 4 and outcome['checkpoint_reload_verified']
    assert outcome['descriptor_spec']['ph_blocks'] == ['5.0']
    assert outcome['task_weights'] == [1.]*7 and outcome['lambda_value'] == 0.
    assert outcome['scaler_hash'] != reference['scaler_hash']
    assert trainer.run_fit(request, release, structure, tmp_path/'single', 'cpu') == outcome
    with pytest.raises(ValueError, match='Unknown ionisation arm'):
        trainer.run_fit(ref_request, release, structure, tmp_path/'invalid', 'cpu')
    with pytest.raises(ValueError, match='lambda_value'):
        trainer.run_fit(dict(request,lambda_value=.1), release, structure, tmp_path/'invalid', 'cpu')
    with pytest.raises(ValueError, match='task weights'):
        trainer.run_fit(dict(request,task_weights=[2,1,1,1,1,1,1]), release, structure, tmp_path/'invalid', 'cpu')
