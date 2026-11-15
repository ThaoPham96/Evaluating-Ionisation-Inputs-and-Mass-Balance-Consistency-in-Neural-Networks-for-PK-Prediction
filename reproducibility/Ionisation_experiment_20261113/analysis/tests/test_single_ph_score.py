from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ionisation_score as common
import ph_ionisation_score as categorical
import single_ph_score as single


@pytest.fixture(scope="module")
def predictions():
    count = 10
    raw = np.repeat(np.arange(1, count + 1, dtype=float)[:, None], 7, axis=1)
    raw[0, 3] = 0   # F zero retained in MAE
    raw[1, 3] = np.nan
    raw[0, 4] = 0   # PPB zero excluded from AAFE
    raw[1, 4] = np.nan
    data = pd.DataFrame(raw, columns=common.RAW_ENDPOINTS)
    data["group_id"] = [f"g{i // 2}" for i in range(count)]
    data["outer_fold"] = [i // 2 + 1 for i in range(count)]
    data["ionisation_class"] = "neutral"
    data.index = pd.Index([f"s{i:02d}" for i in range(count)], name="structure_id")
    truth = np.nan_to_num(common.transform_targets(raw))
    blocks = []
    for cohort, procedure, backbone, arm in sorted(single.expected_runs()):
        campaign_id = "single-pH" if arm in single.NEW_ARMS else categorical.PRIOR_CAMPAIGN
        for seed in common.SEEDS:
            error = {"REF0": 0.20, "REF74": 0.10}.get(arm, 0.05)
            transformed = truth + error
            physical = common.inverse_physical(transformed)
            block = pd.DataFrame(np.column_stack((transformed, physical)), columns=(*common.PRED_COLUMNS, *common.PHYSICAL_COLUMNS))
            blocks.append(block.assign(
                cohort=cohort, procedure=procedure, backbone=backbone, arm=arm, seed=seed,
                structure_id=data.index, outer_fold=data["outer_fold"].to_numpy(), source_campaign_id=campaign_id,
                source_fit_id=[f"{backbone}-{arm}-{fold}-{seed}" for fold in data["outer_fold"]],
            ))
    return data, pd.concat(blocks, ignore_index=True)


def test_exploratory_pairs_use_common_masks_and_fixed_denominator(predictions):
    data, frame = predictions
    runs, sources = single.validate_predictions(frame, data, "single-pH")
    assert len(runs) == 15 and len(sources) == 675
    tables = single.score(data, runs, n_draws=200)
    contrasts = tables["contrasts"]
    assert len(contrasts) == 147
    assert len(tables["contrast_seeds"]) == 1323
    assert len(tables["contrast_outer_folds"]) == 735
    assert len(tables["endpoint_summary"]) == 105
    assert not contrasts.primary_test.any() and contrasts.crossed_adjusted_high.isna().all()
    for comparator, expected_pct in (("REF0", -75), ("REF74", -50)):
        row = contrasts.query("backbone=='GraphConv' and arm=='PH2' and endpoint=='Vd' and comparator_arm==@comparator").iloc[0]
        assert row.relative_delta_pct == pytest.approx(expected_pct)
        assert row.crossed95_low_pct == pytest.approx(expected_pct)
        assert row.crossed95_high_pct == pytest.approx(expected_pct)
    f = tables["endpoint_metrics"].query("endpoint=='F'")
    ppb = tables["endpoint_metrics"].query("endpoint=='PPB'")
    assert f.n_structures.eq(9).all() and f.n_label_zeros.eq(1).all()
    assert ppb.n_structures.eq(9).all() and ppb.n_aafe.eq(8).all()
    assert set(contrasts.loc[contrasts.endpoint.eq("PPB"), "n_structures"]) == {9}
    first = f.query("arm=='PH2' and backbone=='GraphConv' and seed==101").iloc[0]
    pred = frame.query("arm=='PH2' and backbone=='GraphConv' and seed==101").set_index("structure_id").physical_F
    truth = data["F(percentage)"].dropna()
    assert first.mae == pytest.approx((pred.loc[truth.index] - truth).abs().mean())


@pytest.mark.parametrize("failure", ["seed", "fold", "reference"])
def test_reject_invalid_seed_fold_or_reference_source(predictions, failure):
    data, frame = predictions
    invalid = frame.copy()
    if failure == "seed":
        invalid = invalid.loc[~(invalid.arm.eq("PH65") & invalid.seed.eq(909))]
        message = "nine fixed seeds"
    elif failure == "fold":
        invalid.loc[0, "outer_fold"] = 9
        message = "frozen held-out fold"
    else:
        invalid.loc[invalid.arm.isin(categorical.REFERENCES), "source_campaign_id"] = "different-prior-campaign"
        message = "original frozen numerical campaign"
    with pytest.raises(ValueError, match=message):
        single.validate_predictions(invalid, data, "single-pH")
