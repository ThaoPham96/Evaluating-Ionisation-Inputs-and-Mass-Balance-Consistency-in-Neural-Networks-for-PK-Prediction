import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ionisation_score as common
import ph_ionisation_score as ph


@pytest.fixture(scope="module")
def dataset_and_predictions():
    n = 30
    values = np.tile(np.arange(1, n + 1, dtype=float)[:, None], (1, 7))
    data = pd.DataFrame(values, columns=common.RAW_ENDPOINTS)
    data.loc[0, "PPB(percentage)"] = 0
    data.loc[1, "F(percentage)"] = np.nan
    data["group_id"] = [f"group_{i // 2}" for i in range(n)]
    data["outer_fold"] = [i // 2 % 5 + 1 for i in range(n)]
    data["pka_acid"] = 9.0
    data["pka_base"] = np.repeat([3.5, 5.0, 6.6, 8.0, 0.5], 6)
    data["ionisation_class"] = common.threshold_category(data[["pka_acid", "pka_base"]].to_numpy())
    data.index = pd.Index([f"s{i:03}" for i in range(n)], name="structure_id")
    truth = np.nan_to_num(common.transform_targets(data[list(common.RAW_ENDPOINTS)].to_numpy()))
    blocks = []
    for cohort, procedure, backbone, arm in sorted(ph.expected_runs()):
        reused = procedure == "historical" or arm in ph.REFERENCES
        campaign = "previous" if reused else "ph-follow-up"
        for seed in common.SEEDS:
            transformed = truth + (0.05 if arm == "ALL4" else 0.15)
            physical = common.inverse_physical(transformed)
            block = pd.DataFrame(
                np.column_stack([transformed, physical]), columns=[*common.PRED_COLUMNS, *common.PHYSICAL_COLUMNS],
            )
            blocks.append(block.assign(
                cohort=cohort, procedure=procedure, backbone=backbone, arm=arm, seed=seed,
                structure_id=data.index, outer_fold=data["outer_fold"].to_numpy(), source_campaign_id=campaign,
                source_fit_id=[f"{backbone}-{arm}-{seed}-{fold}" for fold in data["outer_fold"]],
            ))
    return data, pd.concat(blocks, ignore_index=True)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda rows: rows.iloc[1:], "every structure exactly once"),
        (lambda rows: pd.concat([rows, rows.iloc[[0]]]), "Duplicate held-out"),
        (lambda rows: rows.assign(outer_fold=9), "frozen held-out fold"),
        (lambda rows: rows.assign(physical_Vd=1.0), "predictions disagree"),
        (lambda rows: rows.assign(source_fit_id="shared"), "different fit requests"),
    ],
)
def test_rejects_invalid_export(dataset_and_predictions, edit, message):
    data, frame = dataset_and_predictions
    with pytest.raises(ValueError, match=message):
        ph.validate_predictions(edit(frame), data)


def test_same_source_fit_must_keep_predictions_across_procedures(dataset_and_predictions):
    data, frame = dataset_and_predictions
    bad = frame.copy()
    row = bad.index[bad.procedure.eq("tuned")][0]
    bad.loc[row, "pred_Vd"] += 0.2
    bad.loc[row, "physical_Vd"] = 10 ** bad.loc[row, "pred_Vd"]
    with pytest.raises(ValueError, match="same source fit"):
        ph.validate_predictions(bad, data)


def test_six_primary_tests_and_paired_subgroup_outputs(dataset_and_predictions):
    data, frame = dataset_and_predictions
    runs = ph.validate_predictions(frame, data, "ph-follow-up")
    assert len(runs) == 42
    tables = ph.score(data, runs, n_draws=100)
    contrasts = tables["contrasts"]
    primary = contrasts.loc[contrasts.primary_test]
    assert len(primary) == 6
    assert set(primary.endpoint) == {"Vd", "F"}
    assert set(primary.arm) == {"ALL4"} and set(primary.comparator_arm) == {"REP4"}
    assert set(primary.scope) == {"all"} and set(primary.procedure) == {"primary"}
    assert primary.crossed_adjusted_high.lt(0).all()
    assert contrasts.loc[~contrasts.primary_test, "crossed_adjusted_high"].isna().all()
    assert contrasts.loc[contrasts.arm.eq("ADD65") & contrasts.comparator_arm.eq("REP2"), "purpose"].eq("descriptive").all()
    all_vd = primary.loc[primary.endpoint.eq("Vd")]
    np.testing.assert_allclose(all_vd.delta_mean, -0.10)
    np.testing.assert_allclose(all_vd.relative_delta_pct, -100 * 0.10 / 0.15)
    assert set(primary.loc[primary.endpoint.eq("F"), "n_structures"]) == {29}
    counts = tables["coverage"].query("outer_fold=='all' and stratum=='pH2/switcher' and endpoint=='F'")
    assert counts.iloc[0].n_labels == 17
    assert counts.iloc[0].n_label_groups == 9
    small = contrasts.query("arm=='ADD65' and comparator_arm=='REP2' and stratum=='pH6.5/switcher'")
    assert len(small) == 21 and not small.interval_eligible.any()
    assert small.crossed95_high.isna().all() and small.seed_t95_high.isna().all()
    assert not tables["switcher_interactions"].empty
    ppb = tables["endpoint_metrics"].loc[lambda table: table.endpoint.eq("PPB")]
    assert ppb.n_structures.eq(30).all() and ppb.n_aafe.eq(29).all()


def test_switcher_interaction_pairs_both_strata_with_same_bootstrap_draws():
    # Each group contributes one switcher and one stable structure. The arm
    # changes both outcomes by the same amount, so every shared draw is zero.
    memberships = np.repeat(np.arange(5), 4)
    switcher = np.tile([True, False, True, False], 5)
    labelled = np.ones(len(switcher), dtype=bool)
    delta = np.tile(memberships + 1, (len(common.SEEDS), 1))
    draws = common.group_draws(5, len(common.SEEDS), 100, np.random.default_rng(1))
    result = ph.interaction_statistics(delta, labelled, switcher, memberships, draws)
    assert result["contrast_of_deltas"] == 0
    assert result["crossed95_low"] == result["crossed95_high"] == 0


def test_collector_ledger_matches_every_source_fit(dataset_and_predictions, tmp_path):
    _, rows = dataset_and_predictions
    sources = ph.prediction_sources(rows, "ph-follow-up")
    path = tmp_path / "outer_predictions.csv"
    ledger = path.with_suffix(".provenance.json")
    provenance = dict(campaign_id="ph-follow-up", prior_campaign_id=ph.PRIOR_CAMPAIGN,
                      fits=sources.drop(columns="cohort").to_dict("records"))
    ledger.write_text(json.dumps(provenance), encoding="utf-8")
    collector = {"campaign_id": "ph-follow-up", "provenance_sha256": ph.sha256(ledger)}
    assert ph.check_provenance(path, collector, sources) == collector["provenance_sha256"]
    changed = sources.copy()
    changed.loc[0, "source_fit_id"] = "different"
    with pytest.raises(ValueError, match="differs from prediction rows"):
        ph.check_provenance(path, collector, changed)
