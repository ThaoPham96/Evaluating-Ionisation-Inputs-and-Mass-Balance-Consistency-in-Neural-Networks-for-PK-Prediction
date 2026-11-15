"""Evidence, source-link, split and feature invariants for ionisation v2."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ionisation_data as data


@pytest.fixture(scope="module")
def inventory():
    return data.load_inventory(data.MASS_BALANCE / "data", data.HISTORICAL, data.REINSTATEMENT)


@pytest.fixture()
def decisions(inventory):
    result = []
    for candidate in inventory.ledger:
        if not candidate["historical_candidate"]:
            continue
        row = dict.fromkeys(data.DECISION_FIELDS, "")
        row.update(
            structure_id=candidate["structure_id"], candidate_sha256=candidate["candidate_sha256"],
            class_decision="exclude", numerical_decision="exclude", reviewer="Synthetic test fixture",
            review_date="2026-11-13", decision_reason="Synthetic unit test only",
            evidence_reference="fixture://ionisation_data_test",
        )
        # Synthetic acceptances exercise folds and exact feature subsetting. These
        # fixtures never leave temporary directories and have no chemical standing.
        if candidate["historical_two_pkas"] and not candidate["changed_graph_review"]:
            accept(row, candidate)
        result.append(row)
    return result


def accept(row, candidate):
    acid, base = candidate["historical_pka_acid"], candidate["historical_pka_base"]
    row.update(
        class_decision="include", numerical_decision="include" if None not in (acid, base) else "exclude",
        selected_source_rows=candidate["source_rows"], identity_evidence="Synthetic identity review",
        ionisation_class=candidate["historical_class"], class_rule="threshold",
        class_support="complete_descriptors" if None not in (acid, base) else "missing_side",
        permanent_charge="none", polyprotic="uncertain", source_independence="single_source",
        mapping_uncertainty="resolved", condition_uncertainty="compatible",
        source_priority_rule="Fixture rule", numerical_reason="Synthetic unit test only",
    )
    for side, value in (("acid", acid), ("base", base)):
        if value is not None:
            row[f"pka_{side}"] = str(value)
            row[f"pka_{side}_kind"] = "predicted"
            row[f"pka_{side}_source"] = "Fixture calculator and version"
            row[f"pka_{side}_site"] = "Fixture site and type"
            row[f"pka_{side}_conditions"] = "Fixture aqueous conditions"


def choose(inventory, decisions, predicate):
    candidate = next(row for row in inventory.ledger if predicate(row))
    return candidate, next(row for row in decisions if row["structure_id"] == candidate["structure_id"])


def test_inventory_matches_protocol_and_keeps_labels_out_of_review(inventory):
    summary = inventory.summary
    assert summary["corrected_labelled_structures"] == 952
    assert summary["historical_eligible_source_links"] == 508
    assert summary["provisional_class"]["structures"] == 506
    assert summary["provisional_class"]["classes"]["zwitterionic"] == 57
    assert summary["provisional_numerical"]["structures"] == 339
    assert summary["provisional_class_unchanged_graph"]["structures"] == 466
    assert summary["provisional_numerical_unchanged_graph"]["structures"] == 300
    assert summary["changed_graph_structures"] == 40
    assert summary["one_pka_review_structures"] == 167
    assert summary["recovery_only_structures"] == 171
    assert summary["previously_excluded_directional_structures"] == 12
    assert summary["historical_eligible_alias_rows"] == [495, 929, 930]
    assert summary["historical_eligible_unlabelled_structures"] == 70
    assert summary["one_pka_linked_reinstatement_evidence"] == {
        "records": 169,
        "records_with_chembl_distributed_via_pubchem": 56,
        "records_with_typed_iupac_missing_side_measurements": 3,
    }
    assert summary["provisional_numerical"]["minimum_inner_validation_endpoints"]["Cmax_uM"] == 19
    packets = json.dumps(inventory.packets)
    forbidden = {*data.ENDPOINTS, "group_id", "outer_fold", "inner_fold", "partition"}
    assert all(f'"{field}":' not in packets for field in forbidden)
    for row in inventory.ledger:
        if row["historical_candidate"]:
            packet = inventory.packets[row["structure_id"]]
            assert packet["candidate_sha256"] == data.digest({k: v for k, v in packet.items()
                                                               if k != "candidate_sha256"})


def test_source_mapping_is_verified_and_does_not_accept_aliases(inventory):
    source = data.MASS_BALANCE / "data"
    mappings = data.index(data.read_csv(source / "source_row_mapping.csv"), "row_idx", "mapping")
    reviewed = data.index(data.read_csv(source / "reviewed_source_records.csv"), "ORIG_INDEX", "reviewed")
    unlabelled = data.read_csv(source / "unlabelled_structures_not_used.csv")
    mappings["929"]["role"] = "source_record"
    with pytest.raises(ValueError, match="Unexpected source row role"):
        data.source_mapping(inventory.dataset, unlabelled, mappings, reviewed)


def test_duplicate_annotations_do_not_leak_across_sources(inventory):
    rows = [
        {"historical_eligible": True, "historical_supported_class": "cationic",
         "historical_pka_acid": None, "historical_pka_base": 9.0, "graph_changed": False},
        {"historical_eligible": True, "historical_supported_class": "cationic",
         "historical_pka_acid": 8.0, "historical_pka_base": None, "graph_changed": False},
    ]
    assert not data.assessment(rows)["historical_two_pkas"]
    rows[1]["historical_pka_base"] = 10.0
    assert data.assessment(rows)["conflicting_historical_annotations"]
    methylphenidate = next(row for row in inventory.ledger if row["source_rows"] == "[117, 849, 850]")
    assert methylphenidate["historical_class"] == "cationic"
    assert methylphenidate["historical_pka_base"] == 9.09
    assert not methylphenidate["conflicting_historical_annotations"]


def test_pending_stale_duplicate_and_recovery_decisions_block_freeze(inventory, decisions, tmp_path):
    out = tmp_path / "release"
    path = tmp_path / "decisions.csv"
    with pytest.raises(ValueError, match="No evidence decisions"):
        data.validate_decisions(inventory, [], False)
    decisions[0]["class_decision"] = "pending"
    data.write_csv(path, decisions, data.DECISION_FIELDS)
    with pytest.raises(ValueError, match="class decisions remain pending"):
        data.freeze(inventory, path, out, ("class",))
    assert not out.exists()
    decisions[0]["class_decision"] = "exclude"
    with pytest.raises(ValueError, match="Duplicate or empty evidence decision"):
        data.validate_decisions(inventory, [*decisions, decisions[0]], False)
    decisions[0]["candidate_sha256"] = "stale"
    with pytest.raises(ValueError, match="Stale candidate evidence hash"):
        data.validate_decisions(inventory, decisions, False)
    decisions[0]["structure_id"] = inventory.recovery[0]["structure_id"]
    with pytest.raises(ValueError, match="unsupported structures"):
        data.validate_decisions(inventory, decisions, False)


def test_changed_graph_requires_exact_review_and_no_alias_inputs(inventory, decisions):
    candidate, row = choose(inventory, decisions, lambda c: c["provenance_alias_rows"] != "[]" and
                            c["historical_two_pkas"])
    accept(row, candidate)
    with pytest.raises(ValueError, match="needs changed_graph_evidence"):
        data.validate_decisions(inventory, decisions, False)
    row["changed_graph_evidence"] = "Synthetic identity review of corrected structure"
    row["selected_source_rows"] = "[929]"
    with pytest.raises(ValueError, match="genuine source-row link"):
        data.validate_decisions(inventory, decisions, False)


def test_missing_pka_requires_affirmative_missing_side_review(inventory, decisions):
    candidate, row = choose(inventory, decisions, lambda c: c["historical_one_pka"] and
                            c["historical_class"] == "neutral" and not c["changed_graph_review"])
    accept(row, candidate)
    with pytest.raises(ValueError, match="invalid missing_side_kind"):
        data.validate_decisions(inventory, decisions, False)
    row.update(missing_side_kind="structure_site_assessment", missing_side_at_7_4="absent")
    with pytest.raises(ValueError, match="needs missing_side_evidence"):
        data.validate_decisions(inventory, decisions, False)
    row["missing_side_evidence"] = "Synthetic site review of absence at 7.4"
    assert data.validate_decisions(inventory, decisions, False)[row["structure_id"]]["ionisation_class"] == "neutral"
    row["numerical_decision"] = "include"
    with pytest.raises(ValueError, match="two finite numerical descriptors"):
        data.validate_decisions(inventory, decisions, True)


def test_class_rule_requires_consistency_and_flags_threshold(inventory, decisions):
    candidate, row = choose(inventory, decisions, lambda c: c["historical_two_pkas"] and
                            not c["changed_graph_review"] and c["historical_class"] == "neutral")
    row["pka_acid"] = "7.4"
    with pytest.raises(ValueError, match="needs conflict_resolution"):
        data.validate_decisions(inventory, decisions, True)
    row["conflict_resolution"] = "Synthetic source discrepancy resolved"
    with pytest.raises(ValueError, match="invalid acid_boundary_at_7_4"):
        data.validate_decisions(inventory, decisions, True)
    row.update(acid_boundary_at_7_4="present", boundary_evidence="Synthetic source note",
               class_rule="boundary_adjudicated")
    with pytest.raises(ValueError, match="class conflicts with the reviewed pKa/charge rule"):
        data.validate_decisions(inventory, decisions, True)


def test_permanent_positive_site_preserves_supported_zwitterion(inventory, decisions):
    candidate, row = choose(inventory, decisions, lambda c: c["source_rows"] == "[397]")
    accept(row, candidate)
    row.update(missing_side_kind="structure_site_assessment", missing_side_at_7_4="absent",
               missing_side_evidence="Synthetic check for absent titratable basic site",
               permanent_charge="positive", class_rule="permanent_charge")
    with pytest.raises(ValueError, match="needs permanent_charge_evidence"):
        data.validate_decisions(inventory, decisions, True)
    row["permanent_charge_evidence"] = "Synthetic confirmation of quaternary ammonium"
    assert data.validate_decisions(inventory, decisions, True)[row["structure_id"]]["ionisation_class"] == "zwitterionic"


def test_class_only_release_then_numerical_release_use_same_evidence(inventory, decisions, tmp_path):
    path, out = tmp_path / "decisions.csv", tmp_path / "release"
    include = [row for row in decisions if row["class_decision"] == "include"]
    for row in include:
        row["numerical_decision"] = "pending"
    data.write_csv(path, decisions, data.DECISION_FIELDS)
    data.freeze(inventory, path, out, ("class",))
    with pytest.raises(ValueError, match="invalid numerical_decision"):
        data.freeze(inventory, path, out, ("numerical",))
    for row in include:
        row["numerical_decision"] = "include"
    data.write_csv(path, decisions, data.DECISION_FIELDS)
    data.freeze(inventory, path, out, ("numerical",))
    numerical, manifest = data.read_release(out / "numerical", "numerical")
    assert manifest["training_ready"] is True and len(numerical) == 300
    assert manifest["minimum_inner_validation_labels"] == 10
    for sid, row in numerical.items():
        assert {k: row[k] for k in inventory.dataset[sid]} == inventory.dataset[sid]
    outer = data.read_csv(out / "numerical" / "outer.csv")
    inner = data.read_csv(out / "numerical" / "inner.csv")
    assert outer == [row for row in inventory.outer if row["structure_id"] in numerical]
    assert inner == [row for row in inventory.inner if row["structure_id"] in numerical]
    data.validate_folds(numerical, outer, inner)
    coverage = data.read_csv(out / "numerical" / "fold_coverage.csv")
    inner_validation = [row for row in coverage if row["partition"] == "inner_validation"]
    assert len(inner_validation) == 25
    assert min(int(row["Cmax_uM"]) for row in inner_validation) == 16
    assert all(sum(int(row[c]) for c in data.CLASSES) == int(row["structures"]) for row in coverage)


def test_group_crossing_and_low_inner_coverage_fail(inventory, decisions):
    outer = [row.copy() for row in inventory.outer]
    inner = [row.copy() for row in inventory.inner]
    sizes = Counter((row["outer_fold"], row["group_id"]) for row in inner)
    match = next(row for row in inner if sizes[(row["outer_fold"], row["group_id"])] > 1)
    match["inner_fold"] = "2" if match["inner_fold"] == "1" else "1"
    with pytest.raises(ValueError, match="Group crosses inner folds"):
        data.validate_folds(inventory.dataset, outer, inner)
    adjudicated = data.validate_decisions(inventory, decisions, True)
    accepted = {sid for sid, row in adjudicated.items() if row["class_decision"] == "include"}
    omit = {sid for sid, fold in inventory.inner_ids["1"].items() if fold == "1"}
    kept = accepted - omit
    dataset = {sid: inventory.dataset[sid] | adjudicated[sid] for sid in kept}
    outer = [row for row in inventory.outer if row["structure_id"] in kept]
    inner = [row for row in inventory.inner if row["structure_id"] in kept]
    with pytest.raises(ValueError, match="inner_validation.+require 10"):
        data.fold_coverage(dataset, outer, inner)
    constant = {sid: row | {"VD(liter)": "5.0"} for sid, row in dataset.items()}
    with pytest.raises(ValueError, match="positive training variance"):
        data.fold_coverage(constant, outer, inner)


def test_feature_subsets_preserve_order_tensors_and_hashes(inventory, decisions, tmp_path):
    import torch

    release, output = tmp_path / "release", tmp_path / "features"
    path = tmp_path / "decisions.csv"
    data.write_csv(path, decisions, data.DECISION_FIELDS)
    data.freeze(inventory, path, release, ("numerical",))
    source = data.MASS_BALANCE / "features"
    data.subset_features(data.MASS_BALANCE / "data", source, release, output, ("numerical",))
    dataset, frozen = data.read_release(release / "numerical", "numerical")
    sub = output / "numerical"
    manifest = data.read_json(sub / "manifest.json")
    assert manifest["data"]["fingerprint"] == frozen["fingerprint"]
    assert manifest["data"]["manifest.json"] == data.sha256(release / "numerical" / "manifest.json")
    assert manifest["data"]["model_dataset.csv"] == frozen["files"]["model_dataset.csv"]
    assert data.read_json(sub / "structures.json") == list(dataset)
    source_ids = data.read_json(source / "structures.json")
    positions = [source_ids.index(sid) for sid in dataset]
    np.testing.assert_array_equal(np.load(sub / "chemberta.npy"), np.load(source / "chemberta.npy")[positions])
    old = torch.load(source / "graphs.pt", map_location="cpu", weights_only=False)
    new = torch.load(sub / "graphs.pt", map_location="cpu", weights_only=False)
    for i, graph in zip(positions, new, strict=True):
        for key in ("x", "edge_index", "edge_attr"):
            assert torch.equal(old[i][key], graph[key])
    assert data.checked_files(sub, data.FEATURE_FILES, manifest["files"]) == manifest["files"]
    with pytest.raises(ValueError, match="Source feature order differs"):
        data.feature_positions(dataset, inventory.dataset, list(reversed(source_ids)))
    sid = next(iter(dataset))
    corrupted = dataset | {sid: dataset[sid] | {"canonical_smiles": "invalid"}}
    with pytest.raises(ValueError, match="Feature source identity mismatch"):
        data.feature_positions(corrupted, inventory.dataset, source_ids)
    (sub / "structures.json").write_text("[]")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        data.checked_files(sub, data.FEATURE_FILES, manifest["files"])
