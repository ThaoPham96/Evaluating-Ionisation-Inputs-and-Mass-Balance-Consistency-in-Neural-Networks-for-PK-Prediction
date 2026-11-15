"""Reconcile, adjudicate and freeze source-linked ionisation v2 data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory


EXPERIMENT = Path(__file__).resolve().parents[1]
WORKSPACE = EXPERIMENT.parent
MASS_BALANCE = WORKSPACE / "Mass_balance_experiment_20261113"
HISTORICAL = WORKSPACE / "revision_work/analysis/outputs/preprocessing/f57c87745d2d433f"
REINSTATEMENT = WORKSPACE / "revision_work/analysis/outputs/ionisation_reinstatement/20261112"
DATA = EXPERIMENT / "data"
FINGERPRINT = "5413771691c6981d22b6c9ca8e21584dc93cd01244c5e3b143c489dfdc9c7942"
HISTORICAL_FINGERPRINT = "f57c87745d2d433feef6995bcc3f81dea52ec7bc9bf93d7e09aeb60dae17508f"
MASS_DATA_MANIFEST = "01a9b8fc98e934b2fce4b399c63b013c6a21fb09f0c5ba8ddd82e788dacafbb1"
MASS_FEATURE_MANIFEST = "82e8ce3794d01ccc8b19c16bcb8f731797b3b7409bcf03d4ec2fba8d31565718"
PH = 7.4
CLASSES = ("anionic", "cationic", "neutral", "zwitterionic")
ENDPOINTS = (
    "t1/2(hour)", "VD(liter)", "Cl(liter/hour)", "F(percentage)",
    "PPB(percentage)", "Cmax_uM", "Tmax",
)
FOLDS = range(1, 6)
MIN_INNER_VALIDATION = 10
RELEASE_FILES = ("model_dataset.csv", "outer.csv", "inner.csv", "ledger.csv", "fold_coverage.csv")
FEATURE_FILES = ("structures.json", "graphs.pt", "chemberta.npy")

DECISION_FIELDS = (
    "structure_id", "candidate_sha256", "class_decision", "numerical_decision",
    "reviewer", "review_date", "decision_reason", "evidence_reference",
    "selected_source_rows", "identity_evidence", "changed_graph_evidence",
    "conflict_resolution", "ionisation_class", "class_support", "class_rule",
    "permanent_charge", "permanent_charge_evidence", "polyprotic",
    "pka_acid", "pka_acid_kind", "pka_acid_source", "pka_acid_site", "pka_acid_conditions",
    "pka_base", "pka_base_kind", "pka_base_source", "pka_base_site", "pka_base_conditions",
    "source_independence", "mapping_uncertainty", "condition_uncertainty",
    "missing_side_kind", "missing_side_evidence", "missing_side_at_7_4",
    "boundary_evidence", "acid_boundary_at_7_4", "base_boundary_at_7_4",
    "numerical_reason", "source_priority_rule",
)
EVIDENCE_FIELDS = DECISION_FIELDS[2:] + ("min_distance_to_pH_7_4",)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        require(columns and len(columns) == len(set(columns)), f"Invalid header in {path}")
        rows = list(reader)
        require(all(None not in row and None not in row.values() for row in rows),
                f"Malformed CSV rows in {path}")
    return rows


def write_json(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def index(rows, key, label):
    indexed = {}
    for row in rows:
        value = row[key]
        require(value and value not in indexed, f"Duplicate or empty {label}: {value}")
        indexed[value] = row
    return indexed


def check_hash(path, expected):
    require(isinstance(expected, str) and sha256(path) == expected, f"Checksum mismatch: {path}")
    return expected


def checked_files(directory, filenames, hashes):
    return {name: check_hash(directory / name, hashes.get(name)) for name in filenames}


def number(value):
    if value == "":
        return None
    parsed = float(value)
    require(math.isfinite(parsed), f"Nonfinite pKa or endpoint: {value}")
    return parsed


def distance(acid, base):
    return min((abs(v - PH) for v in (acid, base) if v is not None), default=None)


def row_ids(value):
    ids = json.loads(value)
    require(isinstance(ids, list) and all(type(v) is int and v >= 0 for v in ids),
            f"Expected a JSON list of non-negative source rows: {value}")
    require(bool(ids) and len(ids) == len(set(ids)), f"Duplicate or missing source rows: {value}")
    return ids


def structure_identity(rows):
    for row in rows:
        smi = row["canonical_smiles"]
        require(smi and hashlib.sha256(smi.encode()).hexdigest()[:16] == row["structure_id"],
                f"Structure ID/SMILES mismatch: {row['structure_id']}")


def validate_folds(dataset, outer, inner):
    """Validate complete inherited assignments and group exclusivity."""
    outer_index = index(outer, "structure_id", "outer structure")
    require(set(outer_index) == set(dataset), "Outer IDs differ from the model dataset")
    outer_by_group = {}
    for sid, row in outer_index.items():
        group, fold = row["group_id"], row["outer_fold"]
        require(group and fold in {str(v) for v in FOLDS}, f"Invalid outer assignment: {sid}")
        require((group, fold) == (dataset[sid]["group_id"], dataset[sid]["outer_fold"]),
                f"Outer assignment conflicts with the model dataset: {sid}")
        require(outer_by_group.setdefault(group, fold) == fold,
                f"Group crosses outer folds: {group}")
    inner_ids = {str(fold): {} for fold in FOLDS}
    inner_by_group = {}
    for row in inner:
        outer_fold, sid, group, fold = (row[k] for k in (
            "outer_fold", "structure_id", "group_id", "inner_fold"))
        require(outer_fold in inner_ids and fold in inner_ids and sid in outer_index,
                f"Invalid inner assignment: {row}")
        require(group == outer_index[sid]["group_id"] and outer_fold != outer_index[sid]["outer_fold"],
                f"Inner assignment uses a held-out structure or wrong group: {sid}")
        require(sid not in inner_ids[outer_fold], f"Duplicate inner assignment: {outer_fold}/{sid}")
        inner_ids[outer_fold][sid] = fold
        require(inner_by_group.setdefault((outer_fold, group), fold) == fold,
                f"Group crosses inner folds: {outer_fold}/{group}")
    for outer_fold, assigned in inner_ids.items():
        expected = {sid for sid, row in outer_index.items() if row["outer_fold"] != outer_fold}
        require(set(assigned) == expected, f"Missing inner assignments for outer fold {outer_fold}")
    return inner_ids


@dataclass
class Inventory:
    dataset: dict
    outer: list
    inner: list
    inner_ids: dict
    ledger: list
    source_audit: list
    packets: dict
    recovery: list
    summary: dict
    inputs: dict
    mass_data_binding: dict


def load_mass_balance(directory):
    manifest_path = directory / "manifest.json"
    check_hash(manifest_path, MASS_DATA_MANIFEST)
    manifest = read_json(manifest_path)
    require(manifest["fingerprint"] == FINGERPRINT
            and manifest["data_frozen"] is True and manifest["splits_frozen"] is True,
            "Expected the frozen v6 mass-balance dataset and splits")
    names = (
        "model_dataset.csv", "source_row_mapping.csv", "reviewed_source_records.csv",
        "unlabelled_structures_not_used.csv", "outer.csv", "inner.csv", "same_row_complete_triads.csv",
        "identity_resolution_65.csv", "batch_review.csv", "final_identity_amendments.csv",
    )
    hashes = checked_files(directory, names, manifest["files"])
    dataset = index(read_csv(directory / "model_dataset.csv"), "structure_id", "labelled structure")
    require(len(dataset) == 952, f"Expected 952 frozen labelled structures, found {len(dataset)}")
    binding = {"fingerprint": manifest["fingerprint"], "manifest.json": sha256(manifest_path),
               "model_dataset.csv": hashes["model_dataset.csv"]}
    return dataset, hashes, binding


def source_mapping(dataset, unlabelled, mappings, reviewed):
    structures = index([*dataset.values(), *unlabelled], "structure_id", "corrected structure")
    structure_identity(structures.values())
    exploded = {}
    for sid, structure in structures.items():
        for rid in row_ids(structure["source_rows"]):
            require(str(rid) not in exploded, f"Source row {rid} maps to multiple structures")
            exploded[str(rid)] = sid
    genuine = {}
    for rid, row in mappings.items():
        sid, role = row["structure_id"], row["role"]
        require(sid in structures and rid in reviewed, f"Unknown mapped structure or source row: {rid}")
        expected_alias = reviewed[rid]["provenance_alias_only"]
        if role == "provenance_alias_only":
            require(rid not in exploded and expected_alias == "True",
                    f"Alias is present among genuine source rows: {rid}")
        else:
            require(role == "source_record" and expected_alias == "False",
                    f"Unexpected source row role: {rid}/{role}")
            require(reviewed[rid]["canonical_smiles"] == structures[sid]["canonical_smiles"],
                    f"Corrected source graph disagrees with the mapped structure: {rid}")
            genuine[rid] = sid
    require(exploded == genuine, "Exploded source_rows disagree with source_row_mapping.csv")


def historical_rows(directory, reinstatement):
    manifest = read_json(directory / "manifest.json")
    require(manifest["fingerprint"] == HISTORICAL_FINGERPRINT,
            "Expected the f57c87745d2d433f historical ionisation assessment")
    names = ("ionisation_eligibility.csv", "candidate_modelled.csv", "inputs/reference.csv")
    hashes = checked_files(directory, names, manifest["artifacts"])
    hashes["manifest.json"] = sha256(directory / "manifest.json")
    applied = read_json(reinstatement / "applied_review.json")
    names = ("applied_review.json", "all_619_recovery_decisions.csv",
             "source_evidence_by_record.json", "reinstatement_summary.json")
    previous_hashes = {name: sha256(reinstatement / name) for name in names}
    require(applied["reference_sha256"] == hashes["inputs/reference.csv"]
            and applied["evidence_pack_sha256"] == previous_hashes["source_evidence_by_record.json"],
            "Reinstatement evidence is unbound to the archived assessment")
    return hashes, previous_hashes


def reconcile_sources(dataset, directory, historical, reinstatement):
    """Join by original product row and retain old annotations as evidence."""
    unlabelled = read_csv(directory / "unlabelled_structures_not_used.csv")
    mappings = index(read_csv(directory / "source_row_mapping.csv"), "row_idx", "source mapping")
    reviewed = index(read_csv(directory / "reviewed_source_records.csv"), "ORIG_INDEX", "reviewed source")
    source_mapping(dataset, unlabelled, mappings, reviewed)
    old = index(read_csv(historical / "ionisation_eligibility.csv"), "row_idx", "historical assessment")
    modelled = index(read_csv(historical / "candidate_modelled.csv"), "ORIG_INDEX", "historical candidate")
    reference = index(read_csv(historical / "inputs/reference.csv"), "ORIG_INDEX", "archived reference")
    recovery = index(read_csv(reinstatement / "all_619_recovery_decisions.csv"), "row_idx", "historical recovery")
    evidence = read_json(reinstatement / "source_evidence_by_record.json")
    identity_evidence = defaultdict(list)
    for filename, old_key, new_key in (
        ("identity_resolution_65.csv", "old_smiles", "new_smiles"),
        ("batch_review.csv", "staged_smiles", "recommended_smiles"),
        ("final_identity_amendments.csv", "before", "after"),
    ):
        records = index(read_csv(directory / filename), "row_idx", f"identity review in {filename}")
        for rid, row in records.items():
            require(row["Name"] == reviewed[rid]["Name"], f"Identity review name mismatch at source row {rid}")
            identity_evidence[rid].append({
                "file": filename, "before": row[old_key], "after": row[new_key],
                "reason": row["reason"], "references": row["references"],
            })
    require(set(modelled) == {rid for rid, row in old.items() if row["ionisation_eligible"] == "True"},
            "Historical assessment and candidate modelled rows disagree")
    require(set(old) <= set(reviewed) and set(evidence) <= set(recovery) <= set(old),
            "Unmatched historical assessment or reinstatement evidence")
    audit = []
    for rid, identity in sorted(reviewed.items(), key=lambda item: int(item[0])):
        historical_row = old.get(rid, {})
        candidate = modelled.get(rid, {})
        recovered = recovery.get(rid, {})
        prior = evidence.get(rid, {})
        recorded = reference.get(rid, {})
        mapping = mappings.get(rid, {})
        sid = mapping.get("structure_id", "")
        eligible = rid in modelled
        linked = sid in dataset and mapping["role"] == "source_record"
        if historical_row:
            require(identity["Name"] == historical_row["name"], f"Name mismatch at source row {rid}")
        if eligible:
            require((candidate["Name"], candidate["CAS"]) == (identity["Name"], identity["CAS"])
                    and candidate["annotation_sha256"] == historical_row["annotation_sha256"],
                    f"Historical candidate identity or annotation mismatch at source row {rid}")
        if prior:
            require(prior["name"] == identity["Name"], f"Reinstatement identity mismatch at source row {rid}")
        if historical_row.get("ionisation_status") == "adjudicated_supported":
            require(recovered.get("disposition") == "reinstated" and prior.get("decision") == "retain"
                    and recovered["category_after_review"] == historical_row["IonType_for_modelling"],
                    f"Reinstatement decision disagrees with the archived assessment: {rid}")
        acid = number(historical_row.get("pKa_acidic_cx", ""))
        base = number(historical_row.get("pKa_basic_cx", ""))
        n_pka = int(acid is not None) + int(base is not None)
        if eligible:
            require(n_pka in (1, 2) and historical_row["IonType_for_modelling"].lower() in CLASSES,
                    f"Unsupported historical candidate annotation: {rid}")
        old_smiles = candidate.get("canonical_smiles", historical_row.get("SMILES", ""))
        new_smiles = identity["canonical_smiles"]
        changed = bool(old_smiles and old_smiles != new_smiles)
        directional = n_pka == 1 and (acid is not None and acid < PH or base is not None and base > PH)
        audit.append({
            "row_idx": int(rid), "name": identity["Name"], "source_cas": identity["CAS"],
            "structure_id": sid, "mapping_role": mapping.get("role", ""),
            "links_labelled_structure": linked, "modelled_analyte": identity["modelled_analyte_name"],
            "modelled_analyte_cas": identity["modelled_analyte_CAS"],
            "identity_status": identity["identity_status"], "pending_identity": identity["pending_identity"] == "True",
            "historical_smiles": old_smiles, "corrected_smiles": new_smiles, "graph_changed": changed,
            "historical_eligible": eligible, "historical_status": historical_row.get("ionisation_status", ""),
            "historical_annotation_sha256": historical_row.get("annotation_sha256", ""),
            "historical_recorded_class": historical_row.get("IonType_recorded", "").lower(),
            "historical_supported_class": historical_row.get("IonType_for_modelling", "").lower(),
            "historical_pka_acid": acid, "historical_pka_base": base, "historical_pka_count": n_pka,
            "historical_min_distance_to_7_4": distance(acid, base),
            "historical_chembl_id": recorded.get("ChEMBL_ID", ""),
            "historical_review_reasons": historical_row.get("review_reasons", ""),
            "historical_adjudication_evidence": historical_row.get("adjudication_evidence", ""),
            "historical_adjudication_reason": historical_row.get("adjudication_reason", ""),
            "historical_recovery_disposition": recovered.get("disposition", ""),
            "historical_recovery_reason_code": recovered.get("reason_code", ""),
            "historical_recovery_reason": recovered.get("reason", ""),
            "historical_evidence_record": rid if prior else "",
            "historical_evidence_matches_annotation": prior.get("baseline_annotation_sha256") ==
            historical_row.get("annotation_sha256") if prior else "",
            "historical_evidence_matches_graph": prior.get("reviewed_smiles") == old_smiles if prior else "",
            "directional_one_pka": directional,
        })
    return audit, evidence, identity_evidence


def assessment(rows):
    """Summarise historical evidence without filling gaps across source rows."""
    supported = [row for row in rows if row["historical_eligible"]]
    labels = sorted({row["historical_supported_class"] for row in supported})
    pairs = {(row["historical_pka_acid"], row["historical_pka_base"]) for row in supported}
    conflicting = len(labels) > 1 or any(len({pair[side] for pair in pairs if pair[side] is not None}) > 1
                                       for side in (0, 1))
    # Complementary partial records alone do not establish a numerical pair.
    acid, base = next(iter(pairs)) if len(pairs) == 1 else (None, None)
    return {
        "historical_candidate": bool(supported),
        "historical_class": labels[0] if len(labels) == 1 else "",
        "historical_pka_acid": acid, "historical_pka_base": base,
        "historical_two_pkas": any(None not in pair for pair in pairs) and not conflicting,
        "historical_one_pka": bool(supported) and not any(None not in pair for pair in pairs),
        "changed_graph_review": any(row["graph_changed"] for row in supported),
        "conflicting_historical_annotations": conflicting,
    }


def inventory_summary(ledger, audit, dataset, triad_ids, inner_ids, evidence):
    candidate = [row for row in ledger if row["historical_candidate"]]
    unchanged = [row for row in candidate if not row["changed_graph_review"]]
    numerical = [row for row in candidate if row["historical_two_pkas"]]
    both_unchanged = [row for row in numerical if not row["changed_graph_review"]]

    def counts(rows):
        ids = [row["structure_id"] for row in rows]
        classes = Counter(row["historical_class"] for row in rows)
        inner_validation = [
            [sid for sid in ids if assignments.get(sid) == str(inner_fold)]
            for assignments in inner_ids.values() for inner_fold in FOLDS
        ]
        return {
            "structures": len(ids),
            "classes": {name: classes[name] for name in CLASSES},
            "endpoints": {key: sum(bool(dataset[sid][key]) for sid in ids) for key in ENDPOINTS},
            "groups": len({dataset[sid]["group_id"] for sid in ids}),
            "outer_structures": {fold: sum(int(dataset[sid]["outer_fold"]) == fold for sid in ids) for fold in FOLDS},
            "same_row_triads": len(set(ids) & triad_ids),
            "minimum_inner_validation_endpoints": {
                key: min(sum(bool(dataset[sid][key]) for sid in split) for split in inner_validation)
                for key in ENDPOINTS
            },
        }

    eligible = [row for row in audit if row["historical_eligible"]]
    linked = [row for row in eligible if row["links_labelled_structure"]]
    one_pka_evidence = [evidence[str(row["row_idx"])] for row in linked if row["historical_pka_count"] == 1]
    unlabelled = [row for row in eligible if row["mapping_role"] == "source_record"
                  and not row["links_labelled_structure"]]
    return {
        "status": "audit_only", "data_frozen": False, "splits_frozen": False, "training_ready": False,
        "corrected_labelled_structures": len(dataset),
        "historical_eligible_records": len(eligible),
        "historical_eligible_structures": len({row["historical_smiles"] for row in eligible}),
        "historical_eligible_source_links": len(linked),
        "historical_eligible_unlabelled_records": len(unlabelled),
        "historical_eligible_unlabelled_structures": len({row["structure_id"] for row in unlabelled}),
        "historical_eligible_alias_rows": [row["row_idx"] for row in eligible
                                           if row["mapping_role"] == "provenance_alias_only"],
        "historical_eligible_unchanged_links": sum(not row["graph_changed"] for row in linked),
        "provisional_class": counts(candidate),
        "provisional_class_unchanged_graph": counts(unchanged),
        "provisional_numerical": counts(numerical),
        "provisional_numerical_unchanged_graph": counts(both_unchanged),
        "changed_graph_structures": sum(row["changed_graph_review"] for row in candidate),
        "changed_graph_numerical_structures": sum(row["changed_graph_review"] for row in numerical),
        "one_pka_review_structures": sum(row["historical_one_pka"] for row in candidate),
        "one_pka_linked_reinstatement_evidence": {
            "records": len(one_pka_evidence),
            "records_with_chembl_distributed_via_pubchem": sum(
                any(annotation["source_name"] == "ChEMBL" for annotation in packet["pubchem_annotations"])
                for packet in one_pka_evidence
            ),
            "records_with_typed_iupac_missing_side_measurements": sum(
                any(annotation["matches_missing_side_type"] for annotation in packet["iupac_measurements"])
                for packet in one_pka_evidence
            ),
        },
        "duplicate_conflict_structures": sum(row["conflicting_historical_annotations"] for row in candidate),
        "recovery_only_structures": sum(row["queue"] == "optional_directional_recovery" for row in ledger),
        "previously_excluded_directional_structures": sum(row["queue"] == "previously_excluded" for row in ledger),
        "pending_class_decisions": len(candidate),
    }


def review_evidence(value):
    """Omit historical split assignments from packets exposed to reviewers."""
    if isinstance(value, dict):
        return {k: review_evidence(v) for k, v in value.items() if k != "partition"}
    if isinstance(value, list):
        return [review_evidence(v) for v in value]
    return value


def load_inventory(directory, historical, reinstatement):
    dataset, mass_hashes, mass_binding = load_mass_balance(directory)
    historical_hashes, reinstatement_hashes = historical_rows(historical, reinstatement)
    outer, inner = (read_csv(directory / name) for name in ("outer.csv", "inner.csv"))
    inner_ids = validate_folds(dataset, outer, inner)
    audit, evidence, identity_evidence = reconcile_sources(dataset, directory, historical, reinstatement)
    by_structure = defaultdict(list)
    aliases = defaultdict(list)
    for row in audit:
        if row["links_labelled_structure"]:
            by_structure[row["structure_id"]].append(row)
        if row["mapping_role"] == "provenance_alias_only":
            aliases[row["structure_id"]].append(row["row_idx"])
    ledger, recovery = [], []
    packets = {}
    for sid, model in dataset.items():
        rows = by_structure[sid]
        observed = assessment(rows)
        unresolved = any(row["directional_one_pka"] and row["historical_status"] == "unresolved" for row in rows)
        excluded = any(row["directional_one_pka"] and row["historical_status"] == "adjudicated_excluded"
                       for row in rows)
        queue = "outside_inventory"
        if observed["historical_candidate"]:
            queue = "class_adjudication"
        elif unresolved:
            queue = "optional_directional_recovery"
        elif excluded:
            queue = "previously_excluded"
        packet = {
            "structure_id": sid, "canonical_smiles": model["canonical_smiles"],
            "modelled_analyte": model["modelled_analyte_name"],
            "modelled_analyte_cas": model["modelled_analyte_CAS"],
            "source_rows": row_ids(model["source_rows"]), "provenance_alias_rows": aliases[sid],
            "queue": queue, **observed,
            "annotations": [{k: v for k, v in row.items() if k != "links_labelled_structure"} for row in rows],
            "corrected_identity_evidence": {
                str(row["row_idx"]): identity_evidence[str(row["row_idx"])]
                for row in rows if str(row["row_idx"]) in identity_evidence
            },
            "historical_reinstatement_evidence": {
                str(row["row_idx"]): review_evidence(evidence[str(row["row_idx"])])
                for row in rows if str(row["row_idx"]) in evidence
            },
        }
        candidate_hash = digest(packet)
        ledger.append({
            "structure_id": sid, "canonical_smiles": model["canonical_smiles"],
            "modelled_analyte": model["modelled_analyte_name"],
            "modelled_analyte_cas": model["modelled_analyte_CAS"],
            "source_rows": model["source_rows"], "provenance_alias_rows": json.dumps(aliases[sid]),
            "group_id": model["group_id"], "outer_fold": model["outer_fold"],
            "inner_folds": json.dumps({outer_fold: ids[sid] for outer_fold, ids in inner_ids.items() if sid in ids},
                                      sort_keys=True),
            "queue": queue, **observed, "candidate_sha256": candidate_hash,
        })
        if queue != "outside_inventory":
            packets[sid] = packet | {"candidate_sha256": candidate_hash}
        if queue in ("optional_directional_recovery", "previously_excluded"):
            recovery.append({
                "structure_id": sid, "candidate_sha256": candidate_hash,
                "canonical_smiles": model["canonical_smiles"], "modelled_analyte": model["modelled_analyte_name"],
                "source_rows": model["source_rows"], "queue": queue,
            })
    triads = index(read_csv(directory / "same_row_complete_triads.csv"), "structure_id", "same-row triad")
    require(set(triads) <= set(dataset), "Triad includes an unknown labelled structure")
    inputs = {"mass_balance": mass_hashes | {"manifest.json": mass_binding["manifest.json"]},
              "historical_assessment": historical_hashes, "historical_reinstatement": reinstatement_hashes,
              "builder_sha256": sha256(Path(__file__))}
    return Inventory(dataset, outer, inner, inner_ids, ledger, audit, packets, recovery,
                     inventory_summary(ledger, audit, dataset, set(triads), inner_ids, evidence),
                     inputs, mass_binding)


def prepare(inventory, output):
    output.mkdir(parents=True, exist_ok=True)
    template = [
        {"structure_id": row["structure_id"], "candidate_sha256": row["candidate_sha256"],
         "class_decision": "pending", "numerical_decision": "pending"}
        for row in inventory.ledger if row["historical_candidate"]
    ]
    write_csv(output / "candidate_ledger.csv", inventory.ledger)
    write_csv(output / "source_row_audit.csv", inventory.source_audit)
    write_csv(output / "decision_template.csv", template, DECISION_FIELDS)
    write_csv(output / "recovery_queue.csv", inventory.recovery)
    write_json(output / "review_packets.json", inventory.packets)
    write_json(output / "summary.json", inventory.summary)
    names = ("candidate_ledger.csv", "source_row_audit.csv", "decision_template.csv",
             "recovery_queue.csv", "review_packets.json", "summary.json")
    write_json(output / "manifest.json", {
        "schema_version": 1, "status": "audit_only", "data_frozen": False,
        "splits_frozen": False, "training_ready": False, "inputs": inventory.inputs,
        "files": {name: sha256(output / name) for name in names},
    })


def category(negative, positive):
    if negative and positive:
        return "zwitterionic"
    if negative:
        return "anionic"
    if positive:
        return "cationic"
    return "neutral"


def required_text(row, fields):
    for field in fields:
        require(bool(row[field]), f"{row['structure_id']} needs {field}")


def valid_choice(row, field, choices):
    require(row[field] in choices,
            f"{row['structure_id']} has invalid {field}: {row[field]!r}, expected {sorted(choices)}")


def decide_inclusion(row, candidate, packet, numerical_requested):
    sid = row["structure_id"]
    valid_choice(row, "ionisation_class", CLASSES)
    valid_choice(row, "permanent_charge", {"none", "positive", "negative", "both"})
    valid_choice(row, "polyprotic", {"yes", "no", "uncertain"})
    valid_choice(row, "source_independence", {"independent", "shared", "single_source", "uncertain"})
    valid_choice(row, "mapping_uncertainty", {"resolved", "resolved_with_caveat"})
    valid_choice(row, "condition_uncertainty", {"compatible", "compatible_with_caveat", "uncertain"})
    required_text(row, ("identity_evidence", "selected_source_rows", "source_priority_rule"))
    selected_rows = set(row_ids(row["selected_source_rows"]))
    source = {v["row_idx"]: v for v in packet["annotations"]}
    require(selected_rows <= set(source) and any(source[v]["historical_eligible"] for v in selected_rows),
            f"{sid} needs an eligible, genuine source-row link")
    require(not any(source[v]["pending_identity"] for v in selected_rows),
            f"{sid} has a pending identity on a selected source row")
    if candidate["changed_graph_review"]:
        required_text(row, ("changed_graph_evidence",))
    if candidate["conflicting_historical_annotations"]:
        required_text(row, ("conflict_resolution",))

    pka = {side: number(row[f"pka_{side}"]) for side in ("acid", "base")}
    for side, value in pka.items():
        if value is None:
            require(all(not row[f"pka_{side}_{field}"] for field in ("kind", "source", "site", "conditions")),
                    f"{sid} has numeric evidence fields for an empty pka_{side}")
        else:
            valid_choice(row, f"pka_{side}_kind", {"experimental", "predicted"})
            required_text(row, (f"pka_{side}_source", f"pka_{side}_site", f"pka_{side}_conditions"))
        old_values = {source[v][f"historical_pka_{side}"] for v in selected_rows}
        if value not in old_values:
            required_text(row, ("conflict_resolution",))

    if candidate["historical_class"] and row["ionisation_class"] != candidate["historical_class"]:
        required_text(row, ("conflict_resolution",))
    missing = [side for side, value in pka.items() if value is None]
    require(len(missing) < 2, f"{sid} has no finite pKa; class unsupported")
    expected_support = "missing_side" if missing else "complete_descriptors"
    require(row["class_support"] == expected_support,
            f"{sid} class_support disagrees with its finite pKa values")
    if missing:
        valid_choice(row, "missing_side_kind", {"measurement_or_bound", "structure_site_assessment"})
        valid_choice(row, "missing_side_at_7_4", {"present", "absent"})
        required_text(row, ("missing_side_evidence",))
        require(row["missing_side_kind"] != "structure_site_assessment" or row["missing_side_at_7_4"] == "absent",
                f"{sid} requires a measurement or bound for a missing side assessed present at 7.4")
    else:
        require(not any(row[field] for field in ("missing_side_kind", "missing_side_at_7_4", "missing_side_evidence")),
                f"{sid} supplies missing-side adjudication for a complete pair")

    permanent = row["permanent_charge"]
    if permanent != "none":
        required_text(row, ("permanent_charge_evidence",))
    charge_at_ph = {}
    boundary = PH in pka.values()
    for side, value in pka.items():
        boundary_field = f"{side}_boundary_at_7_4"
        if value is None:
            charge_at_ph[side] = row["missing_side_at_7_4"] == "present"
        elif value == PH:
            valid_choice(row, boundary_field, {"present", "absent"})
            required_text(row, ("boundary_evidence",))
            charge_at_ph[side] = row[boundary_field] == "present"
        else:
            charge_at_ph[side] = value < PH if side == "acid" else value > PH
        require(value == PH or not row[boundary_field], f"{sid} supplies a boundary decision away from 7.4")
    pka_class = category(charge_at_ph["acid"], charge_at_ph["base"])
    expected_class = category(charge_at_ph["acid"] or permanent in ("negative", "both"),
                              charge_at_ph["base"] or permanent in ("positive", "both"))
    expected_rule = "boundary_adjudicated" if boundary else "threshold"
    if pka_class != expected_class:
        expected_rule = "permanent_charge"
    require(row["class_rule"] == expected_rule, f"{sid} needs class_rule={expected_rule}")
    require(row["ionisation_class"] == expected_class,
            f"{sid} class conflicts with the reviewed pKa/charge rule (expected {expected_class})")

    if numerical_requested:
        valid_choice(row, "numerical_decision", {"include", "exclude"})
    if row["numerical_decision"] == "include":
        require(not missing, f"{sid} needs two finite numerical descriptors")
        require(row["condition_uncertainty"] != "uncertain", f"{sid} has unresolved numerical conditions")
    if row["numerical_decision"] in {"include", "exclude"}:
        required_text(row, ("numerical_reason",))
    return distance(pka["acid"], pka["base"])


def validate_decisions(inventory, decisions, numerical_requested):
    """Require one signed, identity-bound decision per historical candidate."""
    candidates = {row["structure_id"]: row for row in inventory.ledger if row["historical_candidate"]}
    require(bool(decisions), "No evidence decisions supplied")
    require(all(set(row) == set(DECISION_FIELDS) for row in decisions),
            "Decision CSV columns differ from the current decision template")
    decisions = [{field: row[field].strip() for field in DECISION_FIELDS} for row in decisions]
    decisions = index(decisions, "structure_id", "evidence decision")
    unknown, missing = set(decisions) - set(candidates), set(candidates) - set(decisions)
    require(not unknown, f"Decisions reference unsupported structures: {sorted(unknown)[:5]}")
    require(not missing, f"{len(missing)} candidate structures lack a class decision")
    pending = [sid for sid, row in decisions.items() if row["class_decision"] in {"", "pending"}]
    require(not pending, f"{len(pending)} class decisions remain pending")
    final = {}
    for sid, row in decisions.items():
        candidate = candidates[sid]
        require(row["candidate_sha256"] == candidate["candidate_sha256"],
                f"Stale candidate evidence hash: {sid}")
        valid_choice(row, "class_decision", {"include", "exclude"})
        valid_choice(row, "numerical_decision", {"", "pending", "include", "exclude"})
        required_text(row, ("reviewer", "review_date", "decision_reason", "evidence_reference"))
        require(date.fromisoformat(row["review_date"]).isoformat() == row["review_date"],
                f"{sid} needs a YYYY-MM-DD review date")
        near = None
        if row["class_decision"] == "include":
            near = decide_inclusion(row, candidate, inventory.packets[sid], numerical_requested)
        else:
            require(row["numerical_decision"] != "include" and not row["ionisation_class"],
                    f"{sid} class exclusion conflicts with its ionisation class or numerical inclusion")
        final[sid] = {key: row[key] for key in EVIDENCE_FIELDS if key != "min_distance_to_pH_7_4"}
        final[sid]["min_distance_to_pH_7_4"] = near
    rules = {row["source_priority_rule"] for row in final.values() if row["class_decision"] == "include"}
    require(len(rules) <= 1, "Use one prespecified source_priority_rule for the cohort")
    return final


def fold_coverage(dataset, outer, inner, min_validation=MIN_INNER_VALIDATION):
    """Check the actual v6 training transforms in every inherited partition."""
    from _vendor import ensure_vendor

    ensure_vendor()
    import numpy as np
    from mass_balance_train import RAW_ENDPOINTS, transform_targets

    require(RAW_ENDPOINTS == ENDPOINTS, "PK endpoint order differs from the frozen trainer")
    require(bool(dataset), "No structures passed the cohort evidence gate")
    inner_ids = validate_folds(dataset, outer, inner)
    sids = list(dataset)
    raw = np.asarray([[number(dataset[sid][key]) for key in ENDPOINTS] for sid in sids], dtype=np.float64)
    transformed = dict(zip(sids, transform_targets(raw), strict=True))

    def partition(ids, outer_fold, inner_fold, role, fit_training=False, minimum=0):
        targets = np.asarray([transformed[sid] for sid in ids], dtype=np.float64).reshape(-1, len(ENDPOINTS))
        counts = np.isfinite(targets).sum(axis=0)
        prefix = f"outer={outer_fold}, inner={inner_fold}, partition={role}"
        for j, endpoint in enumerate(ENDPOINTS):
            if fit_training:
                values = targets[np.isfinite(targets[:, j]), j]
                require(len(values) >= 2 and np.ptp(values) > 0,
                        f"{prefix}: {endpoint} requires two labels and positive training variance")
            else:
                require(counts[j] >= minimum,
                        f"{prefix}: {endpoint} has {counts[j]} labels; require {minimum}")
        classes = Counter(dataset[sid]["ionisation_class"] for sid in ids)
        return {
            "outer_fold": outer_fold, "inner_fold": inner_fold, "partition": role,
            "structures": len(ids), "groups": len({dataset[sid]["group_id"] for sid in ids}),
            **{label: classes[label] for label in CLASSES},
            **{key: int(counts[j]) for j, key in enumerate(ENDPOINTS)},
        }

    coverage = []
    for outer_fold in map(str, FOLDS):
        outer_test = [sid for sid in sids if dataset[sid]["outer_fold"] == outer_fold]
        outer_train = [sid for sid in sids if dataset[sid]["outer_fold"] != outer_fold]
        require(outer_test, f"Inherited outer fold {outer_fold} is empty")
        coverage.append(partition(outer_test, outer_fold, "", "outer_test"))
        coverage.append(partition(outer_train, outer_fold, "", "outer_train", fit_training=True))
        for inner_fold in map(str, FOLDS):
            assigned = inner_ids[outer_fold]
            validation = [sid for sid in outer_train if assigned[sid] == inner_fold]
            train = [sid for sid in outer_train if assigned[sid] != inner_fold]
            coverage.append(partition(validation, outer_fold, inner_fold, "inner_validation", minimum=min_validation))
            coverage.append(partition(train, outer_fold, inner_fold, "inner_train", fit_training=True))
    return coverage


def cohort_tables(inventory, adjudicated, cohort):
    accepted = {sid for sid, evidence in adjudicated.items() if evidence["class_decision"] == "include"
                and (cohort == "class" or evidence["numerical_decision"] == "include")}
    dataset = {sid: row | adjudicated[sid] for sid, row in inventory.dataset.items() if sid in accepted}
    outer = [row for row in inventory.outer if row["structure_id"] in accepted]
    inner = [row for row in inventory.inner if row["structure_id"] in accepted]
    coverage = fold_coverage(dataset, outer, inner)
    blank_evidence = dict.fromkeys(EVIDENCE_FIELDS, "")
    ledger = [
        row | adjudicated.get(row["structure_id"], blank_evidence)
        | {"included_in_release": row["structure_id"] in accepted}
        for row in inventory.ledger
    ]
    return {
        "model_dataset.csv": list(dataset.values()), "outer.csv": outer, "inner.csv": inner,
        "ledger.csv": ledger, "fold_coverage.csv": coverage,
    }


def freeze(inventory, decisions_path, output, cohorts):
    require(all(not (output / name).exists() for name in cohorts),
            "Frozen cohort output already exists; use a new release directory")
    evidence_hash = sha256(decisions_path)
    decisions = read_csv(decisions_path)
    check_hash(decisions_path, evidence_hash)
    adjudicated = validate_decisions(inventory, decisions, "numerical" in cohorts)
    tables = {name: cohort_tables(inventory, adjudicated, name) for name in cohorts}
    combined = {name: index(files["model_dataset.csv"], "structure_id", "accepted structure")
                for name, files in tables.items()}
    for cohort in ("class", "numerical"):
        if cohort not in combined and (output / cohort).exists():
            combined[cohort], previous = read_release(output / cohort, cohort)
            require(previous["source"] == inventory.mass_data_binding, "Sibling cohort has a different source release")
    if len(combined) == 2:
        require(set(combined["numerical"]) <= set(combined["class"]), "Numerical cohort must be a subset of class")
        shared_evidence = (
            "ionisation_class", "pka_acid", "pka_base", "class_rule", "class_support",
            "permanent_charge", "polyprotic", "pka_acid_kind", "pka_base_kind",
            "pka_acid_source", "pka_base_source", "source_priority_rule",
        )
        for sid, row in combined["numerical"].items():
            require(all(row[field] == combined["class"][sid][field] for field in shared_evidence),
                    f"Class and numerical releases disagree on approved evidence for {sid}")
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temporary:
        for cohort, files in tables.items():
            stage = Path(temporary) / cohort
            stage.mkdir()
            for name, rows in files.items():
                write_csv(stage / name, rows)
            identity = {
                "schema_version": 1, "cohort": cohort,
                "files": {name: sha256(stage / name) for name in RELEASE_FILES},
                "source": inventory.mass_data_binding, "inputs": inventory.inputs,
                "decision_csv_sha256": evidence_hash, "minimum_inner_validation_labels": MIN_INNER_VALIDATION,
                "target_transform_sha256": sha256(Path(__file__).with_name("vendor_mass_balance")
                                                  / "mass_balance_train.py"),
                "structure_count": len(files["model_dataset.csv"]),
            }
            write_json(stage / "manifest.json", identity | {
                "fingerprint": digest(identity), "data_frozen": True,
                "splits_frozen": True, "training_ready": True,
            })
        for cohort in cohorts:
            (Path(temporary) / cohort).rename(output / cohort)


def ordered_identity(dataset):
    rows = [(sid, row["canonical_smiles"]) for sid, row in dataset.items()]
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_release(directory, cohort):
    manifest_path = directory / "manifest.json"
    release = read_json(manifest_path)
    flags = ("data_frozen", "splits_frozen", "training_ready")
    require(release.get("cohort") == cohort and all(release.get(flag) is True for flag in flags),
            f"Expected a frozen, training-ready {cohort} release")
    identity = {key: value for key, value in release.items() if key not in (*flags, "fingerprint")}
    require(release["fingerprint"] == digest(identity), f"Invalid {cohort} release fingerprint")
    checked_files(directory, RELEASE_FILES, release["files"])
    data = index(read_csv(directory / "model_dataset.csv"), "structure_id", "release structure")
    require(len(data) == release["structure_count"], "Release structure count mismatch")
    structure_identity(data.values())
    return data, release


def feature_positions(dataset, full_dataset, source_ids):
    require(source_ids == list(full_dataset), "Source feature order differs from the mass-balance release")
    positions = {sid: position for position, sid in enumerate(source_ids)}
    for sid, row in dataset.items():
        require(sid in full_dataset and row["canonical_smiles"] == full_dataset[sid]["canonical_smiles"],
                f"Feature source identity mismatch: {sid}")
    return [positions[sid] for sid in dataset]


def subset_features(data_dir, feature_source, release_root, output, cohorts):
    import numpy as np
    import torch

    require(all(not (output / cohort).exists() for cohort in cohorts),
            "Frozen feature output already exists; use a new feature directory")
    full, _, source_binding = load_mass_balance(data_dir)
    source_path = feature_source / "manifest.json"
    check_hash(source_path, MASS_FEATURE_MANIFEST)
    source = read_json(source_path)
    require(source["data"] == source_binding | {"ordered_structure_smiles_sha256": ordered_identity(full)},
            "Source feature manifest refers to a different mass-balance release")
    source_hashes = checked_files(feature_source, FEATURE_FILES, source["files"])
    source_ids = read_json(feature_source / "structures.json")
    require(source_ids == list(full) and source["structure_count"] == len(source_ids),
            "Source feature structure count or order mismatch")
    embeddings = np.load(feature_source / "chemberta.npy", allow_pickle=False)
    require(embeddings.shape == (len(source_ids), 768) and embeddings.dtype == np.float32
            and np.isfinite(embeddings).all(), "Invalid source ChemBERTa embedding matrix")
    # This PyG file is the local mass-balance artifact checked against its release manifest.
    graphs = torch.load(feature_source / "graphs.pt", map_location="cpu", weights_only=False)
    require(isinstance(graphs, list) and len(graphs) == len(source_ids),
            "Invalid source PyG graph list")
    subsets = {}
    for cohort in cohorts:
        release_dir = release_root / cohort
        dataset, release = read_release(release_dir, cohort)
        require(release["source"] == source_binding, "Cohort and features have different source releases")
        subsets[cohort] = dataset, release, feature_positions(dataset, full, source_ids)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temporary:
        for cohort, (dataset, release, positions) in subsets.items():
            release_dir = release_root / cohort
            stage = Path(temporary) / cohort
            stage.mkdir()
            write_json(stage / "structures.json", list(dataset))
            torch.save([graphs[i] for i in positions], stage / "graphs.pt")
            np.save(stage / "chemberta.npy", embeddings[positions], allow_pickle=False)
            write_json(stage / "manifest.json", {
                "schema_version": 1, "cohort": cohort, "structure_count": len(dataset),
                "order": "model_dataset.csv row order for structures.json, graphs.pt and chemberta.npy",
                "data": {
                    "fingerprint": release["fingerprint"], "manifest.json": sha256(release_dir / "manifest.json"),
                    "model_dataset.csv": release["files"]["model_dataset.csv"],
                    "ordered_structure_smiles_sha256": ordered_identity(dataset),
                },
                "mass_balance_features": {"manifest.json": sha256(source_path), "files": source_hashes},
                "builder_sha256": sha256(Path(__file__)),
                "files": {name: sha256(stage / name) for name in FEATURE_FILES},
            })
        for cohort in cohorts:
            (Path(temporary) / cohort).rename(output / cohort)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Export the audit inventory and review packets")
    prepare_parser.add_argument("--output", type=Path, default=DATA / "candidate")
    freeze_parser = commands.add_parser("freeze", help="Freeze the explicitly adjudicated cohort and inherited folds")
    freeze_parser.add_argument("--decisions", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, default=DATA / "release")
    features_parser = commands.add_parser("features", help="Subset the frozen mass-balance feature bundle")
    features_parser.add_argument("--output", type=Path, default=EXPERIMENT / "features")
    features_parser.add_argument("--release", type=Path, default=DATA / "release")
    for command in (freeze_parser, features_parser):
        command.add_argument("--cohort", choices=("class", "numerical", "both"), default="both")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            inventory = load_inventory(MASS_BALANCE / "data", HISTORICAL, REINSTATEMENT)
            prepare(inventory, args.output)
            print(f"Audit only: {inventory.summary['pending_class_decisions']} pending class decisions")
            return
        cohorts = ("class", "numerical") if args.cohort == "both" else (args.cohort,)
        if args.command == "freeze":
            inventory = load_inventory(MASS_BALANCE / "data", HISTORICAL, REINSTATEMENT)
            freeze(inventory, args.decisions, args.output, cohorts)
        else:
            subset_features(MASS_BALANCE / "data", MASS_BALANCE / "features", args.release, args.output, cohorts)
        print(f"Created {args.command} output in {args.output}: {', '.join(cohorts)}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
