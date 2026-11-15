SUPPORTING INFORMATION: IONISATION DESCRIPTORS AND CLASS ASSIGNMENTS

Evaluating Ionisation Inputs and Mass-Balance Consistency in Neural Networks
for Pharmacokinetic Prediction

Table S2: Table_S2_archived_pKa.csv
327 structures with paired, identity-matched ChemAxon descriptors archived
from ChEMBL. CX_MOST_APKA and CX_MOST_BPKA are the exact acidic and basic
pKa values used to construct the numerical-subset input categories. They
correspond to pKa_acidic_cx and pKa_basic_cx in the historical source and
pka_acid and pka_base in the frozen model inputs. Original decimal precision
is retained. These are calculated descriptors, not experimental measurements.
The ChEMBL release, exact calculator build and settings used for this export
were not recorded. No release or settings have been inferred retrospectively.

Table S3: Table_S3_ionisation_classes.csv
494 structures with the exact pH 7.4 class assignments used in the encoding
comparison, together with their assignment evidence. This cohort includes
the 327 structures in Table S2. Its additional records include literature
measurements, evidence for an unreported acidic/basic channel, and permanent
charge. The pKa fields in Table S3 retain their recorded source and kind;
they are not uniformly ChemAxon predictions. Blank numerical values denote
missing values, not zero or absence of ionisation.

FORMAT AND LINKAGE
Files are UTF-8, comma-separated text with one header row and one row per
modelled structure. Fields containing commas, quotation marks or line breaks
are quoted using standard CSV escaping. Import structure identifiers, CAS
numbers and SMILES as text. Join the two files by structure_id. Source-row
lists use JSON array notation. The file order matches the frozen inputs.

CLASS DEFINITIONS AND ENCODINGS
For the numerical subset at pH p, acid activity is CX_MOST_APKA < p and base
activity is CX_MOST_BPKA > p. The four combinations are:

acid active  base active  category         class label    code  charge sign
yes          no           acid-only        anionic          0       -1
no           yes          base-only        cationic         1        1
no           no           neither-active   neutral          2        0
yes          yes          dual-active      zwitterionic     3        0

The one-hot channel order is anionic, cationic, neutral, zwitterionic.
The same class codes index the learned 16-dimensional embedding. The code
itself is categorical; it is not an ordinal molecular quantity.

At pH 7.4, the 494-structure cohort combines these activity indicators with
permanent-charge evidence. A missing numerical channel uses its documented
present/absent decision. A positive permanent charge activates the positive
indicator. These four classes describe the input categorisation and do not
resolve microspecies or the full protonation distribution. They are study
assignments, not an export of ChEMBL's MOLECULAR_SPECIES field.

FIELD DICTIONARY
Fields shared by Tables S2 and S3:
  structure_id: Stable identifier of the modelled canonical-SMILES structure.
  modelled_analyte_name: Name of the reviewed modelled analyte.
  modelled_analyte_CAS: Recorded CAS identifier of that analyte.
  canonical_smiles: Exact canonical SMILES supplied to the structural models,
    retaining the reviewed molecular components and their charge states.
  source_rows: Original zero-based source-row identifiers contributing to the
    collapsed structure (ORIG_INDEX in the archived source).
  selected_source_rows: Source-row identifiers selected for the ionisation
    evidence and its chemical-identity reconciliation.
  reference_pH: pH for the supplied class assignment (7.4).
  ionisation_class: Final class used by the models, with labels defined above.
  class_code: Integer index 0-3 in the class order above.
  charge_sign: Scalar encoding -1, 1 or 0 as defined above.
  group_id: Frozen chemical-group identifier used for grouped cross-validation.
  outer_fold: Frozen outer test-fold assignment, numbered 1-5.
  pka_acid_source / pka_base_source: Recorded source of each numerical channel.
  identity_evidence: Recorded evidence linking the source descriptor or class
    assignment to the modelled analyte, including parent/salt/stereochemistry.

Additional Table S2 fields:
  archived_chembl_id: ChEMBL record identifier associated with both archived
    descriptors. This may identify a salt/product record; the reviewed
    modelled analyte and structure are given separately.
  CX_MOST_APKA: Archived most acidic ChemAxon pKa (dimensionless).
  CX_MOST_BPKA: Archived most basic ChemAxon pKa (dimensionless).

Additional Table S3 fields:
  numerical_pKa_subset: true if the structure also occurs in Table S2.
  class_rule: threshold or permanent_charge, according to the recorded rule
    needed to obtain the final class.
  class_support: complete_descriptors for two numerical channels or
    missing_side when one numerical channel requires other evidence.
  pka_acid / pka_base: Numerical values used as class evidence (dimensionless).
  pka_acid_kind / pka_base_kind: predicted or experimental; blank if missing.
  pka_acid_site / pka_base_site: Recorded site or channel attribution.
  pka_acid_conditions / pka_base_conditions: Recorded measurement/calculation
    conditions and any unavailable conditions, verbatim from the frozen inputs.
  permanent_charge: none or positive in this cohort, referring to the modelled
    organic parent rather than formulation counterions.
  permanent_charge_evidence: Structural or source evidence for that decision.
  missing_side_kind: measurement_or_bound or structure_site_assessment.
  missing_side_evidence: Evidence supporting the channel without a numerical
    pKa; blank when both numerical channels are available.
  missing_side_at_7_4: present or absent for activity of that missing channel.
    This does not represent whether the numerical value is available.
  boundary_evidence: Evidence for a value exactly at pH 7.4, if applicable.
    Empty throughout these cohorts because no value equals this boundary.
  evidence_reference: Recorded compound-specific source references and links.
  conflict_resolution: Recorded resolution of conflicting source information,
    where applicable; blank where no conflict required resolution.

COHORT COUNTS
Table S2: anionic 65, cationic 90, neutral 119, zwitterionic 53 (total 327).
Table S3: anionic 101, cationic 163, neutral 171, zwitterionic 59 (total 494).
Table S3 has 334 complete descriptor pairs and 160 missing-side assignments.
Its two permanent-charge assignments are levocarnitine and miltefosine.

PROVENANCE
The files reproduce the frozen model inputs and their recorded evidence.
The SHA-256 hashes of the source model datasets are:
  Numerical cohort:
  ebe47d2c6285ca54db824fd4d7e37313dca60930679e3527a1a5f90ddf0a3426
  Class cohort:
  41eab67a6df1796fcbeca94a0329c8e1ca19a5c1baffb7975d2be387374e6791
The historical reference containing the original paired pKa values has hash:
  3db0d65b168ca2a80df9de1046833fa26861c23aec88bc11b9357c833636850e

All 327 pairs match that archive and the frozen numerical model inputs.
All exported source fields retain their original text and numerical spelling.
The supplied category codes, scalar encodings and subset membership flags
are explicit representations of the recorded class labels and memberships.
SHA256SUMS.txt gives the hashes of the supplied CSV files and this dictionary.
