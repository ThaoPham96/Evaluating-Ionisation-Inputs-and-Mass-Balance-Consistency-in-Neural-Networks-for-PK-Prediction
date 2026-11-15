MASS-BALANCE LOSS EXPERIMENT — REVIEW AND FIGURE HANDOFF
Prepared 13 November 2026. Start with INTERPRETATION.txt and METHODS.txt.

Question: what changes when an elimination-triad consistency loss is added to
GraphConv, Chemprop and a frozen ChemBERTa representation with an MLP head?

This folder contains this experiment's data, executable code, predictions,
selection records, results and interpretation. All 4,240 fits completed on
Barkla, with zero production failures/retries. No AWS was used.

READING ORDER
1. INTERPRETATION.txt: findings, uncertainty and open analyses.
2. METHODS.txt: completed experimental design and metric definitions.
3. results/overview.html: readable primary results and post-hoc control tables.
4. FIGURE_AND_REVIEW_BRIEF.txt: proposed single multipanel figure and review questions.
5. DATA_DICTIONARY.txt and RUN_COMMANDS.txt: figure inputs and reproduction.

CONTENTS
analysis/                 Exact production training code plus scoring/export scripts and tests.
data/                     Frozen dataset, same-row triads, nested folds and identity/label provenance.
features/                 Frozen graphs and 952 x 768 ChemBERTa embedding matrix.
plans/                    Original machine-readable 4,240-fit plan and configuration grid.
campaign/                 All fit predictions, requests, scalers, learning curves and selection records.
results/summary/          Primary policies, seven fixed-lambda curves, endpoint metrics and seed spread.
results/bootstrap/        10,000 paired group-bootstrap comparisons and saved group draws.
figure_data/              Plot-ready outer predictions, fixed-alpha curves and all-fits index.
provenance/               Execution and numerical verification records.
MANIFEST.json             Relative-path SHA256 inventory of the finished handoff.

SCOPE AND STATUS
Primary comparisons and selected post-hoc controls have been scored and checked.
Fixed-lambda curves are scored; this handoff also derives fixed-alpha curves
on the same first three seeds. Supplementary training is complete, but its
comparative scoring/synthesis, full metric suite and subgroup analyses remain.
ANALYSIS_STATUS.csv records this distinction.

All predictions needed for review and figures are included. Trained checkpoint
weights are retained on university storage and are NOT included in this handoff.
Thus this is sufficient to reproduce the included analyses and supports fresh
training, but it is not a distribution of all fitted models for new inference.
Do not run the campaign's checkpoint-verification command on this metadata copy.

Source manifests remain byte-for-byte provenance records. Some contain original
absolute paths, preflight-era training_ready=false fields, or hashes of narrative
files omitted from this handoff. Those fields are historical, not current status
or required local paths. The present README, execution audit and MANIFEST describe
this package. Use only data/outer.csv and data/inner.csv for evaluation splits.
Historical source partition fields, if present, are provenance only.

The original manuscript, response letters, reviewer comments, earlier results,
obsolete AWS plans and narrative revision protocol are excluded. Files named
batch_review/identity_resolution in data/ concern chemical identity provenance,
not manuscript peer review. MB5 logical IDs are retained identifiers for these
new fits. The pipknet modules are exact production dependencies; execution uses
mass_balance_train.py rather than the package's general training entry points.
