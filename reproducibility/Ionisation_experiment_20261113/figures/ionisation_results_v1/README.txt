IONISATION RESULTS: FIGURES FOR AUTHOR REVIEW
13 November 2026

1. ionisation_overview
   The two primary Vd comparisons and all categorical/numerical encodings.
   Upper panels: paired seed dots and 95% / six-test-adjusted 99.17% intervals.
   Lower panels: absolute Vd errors, nine seeds, means and seed SDs.

2. support/endpoint_effects
   Seven endpoint changes and an equal-weight descriptive average, comparing
   one-hot category with no ionisation and multi-pH with pH 7.4 alone.
   Colours identify endpoints; dots and bars describe seed variability.

3. support/representation_checks
   Scalar, one-hot, embedding, raw pKa and dimension-matched controls, plus
   shared-recipe versus independently tuned headline comparisons.

4. ionisation_diagnostics
   Cohort composition, simple baselines, within-class Vd effects and
   predicted-triad consistency with the mass-balance penalty switched off.

Every figure is supplied as PNG, vector PDF and editable SVG, with its caption
and plotted numerical data. These are large viewing drafts for discussion;
final journal dimensions and figure allocation can be decided later.

HOW TO READ THE UNCERTAINTY
Small dots identify the nine paired training seeds. They reuse the same fixed
outer folds. They are not nine independent datasets. Seed SD bars describe
training variability, whereas bootstrap intervals also resample whole
scaffold/parent groups. Interval types are labelled on each figure.

Relative changes use a fixed mean comparator MAE as denominator. Absolute
paired-difference intervals are divided by that same denominator; this is not
a bootstrap of a random ratio. Percentage reductions in log-scale MAE should
not be read as percentage reductions in the original physical-unit error.

MAE is scored in log10 units for half-life, Vd, clearance, peak concentration
and time to peak; bioavailability and protein binding use percentage points.
The seven-endpoint average averages normalized changes within each seed.

SCIENTIFIC CONTEXT
All plots use completed held-out predictions from the frozen ionisation
campaign 3ffed2efa5ca4a2bdfa3584cb0d1d3d2869d6edc17e1697e6476fa5a356b5e28.
There are five outer and five inner grouped folds, with nine final seeds.
Class cohort: 494 structures, 388 Vd labels. Numerical subset: 327 structures,
259 Vd labels. Compare encodings within each cohort. Both supervised endpoint
weights and target scaling are as documented in the experiment; explicit
endpoint weights are all one and the mass-balance penalty is zero.

The pH profile contains acidic/basic Henderson-Hasselbalch propensities at
pH 2, 5, 6.5 and 7.4. It represents existing predicted pKa information and is
not a validated simulation of PK in different organs or compartments.

Source score tables and their manifest are under ../../outputs/scoring.
Plotting checks source hashes, paired-seed means and input coverage.
No model fitting, AWS compute or changes to the scored predictions or original
mass-balance experiment are involved in preparing these figures.

RENDER AGAIN FROM THE PIPKNET WORKSPACE
revision_work\.venv\Scripts\python.exe Ionisation_experiment_20261113\figures\ionisation_results_v1\render.py
revision_work\.venv\Scripts\python.exe Ionisation_experiment_20261113\figures\ionisation_results_v1\support\render_support.py

The scripts record exact input/output hashes and save captions separately
so that the figures can be revised without rerunning the experiments.
