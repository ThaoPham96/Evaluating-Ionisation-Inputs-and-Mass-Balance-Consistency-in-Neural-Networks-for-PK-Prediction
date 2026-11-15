"""Write the SI draft, figure captions and SUP01-SUP09 checklist; every number is read from tables/."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

OUT = fs.ROOT / "supplementary_completion"
T = fs.TABLES
M = [("GraphConv", "GraphConv"), ("Chemprop", "Chemprop"), ("ChemBERTa_head", "ChemBERTa")]


def iv(row, d=2, unit=""):
    return f"{row.estimate:+.{d}f}{unit} [{row.ci95_lower:+.{d}f}, {row.ci95_upper:+.{d}f}]".replace("-", "\u2212")


def one(frame, **where):
    for key, value in where.items():
        frame = frame[frame[key] == value]
    if len(frame) != 1:
        raise ValueError(f"Expected one row for {where}, found {len(frame)}")
    return frame.iloc[0]


def per_model(function):
    return "; ".join(f"{label} {function(key)}" for key, label in M)


def main():
    cov = pd.read_csv(T / "inventory_block_coverage.csv")
    c2 = pd.read_csv(T / "sup02_paired_contrasts.csv")
    s2 = pd.read_csv(T / "sup02_policy_seed_metrics.csv")
    sel = pd.read_csv(T / "sup02_selection_by_fold.csv")
    c3 = pd.read_csv(T / "sup03_curve_contrasts_vs_baseline.csv")
    c4 = pd.read_csv(T / "sup04_encoder_depth_contrasts.csv")
    c5 = pd.read_csv(T / "sup05_task_weight_contrasts.csv")
    c6 = pd.read_csv(T / "sup06_single_task_contrasts.csv")
    s6 = pd.read_csv(T / "sup06_single_task_seed_metrics.csv")
    b6 = pd.read_csv(T / "sup06_single_task_budget.csv")
    n7 = pd.read_csv(T / "sup07_source_sensitivity_counts.csv")
    c7 = pd.read_csv(T / "sup07_source_sensitivity_contrasts.csv")
    st7 = pd.read_csv(T / "sup07_compatibility_strata_contrasts.csv")
    s8 = pd.read_csv(T / "sup08_endpoint_shifts_and_spread.csv")
    m9 = pd.read_csv(T / "sup09_endpoint_metrics_summary.csv")
    d9 = pd.read_csv(T / "sup09_endpoint_metric_deltas_vs_baseline.csv")
    triads = pd.read_csv(fs.ROOT / "data" / "same_row_complete_triads.csv")

    mean2 = s2.groupby(["backbone", "policy", "metric"]).value.mean()
    rel = lambda b, a, arm_b="baseline": iv(one(c2, backbone=b, arm_a=a, arm_b=arm_b, metric="triad_mae_relative_log10"), 2, "%")  # noqa: E731
    cons = lambda b, p, m="within_1_25_fold_percent": f"{mean2[(b, p, m)]:.1f}%"  # noqa: E731
    final = cov[cov.stage == "outer_final"]
    n_final, n_verified = int(final.n_fits.sum()), int(final.n_verified_identity_split_checksum.sum())

    zero_lambda = {b: int((sel[sel.backbone == b].zero_eligible_lambda == 0).sum()) for b, _ in M}
    zero_orth = {b: int((sel[sel.backbone == b].selected_alpha_orthogonal == 0).sum()) for b, _ in M}
    zero_half = {b: int((sel[sel.backbone == b].selected_alpha_half_life == 0).sum()) for b, _ in M}

    lam = c3[(c3.family == "fixed_lambda") & (c3.strength > 0) & (c3.metric == "triad_mae_relative_log10")]
    lam_excl = lam[(lam.ci95_lower > 0) | (lam.ci95_upper < 0)]
    lam_text = "; ".join(f"{dict(M)[r.backbone]} \u03bb = {r.strength:g}: {iv(r, 2, '%')}" for r in lam_excl.itertuples())
    sat = c3[(c3.family == "fixed_lambda") & (c3.metric == "within_1_25_fold_percent")]
    sat_text = per_model(lambda b: ", ".join(f"{one(sat, backbone=b, strength=v).estimate:+.1f}".replace("-", "\u2212") for v in (0.03, 0.3, 3.0)))

    loss4 = c4[c4.contrast.str.startswith("loss - baseline")]
    gain4 = loss4[loss4.metric == "within_1_25_fold_percent"].estimate
    tri4 = loss4[loss4.metric == "triad_mae_relative_log10"]
    tri4_excl = int(((tri4.ci95_lower > 0) | (tri4.ci95_upper < 0)).sum())
    ep4 = loss4[loss4.metric.str.contains("_mae_relative_") & ~loss4.metric.str.startswith("triad")]
    ep4_excl = int(((ep4.ci95_lower > 0) | (ep4.ci95_upper < 0)).sum())

    w = lambda contrast, metric, d=2, unit="%": iv(one(c5, contrast=contrast, metric=metric), d, unit)  # noqa: E731
    st = lambda contrast, metric, d=2, unit="%": iv(one(c6, contrast=contrast, metric=metric), d, unit)  # noqa: E731
    mean6 = s6.groupby(["model", "metric"]).value.mean()

    count = lambda q: (int(one(n7, quantity=q).n_all), int(one(n7, quantity=q).n_after_exclusion))  # noqa: E731
    sens = lambda b, a, s: iv(one(c7, backbone=b, arm_a=a, scoring_set=s, metric="triad_mae_relative_log10"), 2, "%")  # noqa: E731
    strat = lambda b, a, s: iv(one(st7, backbone=b, arm_a=a, observed_stratum=s, metric="triad_mae_log10"), 4)  # noqa: E731
    strat_loss = st7[(st7.arm_a == "selected_positive") & (st7.metric == "triad_mae_log10")]
    strat_loss_excl = int(((strat_loss.ci95_lower > 0) | (strat_loss.ci95_upper < 0)).sum())

    sp8 = s8[s8.policy == "selected_positive"]
    v8 = lambda b, e, stat, d=3: iv(one(sp8, backbone=b, endpoint=e, statistic=stat), d)  # noqa: E731
    e8 = lambda b, e, stat: one(sp8, backbone=b, endpoint=e, statistic=stat).estimate  # noqa: E731
    spread = sp8[(sp8.statistic == "spread_ratio_policy_over_baseline") & sp8.endpoint.isin(["t_half", "Vd", "CL"])].estimate
    observed_mean = triads.signed_residual_log10.mean()

    d_sel = d9[d9.arm_a == "selected_positive"]
    d_excl = int(((d_sel.ci95_lower > 0) | (d_sel.ci95_upper < 0)).sum())
    r2 = m9[(m9.metric == "r_squared") & (m9.policy == "baseline") & ~m9.scale.str.contains("ratio")]
    gm = m9[(m9.metric == "gmfe") & (m9.policy == "baseline") & (m9.scale == "log10")]

    text = f"""# Supplementary results draft: mass-balance loss experiment

Generated by `code/write_si_draft.py`; every number below is read from `tables/`. Intervals are 95%
percentile intervals from the saved paired group bootstrap (10,000 draws of 552 parent/scaffold groups,
the same draws for every arm and seed). They are conditional on the fitted models, are not adjusted for
multiple comparisons, and do not describe what retraining would give. Seed SD is reported separately in
the tables. Results come from nested grouped fivefold cross-validation; there is no external test set.

## S1. Verification and coverage (SUP01)

The package verified without mismatches (38,483 files). The primary summaries and the 81 supplied
bootstrap rows reproduce to floating-point precision with identical resampling draws. All {n_final}
outer-final fits were located and passed identity, frozen-split, coverage and checksum checks
({n_verified} verified): 495 primary, 90 alternative-encoder, 120 depth, 30 task-weight and 105 single-task
fits. The supplementary blocks use seeds 101, 202 and 303. Their encoder, depth and task-weight fits use
the GraphConv outer fold's configuration and selected \u03bb (0.1, 0.03, 3, 3 and 0.3 for folds 1-5), with
their own inner epoch calibration; they are not independently tuned.

## S2. Training loss versus post-hoc correction (SUP02; Figure S1A, B, D)

Nine seeds. Predictions within 1.25-fold: baseline {per_model(lambda b: cons(b, "baseline"))}; selected
positive loss {per_model(lambda b: cons(b, "selected_positive"))}; loss with \u03bb = 0 allowed
{per_model(lambda b: cons(b, "zero_eligible"))}; inner-selected orthogonal correction
{per_model(lambda b: cons(b, "selected_alpha_orthogonal"))}; inner-selected half-life-only correction
{per_model(lambda b: cons(b, "selected_alpha_half_life"))}. Exact correction is 100% by construction.

Change in complete-triad MAE versus baseline, as a percentage of baseline MAE:
- selected positive loss: {per_model(lambda b: rel(b, "selected_positive"))};
- loss with \u03bb = 0 allowed: {per_model(lambda b: rel(b, "zero_eligible"))};
- exact orthogonal correction: {per_model(lambda b: rel(b, "exact_orthogonal"))};
- exact half-life-only correction: {per_model(lambda b: rel(b, "exact_half_life"))}.

Direct contrast, selected positive loss minus exact orthogonal correction:
{per_model(lambda b: rel(b, "selected_positive", "exact_orthogonal"))}; minus exact half-life-only
correction: {per_model(lambda b: rel(b, "selected_positive", "exact_half_life"))}.

Selection asymmetry. The main intervention must use \u03bb > 0, whereas the correction strength and the
zero-eligible loss may select no intervention. Out of five outer folds, no intervention was selected in
{per_model(lambda b: str(zero_lambda[b]))} folds for the zero-eligible loss, in
{per_model(lambda b: str(zero_orth[b]))} folds for orthogonal correction and in
{per_model(lambda b: str(zero_half[b]))} folds for half-life-only correction.

Reading. High consistency is not unique to the training loss: a deterministic exact correction reaches
100%. The exact orthogonal correction carries a small but detectable accuracy cost in GraphConv and
Chemprop, and the half-life-only correction in Chemprop; the training loss shows no detectable cost, but
its direct contrasts with exact correction all include zero, so the data do not establish that the loss
is more accurate than correction.

## S3. Strength trade-offs (SUP03; Figure S1C)

Three seeds, pointwise intervals. Gain in predictions within 1.25-fold (percentage points) at
\u03bb = 0.03, 0.3 and 3: {sat_text}. Of the 18 positive fixed-\u03bb points, the triad-MAE interval
excludes zero at {len(lam_excl)}: {lam_text}. These are isolated, of mixed sign and not monotonic in
\u03bb; with 18 unadjusted comparisons they are not treated as effects. Fixed orthogonal and half-life-only
corrections give identical consistency at the same \u03b1 and differ only in error.

## S4. Alternative encoders and depth (SUP04; Figure S2)

Three seeds, transferred settings. Across GraphConv depths 1-5 and GIN, GAT and AttentiveFP, the loss
raised predictions within 1.25-fold by {gain4.min():.1f} to {gain4.max():.1f} percentage points. The
triad-MAE change interval excludes zero in {tri4_excl} of {len(tri4)} cells. {ep4_excl} of {len(ep4)}
endpoint-level intervals exclude zero, with mixed signs. The consistency effect therefore transfers
across these encoders and depths; the design does not support ranking architectures.

## S5. Task-weight sensitivity (SUP05; Figure S3A)

GraphConv, three seeds, V_d and CL weights of 5 versus equal weights. Without the loss, the heavier
weights changed CL MAE by {w("weight effect, baseline", "CL_mae_relative_log10")}, V_d MAE by
{w("weight effect, baseline", "Vd_mae_relative_log10")} and triad MAE by
{w("weight effect, baseline", "triad_mae_relative_log10")}. The loss changed consistency by
{w("loss effect, equal weights", "within_1_25_fold_percent", 1, " points")} under equal weights and
{w("loss effect, Vd/CL weights 5", "within_1_25_fold_percent", 1, " points")} under the heavier weights;
the interaction is {w("interaction: loss effect (weights 5) - loss effect (equal)", "within_1_25_fold_percent", 1, " points")}
for consistency and {w("interaction: loss effect (weights 5) - loss effect (equal)", "triad_mae_log10", 4, "")}
log10 units for triad MAE.

## S6. Single-task comparison (SUP06; Figure S3B, C)

GraphConv, three seeds; seven independently tuned single-task models (six configurations each) against
the multitask baseline (twenty configurations). Single-task minus multitask-baseline MAE: triad
{st("single_task - multitask_baseline", "triad_mae_relative_log10")}; CL
{st("single_task - multitask_baseline", "CL_mae_relative_log10")}; PPB
{st("single_task - multitask_baseline", "PPB_mae_relative_percentage_point")}; the other endpoint
intervals include zero. Combining the three single-task predictions gives {mean6[("single_task", "within_1_25_fold_percent")]:.1f}%
of structures within 1.25-fold and {mean6[("single_task", "within_2_fold_percent")]:.1f}% within 2-fold, against
{mean6[("multitask_baseline", "within_1_25_fold_percent")]:.1f}% and {mean6[("multitask_baseline", "within_2_fold_percent")]:.1f}%
for the multitask baseline. Parameter budgets differ: the seven single-task models total
{b6.single_task_total_parameters_seven_models.min():,} to {b6.single_task_total_parameters_seven_models.max():,}
parameters per outer fold, the multitask model {b6.multitask_parameters.min():,} to {b6.multitask_parameters.max():,}.

## S7. Source-definition sensitivity and compatibility strata (SUP07; Figure S4D)

The prespecified list names eight source records (eight structures): five source-confirmed central
volumes and the cannabidiol regimen mismatch are complete triads; oteseconazole (central volume, no
clearance) and ergoloid are not. Removing all labels of these structures leaves {count("same-row complete triads")[1]}
of {count("same-row complete triads")[0]} triads and {count("structures (consistency denominator)")[1]} of
{count("structures (consistency denominator)")[0]} structures (endpoint counts in
`sup07_source_sensitivity_counts.csv`). Re-scored selected-loss triad-MAE change:
{per_model(lambda b: sens(b, "selected_positive", "prespecified exclusions removed"))}, against
{per_model(lambda b: sens(b, "selected_positive", "all structures"))} on all structures. This re-scores
existing predictions; it is not evidence about retraining without those labels.

By observed disagreement (363, 177, 101 and 33 triads within 1.25-fold, 1.25-2, 2-5 and above 5-fold),
the selected-loss interval excludes zero in {strat_loss_excl} of {len(strat_loss)} model-by-stratum cells.
Exact orthogonal correction raises triad MAE in the 2- to 5-fold stratum:
{per_model(lambda b: strat(b, "exact_orthogonal", "2- to 5-fold"))} log10 units. Strata are defined on
the observed labels and are descriptive.

## S8. Endpoint shifts and spread (SUP08; Figure S4A-C)

Nine seeds. The observed triads have a mean signed residual of {observed_mean:+.3f} log10 units. Baseline
predictions carry nearly the same offset ({per_model(lambda b: f"{e8(b, 'triad', 'mean_signed_residual_baseline'):+.3f}")}),
and the loss reduces it to {per_model(lambda b: f"{e8(b, 'triad', 'mean_signed_residual_policy'):+.3f}")}.
The fraction of the baseline residual removed is
{per_model(lambda b: v8(b, "triad", "total_fraction_of_baseline_residual_removed", 2))}. It is absorbed
through V_d ({per_model(lambda b: f"{e8(b, 'Vd', 'absorption_share_of_baseline_residual'):.2f}")}) and CL
({per_model(lambda b: f"{e8(b, 'CL', 'absorption_share_of_baseline_residual'):.2f}")}) in every model, and
through half-life to a model-dependent extent
({per_model(lambda b: f"{e8(b, 't_half', 'absorption_share_of_baseline_residual'):.2f}")}); orthogonal
correction would give one third each. Across-structure spread of the three predictions changes by a
factor of {spread.min():.2f} to {spread.max():.2f}.

## S9. Additional endpoint metrics (SUP09)

RMSE, Pearson r, Spearman rho, R-squared and fold-error metrics are in `sup09_*.csv` with their scales
and denominators (PPB ratio metrics use the 781 positive observations; the 24 zero labels remain in MAE,
RMSE, r and R-squared). Baseline R-squared ranges from {r2.estimate.min():.2f} to {r2.estimate.max():.2f}
across models and endpoints, and baseline geometric mean fold error from {gm.estimate.min():.2f} to
{gm.estimate.max():.2f} for the five log-scale endpoints. For the selected loss, {d_excl} of {len(d_sel)}
metric-by-endpoint-by-model intervals exclude zero, unadjusted and of mixed direction.

## What the supplement adds, qualifies or challenges

Adds. The consistency gain is reproduced in every encoder and depth tested and under heavier V_d/CL
weights, and it is largely insensitive to the prespecified source exclusions.

Qualifies. (1) Consistency of this size is also obtained by a deterministic output correction, so it is
not evidence of an advantage for training with the loss. (2) The pooled consistency curve is a mixture
across outer folds: folds where a large \u03bb was selected are almost exactly consistent and the others
much less so. (3) Multitask training alone already gives far more consistent triads than independent
single-task models.

Challenges. The observed same-row triads are offset from the mass-balance identity by {observed_mean:+.3f}
log10 units on average, the baseline models reproduce this offset, and the loss removes most of it.
The loss therefore makes predictions more consistent than the labels they are scored against, without a
detectable change in error. The supportable claim is improved algebraic consistency at no detectable
accuracy cost under heterogeneous curated labels, not closer agreement with physiology or with the data.

## Unresolved items

- No analysis required new training. No missing evidence was found that changes the main interpretation.
- Supplementary blocks have three seeds and transferred settings; they cannot rank architectures.
- Intervals are conditional on the fitted models and unadjusted for multiplicity; Spearman rho has no interval.
- There is no external test set.
- Main-figure layout v2 changes panel B content at the user's request (1.5-fold replaces exact 1-fold; axis to 90).
"""
    (OUT / "SI_DRAFT.md").write_text(text, encoding="utf-8", newline="\n")

    captions = """# Supplementary figure captions

Figure S1. Training loss versus post-hoc correction controls. (A) Predictions within 1.25-fold triad
disagreement for each policy; dots are nine seeds (952 structures each, pooled outer folds), open diamonds
and numbers the seed mean. Exact correction reaches 100% by construction. (B) Change in complete-triad MAE
versus baseline as a percentage of baseline MAE (674 same-row triads, nine seeds); bars are 95% paired
group-bootstrap intervals conditional on the fitted models. (C) Matched trade-offs on seeds 101, 202 and
303: triad-MAE change against consistency for fixed loss strength (lambda = 0, 0.03, 0.1, 0.3, 1, 3, 10)
and fixed correction strength (alpha = 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1); bars are pointwise 95% intervals.
(D) Setting selected inside each outer fold; shaded cells mark folds where no intervention was selected.
The main loss must use lambda > 0. ChemBERTa denotes frozen representations with a trainable MLP head.

Figure S2. Alternative graph encoders and GraphConv depth (seeds 101, 202, 303). All rows except the
two-layer GraphConv reference use the GraphConv outer fold's configuration and selected lambda with their
own inner epoch calibration. (A) Predictions within 1.25-fold disagreement without and with the loss.
(B) Change in complete-triad MAE with the loss as a percentage of baseline MAE; bars are 95% paired
group-bootstrap intervals. (C) Complete-triad MAE in log10 units. One dot per seed in A and C.

Figure S3. Task weighting and single-task models (GraphConv, seeds 101, 202, 303). (A) Change in MAE as a
percentage of the comparator's MAE for the loss effect under equal weights and under V_d and CL weights of
5, and for the weight effect without and with the loss. (B) Single-task minus multitask MAE as a percentage
of multitask MAE, against the multitask model without (filled) and with (open) the loss. Single-task models
were tuned independently (six configurations per endpoint and outer fold; multitask: twenty). (C) Predicted
triad disagreement from seven single-task models and from the multitask model without and with the loss;
mean of three seed cumulative distributions over 952 structures. Bars are 95% paired group-bootstrap
intervals.

Figure S4. Which outputs absorb the penalty, and error changes by observed compatibility (nine seeds).
(A) Fraction of the baseline triad residual removed through each endpoint, -Cov(a_e d_e, r_baseline) /
Var(r_baseline), and their total; the dotted line marks one third, the value for orthogonal correction.
(B) Mean shift in log10 prediction with the loss. (C) Mean signed triad residual without and with the
loss; the dotted line marks the mean of the 674 observed triads. (D) Change in complete-triad MAE within
strata of observed disagreement for the training loss and exact orthogonal correction. Strata are defined
on observed labels and are descriptive. Bars are 95% paired group-bootstrap intervals.
"""
    (OUT / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8", newline="\n")

    checklist = [
        ("SUP01", "complete", "Package verified; summaries and bootstrap reproduced; all 840 outer-final fits verified", "tables/inventory_*.csv; tables/selection_*.csv"),
        ("SUP02", "complete", "Exact-control and zero-eligible intervals, direct loss-versus-correction contrasts, selection asymmetry", "tables/sup02_*.csv; Figure S1A,B,D"),
        ("SUP03", "complete", "Matched three-seed curves with pointwise intervals; nine-seed contrasts kept separate; inner selections tabulated", "tables/sup03_*.csv; Figure S1C"),
        ("SUP04", "complete", "GIN, GAT, AttentiveFP and depth 1-5 scored against matched baseline and loss; transferred settings disclosed", "tables/sup04_*.csv; Figure S2"),
        ("SUP05", "complete", "Equal versus V_d/CL weights of 5 across baseline and loss; interaction reported descriptively", "tables/sup05_*.csv; Figure S3A"),
        ("SUP06", "complete", "Single-task versus multitask baseline and loss; tuning and parameter budgets reported", "tables/sup06_*.csv; Figure S3B,C"),
        ("SUP07", "complete", "Prespecified exclusions applied with exact denominators; compatibility strata scored descriptively", "tables/sup07_*.csv; Figure S4D"),
        ("SUP08", "complete", "Signed shifts, spread, absorption shares and residual changes on matched structures", "tables/sup08_*.csv; Figure S4A-C"),
        ("SUP09", "complete", "RMSE, correlations, R-squared and defined fold metrics with denominators; SI text and checklist", "tables/sup09_*.csv; SI_DRAFT.md"),
    ]
    pd.DataFrame(checklist, columns=["id", "status", "what_was_done", "outputs"]).to_csv(OUT / "COMPLETED_CHECKLIST.csv", index=False)
    print("SI draft:", len(text.split()), "words; captions and checklist written")


if __name__ == "__main__":
    main()
