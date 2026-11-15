# Reporting the unified numerical-pKa campaign

From the workspace root on Windows, stage the figures after `analysis/collect_unified.py` has collected all 15 units:

```powershell
& revision_work/.venv/Scripts/python.exe -B Unified_ionisation_combined_20261114/reporting/build.py --export-dir <collector-destination>
```

The collector destination must contain `plan.json`, `manifest.json`, `combined/`, and `ionisation/`. The driver checks the pinned campaign identity, export and plan checksums, exact row and fit coverage, and the frozen numerical release and same-row triads. It validates the combined final fit allocations, reproduces the inner lambda selection, and requires exact equality of the pH 7.4 reference predictions and fit IDs between the combined `00` and `10` cells and both ionisation exports (`P0/P1` and `REF0/REF74`). Five positive endpoints use `pred_*`/`q_*` log10 predictions and F/PPB use `physical_*`/`q_*` percentages. All comparisons retain the original observed endpoint masks. Percent effects use fixed nine-seed mean comparator MAEs. The same 10,000 whole-group and whole-seed draws feed every endpoint; average-row intervals average the seven within-draw effects before taking percentiles. Separate direct comparisons to `REF74` use that reference's own fixed denominator.

The build publishes `reporting/output/` only after scoring, rendering and identity checks succeed. Use `--output-dir` with a new path inside `reporting/` to run again. Nothing writes to the earlier experiment or manuscript directories.

| Staged path | Contents |
| --- | --- |
| `fig2/ionisation_main.pdf` | Refreshed Figure 2. Older plotting filenames such as `panel_c_estimates.csv` still refer to panel D. `caption.txt` contains updated average values and cross-figure prediction provenance. |
| `fig3/combined_main.pdf` | Refreshed Figure 3 B/C with the accepted vector schematic and layout. `plot_data.csv` has 840 rows. |
| `figS2/ionisation_supplementary_selected.pdf` | Refreshed scatter and heatmap panels A/B/D. The two encoding files for panel C are byte-for-byte copies of the accepted 494-structure result. |
| `figure2_figure3_shared_ph74.csv` | Common figures' pH 7.4 results: 24 endpoint/average estimates and confidence intervals, and 216 seed values. Identity checks include all 21 endpoint normalizers. |
| `manuscript_claims.json` | Endpoint and average effects with intervals, paired changes on the common baseline scale, residual means and fold conversions, R², pH 2.0/8.0 comparisons, counts of intervals spanning zero, and selected lambdas. |
| `scores/` | Full factorial and pH contrast tables, seed metrics, adjusted factorial contrasts, lambda selection audit and residual data. |
| `manifest.json` | Source and output SHA-256 hashes plus the cross-figure identity and layout checks. A report records any heatmap values beyond the approved colour scale and any necessary widening of the residual axis. |

The same PDF directories contain PNG, SVG, caption and plotted CSV outputs. Shared Figure 2/3 pH 7.4 confidence bounds come from the factorial scorer once; Figure 2 B/D and S2 D also share those point and seed estimates. The Figure 3 panel C x coordinate retains each seed's variation about the fixed structure-only mean, as in the accepted plot. Figure 2 C and S2 A/B use seed-mean out-of-fold log predictions. The reporting adapter leaves the approved plot functions, cohort, labels, scheme and colours in place. The copied factorial scorer keeps zero-eligible lambda sensitivity; the unified report omits the auxiliary orthogonal and half-life-only post-hoc correction controls.

Run the disposable regression fixture with:

```powershell
& revision_work/.venv/Scripts/python.exe -B Unified_ionisation_combined_20261114/reporting/verify_archived_fixture.py
```

The fixture validates the collector-facing table contract, renders all three figures and checks that all 840 Figure 3 source rows reproduce the accepted archived values at absolute tolerance 1e-12. The fixture is not the unified rerun and does not leave any publication outputs.
