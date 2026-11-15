# Reproducing the analyses

Run all commands from the repository root. Python 3.11 or newer is sufficient for data reconstruction and point-metric scoring; these commands use only its standard library. Docker with a Linux amd64 engine is additionally required for scientific dependencies, rendering and training.

```sh
python scripts/pipeline.py validate
python scripts/pipeline.py rescore
```

The first invocation expands the losslessly compressed CSV tables to their original locations. Compressed and expanded SHA-256 hashes are checked. Existing changed tables cause an error rather than being overwritten. The expanded paths and generated outputs are ignored by Git; the tracked gzip files contain all required bytes, so no external data download is needed.

`validate` reconstructs 952 structures from reviewed source rows and two endpoint identity masks, checks 552 groups and all nested splits for the 952-, 327- and 494-structure cohorts, and compares exact archived descriptor/class memberships. It independently recomputes 3,618 historical point metrics from 281,934 saved prediction rows. `rescore` additionally writes `outputs/rescore/metrics.csv`, including computed and archived values and their differences. The report is `outputs/rescore/validation.json`. These commands cover point metrics; use `statistics` for the full revised intervals and matrix.

## Scientific runtime

```sh
python scripts/build_runtime.py
python scripts/pipeline.py preflight
python scripts/pipeline.py validate-robustness
```

The build uses a public PyTorch base pinned by digest and 57 hash-pinned overlay wheels. No private registry account or cloud credentials are required. The build verifies 169 Python distribution versions, all 144 conda artifact URLs, and ten pinned operating-system package versions. The conda explicit list records the base environment; it is not reinstalled on top of the base. The pip overlay was derived from the actual installed inventories and does not treat a raw pip listing as a portable lock.

The local image tag is `pipknet-reproduction:20261114-v1`. A clean build needs approximately 17 GiB of available Docker-storage capacity including the 5 GiB reserve; a cached base reduces the prerequisite. `--storage-path` identifies the host volume containing Docker storage when it differs from the default. The script does not remove unrelated images or change host settings. Public package repositories must be reachable during the build; analysis containers run with networking disabled.

The upstream base has one known `pip check` wheel-tag diagnostic for Ninja 1.11.1.1. The installed Ninja executable runs successfully. The environment verifier accepts only that exact independently reproduced upstream diagnostic and rejects additional dependency errors. Equal package versions do not prove full binary equivalence with the historical GPU environment. Historical mass fits used Python 3.11.9; unified fits used Python 3.11.10. Exact GPU fitting replay also depends on hardware and numerical libraries; independent saved-prediction scoring does not require a GPU.

## Figures

```sh
python scripts/pipeline.py figures --output outputs/figures
python scripts/pipeline.py figures --current-base --output outputs/figures-current-base
python scripts/pipeline.py figures --historical --output outputs/figures-historical
```

The default command regenerates Figures 1, 2, 3, S1, S2 and S3. Figure 1C displays the unchanged saved cumulative distributions on a logarithmic axis through fivefold disagreement, with reference guides and mean percentages at 1.5-, 2- and 5-fold. The 18 displayed values are exported as `fig1/predicted_fold_thresholds.csv`; evaluation uses the empirical step values without interpolation. Figure S3 uses the complete verified F/T matrix statistics, including all prespecified input, loss and interaction contrasts. Its 96 displayed rows are saved beside the figure.

`--current-base` regenerates current Figures 1, 2, 3, S1 and S2 from retained source tables and drawing functions. Figures 2 and 3 use the revised primary whole-group bootstrap intervals in `Review_statistics_20261114/outputs/contrasts.csv`; the renderer verifies every displayed endpoint/average estimate and bound against that table. The other three figures retain their existing scientific values. The public runtime falls back to DejaVu Sans when Arial is unavailable; two S1 labels use shorter equivalent wording so the layout checks pass. Numerical coordinates are preserved, but binary-identical PDF files are not promised across fonts and renderers.

The explicit `--historical` command reruns the archived unified builder, including 10,000 bootstrap draws, and regenerates historical Figures 2, 3 and S2 plus plotted tables. Its intervals precede the revised manuscript analysis. `--current-base` selects the five current figures preceding S3. See `reproducibility/RELEASE_STATUS.json` for validation status.

## Full statistics replay

```sh
python scripts/pipeline.py statistics --output outputs/statistics
```

This command reconstructs the checkpoint-free mass-fit archive from three tracked chunks, checks every chunk/member hash, and replays all historical and new primary/supplementary predictions with the revised 10,000-draw bootstrap and algebraic endpoint-derivation analyses. It produces 9,520 contrasts: 5,428 earlier contrasts and 4,092 new matrix contrasts. It compares the archived intervals, verifies all 141,264 same-input reference identities and independently checks the new arithmetic. Extraction needs about 0.5 GiB of additional disk space and creates many small files; the first run can take several minutes. Subsequent runs verify existing files before use. All computation remains local, in the pinned container. `--historical` selects the earlier 5,428-contrast analysis. `--draws 20` is an optional diagnostic run; it deliberately omits the exact manuscript interval comparison.

The compact replay keeps original requests, completion markers, publication markers, predictions, selection records, scalers and summaries. It excludes checkpoint weights, duplicate assignment copies, operational environment receipts and learning curves. Original checkpoint hashes remain in completion markers. The reconstruction does not reinterpret archived predictions as fresh fits under the public packaging identity.

`validate-robustness` independently checks the three compressed fit-provenance bundles for all 6,105 new fits. Original raw request, summary, scaler and prediction bytes are checked against their completion markers; all 60 architecture/lambda/epoch selections are reconstructed exactly using the archived scientific plan. Every final exported prediction is compared with its source fit, including reused F references. The command reports the omitted checkpoint/learning-curve scope explicitly. It uses Linux numerical libraries because exact selection replay can differ by a final floating-point bit on other platforms.

## Training metadata and matched calibration

```sh
python scripts/pipeline.py hydrate
python reproducibility/current/build_training_metadata.py --output outputs/training-metadata
```

This command requires NumPy and pandas (`python -m pip install numpy==1.26.4 pandas==2.3.3` in a separate Python 3.11 environment, matching the scientific runtime). It runs locally on CPU and rebuilds and verifies the 30 fixed-capacity epoch-policy records, 90 training-partition records, 525 paired F/T P0 calibration summaries and 15 selection comparisons in Supplementary Data 3 and 4. It performs no model fitting. The compact `current/paired_calibration.jsonl.gz` archive adds the learning curves for these paired comparisons, including their original byte hashes and the scientific request/summary fields used for comparison. The broader fit-provenance archives retain their documented omission of other learning curves.

The matched records agree in scientific requests, initial-state/data/scaler/environment hashes and batch/dropout streams. Chemprop and ChemBERTa trajectories are identical; GraphConv trajectories show numerical variation across the separate executions, which did not enforce deterministic GPU execution. The archive supports comparison of the recorded trajectories without attributing the variation to a particular GPU operator. F and T keep their own fitted reference models.

## Local training

```sh
python scripts/pipeline.py preflight --campaign robustness --device cpu --output outputs/robustness-preflight
python scripts/pipeline.py train --campaign robustness --unit GraphConv-outer1-P2 --device cuda --output outputs/robustness-training --dry-run
python scripts/pipeline.py train --campaign robustness --unit GraphConv-outer1-P2 --device cuda --output outputs/robustness-training
```

The robustness wrapper preserves architecture selection, positive-lambda selection, scaling, losses, stopping and split logic. It binds the actual Python/PyTorch/CUDA/device environment into a new campaign identity and every fit request. CPU and GPU refits therefore have explicit provenance and cannot be mistaken for the archived fits. The strict archived validator is retained. The F anchor predictions remain the exact original references; fresh T/R0/R2 fits are distinct results, whose numerical values can vary with hardware and runtime. The public CPU parity tests use tiny two-epoch fixtures; no full research refit is claimed by repository validation.

The other shipped training procedures have the same wrapper interface:

| `--campaign` | Inputs and training scope | Unit example |
|---|---|---|
| `mass` | Full 952-structure plan: 4,240 logical fits, including primary backbones, GIN/GAT/AttentiveFP, depth, task weighting and single-task comparisons | The coordinator processes the complete dependency graph |
| `class` | 494-structure class encodings and their tuning comparisons, 15 outer/model units | `class-GraphConv-outer1` |
| `historical` | Original 327-structure unified categorical/pH/loss campaign, 15 units and 2,430 fits | `GraphConv-outer1` |
| `robustness` | Four inputs with input-specific tuning and the F/R0/R2 bridges, 60 units | `GraphConv-outer1-P2` |

```sh
python scripts/pipeline.py preflight --campaign mass --output outputs/mass-training
python scripts/pipeline.py train --campaign mass --device cuda --output outputs/mass-training
python scripts/pipeline.py preflight --campaign class --output outputs/class-training
python scripts/pipeline.py train --campaign class --unit class-GraphConv-outer1 --device cuda --output outputs/class-training
python scripts/pipeline.py train --unit GraphConv-outer1 --device cuda --output outputs/training --dry-run
python scripts/pipeline.py train --unit GraphConv-outer1 --device cuda --output outputs/training
```

`--dry-run` prints the container invocation without starting training. `preflight` initializes or verifies the corresponding plan without fitting research models. The full mass plan initialized successfully with all 4,240 logical fits; 19 CPU trainer tests covered all six backbones, masking, single-task training and artifact validation. The class plan and 19 CPU trainer tests passed; six robustness parity/scaling tests and its 3,000 architecture-request split checks passed. Historical and mass coordinators retain their original nominal request identities where inputs are unchanged; actual runtime and resolved-artifact hashes remain recorded in each new fit directory. Always use a separate output directory for new fitting.

A unit includes inner selection and final fits; it is not a small demonstration. The mass worker processes the full dependency graph and can run much longer than one outer unit. GPU training requires NVIDIA Container Toolkit and a supported GPU; `--device cpu` is available but substantially slower. The wrapper starts local Docker only. Training writes to the requested output directory, with the source/data mounted read-only. No trained checkpoint weights are required to rescore supplied predictions.

The repository includes selection, configuration, lambda, epoch and fit-ID metadata for 8,923 earlier fits plus 6,105 robustness fits. Checkpoint SHA-256 values are provenance only; weights are intentionally not distributed, and no download service is promised. Compact raw-prediction/provenance archives support scoring and selection replay. Training replay is separate from reconstruction/scoring validation.

## Integrity and packaging

`reproducibility/manifest.json` covers the hydrated scientific payload; `artifacts.json` maps compressed tables to their public bytes. `.gitattributes` disables end-of-line rewriting for the checksum-covered snapshot. `PACKAGING.json` records excluded ancillary files and metadata changes. Operational host/job/token fields and author workspace prefixes were removed from metadata. Eight tables also use a neutral reviewer display field. `PUBLIC_METADATA_IDENTITY.json` records original/public hashes and verifies that all other cells and row/column order are unchanged; the public cohort fingerprints and dependent hash constants were refrozen. Scientific labels, predictions, historical fit IDs and original artifact hashes were preserved. Fresh robustness refits explicitly bind the actual environment into new campaign/request identities. Original archival hashes in sanitized fit metadata describe the original files, not the sanitized JSON representation.

The robustness export retains archived campaign `2109d71c4127900d3ee9f1d6eb2acc9bf2469f9e46b77c9ed6958d7f66a74eb3` and the exact final-prediction SHA-256 in its manifest. Its public plan removes a private registry hostname while retaining the immutable image digest; the original plan hash and public identity mapping make that metadata transformation explicit. Source-specific terms are described in [data-terms.md](data-terms.md) and [NOTICE.md](../NOTICE.md).
