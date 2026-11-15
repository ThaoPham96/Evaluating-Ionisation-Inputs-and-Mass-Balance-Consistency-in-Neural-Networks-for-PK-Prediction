# Reproducing the analyses

Run all commands from the repository root. Python 3.11 or newer is sufficient for data reconstruction and point-metric scoring; these commands use only its standard library. Docker with a Linux amd64 engine is additionally required for scientific dependencies, rendering and training.

```sh
python scripts/pipeline.py validate
python scripts/pipeline.py rescore
```

The first invocation expands 22 losslessly compressed CSV tables to their original locations. Compressed and expanded SHA-256 hashes are checked. Existing changed tables cause an error rather than being overwritten. The expanded paths and generated outputs are ignored by Git; the tracked gzip files contain all required bytes, so no external data download is needed.

`validate` reconstructs 952 structures from reviewed source rows and two endpoint identity masks, checks 552 groups and all nested splits for the 952-, 327- and 494-structure cohorts, and compares exact archived descriptor/class memberships. It independently recomputes 3,618 historical point metrics from 281,934 saved prediction rows. `rescore` additionally writes `outputs/rescore/metrics.csv`, including computed and archived values and their differences. The report is `outputs/rescore/validation.json`. These commands do not claim to recalculate pending revision intervals.

## Scientific runtime

```sh
python scripts/build_runtime.py
python scripts/pipeline.py preflight
```

The build uses a public PyTorch base pinned by digest and 57 hash-pinned overlay wheels. No private registry account or cloud credentials are required. The build verifies 169 Python distribution versions, all 144 conda artifact URLs, and ten pinned operating-system package versions. The conda explicit list records the base environment; it is not reinstalled on top of the base. The pip overlay was derived from the actual installed inventories and does not treat a raw pip listing as a portable lock.

The local image tag is `pipknet-reproduction:20261114-v1`. A clean build needs approximately 17 GiB of available Docker-storage capacity including the 5 GiB reserve; a cached base reduces the prerequisite. `--storage-path` identifies the host volume containing Docker storage when it differs from the default. The script does not remove unrelated images or change host settings. Public package repositories must be reachable during the build; analysis containers run with networking disabled.

The upstream base has one known `pip check` wheel-tag diagnostic for Ninja 1.11.1.1. The installed Ninja executable runs successfully. The environment verifier accepts only that exact independently reproduced upstream diagnostic and rejects additional dependency errors. Equal package versions do not prove full binary equivalence with the historical GPU environment. Historical mass fits used Python 3.11.9; unified fits used Python 3.11.10. Exact GPU fitting replay also depends on hardware and numerical libraries; independent saved-prediction scoring does not require a GPU.

## Figures

```sh
python scripts/pipeline.py figures --current-base --output outputs/figures-current-base
python scripts/pipeline.py figures --historical --output outputs/figures-historical
```

`--current-base` regenerates current Figures 1, 2, 3, S1 and S2 from retained source tables and drawing functions. Figures 2 and 3 use the revised primary whole-group bootstrap intervals in `Review_statistics_20261114/outputs/contrasts.csv`; the renderer verifies every displayed endpoint/average estimate and bound against that table. The other three figures retain their existing scientific values. The public runtime falls back to DejaVu Sans when Arial is unavailable; two S1 labels use shorter equivalent wording so the layout checks pass. Numerical coordinates are preserved, but binary-identical PDF files are not promised across fonts and renderers.

The explicit `--historical` command reruns the archived unified builder, including 10,000 bootstrap draws, and regenerates historical Figures 2, 3 and S2 plus plotted tables. Its intervals precede the revised manuscript analysis. Neither option includes the pending robustness figure. The default `figures` command currently stops until that final bundle is integrated. See `reproducibility/RELEASE_STATUS.json` for status.

## Full statistics replay

```sh
python scripts/pipeline.py statistics --output outputs/statistics
```

This command reconstructs the checkpoint-free mass-fit archive from three tracked chunks, checks every chunk/member hash, and replays all historical primary and supplementary predictions with the revised 10,000-draw bootstrap and algebraic endpoint-derivation analyses. It then compares the archived intervals and generates evidence tables. Extraction needs about 0.5 GiB of additional disk space and creates many small files; the first run can take several minutes. Subsequent runs verify existing files before use. All computation remains local, in the pinned container. `--draws 20` is an optional diagnostic run; it deliberately omits the exact manuscript interval comparison.

The compact replay keeps original requests, completion markers, publication markers, predictions, selection records, scalers and summaries. It excludes checkpoint weights, duplicate assignment copies, operational environment receipts and learning curves. Original checkpoint hashes remain in completion markers. The reconstruction does not reinterpret archived predictions as fresh fits under the public packaging identity.

## Local training

```sh
python scripts/pipeline.py train --unit GraphConv-outer1 --device cuda --output outputs/training --dry-run
python scripts/pipeline.py train --unit GraphConv-outer1 --device cuda --output outputs/training
```

The first command prints the container invocation without starting training. The second runs a real historical unified nested-selection unit using the frozen inputs and existing trainer. `preflight` lists all 15 units and validates the frozen plan. A unit includes inner selection and final fits; it is not a small demonstration. GPU training requires NVIDIA Container Toolkit and a supported GPU; `--device cpu` is available but substantially slower. The wrapper starts local Docker only. Training writes to the requested output directory, with the source/data mounted read-only. No trained checkpoint weights are required to rescore the supplied predictions.

The repository includes selection, configuration, lambda, epoch and fit-ID metadata for 8,923 historical fits. Checkpoint SHA-256 values are provenance only; weights are intentionally not distributed, and no download service is promised. A compact raw-prediction/provenance archive supports the full statistics replay. Training replay is separate from reconstruction/scoring validation.

## Integrity and packaging

`reproducibility/manifest.json` covers the hydrated scientific payload; `artifacts.json` maps compressed tables to their public bytes. `.gitattributes` disables end-of-line rewriting for the checksum-covered snapshot. `PACKAGING.json` records excluded ancillary files and metadata changes. Operational host/job/token fields and author workspace prefixes were removed from metadata. Six tables also use a neutral reviewer display field. `PUBLIC_METADATA_IDENTITY.json` records original/public hashes and verifies that all other cells and row/column order are unchanged; the public cohort fingerprints and dependent hash constants were refrozen. Scientific labels, predictions, historical fit IDs and original artifact hashes were preserved. Fresh public training produces its own campaign identity. Original archival hashes in sanitized fit metadata describe the original files, not the sanitized JSON representation.

The new robustness campaign remains pending until its full export and final statistics are independently verified. Historical results and pending analyses are separate in the release status. Source-specific terms are described in [data-terms.md](data-terms.md) and [NOTICE.md](../NOTICE.md).
