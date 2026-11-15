"""Validate complete combined-campaign exports and score the matched factorial."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from combined_features import FINGERPRINT, FOLDS, load_splits
from vendor_ionisation import ionisation_score as prior
from vendor_ionisation.ionisation_data import read_release, sha256
from vendor_ionisation.vendor_mass_balance.mass_balance_controls import (
    ALPHA_GRID,
    partial_correction,
    residual_log10,
)

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = prior.BACKBONES
SEEDS = prior.SEEDS
N_BOOTSTRAP = prior.N_BOOTSTRAP
BOOTSTRAP_SEED = prior.BOOTSTRAP_SEED
ENDPOINTS = ("t_half", "Vd", "CL", "Cmax", "Tmax", "F", "PPB")
RAW_ENDPOINTS = tuple(dict(zip(prior.ENDPOINTS, prior.RAW_ENDPOINTS))[ep] for ep in ENDPOINTS)
LABEL_COUNTS = (316, 259, 246, 190, 249, 172, 292)
CELLS = ("00", "10", "01", "11")
ARMS = ("P0", "P1")
CELL_FOR_ARM_LOSS = dict(zip((("P0", False), ("P1", False), ("P0", True), ("P1", True)), CELLS))
LAMBDAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
TOLERANCE = 1e-8
PRED_COLUMNS = [f"q_{ep}" for ep in ENDPOINTS]
TRUE_COLUMNS = [f"true_{ep}" for ep in ENDPOINTS]
OBS_COLUMNS = [f"observed_{ep}" for ep in ENDPOINTS]
TRIAD_FLAG = "same_row_triad_eligible"
CENTRAL_FLAG = "source_definition_sensitivity_exclude"
TRIAD_ID_SHA256 = "e419d0ee8fb2f369f23cb2ab739e05bc7358ba5c6971e3412ca57683b2ef78fe"
TRIAD = "triad_macro_log10_mae"
SENSITIVITY = "triad_macro_log10_mae_excluding_central"
RESIDUAL = "mean_abs_log10_residual"
ACCURACY_INTERACTION = (TRIAD, "Vd_mae", "CL_mae")
CONTROL_METHODS = ("orthogonal", "half_life")
EXPORT_FILES = ("final_predictions.csv", "inner_p1_zero_predictions.csv",
                "inner_grid_predictions.csv", "selections.json")


@dataclass(frozen=True)
class Cohort:
    rows: pd.DataFrame
    truth: np.ndarray
    triads: np.ndarray
    central: np.ndarray
    group: np.ndarray
    inner: dict[int, dict[str, int]]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={name: "string" for name in ("structure_id", "group_id", "cell", "fit_id")},
                       keep_default_na=False, na_values=[""], float_precision="round_trip")


def ordered_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame["structure_id"].isna().any():
        raise ValueError("Missing structure ID")
    return frame.set_index("structure_id", verify_integrity=True).sort_index()


def load_cohort(release: Path, triads_path: Path) -> Cohort:
    _, release_manifest = read_release(release, "numerical")
    outer, inner, groups = load_splits(release)
    rows = ordered_index(read_csv(release / "model_dataset.csv"))
    triad_rows = ordered_index(read_csv(triads_path))
    if (release_manifest["fingerprint"] != FINGERPRINT or len(rows) != 327
            or set(rows.index) != outer.keys() or len(set(groups.values())) != 238
            or len(triad_rows) != 242 or not triad_rows.index.isin(rows.index).all()
            or hashlib.sha256("\n".join(triad_rows.index).encode()).hexdigest() != TRIAD_ID_SHA256):
        raise ValueError("Expected the frozen 327 structures, 238 groups and 242 same-row triads")
    if (not np.array_equal(rows["outer_fold"], rows.index.map(outer))
            or not np.array_equal(rows["group_id"], rows.index.map(groups))
            or not rows["ionisation_class"].isin(prior.CLASSES).all()):
        raise ValueError("Frozen release labels and split assignments disagree")
    source_columns = ["group_id", "outer_fold", "volume_definition", CENTRAL_FLAG, *RAW_ENDPOINTS[:3]]
    if (not rows.loc[triad_rows.index, source_columns].equals(triad_rows[source_columns])
            or not triad_rows[list(RAW_ENDPOINTS[:3])].notna().all().all()
            or triad_rows["group_id"].nunique() != 181
            or tuple(triad_rows.groupby("outer_fold").size().reindex(FOLDS)) != (48, 48, 57, 47, 42)
            or not rows[CENTRAL_FLAG].isin([True, False]).all()):
        raise ValueError("Frozen same-row triad truth or fold assignments disagree with the numerical release")
    truth = rows[list(RAW_ENDPOINTS)].to_numpy(dtype=float, copy=True)
    if (tuple(np.isfinite(truth).sum(axis=0)) != LABEL_COUNTS or np.isinf(truth).any()
            or np.any(truth[:, :5] <= 0) or np.any((truth[:, 5:] < 0) | (truth[:, 5:] > 100))
            or np.count_nonzero(truth[:, -1] == 0) != 8):
        raise ValueError("Frozen endpoint counts or domains changed")
    truth[:, :5] = np.log10(truth[:, :5])
    triads = rows.index.isin(triad_rows.index)
    central = rows[CENTRAL_FLAG].to_numpy(dtype=bool)
    if (np.count_nonzero(triads & central) != 4
            or not np.array_equal(central, rows["volume_definition"].eq("source_confirmed_central"))):
        raise ValueError("Expected four source-confirmed central-volume triads")
    _, group = np.unique(rows["group_id"], return_inverse=True)
    return Cohort(rows, truth, triads, central, group, inner)


def read_export(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (not manifest.get("campaign_id") or manifest.get("inner_fits") != 1050
            or manifest.get("final_fits") != 540):
        raise ValueError("Expected a complete 1,590-fit combined campaign export")
    for filename in EXPORT_FILES:
        if sha256(path / filename) != manifest.get("files", {}).get(filename):
            raise ValueError(f"Campaign export checksum mismatch for {filename}")
    frames = [read_csv(path / filename) for filename in EXPORT_FILES[:3]]
    for key, frame in zip(("final_rows", "inner_p1_zero_rows", "inner_grid_rows"), frames):
        if manifest.get(key) != len(frame):
            raise ValueError(f"Campaign export row count mismatch for {key}")
    selections = json.loads((path / "selections.json").read_text(encoding="utf-8"))
    return manifest, *frames, selections


def validate_prediction_rows(frame: pd.DataFrame, cohort: Cohort) -> pd.DataFrame:
    keys = ["structure_id", "group_id", "backbone", "outer_fold", "inner_fold", "seed", "cell",
            "feature_arm", "lambda_value", "fit_id", "ionisation_class", TRIAD_FLAG, CENTRAL_FLAG]
    prior.require_columns(frame, [*keys, *PRED_COLUMNS, *TRUE_COLUMNS, *OBS_COLUMNS], "predictions")
    frame = frame.loc[:, [*keys, *PRED_COLUMNS, *TRUE_COLUMNS, *OBS_COLUMNS]].copy()
    positions = cohort.rows.index.get_indexer(frame["structure_id"])
    truth = cohort.truth[positions]
    if (np.any(positions < 0) or frame["fit_id"].isna().any()
            or not frame["backbone"].isin(BACKBONES).all() or not frame["cell"].isin(CELLS).all()
            or not frame["feature_arm"].isin(ARMS).all() or not frame["lambda_value"].isin(LAMBDAS).all()
            or not frame["outer_fold"].isin(FOLDS).all() or frame["group_id"].isna().any()):
        raise ValueError("Prediction rows contain unsupported fit metadata or structure IDs")
    groups_and_classes = ["group_id", "ionisation_class"]
    if (not np.array_equal(frame[groups_and_classes], cohort.rows.iloc[positions][groups_and_classes])
            or not np.array_equal(frame[TRIAD_FLAG], cohort.triads[positions])
            or not np.array_equal(frame[CENTRAL_FLAG], cohort.central[positions])
            or not np.array_equal(frame[OBS_COLUMNS].to_numpy(), np.isfinite(truth))
            or not np.allclose(frame[TRUE_COLUMNS].to_numpy(dtype=float), truth,
                               rtol=0, atol=1e-12, equal_nan=True)):
        raise ValueError("Exported prediction labels, groups, masks or classes disagree with the frozen cohort")
    q = frame[PRED_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(q).all() or np.any((q[:, 5:] < 0) | (q[:, 5:] > 100)):
        raise ValueError("Exported predictions violate endpoint domains")
    frame[PRED_COLUMNS] = q
    expected_cells = [CELL_FOR_ARM_LOSS[arm, lam > 0]
                      for arm, lam in zip(frame["feature_arm"], frame["lambda_value"])]
    if not np.array_equal(frame["cell"], expected_cells):
        raise ValueError("Cell and feature/loss settings disagree")
    return frame


def validate_fit_ids(frame: pd.DataFrame, keys: list[str], expected: int) -> None:
    fits = frame[[*keys, "fit_id"]].drop_duplicates()
    if len(fits) != expected or fits["fit_id"].duplicated().any() or fits.duplicated(keys).any():
        raise ValueError("Expected one distinct fit ID per planned fit")


def validate_inner(grid: pd.DataFrame, p1_zero: pd.DataFrame, cohort: Cohort) -> pd.DataFrame:
    grid = validate_prediction_rows(grid, cohort)
    p1_zero = validate_prediction_rows(p1_zero, cohort)
    if not grid["seed"].eq(42).all() or not grid["inner_fold"].isin(FOLDS).all():
        raise ValueError("Inner grid requires frozen folds and seed 42")
    keys = ["backbone", "outer_fold", "feature_arm", "lambda_value"]
    validate_fit_ids(grid, [*keys, "inner_fold"], 1050)
    combinations = grid.groupby(keys, sort=False)
    if combinations.ngroups != len(BACKBONES) * len(FOLDS) * len(ARMS) * len(LAMBDAS):
        raise ValueError("Missing inner-grid combinations")
    for (backbone, outer, arm, lam), part in combinations:
        expected = cohort.inner[outer]
        if (part["structure_id"].duplicated().any() or set(part["structure_id"]) != expected.keys()
                or not np.array_equal(part["inner_fold"], part["structure_id"].map(expected))):
            raise ValueError(f"Incorrect inner OOF coverage for {(backbone, outer, arm, lam)}")
    sort = ["backbone", "outer_fold", "structure_id"]
    expected_zero = grid.loc[grid["cell"].eq("10")].sort_values(sort).reset_index(drop=True)
    if not p1_zero.sort_values(sort).reset_index(drop=True).equals(expected_zero):
        raise ValueError("Dedicated P1 lambda-zero predictions disagree with the full inner grid")
    return grid


def select_smaller(scores: dict[float, float]) -> float:
    best = min(scores.values())
    return min(candidate for candidate, score in scores.items() if score <= best + TOLERANCE)


def audit_selections(grid: pd.DataFrame, selections: list, campaign_id: str) -> tuple[dict, list[dict]]:
    expected = {(b, fold) for b in BACKBONES for fold in FOLDS}
    if not isinstance(selections, list) or len(selections) != len(expected):
        raise ValueError("Expected 15 complete selection records")
    lookup = {(record["backbone"], record["outer_fold"]): record for record in selections}
    if set(lookup) != expected:
        raise ValueError("Selection records do not cover the planned backbones and outer folds")
    eligible = grid[TRIAD_FLAG].astype(bool)
    keys = ["backbone", "outer_fold", "lambda_value", "feature_arm"]
    triad_names = list(ENDPOINTS[:3])
    errors = np.abs(grid.loc[eligible, PRED_COLUMNS[:3]].to_numpy()
                    - grid.loc[eligible, TRUE_COLUMNS[:3]].to_numpy(dtype=float))
    scored_rows = grid.loc[eligible, keys].copy()
    scored_rows[triad_names] = errors
    score_groups = scored_rows.groupby(keys)
    inner_scores, inner_counts = score_groups[triad_names].mean(), score_groups.size()
    fit_ids = grid.groupby([*keys, "inner_fold"])["fit_id"].first()
    inner_sizes = grid.groupby(["backbone", "outer_fold", "lambda_value", "feature_arm"]).size()
    result, audit = {}, []
    for (backbone, outer), record in lookup.items():
        key = (backbone, outer, 0.0, "P1")
        if (record["campaign_id"] != campaign_id or record["n_observed_triads"] != inner_counts.loc[key]
                or record["n_validation"] != inner_sizes.loc[key]):
            raise ValueError(f"Selection record metadata disagrees with the inner grid for {(backbone, outer)}")
        declared_grid = {entry["lambda_value"]: entry for entry in record["grid"]}
        if len(record["grid"]) != len(LAMBDAS) or set(declared_grid) != set(LAMBDAS):
            raise ValueError(f"Invalid declared lambda grid for {(backbone, outer)}")
        mean_scores = {}
        for lam, declared in declared_grid.items():
            if set(declared["arms"]) != set(ARMS):
                raise ValueError("Lambda record must contain both feature arms")
            arm_scores = [float(inner_scores.loc[backbone, outer, lam, arm].mean()) for arm in ARMS]
            mean_scores[lam] = float(np.mean(arm_scores))
            if (not np.isclose(mean_scores[lam], declared["mean_score"], rtol=0, atol=1e-10)
                    or not np.allclose(arm_scores, [declared["arms"][arm]["score"] for arm in ARMS],
                                       rtol=0, atol=1e-10)):
                raise ValueError(f"Inner OOF scores do not reproduce recorded selection for {(backbone, outer, lam)}")
            for arm, score in zip(ARMS, arm_scores):
                key = (backbone, outer, lam, arm)
                arm_record = declared["arms"][arm]
                epochs = arm_record["epochs"]
                best_epochs = [entry["best_epoch"] for entry in epochs]
                endpoint_mae = inner_scores.loc[key]
                if (len(epochs) != len(FOLDS) or {entry["inner_fold"] for entry in epochs} != set(FOLDS)
                        or any(type(epoch) is not int or not 1 <= epoch <= 200 for epoch in best_epochs)
                        or arm_record["median_epoch"] != round(float(np.median(best_epochs)))
                        or any(entry["fit_id"] != fit_ids.loc[(*key, entry["inner_fold"])] for entry in epochs)
                        or not np.allclose(endpoint_mae, [arm_record["endpoint_log10_mae"][ep] for ep in triad_names],
                                           rtol=0, atol=1e-10)):
                    raise ValueError(f"Inner epoch or endpoint-score metadata disagree for {key}")
                audit.append(dict(backbone=backbone, outer_fold=outer, lambda_value=lam, feature_arm=arm,
                                  inner_triad_mae=score, jointly_averaged_mae=mean_scores[lam],
                                  n_inner_triads=int(inner_counts.loc[key]), best_epochs=json.dumps(epochs),
                                  median_epoch=arm_record["median_epoch"],
                                  **{f"inner_{ep}_mae": endpoint_mae[ep] for ep in triad_names}))
        selected = select_smaller({lam: mean_scores[lam] for lam in LAMBDAS[1:]})
        zero_eligible = select_smaller({0.0: mean_scores[0.0], selected: mean_scores[selected]})
        if (record["selected_lambda"], record["zero_eligible_lambda"]) != (selected, zero_eligible):
            raise ValueError(f"Inner OOF predictions do not reproduce selected lambdas for {(backbone, outer)}")
        result[backbone, outer] = (selected, zero_eligible)
    for row in audit:
        selected, zero_eligible = result[row["backbone"], row["outer_fold"]]
        row.update(selected_positive=row["lambda_value"] == selected,
                   selected_zero_eligible=row["lambda_value"] == zero_eligible)
    return result, audit


def validate_final(frame: pd.DataFrame, cohort: Cohort, selections: dict) -> dict[str, np.ndarray]:
    frame = validate_prediction_rows(frame, cohort)
    if not frame["inner_fold"].isna().all() or not frame["seed"].isin(SEEDS).all():
        raise ValueError("Final predictions must use the nine final seeds and no inner fold")
    validate_fit_ids(frame, ["backbone", "outer_fold", "cell", "seed"], 540)
    expected_index = pd.MultiIndex.from_product((BACKBONES, CELLS, SEEDS, cohort.rows.index),
                                               names=["backbone", "cell", "seed", "structure_id"])
    ordered = frame.set_index(expected_index.names, verify_integrity=True).reindex(expected_index)
    if len(frame) != len(ordered) or ordered["fit_id"].isna().any():
        raise ValueError("Final predictions must cover every structure once per backbone, cell and seed")
    expected_folds = np.tile(cohort.rows["outer_fold"], len(BACKBONES) * len(CELLS) * len(SEEDS))
    if not np.array_equal(ordered["outer_fold"], expected_folds):
        raise ValueError("Final predictions differ from the frozen outer folds")
    # Both positive cells must use the same independently reproduced strength per outer fold.
    expected_lambdas = np.concatenate([
        np.tile([selections[backbone, int(fold)][0] if cell[1] == "1" else 0.0
                 for fold in cohort.rows["outer_fold"]], len(SEEDS))
        for backbone in BACKBONES for cell in CELLS
    ])
    if not np.array_equal(ordered["lambda_value"], expected_lambdas):
        raise ValueError("Final prediction lambda differs from the shared inner-selected lambda")
    q = ordered[PRED_COLUMNS].to_numpy(dtype=float)
    q = q.reshape(len(BACKBONES), len(CELLS), len(SEEDS), len(cohort.rows), len(ENDPOINTS))
    return dict(zip(BACKBONES, q))


def zero_eligible_predictions(q: np.ndarray, cohort: Cohort, selections: dict, backbone: str) -> dict[str, np.ndarray]:
    """Reuse a zero-loss cell in folds where the inner sensitivity selects zero."""
    by_cell = {cell: values for cell, values in zip(CELLS, q)}
    folds = cohort.rows["outer_fold"].to_numpy()
    use_zero = np.isin(folds, [fold for fold in FOLDS if selections[backbone, fold][1] == 0])
    by_cell["zero_eligible_01"] = np.where(use_zero[None, :, None], by_cell["00"], by_cell["01"])
    by_cell["zero_eligible_11"] = np.where(use_zero[None, :, None], by_cell["10"], by_cell["11"])
    return by_cell


def control_predictions(
    q: np.ndarray, inner: pd.DataFrame, cohort: Cohort, selections: dict, backbone: str,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Fit correction strengths on the P1 zero-loss inner OOF triads, then apply by fold."""
    by_cell = zero_eligible_predictions(q, cohort, selections, backbone)
    folds = cohort.rows["outer_fold"].to_numpy()
    alpha_audit = []
    for method in CONTROL_METHODS:
        tuned = by_cell["10"].copy()
        exact = by_cell["10"].copy()
        exact[..., :3] = partial_correction(exact[..., :3], 1.0, method)
        for fold in FOLDS:
            eligible = inner.loc[inner["outer_fold"].eq(fold) & inner[TRIAD_FLAG].astype(bool)]
            p = eligible[PRED_COLUMNS[:3]].to_numpy()
            y = eligible[TRUE_COLUMNS[:3]].to_numpy(dtype=float)
            scores = {alpha: float(np.abs(partial_correction(p, alpha, method) - y).mean())
                      for alpha in ALPHA_GRID}
            chosen = select_smaller(scores)
            alpha_audit.extend(dict(backbone=backbone, outer_fold=fold, method=method, alpha=alpha,
                                    inner_triad_mae=score, n_inner_triads=len(eligible), selected=alpha == chosen)
                               for alpha, score in scores.items())
            held_out = folds == fold
            tuned[:, held_out, :3] = partial_correction(tuned[:, held_out, :3], chosen, method)
        by_cell[f"{method}_selected"] = tuned
        by_cell[f"{method}_exact"] = exact
    return by_cell, alpha_audit


def crossed_means(values: np.ndarray, group: np.ndarray, draws: prior.BootstrapDraws) -> np.ndarray:
    """Cross whole-group multiplicities with whole-seed multiplicities.

    Each cell's pooled endpoint denominator is the number of eligible structures
    in the group draw. The final average weights the resampled seed trajectories equally.
    """
    n_cells, n_seeds, n_structures = values.shape
    n_groups = draws.group_counts.shape[1]
    counts = np.bincount(group, minlength=n_groups)
    group_sums = np.column_stack([
        np.bincount(group, weights=row, minlength=n_groups)
        for row in values.reshape(n_cells * n_seeds, n_structures)
    ])
    totals = draws.group_counts @ np.column_stack((counts, group_sums))
    denominators = totals[:, 0]
    if np.any(denominators == 0):
        raise ValueError("A crossed draw contained no eligible structures")
    by_seed = totals[:, 1:].reshape(len(totals), n_cells, n_seeds) / denominators[:, None, None]
    return np.einsum("dcs,ds->cd", by_seed, draws.seed_counts) / draws.seed_counts.sum(axis=1)


def summary_stats(values: np.ndarray) -> dict:
    defined = values[~np.isnan(values)]
    return dict(
        mean=float(defined.mean()) if len(defined) else np.nan,
        seed_sd=float(defined.std(ddof=1)) if len(defined) > 1 and np.isfinite(defined).all() else np.nan,
        n_seeds_defined=len(defined),
    )


def endpoint_descriptors(q: np.ndarray, y: np.ndarray, percentage: bool) -> dict[str, np.ndarray]:
    """Per-seed SI descriptors on the same scale as endpoint MAE, except fold error."""
    ss_total = np.square(y - y.mean()).sum()
    r2 = 1 - np.square(q - y).sum(axis=1) / ss_total if ss_total > 0 else np.full(len(q), np.nan)
    rho = np.array([spearmanr(y, values).statistic if ss_total > 0 and np.ptp(values) > 0 else np.nan
                    for values in q])
    if percentage:
        positive = y > 0
        with np.errstate(divide="ignore", over="ignore"):
            aafe = np.exp(np.abs(np.log(q[:, positive]) - np.log(y[positive])).mean(axis=1))
    else:
        aafe = np.power(10.0, np.abs(q - y).mean(axis=1))
    return {"r2": r2, "spearman": rho, "aafe": aafe}


def metrics(
    predictions: np.ndarray, cohort: Cohort,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, str]], np.ndarray, np.ndarray]:
    residual = residual_log10(predictions[..., :3])
    with np.errstate(over="ignore"):
        disagreement = np.power(10.0, np.abs(residual))
    if not np.isfinite(disagreement).all():
        raise ValueError("Triad fold disagreement exceeds the finite scoring domain")
    errors = np.abs(predictions - cohort.truth)
    triad_errors = errors[..., :3].mean(axis=-1)
    all_structures = np.ones(len(cohort.rows), dtype=bool)
    outcomes = {
        TRIAD: (triad_errors, cohort.triads, "log10"),
        SENSITIVITY: (triad_errors, cohort.triads & ~cohort.central, "log10"),
        RESIDUAL: (np.abs(residual), all_structures, "log10"),
    }
    for index, endpoint in enumerate(ENDPOINTS):
        outcomes[f"{endpoint}_mae"] = (errors[..., index], np.isfinite(cohort.truth[:, index]),
                                       "log10" if index < 5 else "percentage_points")
    return outcomes, residual, disagreement


def contrast_specifications(metric: str, *, posthoc: bool = True) -> list[tuple[str, dict[str, int], str | None]]:
    simple_effects = (("10", "00"), ("01", "00"), ("11", "00"), ("11", "10"), ("11", "01"))
    contrasts = [(f"{a}_minus_{b}", {a: 1, b: -1},
                  "primary" if (a, b) == ("11", "10") and metric in (TRIAD, RESIDUAL) else None)
                 for a, b in simple_effects]
    interaction = {"00": 1, "10": -1, "01": -1, "11": 1}
    if metric in (*ACCURACY_INTERACTION, RESIDUAL, SENSITIVITY):
        family = "secondary_accuracy" if metric in ACCURACY_INTERACTION else None
        contrasts.append(("interaction", interaction, family))
        contrasts.extend([
            ("zero_eligible_01_minus_00", {"zero_eligible_01": 1, "00": -1}, None),
            ("zero_eligible_11_minus_10", {"zero_eligible_11": 1, "10": -1}, None),
            ("zero_eligible_interaction", {"00": 1, "10": -1, "zero_eligible_01": -1,
                                           "zero_eligible_11": 1}, None),
        ])
    if posthoc and metric in (*ACCURACY_INTERACTION, RESIDUAL, SENSITIVITY, "t_half_mae"):
        for method in CONTROL_METHODS:
            for strength in ("selected", "exact"):
                cell = f"{method}_{strength}"
                contrasts.append((f"{cell}_minus_10", {cell: 1, "10": -1}, None))
                contrasts.append((f"11_minus_{cell}", {"11": 1, cell: -1}, None))
    return contrasts


def add_seed_metric(
    tables: dict, meta: dict, values: np.ndarray, bootstrap: np.ndarray | None = None,
) -> None:
    stats = summary_stats(values)
    if bootstrap is not None:
        stats["crossed95_low"], stats["crossed95_high"] = prior.interval(bootstrap)
    tables["metrics"].append(dict(meta, **stats))
    tables["seed_metrics"].extend(dict(meta, seed=seed, value=value) for seed, value in zip(SEEDS, values))


def add_contrasts(
    tables: dict, meta: dict, cells: list[str], by_seed: np.ndarray, by_draw: np.ndarray,
    *, posthoc: bool = True,
) -> None:
    """Apply the same contrast to paired seed values and within each crossed draw."""
    baseline = float(by_seed[cells.index("00")].mean())
    relative_scale = 100.0 / baseline if baseline > 0 else np.nan
    for name, weights, family in contrast_specifications(meta["metric"], posthoc=posthoc):
        coefficients = np.array([weights.get(cell, 0) for cell in cells])
        seed_values = coefficients @ by_seed
        draw_values = coefficients @ by_draw
        lo, hi = prior.interval(draw_values)
        point = summary_stats(seed_values)
        record = dict(meta, contrast=name, family=family, **point, baseline_00_mean=baseline,
                      relative_change_pct=relative_scale * point["mean"],
                      crossed95_low=lo, crossed95_high=hi,
                      relative_crossed95_low_pct=relative_scale * lo,
                      relative_crossed95_high_pct=relative_scale * hi,
                      requested_draws=len(draw_values), valid_draws=int(np.isfinite(draw_values).sum()))
        for cell in cells:
            record[f"mean_{cell}"] = float(by_seed[cells.index(cell)].mean())
        if family is not None:
            level = 1 - 0.05 / (6 if family == "primary" else 9)
            lo, hi = prior.interval(draw_values, level)
            direction = "lower" if hi < 0 else "higher" if lo > 0 else "unresolved"
            record.update(adjusted_level=level, adjusted_low=lo, adjusted_high=hi,
                          relative_adjusted_low_pct=relative_scale * lo,
                          relative_adjusted_high_pct=relative_scale * hi,
                          adjusted_direction=direction)
        tables["contrasts"].append(record)
        tables["seed_contrasts"].extend(dict(meta, contrast=name, family=family, seed=seed, value=value)
                                        for seed, value in zip(SEEDS, seed_values))


def cdf_rows(backbone: str, cell: str, disagreement: np.ndarray) -> pd.DataFrame:
    """All distinct support points of the mean of the nine seed empirical CDFs."""
    values, counts = np.unique(disagreement, return_counts=True)
    return pd.DataFrame(dict(backbone=backbone, cell=cell, disagreement_fold=values,
                             mean_seed_cdf=np.cumsum(counts) / disagreement.size,
                             n_structures_per_seed=disagreement.shape[1], n_seeds=disagreement.shape[0]))


def add_disagreement(
    tables: dict, backbone: str, cell: str, residual: np.ndarray, disagreement: np.ndarray, cohort: Cohort,
) -> None:
    n_structures = len(cohort.rows)
    meta = dict(backbone=backbone, cell=cell, n_structures=n_structures,
                n_groups=len(np.unique(cohort.group)), unit="fold")
    for quantile, name in ((0.5, "median_fold_disagreement"), (0.9, "p90_fold_disagreement")):
        add_seed_metric(tables, dict(meta, metric=name), np.quantile(disagreement, quantile, axis=1))
    tables["disagreement_cdf"].append(cdf_rows(backbone, cell, disagreement))
    if cell in CELLS:
        categories = cohort.rows["ionisation_class"].to_numpy()
        tables["signed_residuals"].append(pd.DataFrame(dict(
            backbone=backbone, cell=cell, seed=np.repeat(SEEDS, n_structures),
            structure_id=np.tile(cohort.rows.index, len(SEEDS)),
            group_id=np.tile(cohort.rows["group_id"], len(SEEDS)),
            outer_fold=np.tile(cohort.rows["outer_fold"], len(SEEDS)),
            ionisation_class=np.tile(categories, len(SEEDS)),
            signed_residual_log10=residual.ravel(), disagreement_fold=disagreement.ravel(),
        )))
        for category in prior.CLASSES:
            included = categories == category
            signed = residual[:, included]
            subgroup = dict(meta, ionisation_class=category, unit="log10",
                            n_structures=int(included.sum()), n_groups=len(np.unique(cohort.group[included])))
            add_seed_metric(tables, dict(subgroup, metric="mean_signed_log10_residual"), signed.mean(axis=1))
            add_seed_metric(tables, dict(subgroup, metric="median_signed_log10_residual"), np.median(signed, axis=1))


def score_predictions(
    final: dict[str, np.ndarray], inner: pd.DataFrame, cohort: Cohort, selections: dict,
    *, posthoc: bool = False, draws: prior.BootstrapDraws | None = None,
) -> dict[str, pd.DataFrame]:
    tables = {name: [] for name in ("metrics", "seed_metrics", "contrasts", "seed_contrasts",
                                    "disagreement_cdf", "signed_residuals", "alpha_selection")}
    if draws is None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws = prior.group_draws(len(np.unique(cohort.group)), len(SEEDS), N_BOOTSTRAP, rng)
    for backbone, q in final.items():
        if posthoc:
            p1_zero = inner.loc[inner["backbone"].eq(backbone) & inner["cell"].eq("10")]
            by_cell, alpha = control_predictions(q, p1_zero, cohort, selections, backbone)
            tables["alpha_selection"].extend(alpha)
        else:
            by_cell = zero_eligible_predictions(q, cohort, selections, backbone)
        cells = list(by_cell)
        predictions = np.stack(list(by_cell.values()))
        outcomes, residual, disagreement = metrics(predictions, cohort)
        for name, (values, included, unit) in outcomes.items():
            values = values[..., included]
            groups = cohort.group[included]
            meta = dict(backbone=backbone, metric=name, unit=unit,
                        n_structures=int(included.sum()), n_groups=len(np.unique(groups)))
            by_seed = values.mean(axis=-1)
            by_draw = crossed_means(values, groups, draws)
            for i, cell in enumerate(cells):
                add_seed_metric(tables, dict(meta, cell=cell), by_seed[i], by_draw[i])
            add_contrasts(tables, meta, cells, by_seed, by_draw, posthoc=posthoc)
        for i, cell in enumerate(cells):
            add_disagreement(tables, backbone, cell, residual[i], disagreement[i], cohort)
            for j, ep in enumerate(ENDPOINTS):
                mask = np.isfinite(cohort.truth[:, j])
                y = cohort.truth[mask, j]
                descriptors = endpoint_descriptors(predictions[i][:, mask, j], y, percentage=j >= 5)
                for descriptor, values in descriptors.items():
                    scored = mask if descriptor != "aafe" or j < 5 else mask & (cohort.truth[:, j] > 0)
                    unit = "fold" if descriptor == "aafe" else "dimensionless"
                    meta = dict(backbone=backbone, cell=cell, metric=f"{ep}_{descriptor}", unit=unit,
                                n_structures=int(scored.sum()), n_groups=len(np.unique(cohort.group[scored])))
                    add_seed_metric(tables, meta, values)
    tables["disagreement_cdf"] = pd.concat(tables["disagreement_cdf"], ignore_index=True)
    tables["signed_residuals"] = pd.concat(tables["signed_residuals"], ignore_index=True)
    return {name: pd.DataFrame(rows) for name, rows in tables.items() if name != "alpha_selection" or posthoc}


def run(export_dir: Path, output_dir: Path, release_dir: Path, triads: Path) -> None:
    cohort = load_cohort(release_dir, triads)
    manifest, final, inner_p1_zero, inner_grid, selection_records = read_export(export_dir)
    if set(final["fit_id"]) & set(inner_grid["fit_id"]):
        raise ValueError("Final and inner fits must have distinct fit IDs")
    inner = validate_inner(inner_grid, inner_p1_zero, cohort)
    selections, lambda_audit = audit_selections(inner, selection_records, manifest["campaign_id"])
    predictions = validate_final(final, cohort, selections)
    with threadpool_limits(limits=2, user_api="blas"):
        tables = score_predictions(predictions, inner, cohort, selections)
    tables["lambda_selection"] = pd.DataFrame(lambda_audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").unlink(missing_ok=True)
    checksums = {}
    for name, table in tables.items():
        destination = output_dir / f"{name}.csv"
        table.to_csv(destination, index=False)
        checksums[destination.name] = sha256(destination)
    summary = dict(
        status="complete", campaign_id=manifest["campaign_id"], release_fingerprint=FINGERPRINT,
        plan_sha256=manifest.get("plan_sha256"),
        export_manifest_sha256=sha256(export_dir / "manifest.json"), source_triads_sha256=sha256(triads),
        scorer_sha256=sha256(Path(__file__)), final_rows=len(final), inner_grid_rows=len(inner_grid),
        fit_counts={"inner": manifest["inner_fits"], "final": manifest["final_fits"]},
        bootstrap_draws=N_BOOTSTRAP, bootstrap_seed=BOOTSTRAP_SEED, files=checksums,
    )
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, default=ROOT / "data" / "release" / "numerical")
    parser.add_argument("--triads", type=Path, default=ROOT / "runtime" / "same_row_complete_triads.csv")
    args = parser.parse_args()
    if args.export_dir.resolve() == args.output_dir.resolve():
        parser.error("Use a separate directory for scoring outputs")
    run(args.export_dir, args.output_dir, args.release_dir, args.triads)


if __name__ == "__main__":
    main()
