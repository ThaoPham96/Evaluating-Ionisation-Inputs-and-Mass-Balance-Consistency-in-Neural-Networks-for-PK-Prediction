"""Validate and score the frozen v2 ionisation outer predictions."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from threadpoolctl import threadpool_limits

ENDPOINTS = ("t_half", "Vd", "CL", "F", "PPB", "Cmax", "Tmax")
RAW_ENDPOINTS = (
    "t1/2(hour)", "VD(liter)", "Cl(liter/hour)", "F(percentage)",
    "PPB(percentage)", "Cmax_uM", "Tmax",
)
PERCENT = (3, 4)
POSITIVE = (0, 1, 2, 5, 6)
PRED_COLUMNS = tuple(f"pred_{name}" for name in ENDPOINTS)
PHYSICAL_COLUMNS = tuple(f"physical_{name}" for name in ENDPOINTS)
CLASSES = ("anionic", "cationic", "neutral", "zwitterionic")
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909)
BACKBONES = ("GraphConv", "Chemprop", "ChemBERTa_head")
FOLDS = (1, 2, 3, 4, 5)
COHORTS = ("class", "numerical")
ARMS = {"class": ("C0", "C1", "C2", "C3"), "numerical": ("P0", "P1", "P2", "P3", "P4", "P5")}
TUNED_ARMS = {"class": ("C2",), "numerical": ("P3", "P4")}
CONTROL_ARMS = {
    "class": ("overall_median", "class_median", "morgan_ridge", "morgan_class_ridge"),
    "numerical": ("overall_median", "class_median", "descriptor_ridge"),
}
RUN_COLUMNS = ("cohort", "procedure", "backbone", "arm")
ADJUSTED_LEVEL = 1.0 - 0.05 / 6
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20261113
LOG10_LN2 = float(np.log10(np.log(2)))


@dataclass(frozen=True)
class Predictions:
    seeds: tuple[int, ...]
    transformed: np.ndarray  # seed trajectory, sorted structure, endpoint
    physical: np.ndarray


@dataclass(frozen=True)
class BootstrapDraws:
    group_counts: np.ndarray
    seed_counts: np.ndarray


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"structure_id": "string", "group_id": "string"},
                       keep_default_na=False, na_values=[""], float_precision="round_trip")


def frozen_release_fingerprints(release: Path, cohorts, *, diagnostic: bool) -> dict[str, str]:
    if diagnostic:
        return {}
    from ionisation_data import read_release

    return {cohort: read_release(release / cohort, cohort)[1]["fingerprint"] for cohort in cohorts}


def checked_controls(path: Path, fingerprints: dict[str, str]) -> None:
    from ionisation_data import sha256

    manifest = json.loads(path.with_name("controls_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("release_fingerprints") != fingerprints:
        raise ValueError("Controls must be complete and match the selected frozen release fingerprints")
    if manifest.get("predictions_sha256") != sha256(path):
        raise ValueError("Control predictions differ from their completed manifest")
    if manifest.get("selection_sha256") != sha256(path.with_name("control_selection.csv")):
        raise ValueError("Control selection audit differs from its completed manifest")


def checked_campaign(path: Path, fingerprints: dict[str, str]) -> str | None:
    from ionisation_data import sha256

    sidecar = path.with_suffix(".manifest.json")
    if not sidecar.is_file():
        return None
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if manifest.get("sha256") != sha256(path):
        raise ValueError("Neural predictions differ from their campaign export manifest")
    if "release_fingerprints" in manifest and manifest["release_fingerprints"] != fingerprints:
        raise ValueError("Neural predictions are from a different frozen release")
    return manifest.get("campaign_id")


def require_columns(frame: pd.DataFrame, columns, name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns {missing}")


def integer_column(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="raise")
    if values.isna().any() or not np.isfinite(values).all() or (values % 1 != 0).any():
        raise ValueError(f"Expected integer {series.name} values")
    return values.astype(int)


def transform_targets(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(ENDPOINTS) or np.isinf(raw).any():
        raise ValueError("Expected seven PK columns with finite observed values")
    if np.any(raw[:, POSITIVE] <= 0) or np.any((raw[:, PERCENT] < 0) | (raw[:, PERCENT] > 100)):
        raise ValueError("Positive endpoint labels must exceed zero and percentages must be in [0, 100]")
    result = raw.copy()
    result[:, POSITIVE] = np.log10(raw[:, POSITIVE])
    clipped = np.clip(raw[:, PERCENT], 0.1, 99.9)
    result[:, PERCENT] = np.log10(clipped / (100 - clipped))
    return result


def inverse_physical(transformed: np.ndarray) -> np.ndarray:
    raw = np.asarray(transformed, dtype=float).copy()
    with np.errstate(over="ignore"):
        raw[..., POSITIVE] = np.power(10.0, raw[..., POSITIVE])
        logits = raw[..., PERCENT] * np.log(10.0)
        raw[..., PERCENT] = 100 * np.exp(-np.logaddexp(0, -logits))
    return raw


def load_cohort(release: Path, cohort: str, *, diagnostic: bool = False) -> pd.DataFrame:
    folder = release / cohort
    data = read_csv(folder / "model_dataset.csv")
    outer = read_csv(folder / "outer.csv")
    require_columns(data, ("structure_id", "ionisation_class", *RAW_ENDPOINTS), f"{cohort} data")
    require_columns(outer, ("structure_id", "group_id", "outer_fold"), f"{cohort} outer folds")
    for frame in (data, outer):
        if frame.empty or frame["structure_id"].isna().any() or frame["structure_id"].eq("").any() or frame["structure_id"].duplicated().any():
            raise ValueError(f"{cohort} structure IDs must be nonempty and unique")
    outer = outer.set_index("structure_id").sort_index()
    data = data.set_index("structure_id").sort_index()
    if not data.index.equals(outer.index):
        raise ValueError(f"{cohort} cohort data and outer-fold structure IDs differ")
    outer["outer_fold"] = integer_column(outer["outer_fold"])
    if outer["group_id"].isna().any() or outer.groupby("group_id")["outer_fold"].nunique().gt(1).any():
        raise ValueError(f"{cohort} groups must have one outer fold each")
    folds = set(outer["outer_fold"])
    if not diagnostic and folds != set(FOLDS):
        raise ValueError(f"{cohort} must contain the frozen outer folds {FOLDS}")
    for column in ("outer_fold", "group_id"):
        if column in data:
            if (data[column] != outer[column]).fillna(True).any():
                raise ValueError(f"{cohort} {column} disagrees between data and frozen outer folds")
        data[column] = outer[column]
    if not data["ionisation_class"].isin(CLASSES).all():
        raise ValueError(f"{cohort} contains an unsupported ionisation class")
    for endpoint in RAW_ENDPOINTS:
        data[endpoint] = pd.to_numeric(data[endpoint], errors="raise")
    transformed = transform_targets(data[list(RAW_ENDPOINTS)].to_numpy())
    if not diagnostic and not np.isfinite(transformed).any(axis=0).all():
        raise ValueError(f"{cohort} has an entirely unlabelled endpoint")
    if cohort == "numerical":
        require_columns(data, ("pka_acid", "pka_base"), f"{cohort} data")
        if not np.isfinite(data[["pka_acid", "pka_base"]].to_numpy(dtype=float)).all():
            raise ValueError("Numerical cohort requires two finite pKa values per structure")
    return data


def validate_nested_cohorts(cohorts: dict[str, pd.DataFrame]) -> None:
    if "numerical" not in cohorts or "class" not in cohorts:
        return
    class_data, numerical = cohorts["class"], cohorts["numerical"]
    if not numerical.index.isin(class_data.index).all():
        raise ValueError("Numerical structures must be a subset of the class cohort")
    common_columns = ["group_id", "outer_fold", "ionisation_class", *RAW_ENDPOINTS]
    common_columns.extend(column for column in ("canonical_smiles", "pka_acid", "pka_base")
                          if column in class_data and column in numerical)
    expected = class_data.loc[numerical.index, common_columns]
    actual = numerical[common_columns]
    if not (expected.eq(actual) | (expected.isna() & actual.isna())).fillna(False).all().all():
        raise ValueError("Shared class/numerical structures have inconsistent labels, classes, descriptors or folds")


def expected_runs(cohorts, *, control: bool) -> set[tuple[str, str, str, str]]:
    if control:
        return {(c, "control", "CPU", a) for c in cohorts for a in CONTROL_ARMS[c]}
    return {
        (c, procedure, backbone, arm)
        for c in cohorts
        for procedure, arms in (("primary", ARMS[c]), ("tuned", TUNED_ARMS[c]))
        for backbone in BACKBONES
        for arm in arms
    }


def validate_predictions(
    frame: pd.DataFrame,
    cohorts: dict[str, pd.DataFrame],
    *,
    diagnostic: bool = False,
    control: bool = False,
) -> dict[tuple[str, str, str, str], Predictions]:
    required = (*RUN_COLUMNS, "seed", "structure_id", "outer_fold", *PRED_COLUMNS, *PHYSICAL_COLUMNS)
    require_columns(frame, required, "outer predictions")
    frame = frame.loc[:, required].copy()
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Outer predictions must contain complete, nonempty rows")
    for column in ("seed", "outer_fold"):
        frame[column] = integer_column(frame[column])
    if frame.duplicated([*RUN_COLUMNS, "seed", "structure_id"]).any():
        raise ValueError("Duplicate held-out prediction for a structure and seed trajectory")
    known = expected_runs(cohorts, control=control)
    observed = set(frame[list(RUN_COLUMNS)].itertuples(index=False, name=None))
    if observed - known or (not diagnostic and known - observed):
        raise ValueError(f"Outer prediction run mismatch. Missing {sorted(known - observed)}. Unexpected {sorted(observed - known)}")
    aligned = {}
    cohort_seeds = {}
    for run, rows in frame.groupby(list(RUN_COLUMNS), sort=True):
        cohort, _, _, arm = run
        data = cohorts[cohort]
        seeds = tuple(sorted(rows["seed"].unique()))
        expected_seeds = (0,) if control else SEEDS
        if seeds != expected_seeds and (control or not diagnostic):
            raise ValueError(f"{run} seed trajectories must be exactly {expected_seeds}; found {seeds}")
        if seeds != cohort_seeds.setdefault(cohort, seeds):
            raise ValueError(f"{run} has different seed trajectories from its cohort comparators")
        index = pd.MultiIndex.from_product([seeds, data.index], names=["seed", "structure_id"])
        rows = rows.set_index(["seed", "structure_id"]).sort_index()
        if not rows.index.equals(index):
            raise ValueError(f"{run} must contain every cohort structure exactly once per seed")
        if not np.array_equal(rows["outer_fold"].to_numpy(), np.tile(data["outer_fold"], len(seeds))):
            raise ValueError(f"{run} has a prediction on a different fold from the frozen held-out assignment")
        shape = (len(seeds), len(data), len(ENDPOINTS))
        pred = rows[list(PRED_COLUMNS)].to_numpy(dtype=float).reshape(shape)
        physical = rows[list(PHYSICAL_COLUMNS)].to_numpy(dtype=float).reshape(shape)
        if not (np.isfinite(pred).all() and np.isfinite(physical).all()):
            raise ValueError(f"{run} contains nonfinite predictions")
        if np.any(physical[..., POSITIVE] <= 0) or np.any((physical[..., PERCENT] < 0) | (physical[..., PERCENT] > 100)):
            raise ValueError(f"{run} physical predictions fall outside the endpoint domains")
        if control and arm in {"overall_median", "class_median"}:
            valid_scale = np.allclose(pred[0], transform_targets(physical[0]), rtol=1e-6, atol=1e-7)
        else:
            valid_scale = np.allclose(physical, inverse_physical(pred), rtol=1e-6, atol=0)
        if not valid_scale:
            raise ValueError(f"{run} transformed and physical predictions disagree")
        aligned[run] = Predictions(seeds, pred, physical)
    return aligned


def errors(pred: Predictions, raw: np.ndarray, transformed: np.ndarray) -> np.ndarray:
    absolute = np.abs(pred.transformed - transformed)
    absolute[..., PERCENT] = np.abs(pred.physical[..., PERCENT] - raw[..., PERCENT])
    return absolute


def group_draws(n_groups: int, n_seeds: int, n_draws: int, rng: np.random.Generator) -> BootstrapDraws:
    groups = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n_draws).astype(float)
    seeds = rng.multinomial(n_seeds, np.full(n_seeds, 1 / n_seeds), size=n_draws)
    return BootstrapDraws(groups, seeds)


def paired_bootstrap(delta: np.ndarray, group: np.ndarray, draws: BootstrapDraws) -> tuple[np.ndarray, np.ndarray]:
    """Apply common group and seed counts to paired seed-by-structure errors."""
    present, membership, group_counts = np.unique(group, return_inverse=True, return_counts=True)
    group_sums = np.column_stack([
        np.bincount(membership, weights=row, minlength=len(present)) for row in delta
    ])
    sampled = draws.group_counts if len(present) == draws.group_counts.shape[1] else draws.group_counts[:, present]
    totals = sampled @ np.column_stack((group_counts, group_sums))
    denom, numer = totals[:, 0], totals[:, 1:]
    by_seed = np.full(numer.shape, np.nan)
    np.divide(numer, denom[:, None], out=by_seed, where=denom[:, None] > 0)
    crossed = np.einsum("rs,rs->r", by_seed, draws.seed_counts) / draws.seed_counts.sum(axis=1)
    return crossed, by_seed.mean(axis=1)


def interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    if not len(valid):
        return np.nan, np.nan
    tail = (1 - level) / 2
    return tuple(np.quantile(valid, [tail, 1 - tail]))


def interval_direction(bounds: tuple[float, float]) -> str:
    low, high = bounds
    if not np.isfinite(bounds).all():
        return "unavailable"
    if high < 0:
        return "lower_error"
    if low > 0:
        return "higher_error"
    return "unresolved"


def seed_summary(delta: np.ndarray) -> dict:
    if not len(delta) or not np.isfinite(delta).all():
        return dict(delta_mean=np.nan, seed_sd=np.nan, seed_min=np.nan, seed_max=np.nan, seeds_below_zero=0,
                    seeds_equal_zero=0, seeds_above_zero=0)
    return dict(delta_mean=delta.mean(), seed_sd=delta.std(ddof=1) if len(delta) > 1 else np.nan,
                seed_min=delta.min(), seed_max=delta.max(), seeds_below_zero=int((delta < 0).sum()),
                seeds_equal_zero=int((delta == 0).sum()), seeds_above_zero=int((delta > 0).sum()))


def contrast_statistics(arm_error: np.ndarray, base_error: np.ndarray, group: np.ndarray,
                        draws: BootstrapDraws, *, primary: bool) -> tuple[dict, np.ndarray]:
    if arm_error.shape != base_error.shape or not (np.isfinite(arm_error).all() and np.isfinite(base_error).all()):
        raise ValueError("Contrasts require the same finite scored observations in both arms")
    available = arm_error.shape[1] > 0
    delta = arm_error - base_error
    by_seed = delta.mean(axis=1) if available else np.full(delta.shape[0], np.nan)
    result = seed_summary(by_seed)
    arm_mae = arm_error.mean() if available else np.nan
    base_mae = base_error.mean() if available else np.nan
    result.update(arm_mae=arm_mae, comparator_mae=base_mae,
                  relative_delta_pct=100 * result["delta_mean"] / base_mae if base_mae > 0 else np.nan,
                  n_structures=delta.shape[1], n_groups=len(np.unique(group)), n_seeds=delta.shape[0])
    crossed, conditional = paired_bootstrap(delta, group, draws)
    crossed95, group95 = interval(crossed), interval(conditional)
    adjusted = interval(crossed, ADJUSTED_LEVEL) if primary else (np.nan, np.nan)
    seed95 = (np.nan, np.nan)
    df = len(by_seed) - 1
    if df and available:
        half_width = t.ppf(0.975, df) * result["seed_sd"] / np.sqrt(len(by_seed))
        seed95 = (result["delta_mean"] - half_width, result["delta_mean"] + half_width)
    directions = [interval_direction(bounds) for bounds in (crossed95, group95, seed95)]
    result.update(
        crossed95_low=crossed95[0], crossed95_high=crossed95[1],
        crossed_adjusted_low=adjusted[0], crossed_adjusted_high=adjusted[1],
        group95_low=group95[0], group95_high=group95[1],
        seed_t95_low=seed95[0], seed_t95_high=seed95[1], seed_t_df=df if df else np.nan,
        valid_bootstrap_draws=int(np.isfinite(crossed).sum()), requested_bootstrap_draws=len(crossed),
        primary_lower_error=interval_direction(adjusted) == "lower_error" if primary else None,
        crossed95_direction=directions[0], group95_direction=directions[1], seed_t95_direction=directions[2],
        interval_direction_disagreement=len(set(directions) - {"unavailable"}) > 1,
    )
    return result, by_seed


def vd_correlations(observed: np.ndarray, predicted: np.ndarray, n_groups: int, *, subgroup: bool,
                    class_median: bool) -> tuple[float, float]:
    if (subgroup and (len(observed) < 10 or n_groups < 5)) or len(observed) < 2:
        return np.nan, np.nan
    observed_variable = np.ptp(observed) > 0
    predicted_variable = np.ptp(predicted) > 0
    rho = np.nan
    if observed_variable and predicted_variable and not (subgroup and class_median):
        rho = float(spearmanr(observed, predicted).statistic)
    r2 = np.nan
    if observed_variable and (predicted_variable or not subgroup):
        r2 = float(1 - np.sum((observed - predicted) ** 2) / np.sum((observed - observed.mean()) ** 2))
    return rho, r2


def endpoint_metric(pred: Predictions, raw: np.ndarray, error: np.ndarray, seed: int, j: int) -> dict:
    observed = np.isfinite(raw[:, j])
    positive = observed & (raw[:, j] > 0)
    p = pred.physical[seed, positive, j]
    y = raw[positive, j]
    zero_predictions = int((p == 0).sum())
    if not len(y):
        aafe = np.nan
    elif zero_predictions:
        aafe = np.inf
    else:
        aafe = float(10 ** np.abs(np.log10(p) - np.log10(y)).mean())
    return dict(n_structures=int(observed.sum()), n_label_zeros=int((raw[:, j] == 0).sum()),
                n_aafe=int(positive.sum()), n_aafe_zero_predictions=zero_predictions,
                mae=float(error[seed, observed, j].mean()) if observed.any() else np.nan, aafe=aafe)


def evidence_strata(data: pd.DataFrame) -> dict[str, np.ndarray]:
    if "class_support" not in data:
        return {}
    support = data["class_support"]
    kinds = ("pka_acid_kind", "pka_base_kind")
    support_levels = ("complete_descriptors", "missing_side")
    if not support.isin(support_levels).all():
        raise ValueError("Unsupported or missing class_support value in the frozen class cohort")
    strata = {level: support.eq(level).to_numpy() for level in support_levels}
    if not set(kinds).issubset(data.columns):
        return strata
    sources = data[list(kinds)].fillna("")
    if not sources.isin(("experimental", "predicted", "")).all().all():
        raise ValueError("Unsupported pKa source kind in frozen evidence strata")
    experimental = sources.eq("experimental").any(axis=1).to_numpy()
    predicted = sources.eq("predicted").any(axis=1).to_numpy()
    by_source = {
        "experimental_only": experimental & ~predicted,
        "predicted_only": predicted & ~experimental,
        "mixed_experimental_predicted": experimental & predicted,
        "numeric_source_unavailable": ~experimental & ~predicted,
    }
    for level in support_levels:
        for source, mask in by_source.items():
            strata[f"{level}/{source}"] = strata[level] & mask
    return strata


def threshold_category(pka: np.ndarray) -> np.ndarray:
    index = (pka[:, 0] < 7.4).astype(int) + 2 * (pka[:, 1] > 7.4)
    return np.array(("neutral", "anionic", "cationic", "zwitterionic"))[index]


def coherence(pred: Predictions) -> tuple[np.ndarray, np.ndarray]:
    q = pred.transformed
    residual = q[..., 0] - q[..., 1] + q[..., 2] - LOG10_LN2
    with np.errstate(over="ignore"):
        fold = np.power(10.0, np.abs(residual))
    return residual, fold


def coherence_delta(arm: Predictions, comparator: Predictions, mask: np.ndarray) -> dict[str, np.ndarray]:
    arm_r, arm_fold = coherence(arm)
    base_r, base_fold = coherence(comparator)
    log_delta = arm.transformed[:, mask, :3] - comparator.transformed[:, mask, :3]
    signed = log_delta.mean(axis=1)
    arm_abs = np.abs(arm_r[:, mask]).mean(axis=1)
    base_abs = np.abs(base_r[:, mask]).mean(axis=1)
    arm_median_fold = np.median(arm_fold[:, mask], axis=1)
    base_median_fold = np.median(base_fold[:, mask], axis=1)
    return dict(delta_log_t_half=signed[:, 0], minus_delta_log_Vd=-signed[:, 1], delta_log_CL=signed[:, 2],
                delta_signed_log_residual=signed[:, 0] - signed[:, 1] + signed[:, 2],
                arm_abs_log_residual=arm_abs, comparator_abs_log_residual=base_abs,
                delta_abs_log_residual=arm_abs - base_abs, arm_median_fold_disagreement=arm_median_fold,
                comparator_median_fold_disagreement=base_median_fold,
                delta_median_fold_disagreement=arm_median_fold - base_median_fold)


def run_contrasts():
    contrasts = {
        "class": (("C2", "C0"), ("C1", "C0"), ("C2", "C1"), ("C3", "C2")),
        "numerical": (("P4", "P3"), ("P1", "P0"), ("P3", "P1"), ("P4", "P2"), ("P4", "P5")),
    }
    for cohort, pairs in contrasts.items():
        for backbone in BACKBONES:
            for i, (arm, baseline) in enumerate(pairs):
                yield (cohort, "primary", backbone, arm), (cohort, "primary", backbone, baseline), i == 0
            arm, baseline = pairs[0]
            comparator_procedure = "primary" if cohort == "class" else "tuned"
            yield (cohort, "tuned", backbone, arm), (cohort, comparator_procedure, backbone, baseline), False
    for cohort in COHORTS:
        overall = (cohort, "control", "CPU", "overall_median")
        class_median = (cohort, "control", "CPU", "class_median")
        yield class_median, overall, False
        for backbone in BACKBONES:
            for arm in (("C0", "C2") if cohort == "class" else ("P0", "P4")):
                yield (cohort, "primary", backbone, arm), class_median, False
    yield ("class", "control", "CPU", "morgan_class_ridge"), ("class", "control", "CPU", "morgan_ridge"), False
    yield ("numerical", "control", "CPU", "descriptor_ridge"), ("numerical", "control", "CPU", "overall_median"), False


def scope_masks(data: pd.DataFrame, cohort: str) -> list[tuple[str, str, np.ndarray]]:
    masks = [("all", "all", np.ones(len(data), dtype=bool))]
    masks.extend(("class", label, data["ionisation_class"].eq(label).to_numpy()) for label in CLASSES)
    if cohort == "class":
        masks.extend(("evidence", level, mask) for level, mask in evidence_strata(data).items())
    if cohort == "numerical":
        threshold = threshold_category(data[["pka_acid", "pka_base"]].to_numpy(dtype=float))
        agrees = data["ionisation_class"].to_numpy() == threshold
        masks.extend((("class_rule", "agrees_with_numerical_threshold", agrees),
                      ("class_rule", "differs_from_numerical_threshold", ~agrees)))
        if "permanent_charge" in data:
            permanent = data["permanent_charge"].isin(("positive", "negative", "both")).to_numpy()
            masks.append(("class_rule", "permanent_charge_with_threshold_disagreement", permanent & ~agrees))
        if "polyprotic" in data:
            masks.extend(("polyprotic_status", status, data["polyprotic"].eq(status).to_numpy())
                         for status in ("no", "yes", "uncertain"))
    return masks


def label_coverage(cohorts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, groups = [], []
    for cohort, data in cohorts.items():
        for (group, fold), count in data.groupby(["group_id", "outer_fold"], sort=True).size().items():
            groups.append(dict(cohort=cohort, group_id=group, outer_fold=fold, n_structures=count))
        folds = [("all", data), *data.groupby("outer_fold", sort=True)]
        for fold, partition in folds:
            for label in ("all", *CLASSES):
                part = partition if label == "all" else partition[partition["ionisation_class"] == label]
                for endpoint, raw in zip(ENDPOINTS, RAW_ENDPOINTS):
                    observed = part[raw].notna()
                    rows.append(dict(cohort=cohort, outer_fold=fold, ionisation_class=label, endpoint=endpoint,
                                     n_structures=len(part), n_groups=part["group_id"].nunique(),
                                     n_labels=int(observed.sum()), n_label_groups=part.loc[observed, "group_id"].nunique(),
                                     n_label_zeros=int(part[raw].eq(0).sum()), n_aafe=int(part[raw].gt(0).sum())))
    return pd.DataFrame(rows), pd.DataFrame(groups)


def metric_summary(table: pd.DataFrame, keys: list[str], counts: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, block in table.groupby(keys, sort=True):
        row = dict(zip(keys, key), n_seeds=len(block))
        row.update({column: block[column].iloc[0] for column in counts})
        for column in metrics:
            values = block[column].dropna().to_numpy()
            row[f"{column}_mean"] = values.mean() if len(values) else np.nan
            row[f"{column}_seed_sd"] = values.std(ddof=1) if len(values) > 1 and np.isfinite(values).all() else np.nan
            row[f"{column}_n_defined_seeds"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)


def score(cohorts: dict[str, pd.DataFrame], runs: dict, *, n_draws: int = N_BOOTSTRAP,
          bootstrap_seed: int = BOOTSTRAP_SEED, confirmatory: bool = True) -> dict[str, pd.DataFrame]:
    """Score validated arrays and reuse a cohort's resamples across comparisons."""
    rng = np.random.default_rng(bootstrap_seed)
    tables = {name: [] for name in (
        "endpoint_metrics", "within_class_vd", "coherence", "contrasts", "contrast_seeds",
        "contrast_outer_folds", "coherence_contrasts", "coherence_contrast_seeds",
    )}
    for cohort, data in cohorts.items():
        raw = data[list(RAW_ENDPOINTS)].to_numpy()
        transformed = transform_targets(raw)
        memberships, groups = pd.factorize(data["group_id"], sort=True)
        scopes = scope_masks(data, cohort)
        by_run = {key: pred for key, pred in runs.items() if key[0] == cohort}
        all_errors = {key: errors(pred, raw, transformed) for key, pred in by_run.items()}
        for key, pred in by_run.items():
            meta = dict(zip(RUN_COLUMNS, key))
            residual, fold = coherence(pred)
            for i, seed in enumerate(pred.seeds):
                for j, endpoint in enumerate(ENDPOINTS):
                    endpoint_record = endpoint_metric(pred, raw, all_errors[key], i, j)
                    observed = np.isfinite(raw[:, j])
                    endpoint_record["n_groups"] = len(np.unique(memberships[observed]))
                    if endpoint == "Vd":
                        rho, r2 = vd_correlations(transformed[observed, j], pred.transformed[i, observed, j],
                                                  endpoint_record["n_groups"], subgroup=False, class_median=False)
                        endpoint_record.update(vd_spearman=rho, vd_r2=r2)
                    tables["endpoint_metrics"].append(dict(**meta, seed=seed, endpoint=endpoint,
                        unit="percentage_point" if j in PERCENT else "log10", **endpoint_record))
                for axis, level, mask in scopes:
                    if axis not in {"all", "class"}:
                        continue
                    n_groups = len(np.unique(memberships[mask]))
                    tables["coherence"].append(dict(**meta, seed=seed, scope=axis, stratum=level,
                        n_structures=int(mask.sum()), n_groups=n_groups,
                        mean_signed_log_residual=residual[i, mask].mean() if mask.any() else np.nan,
                        mean_absolute_log_residual=np.abs(residual[i, mask]).mean() if mask.any() else np.nan,
                        median_fold_disagreement=np.median(fold[i, mask]) if mask.any() else np.nan))
                    if axis != "class":
                        continue
                    mask = mask & np.isfinite(raw[:, 1])
                    n_groups = len(np.unique(memberships[mask]))
                    rho, r2 = vd_correlations(transformed[mask, 1], pred.transformed[i, mask, 1], n_groups,
                                              subgroup=True, class_median=key[-1] == "class_median")
                    tables["within_class_vd"].append(dict(**meta, seed=seed, ionisation_class=level,
                        n_structures=int(mask.sum()), n_groups=n_groups,
                        vd_mae=all_errors[key][i, mask, 1].mean() if mask.any() else np.nan,
                        vd_spearman=rho, vd_r2=r2))

        n_seeds = max(len(pred.seeds) for pred in by_run.values())
        draws = group_draws(len(groups), n_seeds, n_draws, rng)
        single_seed_draws = BootstrapDraws(draws.group_counts, np.ones((n_draws, 1)))
        for arm_key, base_key, headline in run_contrasts():
            if arm_key not in by_run or base_key not in by_run:
                continue
            arm, base = by_run[arm_key], by_run[base_key]
            seeds = arm.seeds if arm_key[1] != "control" else base.seeds
            pair_draws = draws if len(seeds) > 1 else single_seed_draws
            meta = dict(zip(RUN_COLUMNS, arm_key))
            meta.update(comparator_procedure=base_key[1], comparator_backbone=base_key[2], comparator_arm=base_key[3])
            arm_error = np.broadcast_to(all_errors[arm_key], (len(seeds), *raw.shape))
            base_error = np.broadcast_to(all_errors[base_key], arm_error.shape)
            for axis, level, mask in scopes:
                if axis == "evidence" and not (arm_key[1] == "primary" and arm_key[3] == "C2" and base_key[3] == "C0"):
                    continue
                if axis == "class_rule" and not (arm_key[3] == "P3" and base_key[3] == "P1"):
                    continue
                if axis == "polyprotic_status" and not (arm_key[3] == "P4" and base_key[3] == "P3"):
                    continue
                endpoints = range(len(ENDPOINTS)) if axis == "all" else (1,)
                for j in endpoints:
                    observed = mask & np.isfinite(raw[:, j])
                    primary = headline and j == 1 and axis == "all"
                    record = dict(**meta, scope=axis, stratum=level, endpoint=ENDPOINTS[j], primary_test=primary,
                                  unit="percentage_point" if j in PERCENT else "log10")
                    a, b = arm_error[:, observed, j], base_error[:, observed, j]
                    statistics, by_seed = contrast_statistics(a, b, memberships[observed], pair_draws, primary=primary)
                    if not confirmatory:
                        statistics["primary_lower_error"] = None
                    tables["contrasts"].append(dict(**record, **statistics))
                    for i, (seed, delta) in enumerate(zip(seeds, by_seed)):
                        tables["contrast_seeds"].append(dict(**record, seed=seed, delta=delta,
                            arm_mae=a[i].mean() if observed.any() else np.nan,
                            comparator_mae=b[i].mean() if observed.any() else np.nan))
                    if axis == "all":
                        for fold in sorted(data["outer_fold"].unique()):
                            in_fold = observed & data["outer_fold"].eq(fold).to_numpy()
                            by_fold = arm_error[:, in_fold, j] - base_error[:, in_fold, j]
                            tables["contrast_outer_folds"].append(dict(**record, outer_fold=fold,
                                n_structures=int(in_fold.sum()), n_groups=len(np.unique(memberships[in_fold])),
                                delta_mean=by_fold.mean() if in_fold.any() else np.nan))
                if axis in {"all", "class"} and mask.any():
                    info = dict(**meta, scope=axis, stratum=level, n_structures=int(mask.sum()),
                                n_groups=len(np.unique(memberships[mask])))
                    by_seed = coherence_delta(arm, base, mask)
                    tables["coherence_contrasts"].append(dict(**info,
                        **{column: value.mean() for column, value in by_seed.items()}))
                    for i, seed in enumerate(seeds):
                        tables["coherence_contrast_seeds"].append(dict(**info, seed=seed,
                            **{column: value[i % len(value)] for column, value in by_seed.items()}))
    coverage, groups = label_coverage(cohorts)
    frames = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    frames["endpoint_summary"] = metric_summary(frames["endpoint_metrics"], [*RUN_COLUMNS, "endpoint", "unit"],
        ["n_structures", "n_groups", "n_label_zeros", "n_aafe"], ["mae", "aafe", "vd_spearman", "vd_r2"])
    frames["within_class_vd_summary"] = metric_summary(frames["within_class_vd"], [*RUN_COLUMNS, "ionisation_class"],
        ["n_structures", "n_groups"], ["vd_mae", "vd_spearman", "vd_r2"])
    return {**frames, "coverage": coverage, "group_sizes": groups}


def main() -> None:
    from ionisation_data import sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--class-only", action="store_true", help="Numerical block explicitly deferred")
    parser.add_argument("--neural-only", action="store_true", help="Interim report with the cheap controls pending")
    parser.add_argument("--diagnostic", action="store_true", help="Allow a reduced fixture; never report-ready")
    parser.add_argument("--bootstrap-draws", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    if args.bootstrap_draws < 1 or (not args.diagnostic and args.bootstrap_draws != N_BOOTSTRAP):
        parser.error(f"Production scoring requires exactly {N_BOOTSTRAP:,} bootstrap draws")
    if args.controls and args.neural_only:
        parser.error("Supply --controls or --neural-only")
    if not (args.controls or args.neural_only or args.diagnostic):
        parser.error("Full reporting requires --controls; use --neural-only for an interim report")
    names = ("class",) if args.class_only else COHORTS
    fingerprints = frozen_release_fingerprints(args.release, names, diagnostic=args.diagnostic)
    campaign_id = None if args.diagnostic else checked_campaign(args.predictions, fingerprints)
    cohorts = {name: load_cohort(args.release, name, diagnostic=args.diagnostic) for name in names}
    validate_nested_cohorts(cohorts)
    runs = validate_predictions(read_csv(args.predictions), cohorts, diagnostic=args.diagnostic)
    if args.controls:
        if not args.diagnostic:
            checked_controls(args.controls, fingerprints)
        runs.update(validate_predictions(read_csv(args.controls), cohorts, diagnostic=args.diagnostic, control=True))
    if any(cohort not in {key[0] for key in runs} for cohort in names):
        raise ValueError("At least one complete run is required in every selected cohort")
    report_ready = not (args.diagnostic or args.neural_only)
    with threadpool_limits(limits=1, user_api="blas"):
        tables = score(cohorts, runs, n_draws=args.bootstrap_draws, confirmatory=report_ready)
    metadata = dict(
        status="report_ready" if report_ready else ("diagnostic" if args.diagnostic else "neural_only_interim"),
        cohorts=list(names), numerical_block_deferred=args.class_only, cheap_controls_included=bool(args.controls),
        release_fingerprints=fingerprints, campaign_id=campaign_id,
        neural_predictions_sha256=sha256(args.predictions),
        control_predictions_sha256=sha256(args.controls) if args.controls else None,
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=args.bootstrap_draws, adjusted_level=ADJUSTED_LEVEL,
        primary_family_size=6,
        primary_rule="Crossed equal-tailed adjusted upper bound below zero; production Vd headline contrasts only",
        delta="Arm minus named comparator. Negative MAE deltas denote lower arm error.",
        relative_delta="Mean paired MAE delta divided by comparator MAE averaged over trajectories",
        crossed_interval="Approximate equal-tailed paired percentile interval over whole seed trajectories and frozen groups, conditional on the observed training data, fixed folds and selected settings",
        conditional_group_interval="Equal-tailed paired percentile interval over frozen groups with all trained seeds fixed",
        seed_interval="Paired t interval over whole-trajectory deltas on the fixed observed structures",
        aafe="10 ** mean absolute log10 error on strictly positive observed labels; zero predictions yield infinite AAFE",
        class_median="Training-fold medians scored on the original scale; within-class Spearman omitted",
        scope_class_rule="P3-P1 subgroup contrasts compare the audited category with the strict numeric pKa thresholds. The permanent-charge disagreement subgroup is a subset of all threshold disagreements.",
        empty_draws="Draws containing no labelled structure in a given endpoint/stratum are undefined. Valid draw counts are reported in each row.",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
    metadata["files"] = {f"{name}.csv": sha256(args.out / f"{name}.csv") for name in tables}
    (args.out / "scoring_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
