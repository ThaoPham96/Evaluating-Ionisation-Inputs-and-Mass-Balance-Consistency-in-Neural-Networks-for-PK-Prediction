"""Fit the CPU-only outer-fold controls from the frozen ionisation release."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy.linalg import eigh
from scipy.special import expit
from threadpoolctl import threadpool_limits

from ionisation_score import (
    CLASSES, COHORTS, CONTROL_ARMS, ENDPOINTS, FOLDS, PHYSICAL_COLUMNS, PRED_COLUMNS,
    RAW_ENDPOINTS, frozen_release_fingerprints, integer_column, inverse_physical, load_cohort, read_csv, require_columns,
    threshold_category, transform_targets, validate_nested_cohorts, validate_predictions,
)

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
PROFILE_PH = (2.0, 5.0, 6.5, 7.4)
PH_SWEEP = np.linspace(2, 8, 25)
PKA_SHIFTS = (-1.0, -0.5, 0.0, 0.5, 1.0)
PRODUCTION_RDKIT = "2025.09.6"
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def inner_folds(path: Path, data: pd.DataFrame, *, diagnostic: bool = False) -> dict[int, np.ndarray]:
    inner = read_csv(path)
    require_columns(inner, ("outer_fold", "inner_fold", "group_id", "structure_id"), "inner folds")
    if inner[["group_id", "structure_id"]].isna().any().any():
        raise ValueError("Inner fold IDs must be complete")
    for column in ("outer_fold", "inner_fold"):
        inner[column] = integer_column(inner[column])
    outer = data["outer_fold"].to_numpy()
    if set(inner["outer_fold"]) != set(outer):
        raise ValueError("Inner-fold assignments must cover the frozen outer folds")
    result = {}
    for fold, rows in inner.groupby("outer_fold"):
        train = data.loc[outer != fold]
        rows = rows.set_index("structure_id").sort_index()
        if not rows.index.equals(train.index):
            raise ValueError(f"Outer fold {fold} inner assignments must contain exactly its outer-training IDs")
        if not rows["group_id"].equals(train["group_id"]):
            raise ValueError(f"Outer fold {fold} inner groups disagree with frozen outer groups")
        if rows.groupby("group_id")["inner_fold"].nunique().gt(1).any():
            raise ValueError(f"Outer fold {fold} splits a group across inner folds")
        levels = set(rows["inner_fold"])
        if len(levels) < 2 or (not diagnostic and levels != set(FOLDS)):
            raise ValueError(f"Outer fold {fold} has invalid inner folds {sorted(levels)}")
        result[int(fold)] = rows["inner_fold"].to_numpy()
    return result


def median_predictions(raw_train: np.ndarray, class_train: np.ndarray,
                       class_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]]]:
    if not np.isfinite(raw_train).any(axis=0).all():
        raise ValueError("An outer-training endpoint has no observations for a median control")
    overall = np.nanmedian(raw_train, axis=0)
    marginal = np.broadcast_to(overall, (len(class_test), len(ENDPOINTS))).copy()
    conditional = marginal.copy()
    fallback = {}
    for category in CLASSES:
        labels = raw_train[class_train == category]
        present = np.isfinite(labels).any(axis=0)
        if present.any():
            conditional[np.ix_(class_test == category, present)] = np.nanmedian(labels[:, present], axis=0)
        fallback[category] = [endpoint for endpoint, available in zip(ENDPOINTS, present) if not available]
    return marginal, conditional, fallback


def propensity_profile(pka: np.ndarray, ph: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        exponent = np.stack((ph - pka[:, :1], pka[:, 1:] - ph), axis=-1) * np.log(10.0)
        return expit(exponent).reshape(len(pka), 2 * len(ph))


def numerical_features(data: pd.DataFrame) -> np.ndarray:
    return propensity_profile(data[["pka_acid", "pka_base"]].to_numpy(dtype=float), np.array(PROFILE_PH))


def descriptor_robustness(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stress both numerical inputs independently and tabulate the unshifted pH sweep."""
    pka = data[["pka_acid", "pka_base"]].to_numpy(dtype=float)
    base = numerical_features(data)
    base_class = threshold_category(pka)
    classes = data["ionisation_class"].to_numpy()
    channels = [f"mean_abs_delta_{site}_pH_{h:g}" for h in PROFILE_PH for site in ("acid", "base")]
    rows = []
    for acid_shift, base_shift in product(PKA_SHIFTS, repeat=2):
        if acid_shift == 0 and base_shift == 0:
            continue
        shifted_pka = pka + (acid_shift, base_shift)
        change = np.abs(propensity_profile(shifted_pka, np.array(PROFILE_PH)) - base)
        shifted_class = threshold_category(shifted_pka)
        changed = shifted_class != base_class
        for label in ("all", *CLASSES):
            mask = np.ones(len(data), dtype=bool) if label == "all" else classes == label
            counts = pd.crosstab(base_class[mask], shifted_class[mask])
            counts = {f"{start}->{end}": int(value) for start, record in counts.iterrows()
                      for end, value in record.items() if value}
            means = change[mask].mean(axis=0) if mask.any() else np.full(len(channels), np.nan)
            rows.append(dict(acid_pka_shift=acid_shift, base_pka_shift=base_shift, audited_class=label,
                             n_structures=int(mask.sum()), n_threshold_class_transitions=int(changed[mask].sum()),
                             transition_fraction=float(changed[mask].mean()) if mask.any() else np.nan,
                             threshold_transition_counts=json.dumps(counts, sort_keys=True), **dict(zip(channels, means))))
    sweep = propensity_profile(pka, PH_SWEEP).reshape(-1, 2)
    sweep_frame = pd.DataFrame(dict(structure_id=np.repeat(data.index.to_numpy(), len(PH_SWEEP)),
                                    audited_class=np.repeat(classes, len(PH_SWEEP)),
                                    pH=np.tile(PH_SWEEP, len(data)), acid_propensity=sweep[:, 0],
                                    base_propensity=sweep[:, 1]))
    return pd.DataFrame(rows), sweep_frame


def morgan_features(data: pd.DataFrame) -> np.ndarray:
    require_columns(data, ("canonical_smiles",), "class data")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    features = []
    for structure_id, smiles in data["canonical_smiles"].items():
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES for {structure_id}")
        features.append(generator.GetFingerprintAsNumPy(mol))
    return np.asarray(features, dtype=float)


def standardise_profile(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mean, sd = train.mean(axis=0), train.std(axis=0)
    constant = np.all(train == train[0], axis=0) | (sd == 0)
    denominator = np.where(constant, 1., sd)
    divisor = np.sqrt(len(PROFILE_PH))
    train = (train - mean) / denominator / divisor
    test = (test - mean) / denominator / divisor
    train[:, constant], test[:, constant] = 0, 0
    return train, test, dict(feature_mean=mean.tolist(), feature_sd=np.where(constant, 0, sd).tolist(),
                              constant_features=constant.tolist(), width_divisor=float(divisor))


def target_scaler(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not (np.isfinite(train).sum(axis=0) >= 2).all():
        raise ValueError("Each ridge fit-training endpoint needs at least two observed targets")
    mean, sd = np.nanmean(train, axis=0), np.nanstd(train, axis=0)
    if not (np.isfinite(sd) & (sd > 0)).all():
        raise ValueError("Each ridge fit-training endpoint needs positive variance")
    return mean, sd


def ridge_grid(train_x: np.ndarray, test_x: np.ndarray, train_y: np.ndarray,
               alphas: tuple[float, ...]) -> np.ndarray:
    """Fit intercepts and ridge penalties on each endpoint's observed training rows."""
    gram = train_x @ train_x.T
    test_gram = test_x @ train_x.T
    out = np.empty((len(alphas), len(test_x), len(ENDPOINTS)))
    penalties = np.array(alphas)[None, :]
    for j in range(len(ENDPOINTS)):
        mask = np.isfinite(train_y[:, j])
        k = gram[np.ix_(mask, mask)]
        kt = test_gram[:, mask]
        mean = k.mean(axis=0)
        centered = k - mean[:, None] - mean + mean.mean()
        cross = kt - kt.mean(axis=1, keepdims=True) - mean + mean.mean()
        values, vectors = eigh(centered, check_finite=False)
        y = train_y[mask, j]
        average = y.mean()
        weights = (vectors.T @ (y - average))[:, None] / (np.maximum(values, 0)[:, None] + penalties)
        out[..., j] = ((cross @ vectors) @ weights).T + average
    return out


def fit_ridge_control(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, inner: np.ndarray,
                      *, scale_profile: bool = False, diagnostic: bool = False) -> tuple[np.ndarray, list[dict], dict]:
    sse = np.zeros((len(ALPHAS), len(ENDPOINTS)))
    counts = np.zeros(len(ENDPOINTS), dtype=int)
    audit = []
    for inner_fold in sorted(np.unique(inner)):
        val = inner == inner_fold
        train_x, val_x = x_train[~val], x_train[val]
        feature_audit = {}
        if scale_profile:
            train_x, val_x, feature_audit = standardise_profile(train_x, val_x)
        mean, sd = target_scaler(y_train[~val])
        scaled = (y_train - mean) / sd
        predicted = ridge_grid(train_x, val_x, scaled[~val], ALPHAS)
        observed = np.isfinite(scaled[val])
        fold_counts = observed.sum(axis=0)
        if not diagnostic and np.any(fold_counts < 10):
            raise ValueError(f"Inner fold {inner_fold} has fewer than ten labels for an endpoint")
        sse += np.where(observed, (predicted - scaled[val]) ** 2, 0.0).sum(axis=1)
        counts += fold_counts
        audit.append(dict(stage="inner_scaler", inner_fold=int(inner_fold), target_mean=mean.tolist(),
                          target_sd=sd.tolist(), validation_counts=fold_counts.tolist(), **feature_audit))
    if (counts == 0).any():
        raise ValueError("Every endpoint must have inner-validation observations for ridge selection")
    mean_mse = (sse / counts).mean(axis=1)
    if not np.isfinite(mean_mse).all():
        raise ValueError("Nonfinite ridge selection criterion")
    best = int(mean_mse.argmin())
    for i, alpha in enumerate(ALPHAS):
        audit.append(dict(stage="inner_selection", alpha=alpha, selected=i == best, mean_validation_mse=mean_mse[i],
                          validation_counts=counts.tolist(), mse_by_endpoint=(sse[i] / counts).tolist()))
    feature_audit = {}
    if scale_profile:
        x_train, x_test, feature_audit = standardise_profile(x_train, x_test)
    mean, sd = target_scaler(y_train)
    selected = ALPHAS[best]
    pred = ridge_grid(x_train, x_test, (y_train - mean) / sd, (selected,))[0] * sd + mean
    return pred, audit, dict(stage="outer_refit", alpha=selected, target_mean=mean.tolist(),
                             target_sd=sd.tolist(), **feature_audit)


def build_controls(cohorts: dict[str, pd.DataFrame], splits: dict[str, dict[int, np.ndarray]],
                   *, diagnostic: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    output, audit = [], []
    for cohort, data in cohorts.items():
        raw = data[list(RAW_ENDPOINTS)].to_numpy()
        targets = transform_targets(raw)
        categories = data["ionisation_class"].to_numpy()
        outer = data["outer_fold"].to_numpy()
        ridge_features = {}
        if cohort == "class":
            bits = morgan_features(data)
            one_hot = (categories[:, None] == np.array(CLASSES)).astype(float)
            ridge_features = {"morgan_ridge": bits, "morgan_class_ridge": np.column_stack((bits, one_hot))}
        else:
            ridge_features["descriptor_ridge"] = numerical_features(data)
        for fold, inner in sorted(splits[cohort].items()):
            held = outer == fold
            overall, conditional, fallback = median_predictions(raw[~held], categories[~held], categories[held])
            predictions = {"overall_median": (transform_targets(overall), overall),
                           "class_median": (transform_targets(conditional), conditional)}
            audit.append(dict(cohort=cohort, outer_fold=fold, arm="class_median", stage="median_fallback",
                              class_endpoint_fallback=fallback))
            for arm, x in ridge_features.items():
                pred, details, refit = fit_ridge_control(x[~held], x[held], targets[~held], inner,
                                                         scale_profile=arm == "descriptor_ridge", diagnostic=diagnostic)
                predictions[arm] = pred, inverse_physical(pred)
                audit.extend(dict(cohort=cohort, outer_fold=fold, arm=arm, **row) for row in (*details, refit))
            for arm in CONTROL_ARMS[cohort]:
                transformed, physical = predictions[arm]
                rows = pd.DataFrame(np.column_stack((transformed, physical)), columns=(*PRED_COLUMNS, *PHYSICAL_COLUMNS))
                rows = rows.assign(structure_id=data.index[held], outer_fold=fold, cohort=cohort, arm=arm,
                                   procedure="control", backbone="CPU", seed=0)
                output.append(rows)
    return pd.concat(output, ignore_index=True), pd.DataFrame(audit)


def main() -> None:
    from ionisation_data import sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--class-only", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if rdBase.rdkitVersion != PRODUCTION_RDKIT and not args.diagnostic:
        raise RuntimeError(f"Production requires RDKit {PRODUCTION_RDKIT}; found {rdBase.rdkitVersion}")
    names = ("class",) if args.class_only else COHORTS
    fingerprints = frozen_release_fingerprints(args.release, names, diagnostic=args.diagnostic)
    cohorts = {name: load_cohort(args.release, name, diagnostic=args.diagnostic) for name in names}
    validate_nested_cohorts(cohorts)
    splits = {name: inner_folds(args.release / name / "inner.csv", data, diagnostic=args.diagnostic)
              for name, data in cohorts.items()}
    with threadpool_limits(limits=1, user_api="blas"):
        predictions, audit = build_controls(cohorts, splits, diagnostic=args.diagnostic)
    validate_predictions(predictions, cohorts, diagnostic=False, control=True)
    args.out.mkdir(parents=True, exist_ok=True)
    prediction_file = args.out / "controls_predictions.csv"
    selection_file = args.out / "control_selection.csv"
    predictions.to_csv(prediction_file, index=False)
    audit.to_csv(selection_file, index=False)
    descriptor_outputs = {}
    if "numerical" in cohorts:
        stress, sweep = descriptor_robustness(cohorts["numerical"])
        for name, frame in (("descriptor_stress.csv", stress), ("descriptor_pH_sweep.csv", sweep)):
            frame.to_csv(args.out / name, index=False)
            descriptor_outputs[name] = sha256(args.out / name)
    (args.out / "controls_manifest.json").write_text(json.dumps(dict(
        status="diagnostic" if args.diagnostic else "complete",
        cohorts=list(names), numerical_block_deferred=args.class_only, rdkit=rdBase.rdkitVersion,
        release_fingerprints=fingerprints, predictions_sha256=sha256(prediction_file),
        selection_sha256=sha256(selection_file),
        morgan_radius=MORGAN_RADIUS, morgan_bits=MORGAN_BITS, morgan_use_chirality=False,
        morgan_and_charge_scaling="Unscaled binary fingerprint bits and 0/1 one-hot category",
        one_hot_order=list(CLASSES), descriptor_ph=list(PROFILE_PH),
        descriptor_scaling="Fit-training population SD, constant channels map to zero, divide by sqrt(4)",
        descriptor_outputs=descriptor_outputs,
        descriptor_stress="All nonzero combinations of independent acidic/basic pKa shifts of -1, -0.5, 0, 0.5 or 1. Changes use unscaled propensities. Category transitions apply strict acid pKa <7.4 and base pKa >7.4 to the numerical pair. Audited categories are only used to stratify those transitions.",
        descriptor_sweep="Unshifted descriptor propensities on pH 2 to 8 by 0.25",
        ridge_alphas=list(ALPHAS), ridge_intercept=True,
        alpha_selection="Pool inner-standardised validation squared errors by endpoint, then average the seven endpoint MSEs equally. Ties use the smaller alpha.",
        medians="Original endpoint units. Absent class/endpoint uses the overall outer-training endpoint median.",
    ), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
