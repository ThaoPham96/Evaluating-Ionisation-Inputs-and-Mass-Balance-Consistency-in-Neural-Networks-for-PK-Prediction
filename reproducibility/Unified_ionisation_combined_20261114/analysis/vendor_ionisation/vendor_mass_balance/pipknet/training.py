"""Nested 5-fold scaffold cross-validation training driver.

Reproduces the PIPK-Net training protocol for a single architecture variant:

  outer GroupKFold(5) over Bemis-Murcko scaffolds
    -> inner GroupKFold(5) grid search to select hyperparameters
    -> retrain the best configuration on the full outer-train split
       (with a 90/10 GroupShuffleSplit held out for early stopping)
    -> evaluate on the outer-test fold

Each outer fold writes a checkpoint directory compatible with the inference
CLI: ``best_outer_model.pt``, ``config.json`` (with ``best_hp``), ``scaler.json``,
``predictions.csv`` and ``metrics.csv``. The five outer-fold models form the
inference ensemble.

The default grid matches the published search space; it is large, so training
all variants is compute-intensive. Use ``--quick`` (or pass a reduced
``hp_space``) for a fast smoke run.
"""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch_geometric.loader import DataLoader

from .models import GNNMultitask
from .engine import fit_one, evaluate

# Published hyperparameter search space (manuscript Methods 2.1.4).
FULL_HP_SPACE = {
    "hidden_dim": [64, 96, 128],
    "dropout": [0.1, 0.2],
    "lr": [1e-3, 5e-4],
    "weight_decay": [1e-4, 3e-4],
    "ion_emb_dim": [16, 32],         # variants B_ion / C_physio only
    "lambda_cons": [0.2, 0.5, 1.0],  # variants C_physio / D_physio only
}
# Small grid for smoke testing.
QUICK_HP_SPACE = {"hidden_dim": [64], "dropout": [0.1], "lr": [1e-3],
                  "weight_decay": [1e-4], "ion_emb_dim": [16], "lambda_cons": [0.2]}


def applicable_hp_keys(variant: str) -> List[str]:
    """Hyperparameters that actually affect ``variant``.

    ``ion_emb_dim`` is inert without the ionisation embedding and ``lambda_cons``
    is inert without the mass-balance term, so both are dropped for variants that
    do not use them. Backbone and depth are deliberately absent: they are fixed
    per run, not searched.
    """
    keys = ["hidden_dim", "dropout", "lr", "weight_decay"]
    if variant in ("B_ion", "C_physio"):
        keys.append("ion_emb_dim")
    if variant in ("C_physio", "D_physio"):
        keys.append("lambda_cons")
    return keys


def hp_grid(variant: str, space: Dict[str, list]) -> List[dict]:
    """Enumerate hyperparameter combinations relevant to a variant."""
    keys = applicable_hp_keys(variant)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[space[k] for k in keys])]
    return combos


def run_name(variant: str, backbone: str = "GraphConv", n_layers: int = 2) -> str:
    """Checkpoint directory name for a (variant, backbone, depth) run.

    The published architecture keeps its bare variant name so existing paths and
    checkpoints are untouched; every other architecture gets a suffixed directory
    so an ablation run can never overwrite a published one.
    """
    if backbone == "GraphConv" and n_layers == 2:
        return variant
    return f"{variant}__{backbone}_L{n_layers}"


def inherit_hp(source_dir, variant: str, fold_i: int) -> dict:
    """Read fold ``fold_i``'s selected hyperparameters from a finished run.

    Hyperparameters that do not apply to ``variant`` are dropped (e.g. inheriting
    C_physio's settings into A_baseline discards ion_emb_dim and lambda_cons).
    ``batch_size`` is excluded because the caller sets it from its own argument.
    """
    cfg_path = Path(source_dir) / f"fold_{fold_i}" / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"--inherit-hp-from: no hyperparameters for outer fold {fold_i} at {cfg_path}")
    src_hp = json.load(open(cfg_path))["best_hp"]
    keys = applicable_hp_keys(variant)
    missing = [k for k in keys if k not in src_hp]
    if missing:
        raise KeyError(
            f"--inherit-hp-from: {cfg_path} has no {missing} but variant {variant!r} "
            f"needs them; inherit from a variant that searched those hyperparameters.")
    hp = {k: src_hp[k] for k in keys}
    # The published A_baseline / B_ion configs record placeholder zeros for the
    # hyperparameters their variant does not use. Inheriting those into a variant
    # that *does* use them would silently train with the ionisation embedding
    # zero-width or the mass-balance term switched off, i.e. not that variant at all.
    degenerate = {k: hp[k] for k in ("ion_emb_dim", "lambda_cons") if k in hp and not hp[k]}
    if degenerate:
        raise ValueError(
            f"--inherit-hp-from: {cfg_path} sets {degenerate}, which disables the "
            f"mechanism variant {variant!r} is defined by; inherit from a run that "
            f"actually searched those hyperparameters (e.g. C_physio).")
    return hp


def load_fold_assignment(source, out_root=None) -> Dict[int, int]:
    """``{ORIG_INDEX: outer_fold}`` from a recorded per-drug fold assignment.

    ``source`` may be a CSV, a directory containing ``all_drug_predictions.csv``,
    or a run name under ``out_root``. The CSV needs ``row_idx`` and ``outer_fold``
    columns; a table stacking several variants (all sharing one assignment) is
    collapsed, but genuinely conflicting assignments are rejected.

    This exists because ``GroupKFold`` cannot reproduce the published outer folds:
    82% of the 517 dev scaffold groups are singletons, so the greedy size-ordered
    assignment is decided by sort tie-breaking and differs across library
    versions. Reusing the recorded assignment is the only way a new variant is
    fold-for-fold comparable to the published A/B/C/D runs.
    """
    p = Path(source)
    if not p.exists() and out_root is not None:
        cand = Path(out_root) / str(source) / "all_drug_predictions.csv"
        if cand.exists():
            p = cand
    if p.is_dir():
        # A splits directory (fold{1..5}_test.csv keyed by ORIG_INDEX) is the other
        # canonical shape; prefer it over all_drug_predictions.csv when present.
        per_fold = sorted(p.glob("fold*_test.csv"))
        if per_fold:
            out = {}
            for f in per_fold:
                m = re.search(r"fold(\d+)_test", f.name)
                if not m:
                    continue
                fold = int(m.group(1))
                d = pd.read_csv(f)
                col = "ORIG_INDEX" if "ORIG_INDEX" in d.columns else "row_idx"
                if col not in d.columns:
                    raise ValueError(f"--folds-from: {f} has neither ORIG_INDEX nor row_idx")
                for r in d[col].astype(int):
                    if int(r) in out and out[int(r)] != fold:
                        raise ValueError(f"--folds-from: {p} puts row {int(r)} in both "
                                         f"fold {out[int(r)]} and fold {fold}")
                    out[int(r)] = fold
            if out:
                return out
        p = p / "all_drug_predictions.csv"
    if not p.exists():
        raise FileNotFoundError(f"--folds-from: no fold assignment at {p}")
    df = pd.read_csv(p)
    for col in ("row_idx", "outer_fold"):
        if col not in df.columns:
            raise ValueError(f"--folds-from: {p} has no {col!r} column "
                             f"(found {list(df.columns)[:8]})")
    df = df[["row_idx", "outer_fold"]].drop_duplicates()
    clash = df[df["row_idx"].duplicated(keep=False)]
    if len(clash):
        raise ValueError(f"--folds-from: {p} assigns conflicting folds to "
                         f"{clash['row_idx'].nunique()} rows, e.g. "
                         f"{clash.head(4).to_dict('records')}")
    return {int(r): int(f) for r, f in zip(df["row_idx"], df["outer_fold"])}


def _build_model(variant: str, hp: dict, device, backbone: str = "GraphConv",
                 n_layers: int = 2) -> GNNMultitask:
    return GNNMultitask(
        hidden_dim=hp["hidden_dim"], ion_dim=4,
        ion_emb_dim=hp.get("ion_emb_dim", 16), out_dim=7,
        use_ion=(variant in ("B_ion", "C_physio")), dropout=hp["dropout"],
        backbone=backbone, n_layers=n_layers,
    ).to(device)


def _loader(items, batch_size=128, shuffle=False):
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle)


def _val_score(model, loader, df_ref) -> float:
    metr, _ = evaluate(model, loader, df_ref)
    return float(np.nanmean(metr["SelectionPearson"].values))


def nested_cv(dataset, df_ref, groups, variant: str, out_dir, *,
              hp_space: Optional[dict] = None, n_outer: int = 5, n_inner: int = 5,
              batch_size: int = 128, max_epochs: int = 200, patience: int = 30,
              inner_max_epochs: int = 100, inner_patience: int = 15,
              device=None, seed: int = 42, backbone: str = "GraphConv",
              n_layers: int = 2, inherit_hp_from: Optional[str] = None,
              folds_from: Optional[str] = None,
              grad_clip: Optional[float] = 1.0) -> pd.DataFrame:
    """Run nested CV for one variant and write per-fold checkpoints.

    ``folds_from`` replaces the ``GroupKFold`` outer split with a recorded
    per-drug assignment and restricts the run to exactly the rows it names -- so
    it also pins the dev set, keeping the independent-test compounds out.

    ``backbone`` and ``n_layers`` select the graph encoder and are fixed for the
    whole run (never searched). ``inherit_hp_from`` names a finished run whose
    per-fold hyperparameters are reused verbatim, skipping the inner grid search
    entirely -- this is what makes a backbone comparison an architecture-only
    contrast rather than a confounded architecture-plus-search one.

    Inner search uses 100 epochs / patience 15; final refits use 200 / 30 by
    default, matching cells 57 and 83 of the original training notebook.
    Results are written to ``out_dir / run_name(variant, backbone, n_layers)``.

    Returns a long DataFrame of per-fold, per-task outer-test metrics.
    """
    assert variant in ("A_baseline", "B_ion", "C_physio", "D_physio")
    hp_space = hp_space or FULL_HP_SPACE
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(out_dir)
    run = run_name(variant, backbone, n_layers)
    # Resolve --inherit-hp-from as a path if it is one, else as a run under out_dir.
    hp_src_dir = None
    if inherit_hp_from is not None:
        hp_src_dir = Path(inherit_hp_from)
        if not hp_src_dir.exists():
            hp_src_dir = out_root / inherit_hp_from
        if hp_src_dir.resolve() == (out_root / run).resolve():
            raise ValueError(f"--inherit-hp-from {inherit_hp_from!r} resolves to this "
                             f"run's own output directory ({hp_src_dir})")
    out_dir = out_root / run
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] run={run} backbone={backbone} n_layers={n_layers} hp="
          + (f"inherited from {hp_src_dir}" if hp_src_dir else "inner grid search"))
    mode = "C_physio" if variant in ("C_physio", "D_physio") else variant

    groups = np.asarray(groups, dtype=object)
    combos = hp_grid(variant, hp_space)

    if folds_from is not None:
        fold_map = load_fold_assignment(folds_from, out_root)
        row_ids = np.array([int(d.row_idx) for d in dataset])
        keep = np.flatnonzero(np.isin(row_ids, list(fold_map)))
        if len(keep) == 0:
            raise ValueError(f"--folds-from: none of the {len(dataset)} molecules "
                             f"match the {len(fold_map)} rows in the assignment")
        unmatched = set(fold_map) - set(int(r) for r in row_ids[keep])
        dataset = [dataset[i] for i in keep]
        groups = groups[keep]
        assigned = np.array([fold_map[int(r)] for r in row_ids[keep]])
        print(f"[train] folds from {folds_from}: {len(keep)} of {len(row_ids)} molecules "
              f"kept ({len(row_ids) - len(keep)} not in the assignment"
              + (f", {len(unmatched)} assigned rows absent from the data" if unmatched else "")
              + ")")
        idx_all = np.arange(len(dataset))
        splits = [(int(f), np.flatnonzero(assigned != f), np.flatnonzero(assigned == f))
                  for f in sorted(np.unique(assigned))]
    else:
        idx_all = np.arange(len(dataset))
        n_groups = len(np.unique(groups))
        if n_outer > n_groups:
            print(f"[train] only {n_groups} scaffold groups; reducing outer folds {n_outer} -> {n_groups}")
            n_outer = n_groups
        outer = GroupKFold(n_splits=n_outer)
        splits = [(i, trv, te) for i, (trv, te)
                  in enumerate(outer.split(idx_all, groups=groups), 1)]
    all_metrics = []

    for fold_i, trv_idx, test_idx in splits:
        trv_groups = groups[trv_idx]

        # --- hyperparameters: inherited, trivial, or searched ---
        best_hp, best_inner = None, -np.inf
        if hp_src_dir is not None:
            best_hp = inherit_hp(hp_src_dir, variant, fold_i)
            hp_source = f"inherited:{(hp_src_dir / ('fold_%d' % fold_i) / 'config.json').as_posix()}"
        elif len(combos) == 1:
            best_hp = combos[0]
            hp_source = "single_combo"
        else:
            hp_source = "inner_grid_search"
            inner = GroupKFold(n_splits=min(n_inner, len(np.unique(trv_groups))))
            for hp in combos:
                scores = []
                for itr, ival in inner.split(trv_idx, groups=trv_groups):
                    tr = [dataset[trv_idx[i]] for i in itr]
                    va = [dataset[trv_idx[i]] for i in ival]
                    model = _build_model(variant, hp, device, backbone, n_layers)
                    model, _ = fit_one(model, _loader(tr, batch_size, True), _loader(va, batch_size),
                                       df_ref, mode=mode, lambda_cons=hp.get("lambda_cons", 0.2),
                                       patience=inner_patience, lr=hp["lr"],
                                       weight_decay=hp["weight_decay"], max_epochs=inner_max_epochs,
                                       grad_clip=grad_clip)
                    scores.append(_val_score(model, _loader(va, batch_size), df_ref))
                mean_score = float(np.nanmean(scores))
                if mean_score > best_inner:
                    best_inner, best_hp = mean_score, hp
        best_hp = {**best_hp, "batch_size": batch_size}

        # --- retrain best HP on full outer-train (90/10 early-stopping split) ---
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        rel_tr, rel_va = next(gss.split(trv_idx, groups=trv_groups))
        tr = [dataset[trv_idx[i]] for i in rel_tr]
        va = [dataset[trv_idx[i]] for i in rel_va]
        model = _build_model(variant, best_hp, device, backbone, n_layers)
        model, scaler_stats = fit_one(model, _loader(tr, batch_size, True), _loader(va, batch_size),
                                      df_ref, mode=mode, lambda_cons=best_hp.get("lambda_cons", 0.2),
                                      patience=patience, lr=best_hp["lr"],
                                      weight_decay=best_hp["weight_decay"], max_epochs=max_epochs,
                                      grad_clip=grad_clip)

        # --- evaluate on outer-test + save artefacts ---
        test_items = [dataset[i] for i in test_idx]
        metr, preds = evaluate(model, _loader(test_items, batch_size), df_ref)
        metr["outer_fold"], metr["config"] = fold_i, run
        all_metrics.append(metr)

        fold_dir = out_dir / f"fold_{fold_i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict()}, fold_dir / "best_outer_model.pt")
        json.dump({"variant": variant, "run_name": run, "backbone": backbone,
                   "n_layers": n_layers, "best_hp": best_hp, "outer_fold": fold_i,
                   "hp_source": hp_source,
                   "folds_from": str(folds_from) if folds_from else "GroupKFold",
                   "grad_clip": grad_clip,
                   "inner_max_epochs": inner_max_epochs, "inner_patience": inner_patience,
                   "max_epochs": max_epochs, "patience": patience,
                   "selection_metric": "physical-scale mean validation Pearson"},
                  open(fold_dir / "config.json", "w"), indent=2)
        json.dump(scaler_stats, open(fold_dir / "scaler.json", "w"))
        preds.to_csv(fold_dir / "predictions.csv")
        metr.to_csv(fold_dir / "metrics.csv", index=False)
        print(f"[train] {run} fold {fold_i}: best_hp={best_hp} ({hp_source}) "
              f"-> mean test Pearson={np.nanmean(metr['Pearson'].values):.3f}")

    results = pd.concat(all_metrics, ignore_index=True)
    results.to_csv(out_dir / "nested_cv_results.csv", index=False)
    return results


def train_from_csv(csv_path, variant: str, out_dir, *, quick: bool = False,
                   max_epochs: int = 200, patience: int = 30, batch_size: int = 128,
                   backbone: str = "GraphConv", n_layers: int = 2,
                   inherit_hp_from: Optional[str] = None,
                   folds_from: Optional[str] = None):
    """Convenience entry point: load a CSV and run nested CV for one variant."""
    from .data import load_dataset_from_csv
    dataset, df_ref, groups = load_dataset_from_csv(csv_path)
    print(f"[train] loaded {len(dataset)} molecules, {len(np.unique(groups))} scaffolds")
    return nested_cv(dataset, df_ref, groups, variant, out_dir,
                     hp_space=QUICK_HP_SPACE if quick else FULL_HP_SPACE,
                     max_epochs=max_epochs, patience=patience, batch_size=batch_size,
                     backbone=backbone, n_layers=n_layers,
                     inherit_hp_from=inherit_hp_from, folds_from=folds_from)
