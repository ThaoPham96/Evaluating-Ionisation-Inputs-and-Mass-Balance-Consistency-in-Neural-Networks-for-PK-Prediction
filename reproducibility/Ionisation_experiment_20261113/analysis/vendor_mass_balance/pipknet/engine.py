import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Constants
TASKS = ["t_half", "Vd", "CL", "F", "PPB", "Cmax", "Tmax"]
TIDX = {t: i for i, t in enumerate(TASKS)}
TASKS_WITH_UNITS = ["t_half (h)", "Vd (L)", "CL (L/h)", "F (%)", "PPB (%)", "Cmax (uM)", "Tmax (h)"]

# --- 1. Losses ---

# Per-task loss weights, ordered as TASKS = [t_half, Vd, CL, F, PPB, Cmax, Tmax].
# Vd and CL are up-weighted (5x) to prioritise the two primary dispositional
# drivers of the mass-balance triad; all other endpoints get weight 1. This is
# the {5, 5, 1, 1, 1, 1, 1} scheme for {Vd, Cl, t1/2, F, PPB, Cmax, Tmax} reported
# in the manuscript (Methods; Table 6) and is applied identically to every variant.
DEFAULT_TASK_WEIGHTS = (1, 5, 5, 1, 1, 1, 1)


def loss_supervised(yhat, ytrue, ymask, scaler, task_weights=DEFAULT_TASK_WEIGHTS):
    """Standard Weighted MSE loss using TaskScaler for Z-score normalization."""
    # Standardize both predictions and truths using the scaler
    yhat_std  = (yhat - scaler.means.to(yhat.device)) / scaler.stds.to(yhat.device)
    ytrue_std = (ytrue - scaler.means.to(ytrue.device)) / scaler.stds.to(ytrue.device)
    
    tw = torch.as_tensor(task_weights, device=yhat.device, dtype=yhat.dtype)
    dist_sq = (yhat_std - ytrue_std).pow(2) * ymask.to(yhat.dtype) * tw
    
    return dist_sq.sum() / (ymask.to(yhat.dtype) * tw).sum().clamp(min=1.0)

def loss_with_t12_consistency(yhat, ytrue, ymask, scaler, lambda_cons=0.2,
                              task_weights=DEFAULT_TASK_WEIGHTS):
    """Supervised loss + physics constraint (t1/2 relationship)."""
    L_sup = loss_supervised(yhat, ytrue, ymask, scaler, task_weights)

    # Physics is calculated in log10 space (yhat is already log10/logit)
    log10_ln2 = torch.tensor(-0.1593, device=yhat.device, dtype=yhat.dtype)
    
    lt_calc = log10_ln2 + yhat[:, TIDX["Vd"]] - yhat[:, TIDX["CL"]]
    L_phys = (yhat[:, TIDX["t_half"]] - lt_calc).pow(2).mean()

    return L_sup + lambda_cons * L_phys

# --- 2. Unit Conversion & Metrics ---

def to_original_units(p_log_logit):
    """Converts log10/logit predictions back to physical units."""
    # p_log_logit is (B, 7)
    inv_log = lambda v: 10**v
    inv_logit = lambda v: 1.0 / (1.0 + 10**(-v))
    
    return {
        "t_half (h)": inv_log(p_log_logit[:, TIDX["t_half"]]),
        "Vd (L)":     inv_log(p_log_logit[:, TIDX["Vd"]]),
        "CL (L/h)":   inv_log(p_log_logit[:, TIDX["CL"]]),
        "F (%)":      inv_logit(p_log_logit[:, TIDX["F"]]) * 100.0,
        "PPB (%)":    inv_logit(p_log_logit[:, TIDX["PPB"]]) * 100.0,
        "Cmax (uM)":  inv_log(p_log_logit[:, TIDX["Cmax"]]),
        "Tmax (h)":   inv_log(p_log_logit[:, TIDX["Tmax"]]),
    }

def calculate_metrics(yt, yp, task=None):
    """Physical errors, task-appropriate correlations, and positive-pair GMFE.

    With task=None Pearson uses the physical scale, preserving the released
    training-selection criterion. Reported endpoint Pearson uses log10 for
    unbounded endpoints. Zero percentages remain valid correlation observations.
    """
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0: return [0] + [np.nan]*5
    if task is None and mask.sum() < 3:
        return [int(mask.sum())] + [np.nan]*5
    yt, yp = yt[mask], yp[mask]
    positive = (yt > 0) & (yp > 0)
    cy, cp = (yt[positive], yp[positive]) if task in ('t_half', 'Vd', 'CL', 'Cmax', 'Tmax') else (yt, yp)
    varying = len(cy) >= 2 and np.ptp(cy) > 0 and np.ptp(cp) > 0
    rank_varying = len(yt) >= 2 and np.ptp(yt) > 0 and np.ptp(yp) > 0
    rho = spearmanr(yt, yp)[0] if rank_varying else np.nan
    if task in ('t_half', 'Vd', 'CL', 'Cmax', 'Tmax'):
        cy, cp = np.log10(cy), np.log10(cp)
    r = pearsonr(cy, cp)[0] if varying else np.nan
    mae = np.mean(np.abs(yp - yt))
    rmse = np.sqrt(np.mean((yp - yt)**2))
    gmfe = 10**np.mean(np.abs(np.log10(yp[positive] / yt[positive]))) if positive.any() else np.nan
    
    return [int(mask.sum()), mae, rmse, r, rho, gmfe]

# --- 3. Main Engine Functions ---

@torch.no_grad()
def evaluate(model, loader, df_ref):
    """Generates predictions and calculates per-task metrics."""
    model.eval()
    device = next(model.parameters()).device
    preds, indices = [], []

    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch).cpu().numpy())
        indices.append(batch.row_idx.view(-1).cpu().numpy())

    P_raw = np.vstack(preds)
    IDX = np.concatenate(indices)
    
    # Convert to DataFrames
    pred_units = to_original_units(P_raw)
    pred_df = pd.DataFrame(pred_units, index=pd.Index(IDX, name="row_idx"))
    
    # Build truth table from clinical labels in df_ref
    sub = df_ref.loc[IDX]
    # Resolve the Tmax column name (datasets use "Tmax(hour)" or "Tmax").
    # NB: .get(a, .get(b)) does NOT fall back when column `a` exists but is all-NaN,
    # so resolve by column presence explicitly.
    # Prefer whichever column survives numeric parsing best. In df_ref, "Tmax(hour)" is a
    # STRING column whose entries often carry a trailing " ;" (e.g. "2.5 ;"); to_numeric
    # coerces those to NaN, silently discarding ~25% of the labels, while the clean "Tmax"
    # column parses fully. Picking by parsed count keeps this metric on the same labels the
    # models are trained against (log_Tmax derives from the clean column).
    _cands = [c for c in ("Tmax(hour)", "Tmax") if c in sub.columns]
    tmax_col = max(_cands, key=lambda c: pd.to_numeric(sub[c], errors="coerce").notna().sum())         if _cands else None
    truth_df = pd.DataFrame({
        "t_half (h)": pd.to_numeric(sub.get("t1/2(hour)"), errors="coerce"),
        "Vd (L)":     pd.to_numeric(sub.get("VD(liter)"), errors="coerce"),
        "CL (L/h)":   pd.to_numeric(sub.get("Cl(liter/hour)"), errors="coerce"),
        "F (%)":      pd.to_numeric(sub.get("F(percentage)"), errors="coerce"),
        "PPB (%)":    pd.to_numeric(sub.get("PPB(percentage)"), errors="coerce"),
        "Cmax (uM)":  pd.to_numeric(sub.get("Cmax_uM"), errors="coerce"),
        "Tmax (h)":   pd.to_numeric(sub[tmax_col], errors="coerce") if tmax_col else np.nan,
    }, index=pd.Index(IDX, name="row_idx"))

    # Compute metrics table
    results = []
    selection, counts = [], []
    for short, task in zip(TASKS, TASKS_WITH_UNITS):
        yt, yp = truth_df[task].values, pred_df[task].values
        results.append([task] + calculate_metrics(yt, yp, short))
        # Selection is intentionally independent of publication metric formatting.
        selection.append(calculate_metrics(yt, yp)[3])
        finite = np.isfinite(yt) & np.isfinite(yp)
        positive = finite & (yt > 0) & (yp > 0)
        counts.append((int(positive.sum()), int((finite if short in ('F', 'PPB') else positive).sum()), int(finite.sum())))
    
    metr_df = pd.DataFrame(results, columns=["Task","n","MAE","RMSE","Pearson","Spearman","GMFE"])
    metr_df['SelectionPearson'] = selection
    metr_df['n_GMFE'] = [x[0] for x in counts]
    metr_df['n_Pearson'] = [x[1] for x in counts]
    metr_df['n_Spearman'] = metr_df['n_corr'] = [x[2] for x in counts]
    return metr_df, pred_df

def fit_one(model, train_loader, val_loader, df_ref, mode="C_physio", lambda_cons=0.2,
            patience=20, lr=1e-3, weight_decay=1e-4, max_epochs=200,
            task_weights=DEFAULT_TASK_WEIGHTS, grad_clip=1.0):
    """Train one model with early stopping on validation Pearson r.

    Args:
        model: a ``GNNMultitask`` instance (already on the target device).
        train_loader / val_loader: PyG DataLoaders.
        df_ref: reference DataFrame (indexed by row_idx) holding clinical truth.
        mode: ``"C_physio"`` adds the mass-balance constraint; otherwise plain MSE.
        lambda_cons: weight of the physiology constraint (used when mode=="C_physio").
        patience: early-stopping patience (epochs without improvement).
        lr, weight_decay: AdamW hyperparameters.
        max_epochs: maximum training epochs.
        task_weights: per-task loss weights (default ``DEFAULT_TASK_WEIGHTS``;
            see module note — Vd & CL are up-weighted 5x per the manuscript).
        grad_clip: clip gradient norm to this value before each optimiser step,
            or ``None`` to disable. Defaults to ``1.0``, matching the notebook that
            produced the published checkpoints. The initial release omitted clipping
            entirely; over 15 seeds per arm that lowered macro Spearman by 0.025
            (p=0.0008) and roughly doubled its variance.

    Returns ``(model, {"means": [...], "stds": [...]})`` with the best weights loaded.
    """
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    from .utils import TaskScaler
    scaler = TaskScaler().fit_from_loader(train_loader, device=device).to(device)

    best_score, bad, best_state = -np.inf, 0, None

    for ep in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            yhat, ytrue, ymask = model(batch), batch.y.view(-1, 7), batch.y_mask.view(-1, 7)

            loss = loss_with_t12_consistency(yhat, ytrue, ymask, scaler, lambda_cons, task_weights) \
                if mode == "C_physio" else loss_supervised(yhat, ytrue, ymask, scaler, task_weights)

            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            
        # Validation
        metr, _ = evaluate(model, val_loader, df_ref)
        val_score = np.nanmean(metr["SelectionPearson"].values)

        if val_score > (best_score + 1e-6):
            best_score, bad = val_score, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break

    if best_state: model.load_state_dict(best_state)
    return model, {"means": scaler.means.tolist(), "stds": scaler.stds.tolist()}
