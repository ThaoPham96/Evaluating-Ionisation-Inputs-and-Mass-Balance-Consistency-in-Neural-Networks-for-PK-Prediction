"""No-fit controls for the standalone elimination-triad experiment.

Column order is log10(half-life hours), log10(volume L), log10(clearance L/h).
These functions never use measured labels. Strength is selected separately,
using inner validation only, and then fixed for outer evaluation.
"""
import numpy as np

LOG10_LN2 = float(np.log10(np.log(2.0)))
ALPHA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
NORMAL = np.array([1.0, -1.0, 1.0])

def residual_log10(triad):
    x = np.asarray(triad, dtype=float)
    if x.ndim == 0 or x.shape[-1] != 3:
        raise ValueError("Expected final dimension of three: log half-life, volume, clearance")
    return x @ NORMAL - LOG10_LN2

def partial_correction(triad, alpha, method="orthogonal"):
    """Correct a fraction alpha of residual; preserve invalid rows as invalid.

    orthogonal: equal squared-log-distance projection towards the plane.
    half_life: move half-life only towards log10(ln(2)*V/CL).
    alpha=0 is exactly the input; alpha=1 enforces the identity.
    Invalid inputs are not repaired or dropped by this analysis control.
    """
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a finite scalar in [0, 1]")
    if method not in {"orthogonal", "half_life"}:
        raise ValueError("Unknown correction method")
    x = np.asarray(triad, dtype=float)
    residual_log10(x)  # Validate shape even at alpha zero.
    result = x.copy()
    valid = np.isfinite(x).all(axis=-1)
    delta = float(alpha) * residual_log10(x[valid])
    if method == "orthogonal":
        result[valid] = x[valid] - delta[..., None] * NORMAL / 3.0
    else:
        values = result[valid].copy()
        values[..., 0] -= delta
        result[valid] = values
    return result

def coherent_label_error_floor(observed_triad):
    """Lower bound on equal-triad log-MAE for predictions forced onto the plane.

    Descriptive observed-label diagnostic, not a model or training target.
    Requires a complete, positive triad before taking logs, from one source row.
    """
    x = np.asarray(observed_triad, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() or len(x) == 0:
        raise ValueError("Expected nonempty finite complete observed log triads")
    return float(np.mean(np.abs(residual_log10(x))) / 3.0)
