import numpy as np
import pytest
from mass_balance_controls import ALPHA_GRID, LOG10_LN2, partial_correction, residual_log10, coherent_label_error_floor

def test_physical_identity_and_partial_projection():
    # Physically coherent: V=100 L, CL=10 L/h, half-life=ln(2)*10 h.
    coherent=np.log10([[np.log(2)*10,100,10]])
    np.testing.assert_allclose(residual_log10(coherent),0,atol=1e-15)
    x=np.log10([[4,100,10],[20,50,2],[2,30,9]])
    original=x.copy()
    for alpha in ALPHA_GRID:
        for method in ["orthogonal","half_life"]:
            y=partial_correction(x,alpha,method)
            np.testing.assert_allclose(residual_log10(y),(1-alpha)*residual_log10(x),atol=1e-15)
    np.testing.assert_array_equal(x,original)
    np.testing.assert_array_equal(partial_correction(x,0),x)

def test_derived_half_life_preserves_other_endpoints():
    x=np.log10([[4,100,10]])
    y=10**partial_correction(x,1,"half_life")
    np.testing.assert_allclose(y,[[np.log(2)*10,100,10]])

def test_projection_attains_error_floor_and_shortest_distance():
    x=np.log10([[4,100,10],[20,50,2]])
    projected=partial_correction(x,1)
    derived=partial_correction(x,1,"half_life")
    np.testing.assert_allclose(np.mean(np.abs(projected-x)),coherent_label_error_floor(x))
    assert np.sum((projected-x)**2) <= np.sum((derived-x)**2)

def test_invalid_predictions_are_not_silently_repaired():
    x=np.array([[np.nan,1,2],[1,np.inf,3],[1,2,3]])
    y=partial_correction(x,.5)
    np.testing.assert_array_equal(y[:2],x[:2])
    assert np.isfinite(y[2]).all()

@pytest.mark.parametrize("alpha",[-.1,1.01,float("nan"),float("inf"),[.1,.2]])
def test_invalid_strength(alpha):
    with pytest.raises(ValueError):partial_correction([[1,2,3]],alpha)

def test_log_ensemble_commutes_only_for_common_strength():
    folds=np.array([[[1,2,3]],[[4,5,6]]],dtype=float)
    left=np.mean([partial_correction(x,.5) for x in folds],axis=0)
    np.testing.assert_allclose(left,partial_correction(folds.mean(axis=0),.5))
    distinct=np.mean([partial_correction(folds[0],0),partial_correction(folds[1],1)],axis=0)
    assert not np.allclose(distinct,partial_correction(folds.mean(axis=0),.5))
