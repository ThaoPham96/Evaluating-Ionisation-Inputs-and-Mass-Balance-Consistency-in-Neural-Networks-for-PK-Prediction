import numpy as np
import pandas as pd
import pytest
from evaluate import Bootstrap,Cohort,Evaluator,hard_derive,K
from sources import aligned

def tiny():
    rows=pd.DataFrame(dict(structure_id=['a','b','c','d'],group_id=['x','x','y','z'],outer_fold=[1,1,2,3])).set_index('structure_id')
    truth=np.arange(28,dtype=float).reshape(4,7)/10+1
    return Cohort('test',rows,truth,np.ones(4,bool),np.array([0,0,1,2]))

def test_whole_group_and_missing_labels_manual():
    c=tiny(); b=Bootstrap(c,[101,202],draws=20,seed=17)
    values=np.array([[1.,100.,3.,4.],[2.,200.,6.,8.]])
    mask=np.array([1,0,1,1],bool)
    point,draw=b.means(values,mask)
    manual=[]
    for counts in b.counts:
        w=counts[c.group]*mask
        manual.append((values*w).sum(axis=1)/w.sum())
    np.testing.assert_allclose(draw,manual)
    np.testing.assert_allclose(point,values[:,mask].mean(axis=1))

def test_hard_all_three_directions_and_idempotence():
    q=np.random.default_rng(1).normal(size=(3,9,7))
    for e,j in [('t_half',0),('Vd',1),('CL',2)]:
        h=hard_derive(q,e)
        np.testing.assert_array_equal(h[:,:,[i for i in range(7) if i!=j]],q[:,:,[i for i in range(7) if i!=j]])
        np.testing.assert_allclose(hard_derive(h,e),h)
        np.testing.assert_allclose(h[:,:,0]-h[:,:,1]+h[:,:,2]-K,0,atol=1e-14)

def test_ratio_of_seed_means_not_mean_seed_ratios_and_interaction_scale():
    c=tiny(); b=Bootstrap(c,[101,202],draws=50,seed=7); ev=Evaluator(50)
    values={'00':np.array([[1.,2,3,4],[5.,6,7,8]]),'10':np.array([[2.,2,3,4],[6.,6,7,8]]),'01':np.array([[2.,3,4,5],[6.,7,8,9]]),'11':np.array([[4.,4,5,6],[8.,8,9,10]])}
    stats={}
    for arm,v in values.items():
        p,d=b.means(v,np.ones(4,bool));stats[arm]={('all','t_half'):(p,d,4,'log10')}
    ev.contrast(c,'test','T','interaction',{'11':1,'10':-1,'01':-1,'00':1},'00',stats,b,'crossed_fixed',only_triads=True)
    row=ev.contrasts[0]
    expected=100*(values['11']-values['10']-values['01']+values['00']).mean()/values['00'].mean()
    assert row['relative_pct']==pytest.approx(expected)
    assert row['denominator']=='00'
    manual=100*(stats['11'][('all','t_half')][1]-stats['10'][('all','t_half')][1]-stats['01'][('all','t_half')][1]+stats['00'][('all','t_half')][1]).mean(axis=1)/stats['00'][('all','t_half')][1].mean(axis=1)
    np.testing.assert_allclose([row['primary_low'],row['primary_high']],np.quantile(manual,[.025,.975]))

def test_duplicate_prediction_and_fold_leak_fail_closed():
    c=tiny(); d=pd.DataFrame(dict(structure_id=list(c.rows.index),seed=101,outer_fold=[1,1,2,3],**{'q_'+e:1. for e in ('t_half','Vd','CL','Cmax','Tmax','F','PPB')}))
    assert aligned(d.sample(frac=1,random_state=1),c,[101]).shape==(1,4,7)
    with pytest.raises(ValueError):aligned(pd.concat([d,d.iloc[:1]]),c,[101])
    d.loc[0,'outer_fold']=2
    with pytest.raises(ValueError,match='fold'):aligned(d,c,[101])
