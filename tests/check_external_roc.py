"""Check ties, weighted/repeated observations, feasibility, and analytic areas."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from external_roc import FittedROC

for s, expected in (([.1,.2,.8,.9], 1.), ([.8,.9,.1,.2], .5*(1-.005/.095)),
                    ([.5,.5,.5,.5], .5)):
    got = FittedROC(s, [0,1,2,3]).evaluate()
    np.testing.assert_allclose(got['standardized_pauc'], expected, atol=1e-14)

rng = np.random.default_rng(61)
g = np.array([0,0,1,1,2,2,3,4])
for _ in range(50):
    scores = rng.integers(0,5,len(g))/4
    weights = rng.integers(0,5,len(g))
    weights[0] = max(weights[0],1)
    weights[4] = max(weights[4],1)
    weighted = FittedROC(scores,g).evaluate(weights)
    expanded = FittedROC(np.repeat(scores,weights),np.repeat(g,weights)).evaluate()
    for key in weighted:
        np.testing.assert_allclose(weighted[key],expanded[key],atol=1e-14)
    assert weighted['specificity'] >= .9
    candidates=np.r_[np.nextafter(np.unique(scores),np.inf),np.unique(scores),np.inf]
    feasible=[]
    for t in candidates:
        if (weights*(scores<t))[g<2].sum()/weights[g<2].sum() >= .9:
            feasible.append((weights*(scores>=t))[g>=2].sum()/weights[g>=2].sum())
    np.testing.assert_allclose(weighted['referable_sensitivity'],max(feasible),atol=1e-14)

try:
    FittedROC([.1,.2,.3], [0,2,3]).evaluate([1,.5,1])
except ValueError: pass
else: raise AssertionError('Fractional bootstrap counts must fail')
print('EXTERNAL_ROC_TIES_WEIGHTED_RESAMPLING_AND_BOUNDARY_CHECKS_PASS')
