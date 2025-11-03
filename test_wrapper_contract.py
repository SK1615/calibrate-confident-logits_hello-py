import numpy as np
from data_gen import make_synthetic
import importlib.util

def load_solve(mod):
    spec = importlib.util.spec_from_file_location("solve_mod", mod)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.solve

def test_solve_contract():
    solve = load_solve("solve.py")
    (_, _), (lv, yv), (lt, yt) = make_synthetic(seed=1, n_val=100, n_test=150, K=5)
    p = solve(lv, yv, lt)
    assert p.shape == (lt.shape[0], lt.shape[1])
    assert np.all(p >= -1e-12)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
