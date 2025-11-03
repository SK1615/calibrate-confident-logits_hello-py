import numpy as np
from data_gen import make_synthetic
from metrics import softmax, nll, ece, accuracy

def test_make_synthetic_shapes():
    (_, _), (lv, yv), (lt, yt) = make_synthetic(seed=0, n_val=200, n_test=400, K=5)
    assert lv.shape[0] == yv.shape[0]
    assert lt.shape[0] == yt.shape[0]
    assert lv.shape[1] == lt.shape[1] == 5

def test_softmax_valid():
    logits = np.array([[1.0, 2.0, 3.0]])
    p = softmax(logits)
    assert p.shape == (1,3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(p >= 0.0)

def test_metrics_monotonicity():
    # Make probs increasingly peaked toward class 0
    y = np.array([0,0,0,0,0])
    logits = np.tile(np.array([[0.0, -1.0, -5.0]]), (5,1))
    p1 = softmax(logits)
    p2 = softmax(logits * 2.0)
    assert nll(p2, y) <= nll(p1, y) + 1e-9  # more confident, same correct label → lower NLL
    assert accuracy(p2, y) == 1.0
