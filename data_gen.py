import numpy as np

def make_synthetic(seed=0, n_train=2000, n_val=2000, n_test=4000, K=5):
    """
    Generates synthetic (logits, labels) with seed-dependent over-confidence and
    a mild val->test shift. Labels are integer IDs in [0..K-1].
    """
    rng = np.random.default_rng(seed)

    d = 8
    centers = rng.normal(0, 1.2, size=(K, d))

    def sample_split(n):
        y = rng.integers(0, K, size=n)
        X = centers[y] + rng.normal(0, 1.0, size=(n, d))
        return X, y

    Xtr, ytr = sample_split(n_train)
    Xv,  yv  = sample_split(n_val)
    Xte, yte = sample_split(n_test)

    W = rng.normal(0, 0.8, size=(d, K))
    b = rng.normal(0, 0.2, size=(K,))

    # Over-confidence factor varies per seed (some easy, some hard)
    scale_val  = rng.uniform(1.3, 2.6)
    scale_test = scale_val * rng.uniform(0.85, 1.15)  # mild shift

    # small label noise on VAL ONLY (forces robust fit; still solvable)
    noise_rate = rng.uniform(0.00, 0.06)
    if noise_rate > 0:
        flip = rng.random(len(yv)) < noise_rate
        if np.any(flip):
            yv = yv.copy()
            yv[flip] = rng.integers(0, K, size=np.sum(flip))

    logits_tr  = scale_val  * (Xtr @ W + b)  # train unused by grader
    logits_val = scale_val  * (Xv  @ W + b)
    logits_test= scale_test * (Xte @ W + b)

    return (logits_tr, ytr), (logits_val, yv), (logits_test, yte)
