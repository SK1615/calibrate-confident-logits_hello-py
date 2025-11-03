import importlib.util
import numpy as np
from data_gen import make_synthetic
from metrics import softmax, nll, ece, accuracy

# ===== Thresholds tuned for a ~10–40% pass window on this generator =====
REQ_NLL_IMP  = 8.25   # NLL improvement >= this
REQ_ECE_IMP  = 0.510  # ECE improvement >= this
REQ_CONF_RED = 0.52  # mean max-confidence reduction >= this
MAX_ACC_DROP = 0.004  # accuracy must not drop by more than this

def _mean_max_conf(p: np.ndarray) -> float:
    return float(np.mean(np.max(p, axis=1)))

def load_solution(mod_path="solve.py"):
    spec = importlib.util.spec_from_file_location("solve_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "solve"), f"{mod_path} must define solve(...)"
    return mod.solve

def evaluate_once(seed: int, mod_path: str = None):
    (_, _), (logits_val, y_val), (logits_test, y_test) = make_synthetic(seed=seed)
    probs_base = softmax(logits_test)

    # module to grade (default: wrapper solve.py)
    solve = load_solution(mod_path or "solve.py")
    probs_cal = solve(logits_val, y_val, logits_test)

    # validity checks
    assert probs_cal.shape == probs_base.shape, "Shape mismatch."
    assert np.all(probs_cal >= -1e-12), "Negative probabilities."
    assert np.allclose(probs_cal.sum(axis=1), 1.0, atol=1e-6), "Rows must sum to 1."

    # metrics on TEST
    base_nll, cal_nll = nll(probs_base, y_test), nll(probs_cal, y_test)
    base_ece, cal_ece = ece(probs_base, y_test), ece(probs_cal, y_test)
    base_acc, cal_acc = accuracy(probs_base, y_test), accuracy(probs_cal, y_test)

    nll_improve = base_nll - cal_nll
    ece_improve = base_ece - cal_ece
    acc_drop    = base_acc - cal_acc

    mmc_base = _mean_max_conf(probs_base)
    mmc_cal  = _mean_max_conf(probs_cal)
    conf_red = mmc_base - mmc_cal

    passed = (
        (nll_improve >= REQ_NLL_IMP) and
        (ece_improve >= REQ_ECE_IMP) and
        (acc_drop    <= MAX_ACC_DROP) and
        (conf_red    >= REQ_CONF_RED)
    )

    report = {
        "seed": seed,
        "base": {"nll": float(base_nll), "ece": float(base_ece), "acc": float(base_acc), "mmc": float(mmc_base)},
        "cal":  {"nll": float(cal_nll),  "ece": float(cal_ece),  "acc": float(cal_acc), "mmc": float(mmc_cal)},
        "deltas": {
            "nll_improve": float(nll_improve),
            "ece_improve": float(ece_improve),
            "acc_drop": float(acc_drop),
            "conf_reduction": float(conf_red),
        },
        "passed": bool(passed),
        "thresholds": {
            "REQ_NLL_IMP": REQ_NLL_IMP,
            "REQ_ECE_IMP": REQ_ECE_IMP,
            "MAX_ACC_DROP": MAX_ACC_DROP,
            "REQ_CONF_RED": REQ_CONF_RED,
        },
    }
    return passed, report

if __name__ == "__main__":
    ok, rep = evaluate_once(seed=1234, mod_path="solve.py")
    print(rep)
    print("PASS" if ok else "FAIL")
