import argparse, numpy as np
from grade import evaluate_once

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--q", type=float, default=0.30)  # target pass quantile
    args = ap.parse_args()

    nlls, eces, confs = [], [], []
    for i in range(args.n):
        _, rep = evaluate_once(seed=1234 + i, mod_path="solve.py")
        d = rep["deltas"]
        nlls.append(d["nll_improve"])
        eces.append(d["ece_improve"])
        confs.append(d["conf_reduction"])

    nlls, eces, confs = map(np.array, (nlls, eces, confs))
    q = args.q

    req_nll  = float(np.percentile(nlls, 100*(1-q)))
    req_ece  = float(np.percentile(eces, 100*(1-q)))
    req_conf = float(np.percentile(confs, 100*(q)))  # want to be BELOW the upper tail

    print("\nSuggested thresholds for ~{:.0f}% pass:".format(q*100))
    print(f"REQ_NLL_IMP  = {req_nll:.3f}")
    print(f"REQ_ECE_IMP  = {req_ece:.3f}")
    print(f"REQ_CONF_RED = {req_conf:.3f}  # adjust ±0.01 to fine tune 10–40%")
    print("MAX_ACC_DROP = 0.004  # keep")
    print("\nPercentiles:")
    for name, arr in [("nll", nlls), ("ece", eces), ("conf", confs)]:
        ps = np.percentile(arr, [25, 50, 60, 70, 80])
        print(f"{name}_improve percentiles 25/50/60/70/80: {ps.round(3)}")
