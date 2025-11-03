import argparse, numpy as np
from grade import evaluate_once

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mod_path", default="solve.py")
    args = ap.parse_args()

    passes, nlls, eces, confs = 0, [], [], []
    for i in range(args.n):
        ok, rep = evaluate_once(seed=1234 + i, mod_path=args.mod_path)
        print(rep)
        d = rep["deltas"]
        nlls.append(d["nll_improve"])
        eces.append(d["ece_improve"])
        confs.append(d["conf_reduction"])
        if ok:
            passes += 1

    nlls, eces, confs = map(np.array, (nlls, eces, confs))
    print(f"Passes: {passes}/{args.n} = {passes/args.n:.2%}")
    for name, arr in [("nll_improve", nlls), ("ece_improve", eces), ("conf_reduction", confs)]:
        q = np.percentile(arr, [25, 50, 60, 70, 80])
        print(f"{name} percentiles (25/50/60/70/80): {q.round(4)}")
