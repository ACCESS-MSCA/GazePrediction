"""E6: does entropy flag the held-out behaviour? AUROC(entropy: held-out-task windows vs in-distribution windows)."""
import os, numpy as np
from sklearn.metrics import roc_auc_score
PW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "per_window")
TASKS = ["Free Viewing", "Visual Search", "Saliency", "Track"]
print("| held-out task | AUROC entropy (EHT test) | mean H held-out | mean H in-dist | predicted as (share) | held-out err@20 lobo-dual / full-dual / lobo-gaze / full-gaze (mean over 3 full seeds) |")
print("|---|---|---|---|---|---|")
full_d = [np.load(f"{PW}/eht_test__e1_dual100_s{s}.npz") for s in range(3)]
full_g = [np.load(f"{PW}/eht_test__e1_gaze_s{s}.npz") for s in range(3)]
for k in range(4):
    z = np.load(f"{PW}/eht_test__e6_lobo{k}_dual100.npz"); zg = np.load(f"{PW}/eht_test__e6_lobo{k}_gaze.npz")
    p = np.clip(z["probs"], 1e-12, 1); H = -(p * np.log(p)).sum(1)
    ood = z["task"] == k
    keep = [t for t in range(4) if t != k]
    pred = np.array(keep)[p[ood].argmax(1)]
    share = ", ".join(f"{TASKS[t][:6]} {100*(pred==t).mean():.0f}%" for t in keep)
    e = lambda zz: zz["err"][zz["task"] == k, 19].mean()
    print(f"| {TASKS[k]} | {roc_auc_score(ood, H):.3f} | {H[ood].mean():.3f} | {H[~ood].mean():.3f} | {share} | "
          f"{e(z):.2f} / {np.mean([e(f) for f in full_d]):.2f} / {e(zg):.2f} / {np.mean([e(f) for f in full_g]):.2f} |")
