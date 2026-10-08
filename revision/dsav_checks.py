"""E8: DSAV360 data checks - subset description, stereo/mono split of zero-shot error, Step-0 offset (both datasets)."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_data import build_windows, recording_dirs, angular_error
RAW = "/mnt/DataLake/AlienBackup/access/dsav360_extracted"
PW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "per_window")
out = ["# E8 DSAV360 data checks\n"]
stereo = {}
for v in range(1, 11):
    h = pd.read_csv(f"{RAW}/head_data/head_data/head_video_{v:04d}.csv", usecols=["id", "stereo"]).drop_duplicates()
    for _, r in h.iterrows():
        stereo[(str(v), str(r.id))] = bool(r.stereo)
sp = json.load(open("/mnt/DataLake/AlienBackup/access/processed_dataset/splits/dsav360_splits.json"))
recs = [r for s in sp.values() for r in s]
users = {r.split("/")[1].split("_")[1] for r in recs}
out.append(f"Processed subset: videos 1–10 (of 85), {len(users)} participants, {len(recs)} recordings "
           f"(train {len(sp['train'])} / val {len(sp['val'])} / test {len(sp['test'])}; only test is used, zero-shot). "
           f"Stereo viewers among test recordings: {np.mean([stereo[(r.split('/')[0][6:], r.split('/')[1].split('_')[1])] for r in sp['test']])*100:.1f}%.\n")
# Stereo vs mono error
rows = []
for name in ["e1_gaze_s0", "e2_dual20_s0", "e1_dual100_s0", "base_hmd_cv"]:
    p = f"{PW}/dsav_test__{name}.npz"
    if not os.path.exists(p):
        continue
    z = np.load(p, allow_pickle=True)
    st = np.array([stereo[(str(v), str(u))] for v, u in zip(z["video"], z["user"])])
    for cond, m in (("mono", ~st), ("stereo", st)):
        rows.append(f"| {name} | {cond} | {m.sum()} | {z['err'][m, 9].mean():.2f} | {z['err'][m, 19].mean():.2f} | {z['hmd_err'][m, 9].mean():.2f} | {z['hmd_err'][m, 19].mean():.2f} |")
out.append("## Zero-shot error by viewing condition\n\n| model | condition | N | step10 | step20 | HMD step10 | HMD step20 |\n|---|---|---|---|---|---|---|")
out += rows
# Step-0 offset: angle between gaze and HMD direction at the prediction time f (all test windows)
out.append("\n## Step-0 head–gaze offset at prediction time (great-circle, deg)\n\n| dataset | N windows | mean | median | 90th pct |\n|---|---|---|---|---|")
for ds in ("eht", "dsav"):
    w = build_windows(ds, recording_dirs(ds, "test"), future=20)
    # gt[:,0] is gaze at frame f relative to HMD at f
    e0 = angular_error(np.zeros((len(w["gt"]), 2)), w["gt"][:, 0])
    out.append(f"| {ds} | {len(e0)} | {e0.mean():.2f} | {np.median(e0):.2f} | {np.percentile(e0, 90):.2f} |")
txt = "\n".join(out)
open(os.path.join(os.path.dirname(PW), "dsav_checks.md"), "w").write(txt)
print(txt)
