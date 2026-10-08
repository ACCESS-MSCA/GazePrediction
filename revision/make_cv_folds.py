"""E7: user-level K-fold splits for EHTask (test users disjoint across folds; val users disjoint from train)."""
import json, os, random
from glob import glob
ROOT = "/mnt/DataLake/AlienBackup/access/processed_dataset/sal_cache25/eht"
recs = sorted(os.path.relpath(p, ROOT) for p in glob(os.path.join(ROOT, "Video_*", "User_*_Task_*")))
users = sorted({r.split("/")[1].split("_")[1] for r in recs}, key=int)
random.Random(42).shuffle(users)
K = 5
folds = []
for k in range(K):
    test_u = set(users[k::K])
    rest = [u for u in users if u not in test_u]
    val_u = set(rest[:4])
    part = lambda S: [r for r in recs if r.split("/")[1].split("_")[1] in S]
    folds.append({"test": part(test_u), "val": part(val_u), "train": part(set(rest) - val_u),
                  "test_users": sorted(test_u, key=int), "val_users": sorted(val_u, key=int)})
    print(k, "test users", sorted(test_u, key=int), "val users", sorted(val_u, key=int), {s: len(folds[-1][s]) for s in ("train", "val", "test")})
os.makedirs("results", exist_ok=True)
json.dump(folds, open("results/cv_folds.json", "w"), indent=1)
for k, f in enumerate(folds):
    json.dump({s: f[s] for s in ("train", "val", "test")}, open(f"results/cv_fold{k}.json", "w"), indent=1)
