"""Aggregate retrained runs (participant-exclusive splits) over seeds (E1-R, E2, E3, E6) from per-window outputs."""
import os
import re
import json
import glob
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PW = os.path.join(HERE, "results", "per_window")
TASKS = ["Free Viewing", "Visual Search", "Saliency", "Track"]
TARGETS = ["eht_test", "eht_isvc_holdout", "dsav_test"]


def load(t, n):
    p = os.path.join(PW, f"{t}__{n}.npz")
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


def stats(z):
    e = z["err"]
    d = {"step10": e[:, 9].mean()}
    if e.shape[1] >= 20:
        d.update(step15=e[:, 14].mean(), step20=e[:, 19].mean(), mean20=e.mean())
        for i, t in enumerate(TASKS):
            m = z["task"] == i
            if m.any():
                d[f"{t}@20"] = e[m, 19].mean()
    if z["probs"].size and z["probs"].shape[1] == 4:
        d["acc"] = (z["probs"].argmax(1) == z["task"]).mean()
    return d


def ms(vals):
    v = np.array(vals, float)
    return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v.mean():.2f}"


def main():
    names = sorted({os.path.basename(p).split("__")[1][:-4] for p in glob.glob(os.path.join(PW, "*__e*_*.npz"))})
    groups = defaultdict(list)
    for n in names:
        groups[re.sub(r"_s\d+$", "", n)].append(n)
    out = ["# Retrained runs (user-exclusive split) — mean ± std over seeds\n"]
    js = {}
    for t in TARGETS:
        rows = []
        for g, members in sorted(groups.items()):
            st = [stats(z) for z in (load(t, m) for m in members) if z is not None]
            if not st:
                continue
            keys = [k for k in st[0] if all(k in s for s in st)]
            rows.append({"model": g, "seeds": len(st), **{k: ms([s[k] for s in st]) for k in keys}})
        for b in ("base_hmd_cv", "base_lin_motion", "base_nakashima"):
            z = load(t, b)
            if z is not None:
                rows.append({"model": b, "seeds": "-", **{k: f"{v:.2f}" for k, v in stats(z).items()}})
        z = load(t, "gaze_only")
        if z is not None:
            rows.append({"model": "HMD-centre", "seeds": "-", "step10": f"{z['hmd_err'][:, 9].mean():.2f}",
                         "step20": f"{z['hmd_err'][:, 19].mean():.2f}", "mean20": f"{z['hmd_err'].mean():.2f}"})
        if not rows:
            continue
        cols = ["model", "seeds", "step10", "step15", "step20", "mean20", "acc"] + [f"{x}@20" for x in TASKS]
        out.append(f"\n## {t}\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
        for r in rows:
            out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        # paired seed-matched dual - gaze
        diffs = []
        for s in range(10):
            a, b = load(t, f"e1_dual100_s{s}"), load(t, f"e1_gaze_s{s}")
            if a is not None and b is not None:
                diffs.append((a["err"][:, 19] - b["err"][:, 19]).mean())
        if diffs:
            out.append(f"\nSeed-matched dual(λ=100) − gaze-only at step 20: per seed {[round(x, 3) for x in diffs]}, mean {np.mean(diffs):.3f}")
        js[t] = rows
    txt = "\n".join(out)
    open(os.path.join(HERE, "results", "seeds_summary.md"), "w").write(txt)
    json.dump(js, open(os.path.join(HERE, "results", "seeds_summary.json"), "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
