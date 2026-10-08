#!/bin/bash
# Evaluate CV runs on their own fold test split (+ DSAV), λ=20 seeds on standard targets, then summarise.
cd "$(dirname "$0")"
PY="conda run --no-capture-output -n access_env python"
for k in 0 1 2 3 4; do
  M=""; X=""
  for n in cv${k}_gaze cv${k}_dual20; do [ -f runs/$n/best.pth ] && { M="$M $n"; X="$X $n=$PWD/runs/$n/best.pth"; }; done
  [ -n "$M" ] && $PY eval_unified.py --models $M --extra $X --targets eht:fold${k}_test dsav:test 2>&1 | grep "^\["
done
M=""; X=""; for n in e2_dual20_s1 e2_dual20_s2; do [ -f runs/$n/best.pth ] && { M="$M $n"; X="$X $n=$PWD/runs/$n/best.pth"; }; done
[ -n "$M" ] && $PY eval_unified.py --models $M --extra $X --targets eht:test eht:val eht:isvc_holdout dsav:test 2>&1 | grep "^\["
$PY - <<'PY'
import numpy as np, os
PW="results/per_window"; rows=[]
for k in range(5):
    r={}
    for n in ("gaze","dual20"):
        p=f"{PW}/eht_fold{k}_test__cv{k}_{n}.npz"
        if os.path.exists(p):
            z=np.load(p); r[n]=(z["err"][:,9].mean(), z["err"][:,19].mean(), z["hmd_err"][:,19].mean(), len(z["err"]))
            if n=="dual20": r["acc"]=(z["probs"].argmax(1)==z["task"]).mean()
            r["d"]=z
    rows.append(r)
out=["# E7 user-level 5-fold CV (EHTask)\n","| fold | N | HMD@20 | gaze@10 | gaze@20 | dual20@10 | dual20@20 | acc | dual−gaze@20 |","|---|---|---|---|---|---|---|---|---|"]
G=[];D=[];A=[];H=[]
for k,r in enumerate(rows):
    if "gaze" in r and "dual20" in r:
        g,d=r["gaze"],r["dual20"]; G.append(g);D.append(d);A.append(r["acc"])
        out.append(f"| {k} | {g[3]} | {g[2]:.2f} | {g[0]:.2f} | {g[1]:.2f} | {d[0]:.2f} | {d[1]:.2f} | {r['acc']:.3f} | {d[1]-g[1]:+.2f} |")
if G:
    G=np.array(G);D=np.array(D)
    out.append(f"| **mean ± sd** | | {G[:,2].mean():.2f} ± {G[:,2].std(ddof=1):.2f} | {G[:,0].mean():.2f} ± {G[:,0].std(ddof=1):.2f} | {G[:,1].mean():.2f} ± {G[:,1].std(ddof=1):.2f} | {D[:,0].mean():.2f} ± {D[:,0].std(ddof=1):.2f} | {D[:,1].mean():.2f} ± {D[:,1].std(ddof=1):.2f} | {np.mean(A):.3f} ± {np.std(A,ddof=1):.3f} | {(D[:,1]-G[:,1]).mean():+.2f} ± {(D[:,1]-G[:,1]).std(ddof=1):.2f} |")
open("results/cv_summary.md","w").write("\n".join(out)); print("\n".join(out))
PY
$PY seeds_summary.py > /dev/null && grep -A2 "e2_dual20 " results/seeds_summary.md | head -3
