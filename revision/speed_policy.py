"""E4-f: horizon policy conditioned on recent head speed (4 quantile bins fit on validation) vs predicted behaviour."""
import sys, numpy as np
sys.path.insert(0, '.')
from analyze import load, select_horizons, boot_ci, fmt
from cached_data import build_windows, recording_dirs
rng = np.random.default_rng(0)
def speed(ds, split):
    w = build_windows(ds, recording_dirs(ds, split))
    return np.linalg.norm(w['ts'][:, -5:, :2], axis=2).mean(1)
sp = {k: speed(*k.split(':')) for k in ['eht:val', 'eht:test', 'dsav:test']}
print("| model | target | tau | behaviour policy (mean h) | speed-quartile policy (mean h) | behaviour × speed (8 bins) | fixed matched to behaviour | speed − behaviour |")
print("|---|---|---|---|---|---|---|---|")
for d in ['e2_dual20_s0', 'e1_dual100_s0']:
    va = load('eht_val', d)
    q = np.quantile(sp['eht:val'], [0.25, 0.5, 0.75])
    bins_va = np.digitize(sp['eht:val'], q)
    med = np.median(sp['eht:val'])
    for tau in (9, 10, 11):
        hb = select_horizons(va['err'], va['pred_task'], tau)
        hs = select_horizons(va['err'], bins_va, tau)
        comb_va = va['pred_task'] * 2 + (sp['eht:val'] > med)
        hc = {}
        for c in range(8):
            m = comb_va == c
            curve = va['err'][m].mean(0) if m.sum() else np.full(20, 99.)
            ok = np.where(curve <= tau)[0]; hc[c] = int(ok.max() + 1) if len(ok) else 1
        for t in ['eht_test', 'dsav_test']:
            z = load(t, d); s = sp[t.replace('_', ':')]; n = np.arange(len(s))
            h1 = np.array([hb[c] for c in z['pred_task']]); h2 = np.array([hs[c] for c in np.digitize(s, q)])
            h3 = np.array([hc[c] for c in z['pred_task'] * 2 + (s > med)])
            m1 = h1.mean(); lo = int(m1); hm = np.where(rng.random(len(h1)) < m1 - lo, lo + 1, lo)
            e1, e2, e3, ef = z['err'][n, h1 - 1], z['err'][n, h2 - 1], z['err'][n, h3 - 1], z['err'][n, hm - 1]
            print(f"| {d} | {t} | {tau} | {e1.mean():.2f} ({h1.mean():.1f}) | {e2.mean():.2f} ({h2.mean():.1f}) | {e3.mean():.2f} ({h3.mean():.1f}) | {ef.mean():.2f} | {fmt(boot_ci(e2 - e1, z['user']))} |")

print("\n### Gain of each policy over a fixed horizon with ITS OWN mean (tau=10, lambda=20)\n")
print("| target | policy | mean h | policy err | own matched fixed | gain [95% CI] |\n|---|---|---|---|---|---|")
d = 'e2_dual20_s0'; va = load('eht_val', d)
q = np.quantile(sp['eht:val'], [0.25, 0.5, 0.75])
hb = select_horizons(va['err'], va['pred_task'], 10); hs = select_horizons(va['err'], np.digitize(sp['eht:val'], q), 10)
for t in ['eht_test', 'dsav_test']:
    z = load(t, d); s = sp[t.replace('_', ':')]; n = np.arange(len(s))
    for name, h in (("behaviour", np.array([hb[c] for c in z['pred_task']])), ("head-speed quartile", np.array([hs[c] for c in np.digitize(s, q)]))):
        m = h.mean(); lo = int(m); hm = np.where(rng.random(len(h)) < m - lo, lo + 1, lo)
        e, ef = z['err'][n, h - 1], z['err'][n, hm - 1]
        print(f"| {t} | {name} | {m:.2f} | {e.mean():.2f} | {ef.mean():.2f} | {fmt(boot_ci(e - ef, z['user']))} |")
