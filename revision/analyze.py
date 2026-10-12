"""E0/E4/E5 analyses from per-window outputs written by eval_unified.py.

Usage: python analyze.py --dual dual_l100 --gaze gaze_only [--tag existing]
Writes revision/results/report_<tag>.md and report_<tag>.json
"""
import os
import json
import argparse

import numpy as np

from cached_data import keep_recorded
from scipy.stats import spearmanr, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
PW = os.path.join(HERE, "results", "per_window")
TASKS = ["Free Viewing", "Visual Search", "Saliency", "Track"]
RNG = np.random.default_rng(0)
N_BOOT = 2000


def load(target, name):
    p = os.path.join(PW, f"{target}__{name}.npz")
    if not os.path.exists(p):
        return None
    z = keep_recorded(target, dict(np.load(p, allow_pickle=True)))
    if z["probs"].size:
        pr = np.clip(z["probs"], 1e-12, 1)
        z["entropy"] = -(pr * np.log(pr)).sum(1)
        z["pred_task"] = z["probs"].argmax(1)
    return z


def boot_ci(values, groups, stat=np.mean):
    """Cluster bootstrap over groups (participants). Returns (point, lo, hi)."""
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    pts = []
    for _ in range(N_BOOT):
        s = RNG.choice(ug, len(ug), replace=True)
        ii = np.concatenate([idx[g] for g in s])
        pts.append(stat(values[ii]))
    return float(stat(values)), float(np.percentile(pts, 2.5)), float(np.percentile(pts, 97.5))


def fmt(ci):
    return f"{ci[0]:.2f} [{ci[1]:.2f}, {ci[2]:.2f}]"


def paired_diff_ci(a, b, groups):
    return boot_ci(a - b, groups)


# ---------------------------------------------------------------- E0 tables
def table_errors(models, z_ref, groups, label_key="task"):
    rows = []
    hmd = z_ref["hmd_err"]
    sel_sets = [("Global", np.ones(len(hmd), bool))] + [(t, z_ref[label_key] == i) for i, t in enumerate(TASKS)]
    for sname, m in sel_sets:
        row = {"subset": sname, "N": int(m.sum())}
        for k in (10, 15, 20):
            row[f"HMD@{k}"] = fmt(boot_ci(hmd[m, k - 1], groups[m]))
            for mn, z in models.items():
                if z["err"].shape[1] >= k:
                    row[f"{mn}@{k}"] = fmt(boot_ci(z["err"][m, k - 1], groups[m]))
        row["HMD mean1-20"] = fmt(boot_ci(hmd[m].mean(1), groups[m]))
        for mn, z in models.items():
            if z["err"].shape[1] >= 20:
                row[f"{mn} mean1-20"] = fmt(boot_ci(z["err"][m].mean(1), groups[m]))
        rows.append(row)
    return rows


def md_table(rows, cols=None):
    cols = cols or list(rows[0].keys())
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return out


# ---------------------------------------------------------------- E4 horizon policies
def select_horizons(err_val, labels_val, tau, rule="endpoint"):
    """Largest k (1..20) per class such that the validation endpoint (or interval-mean) error <= tau; min 1."""
    hz = {}
    for c in range(4):
        m = labels_val == c
        curve = err_val[m].mean(0)
        if rule == "interval":
            curve = np.cumsum(curve) / np.arange(1, len(curve) + 1)
        ok = np.where(curve <= tau)[0]
        hz[c] = int(ok.max() + 1) if len(ok) else 1
    return hz


def eval_policy(err, horizons):
    """err: (N,20). horizons: (N,) ints in 1..20 -> per-window endpoint error at the chosen horizon."""
    return err[np.arange(len(err)), horizons - 1]


def policies(dual_te, gaze_te, dual_va, tau, rule):
    lab_va_pred = dual_va["pred_task"]
    hz_pred = select_horizons(dual_va["err"], lab_va_pred, tau, rule)       # selection on predicted labels (deployable)
    hz_gt = select_horizons(dual_va["err"], dual_va["task"], tau, rule)      # oracle-label selection
    h_pred = np.array([hz_pred[c] for c in dual_te["pred_task"]])
    h_oracle = np.array([hz_gt[c] for c in dual_te["task"]])
    mean_h = h_pred.mean()
    res = {"tau": tau, "rule": rule, "horizons_pred": hz_pred, "horizons_gt": hz_gt, "mean_h": float(mean_h)}
    res["adaptive_dual"] = eval_policy(dual_te["err"], h_pred)
    res["adaptive_gaze_only_at_dual_h"] = eval_policy(gaze_te["err"], h_pred)
    res["oracle_task_dual"] = eval_policy(dual_te["err"], h_oracle)
    res["random_same_dist_dual"] = eval_policy(dual_te["err"], RNG.permutation(h_pred))
    lo, hi = int(np.floor(mean_h)), int(np.ceil(mean_h))
    frac = mean_h - lo  # fixed horizon matched in mean: mix of floor/ceil
    hmix = np.where(RNG.random(len(h_pred)) < frac, hi, lo).clip(1, 20)
    res["fixed_matched_dual"] = eval_policy(dual_te["err"], hmix)
    res["fixed_matched_gaze_only"] = eval_policy(gaze_te["err"], hmix)
    res["hmd_at_adaptive"] = eval_policy(dual_te["hmd_err"], h_pred)
    return res


# ---------------------------------------------------------------- E5 entropy analyses
def partial_spearman(x, y, strata):
    """Spearman of x,y after removing stratum means of their ranks (controls for predicted task)."""
    rx, ry = rankdata(x), rankdata(y)
    for s in np.unique(strata):
        m = strata == s
        rx[m] -= rx[m].mean()
        ry[m] -= ry[m].mean()
    return float(np.corrcoef(rx, ry)[0, 1])


def aurc(conf, err):
    """Area under risk-coverage curve: sort by confidence desc, cumulative mean error. Lower = better."""
    o = np.argsort(-conf, kind="stable")
    risk = np.cumsum(err[o]) / np.arange(1, len(err) + 1)
    return float(risk.mean())


def entropy_report(z, step=20, head_speed=None):
    e = z["err"][:, step - 1]
    H = z["entropy"]
    out = {"global_spearman": float(spearmanr(H, e ** 2)[0]),
           "partial_spearman_given_pred_task": partial_spearman(H, e, z["pred_task"]),
           "per_pred_task": {}, "per_gt_task": {}, "quintiles_within_pred_task": {}}
    for i, t in enumerate(TASKS):
        for key, lab in (("per_pred_task", z["pred_task"]), ("per_gt_task", z["task"])):
            m = lab == i
            if m.sum() > 10:
                r, p = spearmanr(H[m], e[m])
                out[key][t] = {"rho": float(r), "p": float(p), "n": int(m.sum())}
        m = z["pred_task"] == i
        if m.sum() > 50:
            q = np.quantile(H[m], [0.2, 0.4, 0.6, 0.8])
            b = np.digitize(H[m], q)
            out["quintiles_within_pred_task"][t] = [float(e[m][b == j].mean()) for j in range(5)]
    # risk-coverage: higher confidence first
    task_mean = {c: e[z["pred_task"] == c].mean() for c in range(4)}  # NB: in-sample task means (optimistic)
    confs = {"neg_entropy": -H, "max_prob": z["probs"].max(1),
             "pred_task_only(in-sample)": -np.array([task_mean[c] for c in z["pred_task"]]),
             "pred_task_then_entropy": -np.array([task_mean[c] for c in z["pred_task"]]) * 1e3 - H,
             "random": RNG.random(len(e)), "oracle": -e}
    if head_speed is not None:
        confs["neg_head_speed"] = -head_speed
    out["aurc"] = {k: aurc(v, e) for k, v in confs.items()}
    return out


# ---------------------------------------------------------------- E9 eye-movement type + hit-rate curves
def movement_labels(gt, sacc_thr=100.0, pursuit_thr=15.0):
    """Label each future window from gaze-in-world angular velocity (gt offsets share a fixed head reference,
    so their frame differences are gaze-in-world motion). 0=fixation-dominant, 1=pursuit, 2=contains saccade."""
    from cached_data import angular_error as ae
    v = ae(gt[:, 1:], gt[:, :-1]) * 30.0  # deg/s, (N,19)
    lab = np.zeros(len(gt), int)
    lab[np.median(v, 1) > pursuit_thr] = 1
    lab[v.max(1) > sacc_thr] = 2
    return lab


def e9_section(models, z_ref, groups):
    lines = ["\n## E9. Error by eye-movement type of the future window (I-VT on 30 Hz gaze-in-world velocity)\n"]
    names = ["fixation-dominant", "pursuit-like", "contains saccade"]
    for thr in ((100, 15), (75, 10), (150, 25)):
        lab = movement_labels(z_ref["gt"], *thr)
        rows = []
        for c, nm in enumerate(names):
            m = lab == c
            if m.sum() < 20:
                continue
            row = {"thresholds (sacc,pursuit) deg/s": thr, "type": nm, "N": int(m.sum()),
                   "share": f"{100 * m.mean():.1f}%", "HMD@20": f"{z_ref['hmd_err'][m, 19].mean():.2f}"}
            for mn, z in models.items():
                if z["err"].shape[1] >= 20:
                    row[f"{mn}@10"] = f"{z['err'][m, 9].mean():.2f}"
                    row[f"{mn}@20"] = f"{z['err'][m, 19].mean():.2f}"
            rows.append(row)
        lines.append(md_table(rows))
    lines.append("\n## E9b. Hit rate: % windows with endpoint error <= r\n")
    rows = []
    for k in (10, 15, 20):
        for mn, z in list(models.items()) + [("HMD", {"err": z_ref["hmd_err"]})]:
            if z["err"].shape[1] >= k:
                row = {"step": k, "model": mn}
                for r in (2.5, 5, 7.5, 10, 15, 20):
                    row[f"<= {r}"] = f"{100 * (z['err'][:, k - 1] <= r).mean():.1f}"
                rows.append(row)
    lines.append(md_table(rows))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dual", default="dual_l100")
    ap.add_argument("--gaze", default="gaze_only")
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--tag", default="existing")
    ap.add_argument("--test", default="eht_test")
    ap.add_argument("--val", default="eht_val")
    a = ap.parse_args()

    rep, js = [], {}
    dual_te, gaze_te = load(a.test, a.dual), load(a.test, a.gaze)
    dual_va = load(a.val, a.dual)
    users = dual_te["user"]
    models = {"gaze": gaze_te, "dual": dual_te}
    for o in a.others:
        z = load(a.test, o)
        if z is not None:
            models[o] = z

    rep.append(f"# Revision analysis — {a.tag}\n\nTest target `{a.test}`, N = {len(users)} windows, "
               f"{len(np.unique(users))} participants. 95% CIs: participant-level cluster bootstrap ({N_BOOT}).\n")
    rows = table_errors(models, dual_te, users, "task")
    rep.append("## E0. Endpoint error at step k and interval mean (deg), grouped by GT task\n")
    cols = ["subset", "N"] + [c for c in rows[0] if "@20" in c or "mean1-20" in c]
    rep.append(md_table(rows, cols))
    cols10 = ["subset", "N"] + [c for c in rows[0] if "@10" in c or "@15" in c]
    rep.append("\n" + md_table(rows, cols10))
    js["E0_gt"] = rows
    d = paired_diff_ci(dual_te["err"][:, 19], gaze_te["err"][:, 19], users)
    dm = paired_diff_ci(dual_te["err"].mean(1), gaze_te["err"].mean(1), users)
    rep.append(f"\nPaired difference dual − gaze-only: step 20 {fmt(d)}; interval mean {fmt(dm)}\n")
    js["dual_minus_gaze"] = {"step20": d, "mean": dm}
    if "probs" in dual_te and dual_te["probs"].size:
        acc = boot_ci((dual_te["pred_task"] == dual_te["task"]).astype(float), users)
        rep.append(f"\nTask accuracy (dual): {fmt(acc)}\n")
        cm = np.zeros((4, 4))
        for g, p in zip(dual_te["task"], dual_te["pred_task"]):
            cm[g, p] += 1
        js["confusion"] = (cm / cm.sum(1, keepdims=True)).round(3).tolist()
        rows_p = table_errors(models, dual_te, users, "pred_task")
        rep.append("\n## E0b. Same, grouped by **predicted** task (as in paper Tables 3–4)\n")
        rep.append(md_table(rows_p, ["subset", "N"] + [c for c in rows_p[0] if "@20" in c or "mean1-20" in c]))
        js["E0_pred"] = rows_p

    # λ sweep etc. in "others"
    if dual_va is not None:
        rep.append("\n## E4. Horizon policies (horizons selected on validation, evaluated on test)\n")
        rep.append("Endpoint error at each window's chosen horizon. `matched` = fixed horizon with the same mean.\n")
        prow = []
        for rule in ("endpoint", "interval"):
            for tau in (6, 7, 8, 9, 10, 11, 12, 14):
                r = policies(dual_te, gaze_te, dual_va, tau, rule)
                row = {"rule": rule, "tau": tau, "horizons (FV,VS,Sal,Tr)": tuple(r["horizons_pred"][c] for c in range(4)),
                       "mean_h": f"{r['mean_h']:.2f}"}
                for k in ("adaptive_dual", "fixed_matched_dual", "random_same_dist_dual", "oracle_task_dual",
                          "adaptive_gaze_only_at_dual_h", "fixed_matched_gaze_only", "hmd_at_adaptive"):
                    row[k] = f"{r[k].mean():.2f}"
                dd = paired_diff_ci(r["adaptive_dual"], r["fixed_matched_dual"], users)
                row["adaptive−matched (dual)"] = fmt(dd)
                prow.append(row)
        rep.append(md_table(prow))
        js["E4"] = prow
        rep.append("\nFixed horizon curves (test, endpoint): " + ", ".join(
            f"k={k}: dual {dual_te['err'][:, k-1].mean():.2f} / gaze {gaze_te['err'][:, k-1].mean():.2f}" for k in range(10, 21)) + "\n")

    if dual_te["probs"].size:
        rep.append("\n## E5. Entropy vs error (step 20)\n")
        ts_speed = None
        er = entropy_report(dual_te, 20, ts_speed)
        rep.append("```json\n" + json.dumps(er, indent=1) + "\n```\n")
        js["E5_test"] = er
        dz = load("dsav_test", a.dual)
        if dz is not None:
            erd = entropy_report(dz, 20)
            rep.append("\n### DSAV360 (zero-shot)\n```json\n" + json.dumps(erd, indent=1) + "\n```\n")
            js["E5_dsav"] = erd

    rep.extend(e9_section(models, dual_te, users))

    dz, gz = load("dsav_test", a.dual), load("dsav_test", a.gaze)
    if dz is not None and gz is not None:
        rep.append("\n## DSAV360 zero-shot (grouped by predicted task)\n")
        rows_d = table_errors({"gaze": gz, "dual": dz}, dict(dz, task=dz["pred_task"]), dz["user"], "task")
        rep.append(md_table(rows_d, ["subset", "N"] + [c for c in rows_d[0] if "@10" in c or "@20" in c or "mean1-20" in c]))
        js["dsav"] = rows_d
        rep.append("\nPredicted-label distribution: " + ", ".join(
            f"{t} {100 * (dz['pred_task'] == i).mean():.1f}%" for i, t in enumerate(TASKS)) + "\n")

    with open(os.path.join(HERE, "results", f"report_{a.tag}.md"), "w") as f:
        f.write("\n".join(rep))
    with open(os.path.join(HERE, "results", f"report_{a.tag}.json"), "w") as f:
        json.dump(js, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\n".join(rep))


if __name__ == "__main__":
    main()
