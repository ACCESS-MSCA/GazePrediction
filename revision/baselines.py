"""E12 - training-free / linear literature baselines, written in eval_unified's per-window format.

  hmd_cv        : constant-velocity head extrapolation (gaze = predicted head direction; viewport-prediction analogue)
  lin_sgaze     : SGaze-style linear eye-head model: future eye-in-head offset = A * [last head velocity, 1]
  lin_motion    : ridge regression from the full 15x4 motion window to the 20x2 future offsets
  nakashima     : saliency-peak gaze prior weighted by a head-centred Gaussian (Nakashima et al. 2015 idea),
                  constant over the horizon; sigma and vertical orientation chosen on the training split
All parameters are fit on the train split only.
"""
import os
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
from cached_data import build_windows, recording_dirs, angular_error  # noqa: E402

OUT = os.path.join(HERE, "results", "per_window")
FOV_H, FOV_V = 108.77, 111.53  # pipelines/create_EHT_dataset.py crop
STEPS = np.arange(1, 21)[:, None] / 30.0


def ridge(X, Y, alpha=1.0):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = np.linalg.solve(Xb.T @ Xb + alpha * np.eye(Xb.shape[1]), Xb.T @ Y)
    return lambda Z: np.hstack([Z, np.ones((len(Z), 1))]) @ A


def sal_grid():
    az = (np.arange(25) + 0.5) / 25 * FOV_H - FOV_H / 2
    el = (np.arange(25) + 0.5) / 25 * FOV_V - FOV_V / 2
    return np.meshgrid(az, el)  # rows = image rows (top first)


def nakashima_pred(sal_last, sigma, flip):
    AZ, EL = sal_grid()
    EL = -EL if flip else EL
    prior = np.exp(-(AZ ** 2 + EL ** 2) / (2 * sigma ** 2))
    s = sal_last - sal_last.reshape(len(sal_last), -1).min(1)[:, None, None]
    score = (s * prior).reshape(len(s), -1)
    k = score.argmax(1)
    return np.stack([AZ.ravel()[k], EL.ravel()[k]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["eht:test", "eht:val", "dsav:test"])
    ap.add_argument("--split_file", default=None, help="fit split (default ehtask_splits.json train)")
    a = ap.parse_args()

    tr = build_windows("eht", recording_dirs("eht", "train"))
    ts_tr, gt_tr = tr["ts"], tr["gt"]
    fits = {}
    fits["lin_motion"] = ridge(ts_tr.reshape(len(ts_tr), -1), gt_tr.reshape(len(gt_tr), -1), alpha=10.0)
    fits["lin_sgaze"] = ridge(ts_tr[:, -1, :2], gt_tr.reshape(len(gt_tr), -1), alpha=1.0)
    # constant-velocity gain fit on train (gain=1 is pure extrapolation; report both)
    best = None
    for sig in (5, 10, 15, 20, 30, 45, 1e6):
        for flip in (False, True):
            p = nakashima_pred(tr["sal"][:, -1], sig, flip)
            e = angular_error(np.repeat(p[:, None], 20, 1), gt_tr).mean()
            if best is None or e < best[0]:
                best = (e, sig, flip)
    print(f"Nakashima prior chosen on train: sigma={best[1]}, flip={best[2]}, train err={best[0]:.2f}")

    os.makedirs(OUT, exist_ok=True)
    for target in a.targets:
        ds, split = target.split(":")
        w = build_windows(ds, recording_dirs(ds, split))
        n = len(w["ts"])
        last_v = w["ts"][:, -1, :2]
        preds = {
            "hmd_cv": last_v[:, None, :] * STEPS[None],
            "lin_sgaze": fits["lin_sgaze"](last_v).reshape(n, 20, 2),
            "lin_motion": fits["lin_motion"](w["ts"].reshape(n, -1)).reshape(n, 20, 2),
            "nakashima": np.repeat(nakashima_pred(w["sal"][:, -1], best[1], best[2])[:, None], 20, 1),
        }
        hmd = angular_error(np.zeros_like(w["gt"]), w["gt"])
        for name, p in preds.items():
            err = angular_error(p, w["gt"])
            np.savez(os.path.join(OUT, f"{ds}_{split}__base_{name}.npz"), err=err.astype(np.float32),
                     hmd_err=hmd.astype(np.float32), pred=p.astype(np.float32), gt=w["gt"], probs=np.zeros((0,)),
                     task=w["task"], user=w["user"], video=w["video"], rec=w["rec"], clip=w["clip"], f=w["f"],
                     ckpt=name, lam=0.0, q=20)
            print(f"[{ds}:{split}] base_{name:<12} step10={err[:, 9].mean():.2f} step20={err[:, 19].mean():.2f} "
                  f"mean={err.mean():.2f} | HMD step20={hmd[:, 19].mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
