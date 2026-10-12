"""DSAV360 video/data synchronisation test.

The gaze files record the video frame shown at each gaze sample (`frame` column), which gives the
actual mapping from data time to video frame. This script compares the mapping used by
pipelines/create_dsav360_dataset.py (frame = round(60 * t)) with the measured mapping and with
alternative playback rates r (frame = round(r * t + offset)), keeping head motion, gaze targets and
windows identical: only the video frames, hence the viewport crops and saliency maps, change.

For every condition it reports
- the endpoint error of the behaviour-aware model (lambda = 20, three seeds) on the DSAV360 test windows,
- NSS: the standardised saliency value at the gaze position in the current viewport saliency map
  (a direct measure of saliency/gaze alignment).

Stage 1 (--stage sal) decodes each video once, cuts the viewport crops exactly as the preprocessing does
and computes UniSal 25x25 maps; stage 2 (--stage eval) evaluates. Saliency arrays are stored locally.
"""
import os
import sys
import json
import argparse
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pipelines"))
from cached_data import build_windows, recording_dirs, angular_error, clip_index  # noqa: E402
from eval_unified import resolve, load_model, predict  # noqa: E402
from build_sal_cache import sal_batch  # noqa: E402
from core import utils  # noqa: E402
import create_dsav360_dataset as prep  # noqa: E402

RAW = "/mnt/DataLake/AlienBackup/access/dsav360_extracted"
PROC = "/mnt/DataLake/AlienBackup/access/processed_dataset/dsav360_processed_vid"
PROC_SYNC = PROC + "_sync"  # re-synchronised data with per-sample gaze validity
OUT = os.path.join(HERE, "results", "sync_sweep")
N = 900            # 30 s at 30 Hz (6 clips of 150 samples)
FPS = 30.0
RATES = [30, 45, 55, 60, 65, 75, 90, 120]
MODELS = [f"e2_dual20_s{s}" for s in range(3)]
OFFSET_VIDEOS = {3, 5, 8, 10}


def conditions(a, b):
    """name -> function of sample time t (s) giving the video frame index."""
    c = {"pipeline: 60 fps, no offset": lambda t: 60.0 * t,
         "measured mapping (fit)": lambda t: a * t + b}
    for r in RATES:
        c[f"{r} fps + measured offset"] = (lambda r: (lambda t: r * t + b))(r)
    return c


def fit_sync(video, user):
    """Linear fit of displayed frame vs data time, with the pipeline's common time origin."""
    h = pd.read_csv(f"{RAW}/head_data/head_data/head_video_{video:04d}.csv", usecols=["t", "id"])
    g = pd.read_csv(f"{RAW}/gaze_data/gaze_video_{video:04d}.csv", usecols=["frame", "t", "id"])
    h, g = h[h.id == user], g[g.id == user]
    t0 = min(h.t.min(), g.t.min() / 1000.0)
    tg, f = g.t.values / 1000.0 - t0, g.frame.values
    m = f < f.max()                     # drop the tail where the last frame is held
    a, b = np.polyfit(tg[m], f[m], 1)
    return float(a), float(b), int(f.min()), int(f.max())


def load_meta(rec):
    """The 900 per-sample records (head, gaze, pipeline frame) of clips 1-6 of a processed recording."""
    files = sorted(glob(os.path.join(PROC, rec, "*.json")), key=clip_index)
    meta = []
    for fpath in files:
        if clip_index(fpath) <= 6:
            meta += json.load(open(fpath))
    assert len(meta) == N, (rec, len(meta))
    return meta


def crop(frame, hor_deg, ver_deg):
    """Viewport crop identical to create_dsav360_dataset.generate_videos (before mp4 encoding)."""
    width, height = frame.shape[1], frame.shape[0]
    yaw, pitch = prep.convert_hor_degre(hor_deg), prep.convert_ver_degree(ver_deg)
    left = int((yaw - prep.fov_horizontal / 2) * width / 360)
    right = int((yaw + prep.fov_horizontal / 2) * width / 360)
    top = int((pitch - prep.fov_vertical / 2) * height / 180)
    bottom = int((pitch + prep.fov_vertical / 2) * height / 180)
    if right > width:
        c = np.hstack((frame[top:bottom, left:], frame[top:bottom, :right - width]))
    elif left < 0:
        c = np.hstack((frame[top:bottom, left:], frame[top:bottom, :right]))
    else:
        c = frame[top:bottom, left:right]
    if top < 0:
        c = np.vstack((np.zeros((-top, c.shape[1], 3), np.uint8), c))
    if bottom > height:
        c = np.vstack((c, np.zeros((bottom - height, c.shape[1], 3), np.uint8)))
    return cv2.resize(c, (384, 288))[:, :, ::-1]          # BGR -> RGB, as build_sal_cache feeds UniSal


def stage_sal():
    os.makedirs(OUT, exist_ok=True)
    recs = recording_dirs("dsav", "test")
    info, metas = [], []
    for rec in recs:
        v, u = int(rec.split("/")[0][6:]), int(rec.split("/")[1].split("_")[1])
        a, b, fmin, fmax = fit_sync(v, u)
        meta = load_meta(rec)
        assert all(m["orig_frame_number"] == round(60.0 * i / FPS) for i, m in enumerate(meta))
        info.append(dict(rec=rec, video=v, user=u, slope=a, offset=b, fmin=fmin, fmax=fmax))
        metas.append(meta)
    names = list(conditions(60, 0).keys())
    t = np.arange(N) / FPS
    S = np.zeros((len(names), len(recs), N, 25, 25), np.float16)
    dev = torch.device("cuda")
    unisal = utils.load_unisal(os.path.join(ROOT, "models", "unisal.pth"), dev)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for v in sorted({i["video"] for i in info}):
        cap = cv2.VideoCapture(f"{RAW}/videos/{v:04d}/{v:04d}_mono_60fps.mp4")
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        req = {}                                            # frame -> [(cond, rec, sample)]
        for ri, inf in enumerate(info):
            if inf["video"] != v:
                continue
            for ci, fn in enumerate(conditions(inf["slope"], inf["offset"]).values()):
                for i, k in enumerate(np.clip(np.round(fn(t)).astype(int), 0, nframes - 1)):
                    req.setdefault(int(k), []).append((ci, ri, i))
        buf, idx = [], []

        def flush():
            if buf:
                s = sal_batch(unisal, np.stack(buf), dev, chunk=256)
                for (ci, ri, i), m in zip(idx, s):
                    S[ci, ri, i] = m
                buf.clear(); idx.clear()

        last = max(req)
        with torch.no_grad():
            for k in range(last + 1):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"video {v}: could not read frame {k}")
                for ci, ri, i in req.get(k, []):
                    m = metas[ri][i]
                    buf.append(crop(frame, m["horz_hmd_deg"], m["vert_hmd_deg"]))
                    idx.append((ci, ri, i))
                if len(buf) >= 1024:
                    flush()
            flush()
        cap.release()
        print(f"video {v}: {sum(len(x) for x in req.values())} crops", flush=True)
    np.save(os.path.join(OUT, "saliency.npy"), S)
    json.dump(dict(conditions=names, recordings=info, rates=RATES), open(os.path.join(OUT, "conditions.json"), "w"), indent=1)


def gaze_valid(rec):
    """Per-sample flag (measured, not interpolated gaze) from the re-synchronised data; None if absent."""
    files = sorted(glob(os.path.join(PROC_SYNC, rec, "*.json")), key=clip_index)
    if not files:
        return None
    return [m["gaze_valid"] for f in files if clip_index(f) <= 6 for m in json.load(open(f))]


def nss(sal, meta, valid):
    """Standardised saliency at the gaze position of each sample (measured gaze inside the viewport only)."""
    vals = []
    for i, (s, m) in enumerate(zip(sal.astype(np.float32), meta)):
        if valid is None or i >= len(valid) or not valid[i]:
            continue
        daz = (m["giw_long"] - m["horz_hmd_deg"] + 180) % 360 - 180
        dlat = m["giw_lat"] - m["vert_hmd_deg"]
        col = (daz / prep.fov_horizontal + 0.5) * 25
        row = (0.5 - dlat / prep.fov_vertical) * 25
        if 0 <= col < 25 and 0 <= row < 25:
            vals.append((s[int(row), int(col)] - s.mean()) / (s.std() + 1e-6))
    return vals


def stage_eval():
    S = np.load(os.path.join(OUT, "saliency.npy"), mmap_mode="r")
    cfg = json.load(open(os.path.join(OUT, "conditions.json")))
    names, info = cfg["conditions"], cfg["recordings"]
    rec_index = {r["rec"]: i for i, r in enumerate(info)}
    w = build_windows("dsav", recording_dirs("dsav", "test"))
    rec_used = set(w["rec"])
    ri = np.array([rec_index[r] for r in w["rec"]])
    start = np.array([(clip_index(c) - 1) * 150 for c in w["clip"]]) + w["f"] - 15
    samples = start[:, None] + np.arange(15)[None]                      # (N_win, 15) sample indices
    off = np.isin(np.array([int(v) for v in w["video"]]), list(OFFSET_VIDEOS))
    dev = torch.device("cuda")
    models = [load_model(*resolve(os.path.join(HERE, "runs", m, "best.pth")), dev) for m in MODELS]
    rows = []
    cached = dict(w)
    metas = [load_meta(r["rec"]) for r in info]
    valids = [gaze_valid(r["rec"]) for r in info]
    for ci, name in enumerate(names + ["cached saliency (mp4 clips, as evaluated)"]):
        if ci < len(names):
            wc = dict(w, sal=np.asarray(S[ci][ri[:, None], samples], np.float32))
            nv = [x for r in range(len(info)) for x in nss(S[ci, r], metas[r], valids[r])]
            nv_off = [x for r in range(len(info)) if info[r]["video"] in OFFSET_VIDEOS
                      for x in nss(S[ci, r], metas[r], valids[r])]
        else:
            wc, nv, nv_off = cached, [], []
        e10, e20, e10o, e20o = [], [], [], []
        for m, use_task in models:
            pred, _ = predict(m, use_task, wc, dev)
            err = angular_error(pred, wc["gt"])
            e10.append(err[:, 9].mean()); e20.append(err[:, 19].mean())
            e10o.append(err[off, 9].mean()); e20o.append(err[off, 19].mean())
        rows.append((name, np.mean(e10), np.std(e10, ddof=1), np.mean(e20), np.std(e20, ddof=1),
                     np.mean(e10o), np.mean(e20o), np.mean(nv) if nv else np.nan, np.mean(nv_off) if nv_off else np.nan))
        print(rows[-1], flush=True)
    ok = [r for r in info if r["rec"] in rec_used]          # recordings with measured-gaze windows
    sl = np.array([r["slope"] for r in ok])
    offs = {v: np.median([r["offset"] for r in ok if r["video"] == v]) for v in sorted({r["video"] for r in ok})}
    lines = ["# DSAV360 synchronisation test\n",
             f"Measured playback rate (displayed frame vs data time, {len(ok)} test recordings with measured-gaze windows): "
             f"median {np.median(sl):.2f} fps (5-95%: {np.percentile(sl, 5):.2f}-{np.percentile(sl, 95):.2f}). "
             "Median start frame by video: " + ", ".join(f"{v}: {abs(o):.0f}" for v, o in offs.items()) + ".\n",
             f"Windows with measured gaze targets: {len(w['gt'])} (videos 3/5/8/10 with a start offset: {off.sum()}). Model: behaviour-aware, lambda = 20, "
             "3 seeds (mean ± s.d.). NSS: standardised saliency at the gaze position (higher = better aligned).\n",
             "| condition | step 10 | step 20 | step 10 (offset videos) | step 20 (offset videos) | NSS all | NSS offset videos |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]:.2f} ± {r[2]:.2f} | {r[3]:.2f} ± {r[4]:.2f} | {r[5]:.2f} | {r[6]:.2f} | "
                     + (f"{r[7]:.3f} | {r[8]:.3f} |" if not np.isnan(r[7]) else "– | – |"))
    open(os.path.join(HERE, "results", "dsav_sync_sweep.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sal", "eval"], required=True)
    a = ap.parse_args()
    stage_sal() if a.stage == "sal" else stage_eval()
