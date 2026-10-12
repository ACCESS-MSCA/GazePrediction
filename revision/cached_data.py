"""Window construction from the saliency cache, mirroring core/dataloader.py and
tools/uncertainty_aware_evaluation.py exactly (same frame indices, features and targets)."""
import os
import json
from glob import glob

import numpy as np
import torch

CACHE_ROOT = "/mnt/DataLake/AlienBackup/access/processed_dataset/sal_cache25"
SPLITS = {
    "eht": "/mnt/DataLake/AlienBackup/access/processed_dataset/splits/ehtask_splits.json",
    "dsav": "/mnt/DataLake/AlienBackup/access/processed_dataset/splits/dsav360_splits.json",
}
TASK_NAMES = ["Free Viewing", "Visual Search", "Saliency", "Track"]
FPS = 30
# DSAV360: head and gaze are recorded for the first 30 s of each video (6 clips of 5 s). The videos are
# 34-60 s long and the processed clips beyond 30 s carry the last recorded sample (frozen head/gaze),
# so only clips 1-6 are used.
DSAV_MAX_CLIP = 6


def clip_index(name):
    """Clip number from a clip file name such as Video_4_User_64_Task_1_7.npz -> 7."""
    return int(os.path.splitext(os.path.basename(str(name)))[0].rsplit("_", 1)[1])


def keep_recorded(target, z):
    """Drop DSAV360 windows from clips past the recorded 30 s. `z` is a per-window dict (eval_unified output)."""
    if z is None or not str(target).startswith("dsav"):
        return z
    m = np.array([clip_index(c) <= DSAV_MAX_CLIP for c in z["clip"]])
    if m.all():
        return z
    n = len(m)
    return {k: (v[m] if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == n else v) for k, v in z.items()}


def motion_features(h_az, h_el, dt=1.0 / FPS):
    """Same as core.utils.compute_spherical_motion: wrapped first differences, first value duplicated."""
    d_az = np.diff(h_az)
    d_az = np.where(d_az > 180, d_az - 360, d_az)
    d_az = np.where(d_az < -180, d_az + 360, d_az)
    d_el = np.diff(h_el)
    d_az = np.concatenate([d_az[:1], d_az])
    d_el = np.concatenate([d_el[:1], d_el])
    return np.stack([d_az / dt, d_el / dt, d_az, d_el], axis=1).astype(np.float32)


def recording_dirs(dataset, split):
    """Recording folders (Video_X/User_Y_Task_Z) for a split; split='all' returns every cached folder."""
    root = os.path.join(CACHE_ROOT, dataset)
    if split == "all":
        return sorted(os.path.relpath(p, root) for p in glob(os.path.join(root, "Video_*", "User_*_Task_*")))
    with open(SPLITS[dataset]) as f:
        return json.load(f)[split]


def build_windows(dataset, rec_dirs, past=15, future=20, stride=15, sal_sampling=1):
    """Returns dict of stacked arrays: sal (N,past,25,25), ts (N,past,4), gt (N,future,2), and per-window ids."""
    root = os.path.join(CACHE_ROOT, dataset)
    out = {k: [] for k in ["sal", "ts", "gt", "task", "user", "video", "rec", "clip", "f"]}
    for rec in rec_dirs:
        for npz in sorted(glob(os.path.join(root, rec, "*.npz"))):
            if dataset == "dsav" and clip_index(npz) > DSAV_MAX_CLIP:
                continue
            z = np.load(npz)
            n = int(z["n_meta"])
            sal = z["sal"]
            h_az, h_el = z["horz_hmd_deg"], z["vert_hmd_deg"]
            g_az, g_el = z["giw_long"], z["giw_lat"]
            feats = motion_features(h_az, h_el)
            for f in range(past, n - future, stride):
                if f > len(sal):
                    continue
                idx = np.arange(f - past, f)
                if sal_sampling > 1:  # emulate tools/evaluation.py saliency_sampling (repeat every Nth map)
                    idx = (f - past) + ((idx - (f - past)) // sal_sampling) * sal_sampling
                t_az = g_az[f:f + future] - h_az[f]
                t_az = (t_az + 180) % 360 - 180
                t_el = g_el[f:f + future] - h_el[f]
                out["sal"].append(sal[idx])
                out["ts"].append(feats[f - past:f])
                out["gt"].append(np.stack([t_az, t_el], 1))
                out["task"].append(int(z["task_id"]) - 1)
                out["user"].append(str(z["user_id"]))
                out["video"].append(str(z["video_id"]))
                out["rec"].append(rec)
                out["clip"].append(os.path.basename(npz))
                out["f"].append(f)
    res = {}
    for k, v in out.items():
        if k in ("sal", "ts", "gt"):
            res[k] = np.stack(v).astype(np.float32)
        else:
            res[k] = np.array(v)
    return res


def angular_error(p, t):
    """Great-circle angle (deg), identical to tools/uncertainty_aware_evaluation.angular_error."""
    p_az, p_el = np.radians(p[..., 0]), np.radians(p[..., 1])
    t_az, t_el = np.radians(t[..., 0]), np.radians(t[..., 1])
    c = np.sin(p_el) * np.sin(t_el) + np.cos(p_el) * np.cos(t_el) * np.cos(p_az - t_az)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


class WindowTensorDataset(torch.utils.data.Dataset):
    def __init__(self, w, task_override=None):
        self.sal = torch.from_numpy(w["sal"]).unsqueeze(2).unsqueeze(2)  # (N,past,1,1,25,25) as model expects
        self.ts = torch.from_numpy(w["ts"])
        self.gt = torch.from_numpy(w["gt"])
        self.task = torch.from_numpy(w["task"] if task_override is None else task_override).long()

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, i):
        return self.sal[i], self.ts[i], self.gt[i], self.task[i]
