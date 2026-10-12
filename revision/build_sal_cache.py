"""Precompute frozen-UniSal saliency maps for every frame of every processed clip.

Stores, per clip, the 25x25 map that Conv_LSTM actually consumes (the model applies
F.interpolate(size=(25, 25)) with nearest mode, which is idempotent at 25x25), so cached
inputs reproduce the original pipeline exactly. Per-clip metadata needed for windowing
(HMD angles, gaze, task, user, video) is stored alongside.

Output: <cache_root>/<dataset>/<Video_X>/<User_Y_Task_Z>/<clip>.npz
"""
import os
import sys
import json
import argparse
from glob import glob

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import utils  # noqa: E402

RGB_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
RGB_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
META_KEYS = ["horz_hmd_deg", "vert_hmd_deg", "giw_long", "giw_lat"]


class ClipFrames(Dataset):
    """Decodes a clip and applies the same BGR->RGB + PIL Lanczos resize as utils.preprocess_sal_img."""

    def __init__(self, clips):
        self.clips = clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        mp4, out = self.clips[idx]
        cap = cv2.VideoCapture(mp4)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            img = PIL.Image.fromarray(rgb).resize((384, 288), PIL.Image.LANCZOS)
            frames.append(np.asarray(img))
        cap.release()
        with open(os.path.splitext(mp4)[0] + ".json") as f:
            meta = json.load(f)
        arr = np.stack(frames) if frames else np.zeros((0, 288, 384, 3), np.uint8)
        return arr, meta, out, mp4


def sal_batch(unisal_wrapped, uint8_frames, device, chunk=64):
    """Batched equivalent of WrappedUNISAL.forward applied to one image at a time."""
    net = unisal_wrapped.unisal_model
    outs = []
    for i in range(0, len(uint8_frames), chunk):
        x = torch.as_tensor(uint8_frames[i:i + chunk]).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - RGB_MEAN.to(device)) / RGB_STD.to(device)
        s = net(x.unsqueeze(1))            # (N, 1, 1, H, W) static mode
        s = s.squeeze(1)                   # (N, 1, H, W)
        # BatchNorm2d(affine=False, track_running_stats=False) on a single image == per-image standardisation
        mean = s.mean(dim=(1, 2, 3), keepdim=True)
        var = s.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        s = (s - mean) / torch.sqrt(var + 1e-5)
        s = s * 2 - 1
        s = F.interpolate(s, size=(25, 25))
        outs.append(s[:, 0].float().cpu())
    return torch.cat(outs).numpy()


def reference_sal(unisal_wrapped, mp4, device, n=4):
    cap = cv2.VideoCapture(mp4)
    res = []
    for _ in range(n):
        ret, frame = cap.read()
        img = utils.preprocess_sal_img(frame).to(device)
        s = unisal_wrapped(img.unsqueeze(0))
        res.append(F.interpolate(s, size=(25, 25))[0, 0].cpu().numpy())
    cap.release()
    return np.stack(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["eht", "dsav"], required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--sal_path", default="./models/unisal.pth")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    # Disable TF32 so batched inference matches the single-image reference bit-for-bit (to ~1e-5).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    unisal_wrapped = utils.load_unisal(args.sal_path, device)

    mp4s = sorted(glob(os.path.join(args.processed_dir, "Video_*", "User_*_Task_*", "*.mp4")))
    n_all = len(mp4s)
    # The original pipeline skips clips without metadata; do the same.
    mp4s = [m for m in mp4s if os.path.exists(os.path.splitext(m)[0] + ".json")]
    print(f"{n_all - len(mp4s)} clips without metadata skipped", flush=True)
    clips = []
    for mp4 in mp4s:
        rel = os.path.relpath(os.path.splitext(mp4)[0], args.processed_dir)
        out = os.path.join(args.cache_root, args.dataset, rel + ".npz")
        if not os.path.exists(out):
            clips.append((mp4, out))
    if args.limit:
        clips = clips[:args.limit]
    print(f"{len(mp4s)} clips found, {len(clips)} to process", flush=True)

    # Equivalence check against the original single-image path.
    with torch.no_grad():
        probe = mp4s[0]
        ref = reference_sal(unisal_wrapped, probe, device)
        arr, _, _, _ = ClipFrames([(probe, None)])[0]
        got = sal_batch(unisal_wrapped, arr[:len(ref)], device).astype(np.float32)
        err = np.abs(ref - got).max()
        print(f"Equivalence check max |diff| = {err:.4e} (float32)", flush=True)
        assert err < 1e-3, "Batched saliency deviates from the reference pipeline"

    loader = DataLoader(ClipFrames(clips), batch_size=None, shuffle=False,
                        num_workers=args.workers, prefetch_factor=2)
    with torch.no_grad():
        for arr, meta, out, mp4 in tqdm(loader, total=len(clips), mininterval=30):
            if len(arr) == 0:
                print(f"WARNING empty clip {mp4}", flush=True)
                continue
            sal = sal_batch(unisal_wrapped, arr, device)
            first = meta[0]
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(
                out,
                sal=sal,
                **{k: np.array([r[k] for r in meta], np.float32) for k in META_KEYS},
                task_id=np.int64(first["task_id"]),
                user_id=str(first["user_id"]),
                video_id=str(first["video_id"]),
                n_meta=np.int64(len(meta)),
                # DSAV360 only: whether the gaze of each frame is measured (not interpolated across a gap)
                gaze_valid=np.array([bool(r.get("gaze_valid", True)) for r in meta]),
            )


if __name__ == "__main__":
    main()
