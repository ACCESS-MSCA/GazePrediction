"""E0 - Unified evaluation. Runs every registered model on identical windows and stores per-window
outputs (step-wise great-circle error, HMD-centre error, task probabilities) so that all tables and
figures are derived from one source.

Output: revision/results/per_window/<dataset>_<split>__<model>.npz
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.models import Conv_LSTM  # noqa: E402
from cached_data import build_windows, recording_dirs, angular_error, WindowTensorDataset  # noqa: E402

EXP = "/mnt/DataLake/AlienBackup/access/experiments"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "per_window")

# name -> (checkpoint, config json)
REGISTRY = {
    "dual_l100": f"{EXP}/eht_scratch_improved_head/dsav_scratch/conv_lstm_past_15_future_20_window_15_samp1_20260416_161628/model_epoch_10_val_loss_172.1438.pth",
    "gaze_only": f"{EXP}/eht_scratch_no_task_head/dsav_scratch/conv_lstm_past_15_future_20_window_15_samp1_20260424_161043/model_epoch_10_val_loss_74.7576.pth",
    "dual_l20": f"{EXP}/eht_scratch_improved_head/dsav_scratch/conv_lstm_past_15_future_20_window_15_samp1_20260415_100741/model_epoch_10_val_loss_95.1892.pth",
    "dual_l10": f"{EXP}/eht_scratch_improved_head/dsav_scratch/conv_lstm_past_15_future_20_window_15_samp1_20260410_165823/model_epoch_9_val_loss_85.3864.pth",
    "dual_l2": f"{EXP}/eht_scratch_improved_head/dsav_scratch/conv_lstm_past_15_future_20_window_15_samp1_20260408_163226/model_epoch_5_val_loss_78.1268.pth",
    "dual_l0.5_samp3": f"{EXP}/eht_scratch_improved_head/dsav_scratch/conv_lstm_past_15_future_20_window_15_samp3_20260402_190655/BEST",
    # ISVC conference model (q=10), trained with the original conference pipeline.
    "isvc_q10_oldsplit": "/mnt/DataLake/AlienBackup/access/eht_gaze_prediction/backup_scripts/trained_models_combined/conv_lstm_past_15_future_10_window_15_20250113_182736/model_epoch_5_val_loss_49.1556.pth",
}


ISVC_HOLDOUT = ["Video_8/User_8_Task_1", "Video_8/User_13_Task_1", "Video_1/User_26_Task_1",
                "Video_1/User_1_Task_1", "Video_10/User_4_Task_1", "Video_10/User_9_Task_1"]


def split_dirs(dataset, split):
    if split == "isvc_holdout":
        return ISVC_HOLDOUT
    if split.startswith("fold"):  # e.g. fold3_test from revision/results/cv_folds.json
        with open(os.path.join(HERE, "results", "cv_folds.json")) as f:
            k, part = split[4:].split("_")
            return json.load(f)[int(k)][part]
    return recording_dirs(dataset, split)


def resolve(ckpt):
    """'BEST' -> lowest val-loss checkpoint in the folder."""
    d = os.path.dirname(ckpt)
    if os.path.basename(ckpt) == "BEST":
        pths = [p for p in os.listdir(d) if p.endswith(".pth")]
        ckpt = os.path.join(d, min(pths, key=lambda p: float(p.split("val_loss_")[1][:-4])))
    cfg_path = [os.path.join(d, p) for p in os.listdir(d) if p.startswith("conv_") and p.endswith(".json")][0]
    with open(cfg_path) as f:
        cfg = json.load(f)
    return ckpt, cfg


def load_model(ckpt, cfg, device):
    use_task = bool(cfg.get("use_task_output", False))
    m = Conv_LSTM(num_time_features=cfg["num_time_features"], conv_output_embeddings=cfg["conv_output_embeddings"],
                  sequence_length=cfg["past_timepoints"], prediction_length=cfg["future_timepoints"],
                  prediction_features=cfg["prediction_features"], lstm_hidden_size=cfg["lstm_hidden_states"],
                  lstm_num_layers=cfg.get("lstm_layers", 1), use_task_output=use_task).to(device)
    if use_task and cfg.get("n_classes", 4) != 4:
        m.task_head[-1] = torch.nn.Linear(64, cfg["n_classes"]).to(device)
    state = torch.load(ckpt, map_location=device)
    missing, unexpected = m.load_state_dict(state, strict=False)
    if missing or unexpected:  # the original scripts use strict=False silently; we refuse partial loads
        raise RuntimeError(f"State dict mismatch for {ckpt}: missing={missing} unexpected={unexpected}")
    m.eval()
    return m, use_task


@torch.no_grad()
def predict(model, use_task, w, device, bs=512, cfg=None):
    if cfg is not None and cfg.get("ts_mean") is not None:
        w = dict(w, ts=((w["ts"] - np.array(cfg["ts_mean"], np.float32)) / np.array(cfg["ts_std"], np.float32)).astype(np.float32))
    ds = WindowTensorDataset(w)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
    preds, probs = [], []
    for sal, ts, _, _ in dl:
        out = model(sal.to(device), ts.to(device))
        if use_task:
            g, logits = out
            probs.append(torch.softmax(logits, 1).cpu().numpy())
        else:
            g = out
        preds.append(g.cpu().numpy())
    return np.concatenate(preds), (np.concatenate(probs) if probs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(REGISTRY))
    ap.add_argument("--targets", nargs="+", default=["eht:test", "eht:val", "dsav:test"])
    ap.add_argument("--extra", nargs="*", default=[], help="name=path/to/ckpt.pth for new runs")
    args = ap.parse_args()

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    reg = dict(REGISTRY)
    for e in args.extra:
        k, v = e.split("=", 1)
        reg[k] = v
        if k not in args.models:
            args.models.append(k)
    os.makedirs(OUT_DIR, exist_ok=True)

    for target in args.targets:
        dataset, split = target.split(":")
        windows = {}
        for name in args.models:
            ckpt, cfg = resolve(reg[name])
            q = cfg["future_timepoints"]
            samp = int(cfg.get("saliency_sampling") or 1)
            key = (q, samp)
            if key not in windows:
                # q=10 models are evaluated on the same window starts as q=20 (future=20 windows, first q steps)
                windows[key] = build_windows(dataset, split_dirs(dataset, split), future=20, sal_sampling=samp)
            w = windows[key]
            model, use_task = load_model(ckpt, cfg, device)
            pred, prob = predict(model, use_task, w, device, cfg=cfg)
            gt = w["gt"][:, :q]
            err = angular_error(pred, gt)
            hmd = angular_error(np.zeros_like(w["gt"]), w["gt"])
            np.savez(os.path.join(OUT_DIR, f"{dataset}_{split}__{name}.npz"),
                     err=err.astype(np.float32), hmd_err=hmd.astype(np.float32), pred=pred.astype(np.float32),
                     gt=w["gt"], probs=prob if prob is not None else np.zeros((0,)),
                     task=w["task"], user=w["user"], video=w["video"], rec=w["rec"], clip=w["clip"], f=w["f"],
                     ckpt=ckpt, lam=cfg.get("task_weight", 0.0), q=q)
            msg = f"[{dataset}:{split}] {name:<20} N={len(err)} step10={err[:, 9].mean():.2f}"
            if q >= 20:
                msg += f" step15={err[:, 14].mean():.2f} step20={err[:, 19].mean():.2f} mean1-20={err.mean():.2f}"
            if prob is not None:
                if prob.shape[1] == 4:
                    msg += f" acc={(prob.argmax(1) == w['task']).mean():.4f}"
            msg += f" | HMD step20={hmd[:, 19].mean():.2f}"
            print(msg, flush=True)


if __name__ == "__main__":
    main()
