"""Cached re-implementation of train.py with explicit, participant-exclusive splits.

Same model (core.models.Conv_LSTM), loss (per-axis SphericalMSELoss + lambda * CE), optimiser (Adam 1e-3),
scheduler (ReduceLROnPlateau, patience 5) and checkpoint criterion (val gaze loss + lambda * val CE) as
train.py. Differences, all deliberate and reported:
  * splits come from processed_dataset/splits/ehtask_splits.json (user-exclusive test) or a CV fold file,
  * saliency is read from the frozen-UniSal cache (numerically identical inputs),
  * windows are shuffled globally instead of within groups of 16 clips,
  * explicit seed, longer epoch budget with working early stopping,
  * optional task exclusion (leave-one-behaviour-out) with a reduced-class head.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(HERE))
sys.path.append(HERE)
from core.models import Conv_LSTM  # noqa: E402
from cached_data import build_windows, recording_dirs, WindowTensorDataset, SPLITS  # noqa: E402

# The loss classes live in train.py; import them without running its argparse.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("orig_train", os.path.join(os.path.dirname(HERE), "train.py"))
_orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_orig)
SphericalMSELoss = _orig.SphericalMSELoss


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--split_file", default=SPLITS["eht"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task_weight", type=float, default=0.0, help="lambda; 0 disables the task head")
    ap.add_argument("--future", type=int, default=20)
    ap.add_argument("--exclude_task", type=int, default=None, help="0..3, drop this task from train/val (LOBO)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr_patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--normalize_ts", action="store_true", help="z-score HMD features with train statistics")
    return ap.parse_args()


def load_split(path, name):
    with open(path) as f:
        return json.load(f)[name]


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    out_dir = os.path.join(a.out_root, a.name)
    os.makedirs(out_dir, exist_ok=True)

    use_task = a.task_weight > 0
    w_tr = build_windows("eht", load_split(a.split_file, "train"), future=a.future)
    w_va = build_windows("eht", load_split(a.split_file, "val"), future=a.future)

    task_map = None
    if a.exclude_task is not None:
        keep = [t for t in range(4) if t != a.exclude_task]
        task_map = {t: i for i, t in enumerate(keep)}
        for w in (w_tr, w_va):
            m = w["task"] != a.exclude_task
            for k in list(w):
                w[k] = w[k][m]
            w["task"] = np.array([task_map[t] for t in w["task"]])
    n_classes = 4 if task_map is None else 3

    ts_mean = ts_std = None
    if a.normalize_ts:
        ts_mean = w_tr["ts"].reshape(-1, 4).mean(0)
        ts_std = w_tr["ts"].reshape(-1, 4).std(0) + 1e-6
        for w in (w_tr, w_va):
            w["ts"] = (w["ts"] - ts_mean) / ts_std

    print(f"[{a.name}] train windows {len(w_tr['ts'])}, val windows {len(w_va['ts'])}, classes {n_classes}", flush=True)
    g = torch.Generator().manual_seed(a.seed)
    dl_tr = torch.utils.data.DataLoader(WindowTensorDataset(w_tr), batch_size=a.batch, shuffle=True,
                                        drop_last=True, generator=g)
    dl_va = torch.utils.data.DataLoader(WindowTensorDataset(w_va), batch_size=512, shuffle=False)

    model = Conv_LSTM(4, 10, 15, a.future, 2, lstm_hidden_size=a.hidden, lstm_num_layers=1,
                      use_task_output=use_task).to(device)
    if use_task and n_classes != 4:
        model.task_head[-1] = nn.Linear(64, n_classes).to(device)
    crit_g = SphericalMSELoss("mean")
    crit_t = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=a.lr_patience)

    best, bad, log = float("inf"), 0, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        tr_g = tr_t = 0.0
        nb = 0
        for sal, ts, gt, task in dl_tr:
            sal, ts, gt, task = sal.to(device), ts.to(device), gt.to(device), task.to(device)
            opt.zero_grad()
            out = model(sal, ts)
            if use_task:
                pred, logits = out
                lg, lt = crit_g(pred, gt), crit_t(logits, task)
                loss = lg + a.task_weight * lt
                tr_t += lt.item()
            else:
                pred = out
                lg = crit_g(pred, gt)
                loss = lg
            loss.backward()
            opt.step()
            tr_g += lg.item()
            nb += 1
        model.eval()
        va_g = va_t = 0.0
        acc_n = acc_d = 0
        nvb = 0
        with torch.no_grad():
            for sal, ts, gt, task in dl_va:
                sal, ts, gt, task = sal.to(device), ts.to(device), gt.to(device), task.to(device)
                out = model(sal, ts)
                if use_task:
                    pred, logits = out
                    va_t += crit_t(logits, task).item()
                    acc_n += (logits.argmax(1) == task).sum().item()
                    acc_d += len(task)
                else:
                    pred = out
                va_g += crit_g(pred, gt).item()
                nvb += 1
        va_g /= nvb
        va_t /= nvb
        crit = va_g + (a.task_weight * va_t if use_task else 0.0)
        sched.step(va_g)
        rec = dict(epoch=ep, train_gaze=tr_g / nb, train_task=tr_t / nb, val_gaze=va_g, val_task=va_t,
                   val_acc=(acc_n / acc_d if acc_d else None), criterion=crit, lr=opt.param_groups[0]["lr"],
                   sec=round(time.time() - t0, 1))
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if crit < best:
            best, bad = crit, 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
        else:
            bad += 1
            if bad >= a.patience:
                break

    cfg = dict(vars(a), past_timepoints=15, future_timepoints=a.future, num_time_features=4, conv_output_embeddings=10,
               prediction_features=2, lstm_hidden_states=a.hidden, lstm_layers=1, use_task_output=use_task,
               n_classes=n_classes, task_map=task_map, saliency_sampling=1,
               ts_mean=None if ts_mean is None else ts_mean.tolist(), ts_std=None if ts_std is None else ts_std.tolist(),
               best_criterion=best, training_logs=log)
    with open(os.path.join(out_dir, f"conv_{a.name}.json"), "w") as f:
        json.dump(cfg, f, indent=1)


if __name__ == "__main__":
    main()
