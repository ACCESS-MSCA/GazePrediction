import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
import json
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import Conv_LSTM
import core.utils as utils

# Reuse Dataset logic from evaluation.py
class OptimizedTimedDataset(Dataset):
    def __init__(self, video_paths, past_timepoints=15, future_timepoints=20, sliding_window=15):
        self.video_paths = video_paths
        self.past_timepoints = past_timepoints
        self.future_timepoints = future_timepoints
        self.sliding_window = sliding_window

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        json_meta_path = os.path.splitext(video_path)[0] + ".json"
        
        if not (os.path.exists(video_path) and os.path.exists(json_meta_path)):
            return None

        with open(json_meta_path, "r") as file:
            metadata = pd.DataFrame(json.load(file))

        task_id = int(metadata["task_id"].iloc[0]) - 1
        
        cap = cv2.VideoCapture(video_path)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            all_frames.append(utils.preprocess_sal_img(frame))
        cap.release()

        if not all_frames: return None

        angles = torch.tensor(np.column_stack((
            metadata["horz_hmd_deg"].astype(np.float32), 
            metadata["vert_hmd_deg"].astype(np.float32)
        )))
        all_gaze = torch.tensor(np.column_stack((
            metadata["giw_long"].astype(np.float32), 
            metadata["giw_lat"].astype(np.float32)
        )))
        
        vel_diff = utils.compute_spherical_motion(angles)
        hmd_features = np.column_stack((
            vel_diff['horz_vel'], vel_diff['vert_vel'],
            vel_diff['horz_diff'], vel_diff['vert_diff']
        )).astype(np.float32)

        sequences = []
        for fnumber in range(self.past_timepoints, len(metadata) - self.future_timepoints, self.sliding_window):
            img_seq = torch.stack(all_frames[fnumber - self.past_timepoints : fnumber])
            hmd_seq = hmd_features[fnumber - self.past_timepoints : fnumber]
            reference_hmd = metadata.loc[fnumber, ["horz_hmd_deg", "vert_hmd_deg"]].values.astype(np.float32)
            
            target_gaze = all_gaze[fnumber : fnumber + self.future_timepoints].numpy() - reference_hmd
            target_gaze[:, 0] = (target_gaze[:, 0] + 180) % 360 - 180
            
            sequences.append({
                'images': img_seq,
                'timeseries': torch.tensor(hmd_seq, dtype=torch.float32),
                'gt_gaze': torch.tensor(target_gaze, dtype=torch.float32),
                'task_id': task_id
            })
            
        return sequences

def angular_error(y_pred, y_true):
    p_az, p_el = np.radians(y_pred[..., 0]), np.radians(y_pred[..., 1])
    t_az, t_el = np.radians(y_true[..., 0]), np.radians(y_true[..., 1])
    cos_d = (np.sin(p_el) * np.sin(t_el) + np.cos(p_el) * np.cos(t_el) * np.cos(p_az - t_az))
    return np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))

def run_uncertainty_aware_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Optimal Horizon Mapping (Task Index -> Steps)
    # EHTask Task IDs: 0: Free Viewing, 1: Visual Search, 2: Saliency, 3: Track
    horizon_map = {
        0: args.horizon_free,
        1: args.horizon_search,
        2: args.horizon_saliency,
        3: args.horizon_track
    }
    
    # 2. Setup Data
    dataset_name = "dsav" if "dsav" in args.dataset_path.lower() else "eht"
    model_dir = os.path.dirname(args.dual_model_path)
    output_json = os.path.join(model_dir, f"ua_results_{dataset_name}.json")

    run_eval = False
    if os.path.exists(output_json) and not args.force:
        print(f"Skipping evaluation. Loading cached results from: {output_json}")
        try:
            with open(output_json, 'r') as f:
                cached_data = json.load(f)
            summary = cached_data['summary']
            task_percentages = cached_data['task_percentages']
            per_task_metrics = cached_data.get('per_task_metrics', {})
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Cached results at {output_json} are corrupted or invalid ({e}). Re-evaluating...")
            run_eval = True
    else:
        run_eval = True

    if run_eval:
        if "dsav" in args.dataset_path.lower():
            split_file = "../processed_dataset/splits/dsav360_splits.json"
        else:
            split_file = "../processed_dataset/splits/ehtask_splits.json"
            
        if not os.path.exists(split_file):
            print(f"Error: Splits not found at {split_file}. Run splitting first.")
            return

        with open(split_file, 'r') as f:
            test_paths = json.load(f)['test']
        # Prepend dataset path
        test_vd_paths = [os.path.join(args.dataset_path, p) for p in test_paths]
        # Filter for valid existing paths
        test_vd_paths = [p for p in test_vd_paths if os.path.exists(p)]

        # Expand to all .mp4 files in each directory
        all_clips = []
        for p in test_vd_paths:
            clips = glob(os.path.join(p, "*.mp4"))
            all_clips.extend(clips)
        test_vd_paths = all_clips

        print(f"Evaluating {len(test_vd_paths)} video clips...")
        dataset = OptimizedTimedDataset(test_vd_paths)

        # 3. Load Models
        unisal_model = utils.load_unisal(args.sal_path, device)
        
        # Dual-Task Model
        dual_model = Conv_LSTM(num_time_features=4, conv_output_embeddings=10, 
                              sequence_length=15, prediction_length=20, 
                              prediction_features=2, lstm_hidden_size=32, use_task_output=True).to(device)
        dual_model.load_state_dict(torch.load(args.dual_model_path, map_location=device), strict=False)
        dual_model.eval()

        # Gaze-Only Model
        gaze_only_model = Conv_LSTM(num_time_features=4, conv_output_embeddings=10, 
                                   sequence_length=15, prediction_length=20, 
                                   prediction_features=2, lstm_hidden_size=32, use_task_output=False).to(device)
        gaze_only_model.load_state_dict(torch.load(args.gaze_only_model_path, map_location=device), strict=False)
        gaze_only_model.eval()

        # 4. Evaluation Loop
        results = {"ua_err": [], "fixed10_err": [], "hmd_ua_err": [], "hmd_fixed10_err": []}
        steps_used = []
        
        # Per-Task Results
        # task_id -> { "ua_err": [], "go10_err": [], "go15_err": [], "hmd_ua_err": [] }
        task_specific_results = {i: {"ua_err": [], "go10_err": [], "go15_err": [], "hmd_ua_err": []} for i in range(4)}

        with torch.no_grad():
            for vid_idx in tqdm(range(len(dataset)), desc="Evaluating Sequences"):
                sequences = dataset[vid_idx]
                if not sequences: continue
                
                for seq in sequences:
                    imgs = seq['images'].to(device)
                    ts = seq['timeseries'].unsqueeze(0).to(device)
                    gt = seq['gt_gaze'].cpu().numpy()
                    
                    # Saliency
                    sal_list = [unisal_model(img.unsqueeze(0)) for img in imgs]
                    stacked_sal = torch.stack(sal_list).unsqueeze(0)
                    
                    # UA Model Inference
                    ua_out, task_logits = dual_model(stacked_sal, ts)
                    pred_task = torch.argmax(task_logits, dim=1).item()
                    ua_gaze = ua_out.squeeze().cpu().numpy()
                    
                    # Gaze-Only Model Inference
                    go_gaze = gaze_only_model(stacked_sal, ts).squeeze().cpu().numpy()
                    
                    # Apply Dynamic Strategy
                    ua_step = horizon_map[pred_task]
                    steps_used.append(ua_step)
                    
                    # Indices: Step 10 (idx 9), Step 15 (idx 14), Step 20 (idx 19), Step UA (idx UA-1)
                    ua_idx = ua_step - 1
                    idx10 = 9
                    idx15 = 14
                    idx20 = 19
                    
                    # Error Calculations
                    err_ua = angular_error(ua_gaze[ua_idx], gt[ua_idx])
                    err_hmd_ua = angular_error(np.array([0, 0]), gt[ua_idx])
                    err_go10 = angular_error(go_gaze[idx10], gt[idx10])
                    err_hmd10 = angular_error(np.array([0, 0]), gt[idx10])
                    err_go15 = angular_error(go_gaze[idx15], gt[idx15])
                    err_hmd15 = angular_error(np.array([0, 0]), gt[idx15])
                    err_go20 = angular_error(go_gaze[idx20], gt[idx20])
                    err_hmd20 = angular_error(np.array([0, 0]), gt[idx20])

                    results["ua_err"].append(err_ua)
                    results["hmd_ua_err"].append(err_hmd_ua)
                    results.setdefault("go10_err", []).append(err_go10)
                    results.setdefault("hmd10_err", []).append(err_hmd10)
                    results.setdefault("go15_err", []).append(err_go15)
                    results.setdefault("hmd15_err", []).append(err_hmd15)
                    results.setdefault("go20_err", []).append(err_go20)
                    results.setdefault("hmd20_err", []).append(err_hmd20)
                    results.setdefault("pred_tasks", []).append(pred_task)
                    
                    # Per-Task storage
                    task_specific_results[pred_task]["ua_err"].append(err_ua)
                    task_specific_results[pred_task]["go10_err"].append(err_go10)
                    task_specific_results[pred_task]["go15_err"].append(err_go15)
                    task_specific_results[pred_task]["hmd_ua_err"].append(err_hmd_ua)

        # 5. Aggregate Results
        summary = {
            "MAE UA-Model (Dynamic)": float(np.mean(results["ua_err"])),
            "MAE UA-HMD Baseline": float(np.mean(results["hmd_ua_err"])),
            "MAE Gaze-Only (Step 10)": float(np.mean(results["go10_err"])),
            "MAE HMD Baseline (Step 10)": float(np.mean(results["hmd10_err"])),
            "MAE Gaze-Only (Step 15)": float(np.mean(results["go15_err"])),
            "MAE HMD Baseline (Step 15)": float(np.mean(results["hmd15_err"])),
            "MAE Gaze-Only (Step 20)": float(np.mean(results["go20_err"])),
            "MAE HMD Baseline (Step 20)": float(np.mean(results["hmd20_err"])),
            "Mean Future Steps Used": float(np.mean(steps_used))
        }

        # Calculate Task Percentages
        total_preds = len(results["pred_tasks"])
        task_names = {0: "Free Viewing", 1: "Visual Search", 2: "Saliency", 3: "Track"}
        task_percentages = {task_names[k]: float((results["pred_tasks"].count(k) / total_preds) * 100) for k in range(4)}

        # Aggregate Per-Task Metrics
        per_task_metrics = {}
        for t_id, t_name in task_names.items():
            if len(task_specific_results[t_id]["ua_err"]) > 0:
                per_task_metrics[t_name] = {
                    "MAE UA-Model": float(np.mean(task_specific_results[t_id]["ua_err"])),
                    "MAE UA-HMD": float(np.mean(task_specific_results[t_id]["hmd_ua_err"])),
                    "MAE Gaze-Only (10)": float(np.mean(task_specific_results[t_id]["go10_err"])),
                    "MAE Gaze-Only (15)": float(np.mean(task_specific_results[t_id]["go15_err"])),
                    "Samples": int(len(task_specific_results[t_id]["ua_err"]))
                }

        # Save to Cache
        with open(output_json, 'w') as f:
            json.dump({
                'summary': summary, 
                'task_percentages': task_percentages,
                'per_task_metrics': per_task_metrics
            }, f, indent=4)
    
    # 6. Final Results Presentation (both paths)
    print("\n" + "="*60)
    print(f"UNCERTAINTY-AWARE PERFORMANCE SUMMARY: {dataset_name}")
    print("="*60)
    for k, v in summary.items():
        print(f"{k:<40}: {v:.4f}")
    
    print("\nPredicted Task Distribution:")
    for task, pct in task_percentages.items():
        print(f"  {task:<20}: {pct:.1f}%")

    if per_task_metrics:
        print("\nPer-Task Performance (MAE):")
        print(f"{'Task':<20} | {'UA-Model':<10} | {'UA-HMD':<10} | {'GO-10':<10} | {'GO-15':<10}")
        print("-" * 65)
        for t_name, metrics in per_task_metrics.items():
            print(f"{t_name:<20} | {metrics['MAE UA-Model']:<10.2f} | {metrics['MAE UA-HMD']:<10.2f} | {metrics['MAE Gaze-Only (10)']:<10.2f} | {metrics['MAE Gaze-Only (15)']:<10.2f}")
    print("="*60)

    # LaTeX Table Output
    print("\n--- LaTeX Table Output ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Comparison of Uncertainty-Aware Dynamic Horizon vs Fixed Baselines ({dataset_name}).}}")
    print("\\begin{tabular}{|l|c|c|}")
    print("\\hline")
    print("\\textbf{Model / Baseline} & \\textbf{MAE (deg)} & \\textbf{Horizon} \\\\")
    print("\\hline")
    print(f"Uncertainty-Aware Model (Dynamic) & {summary['MAE UA-Model (Dynamic)']:.2f}$^\\circ$ & {summary['Mean Future Steps Used']:.1f} steps ({summary['Mean Future Steps Used']*33.3:.0f}ms) \\\\")
    print(f"HMD-Centre Baseline (Dynamic)     & {summary['MAE UA-HMD Baseline']:.2f}$^\\circ$ & - \\\\")
    print("\\hline")
    print(f"Gaze-Only Model (Step 10)         & {summary['MAE Gaze-Only (Step 10)']:.2f}$^\\circ$ & 10 steps (333ms) \\\\")
    print(f"HMD-Centre Baseline (Step 10)     & {summary['MAE HMD Baseline (Step 10)']:.2f}$^\\circ$ & - \\\\")
    print("\\hline")
    print(f"Gaze-Only Model (Step 15)         & {summary['MAE Gaze-Only (Step 15)']:.2f}$^\\circ$ & 15 steps (500ms) \\\\")
    print(f"HMD-Centre Baseline (Step 15)     & {summary['MAE HMD Baseline (Step 15)']:.2f}$^\\circ$ & - \\\\")
    print("\\hline")
    print(f"Gaze-Only Model (Step 20)         & {summary['MAE Gaze-Only (Step 20)']:.2f}$^\\circ$ & 20 steps (666ms) \\\\")
    print(f"HMD-Centre Baseline (Step 20)     & {summary['MAE HMD Baseline (Step 20)']:.2f}$^\\circ$ & - \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\label{tab:ua_comparison}")
    print("\\end{table}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dual_model_path', type=str, required=True)
    parser.add_argument('--gaze_only_model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--sal_path', type=str, default="./models/unisal.pth")
    parser.add_argument('--force', action='store_true', help='Force recalculation')
    # Task-specific Horizons
    parser.add_argument('--horizon_track', type=int, default=20)
    parser.add_argument('--horizon_search', type=int, default=15)
    parser.add_argument('--horizon_free', type=int, default=15)
    parser.add_argument('--horizon_saliency', type=int, default=10)
    args = parser.parse_args()
    
    run_uncertainty_aware_eval(args)
