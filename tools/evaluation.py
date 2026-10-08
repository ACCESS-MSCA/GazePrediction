import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
import json
import time
import random
import math
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from the project
from core.models import Conv_tsmixer, Conv_LSTM
from train import SphericalMSELoss
import core.utils as utils

# ==============================================================================
# 1. Optimized Timed Dataloader
# ==============================================================================
class OptimizedTimedDataset(Dataset):
    def __init__(self, video_paths, past_timepoints=15, future_timepoints=10, sliding_window=15):
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

        # Extract Task ID
        task_id = int(metadata["task_id"].iloc[0]) - 1
        
        # Load video into memory once
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            all_frames.append(utils.preprocess_sal_img(frame))
        cap.release()

        if not all_frames: return None

        # Precompute motion and relative gaze features
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
            
            # REFERENCE: HMD at the present frame
            reference_hmd = metadata.loc[fnumber, ["horz_hmd_deg", "vert_hmd_deg"]].values.astype(np.float32)
            
            # Future Gaze (Relative to Step 0 HMD)
            target_gaze = all_gaze[fnumber : fnumber + self.future_timepoints].numpy() - reference_hmd
            target_gaze[:, 0] = (target_gaze[:, 0] + 180) % 360 - 180
            
            # Past Gaze (Relative to Step 0 HMD)
            past_gaze_abs = all_gaze[fnumber - self.past_timepoints : fnumber].numpy()
            past_rel_gaze = past_gaze_abs - reference_hmd
            past_rel_gaze[:, 0] = (past_rel_gaze[:, 0] + 180) % 360 - 180

            sequences.append({
                'images': img_seq,
                'timeseries': torch.tensor(hmd_seq, dtype=torch.float32),
                'gt_gaze': torch.tensor(target_gaze, dtype=torch.float32),
                'past_rel_gaze': torch.tensor(past_rel_gaze, dtype=torch.float32),
                'task_id': task_id
            })
            
        return sequences

def collate_study(batch):
    batch = [s for s in batch if s is not None]
    if not batch: return None
    # Flatten sequences from multiple videos into one flat list
    return [item for sublist in batch for item in sublist]

def angular_error(y_pred, y_true):
    """
    Calculate the Great Circle Distance (Angular Error) in degrees.
    y_pred/y_true: [..., 2] where [..., 0] is azimuth, [..., 1] is elevation (both in degrees)
    """
    # Convert to radians
    p_az = y_pred[..., 0] * np.pi / 180.0
    p_el = y_pred[..., 1] * np.pi / 180.0
    t_az = y_true[..., 0] * np.pi / 180.0
    t_el = y_true[..., 1] * np.pi / 180.0
    
    # Formula: cos(d) = sin(phi1)sin(phi2) + cos(phi1)cos(phi2)cos(delta_lambda)
    # Using the coordinates from the project: Elevation is lat, Azimuth is long
    cos_d = (np.sin(p_el) * np.sin(t_el) + 
             np.cos(p_el) * np.cos(t_el) * np.cos(p_az - t_az))
    
    # Clamp to avoid precision issues
    cos_d = np.clip(cos_d, -1.0, 1.0)
    return np.degrees(np.arccos(cos_d))

# ==============================================================================
# 2. Main Evaluation Logic
# ==============================================================================
def run_study(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Study Mode Active. Using device: {device}")

    # ... (Auto-config logic remains same) ...

    # --- AUTO-CONFIG DETECTION ---
    model_dir = os.path.dirname(args.lstm_model_path)
    config_files = glob(os.path.join(model_dir, "*.json"))
    
    # Defaults (Fallback)
    hidden_size = 32
    use_task = True 
    
    if config_files:
        try:
            with open(config_files[0], 'r') as f:
                config = json.load(f)
                hidden_size = config.get('lstm_hidden_states', hidden_size)
                use_task = config.get('use_task_output', use_task)
                print(f"Auto-configured from JSON: Hidden={hidden_size}, TaskHead={use_task}")
        except Exception as e:
            print(f"Warning: Failed to load config JSON: {e}")
    # -----------------------------

    # Load Splits - Detect dataset type
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

    # --- CACHING LOGIC ---
    dataset_name = "dsav" if "dsav" in args.dataset_path.lower() else "eht"
    output_json = os.path.join(model_dir, f"optimal_horizon_study_{dataset_name}.json")
    
    if os.path.exists(output_json) and not getattr(args, 'force', False):
        print(f"Skipping evaluation. Results found at: {output_json}")
        return
    # ---------------------

    print(f"Evaluating {len(test_vd_paths)} video clips...")

    # Load Models
    print(f"Loading Saliency model from {args.sal_path}")
    unisal_model = utils.load_unisal(args.sal_path, device)
    
    model = Conv_LSTM(
        num_time_features=4, conv_output_embeddings=10, 
        sequence_length=15, prediction_length=args.future_timepoints, 
        prediction_features=2, 
        lstm_hidden_size=hidden_size, # Auto-detected
        use_task_output=use_task # Auto-detected
    ).to(device)
    
    model.load_state_dict(torch.load(args.lstm_model_path, map_location=device), strict=False)
    model.eval()

    criterion = SphericalMSELoss(reduction='none')
    
    # Results
    task_results = {
        0: {"lstm": [], "hmd_center": [], "mean_past": []},
        1: {"lstm": [], "hmd_center": [], "mean_past": []},
        2: {"lstm": [], "hmd_center": [], "mean_past": []},
        3: {"lstm": [], "hmd_center": [], "mean_past": []}
    }
    all_lstm_errors = []
    all_hmd_center_errors = []
    all_mean_hmd_errors = []
    
    # Confusion Matrix Data
    all_gt_tasks = []
    all_pred_tasks = []
    
    task_names = {0: "Free Viewing", 1: "Visual Search", 2: "Saliency", 3: "Track"}

    dataset = OptimizedTimedDataset(test_vd_paths, future_timepoints=args.future_timepoints)
    
    with torch.no_grad():
        for vid_idx in tqdm(range(len(dataset)), desc="Videos"):
            sequences = dataset[vid_idx]
            if not sequences: continue
            
            for seq in sequences:
                imgs = seq['images'].to(device) 
                ts = seq['timeseries'].unsqueeze(0).to(device)
                gt = seq['gt_gaze'].unsqueeze(0).to(device)
                tid = seq['task_id']

                # 1. Baseline Calculations
                center_hmd_preds = torch.zeros_like(gt)
                avg_past_gaze = torch.mean(seq['past_rel_gaze'], dim=0) # (2,)
                mean_pos = avg_past_gaze.unsqueeze(0).expand(args.future_timepoints, -1).unsqueeze(0).to(device)

                # 2. Generate Saliency Maps
                sal_list = []
                if args.saliency_sampling > 1:
                    sampled_indices = list(range(0, len(imgs), args.saliency_sampling))
                    for idx in sampled_indices:
                        sal = unisal_model(imgs[idx].unsqueeze(0))
                        for _ in range(args.saliency_sampling):
                            if len(sal_list) < len(imgs):
                                sal_list.append(sal)
                else:
                    for img in imgs:
                        sal = unisal_model(img.unsqueeze(0))
                        sal_list.append(sal)
                stacked_sal = torch.stack(sal_list).unsqueeze(0)

                # 3. LSTM Inference
                model_out = model(stacked_sal, ts)
                if isinstance(model_out, tuple):
                    pred_gaze_tensor, task_logits = model_out
                    pred_gaze = pred_gaze_tensor.squeeze().cpu().numpy()
                    
                    # Capture Task Prediction
                    pred_task = torch.argmax(task_logits, dim=1).item()
                    all_gt_tasks.append(tid)
                    all_pred_tasks.append(pred_task)
                else:
                    pred_gaze = model_out.squeeze().cpu().numpy()
                
                gt_np = gt.squeeze().cpu().numpy()
                hmd_center_np = center_hmd_preds.squeeze().cpu().numpy()
                mean_past_np = mean_pos.squeeze().cpu().numpy()

                # 4. Calculate Step-wise Angular Error
                # Determine evaluation horizon
                horizon = args.eval_horizon if args.eval_horizon is not None else args.future_timepoints
                
                lstm_step_err = angular_error(pred_gaze[:horizon, :], gt_np[:horizon, :])
                hmd_center_err = angular_error(hmd_center_np[:horizon, :], gt_np[:horizon, :])
                mean_past_err = angular_error(mean_past_np[:horizon, :], gt_np[:horizon, :])

                task_results[tid]["lstm"].append(lstm_step_err)
                task_results[tid]["hmd_center"].append(hmd_center_err)
                task_results[tid]["mean_past"].append(mean_past_err)
                
                all_lstm_errors.append(lstm_step_err)
                all_hmd_center_errors.append(hmd_center_err)
                all_mean_hmd_errors.append(mean_past_err)

    # --- REPORTING ---
    model_dir = os.path.dirname(args.lstm_model_path)
    
    # Save Confusion Matrix Data if available
    if all_gt_tasks:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_gt_tasks, all_pred_tasks, labels=list(task_names.keys()))
        
        suffix = "dsav" if "dsav" in args.dataset_path.lower() else "eht"
        cm_path = os.path.join(model_dir, f"task_confusion_matrix_{suffix}.json")
        
        with open(cm_path, 'w') as f:
            json.dump({
                "matrix": cm.tolist(),
                "labels": list(task_names.values())
            }, f, indent=4)
        print(f"Confusion matrix data saved to: {cm_path}")
        
        # --- Pretty Print Confusion Matrix ---
        print("\nTASK CLASSIFICATION CONFUSION MATRIX:")
        labels = list(task_names.values())
        header = " " * 20 + " | ".join([f"{l[:10]:>10}" for l in labels])
        print(header)
        print("-" * len(header))
        for i, row in enumerate(cm):
            row_str = f"{labels[i][:20]:<20} | " + " | ".join([f"{val:>10}" for val in row])
            print(row_str)
        
        # Calculate Accuracy
        accuracy = np.mean(np.array(all_gt_tasks) == np.array(all_pred_tasks))
        print(f"\nTask Classification Accuracy: {accuracy:.4f}")

    if args.study_mode:
        eval_h = args.eval_horizon if args.eval_horizon is not None else args.future_timepoints
        print("\n" + "="*60)
        print(f"PHASE 1: OPTIMAL HORIZON STUDY (Per Task Analysis, Horizon: {eval_h})")
        print("="*60)
        summary_data = {}
        for tid, name in task_names.items():
            if not task_results[tid]["lstm"]: continue
            
            avg_lstm = np.mean(task_results[tid]["lstm"], axis=0)
            avg_pers = np.mean(task_results[tid]["hmd_center"], axis=0)
            avg_mean = np.mean(task_results[tid]["mean_past"], axis=0)
            
            print(f"\nTask: {name}")
            print(f"  LSTM Avg Error: {np.mean(avg_lstm[:eval_h]):.4f} deg")
            print(f"  HMD-center Avg: {np.mean(avg_pers[:eval_h]):.4f} deg")
            
            summary_data[name] = {
                "lstm_err": avg_lstm[:eval_h].tolist(),
                "hmd_center_err": avg_pers[:eval_h].tolist(),
                "mean_past_err": avg_mean[:eval_h].tolist()
            }

        # --- Overall Summary for run_all.sh Parsing & JSON storage ---
        avg_lstm_overall = np.mean(all_lstm_errors, axis=0)
        avg_pers_overall = np.mean(all_hmd_center_errors, axis=0)
        avg_mean_overall = np.mean(all_mean_hmd_errors, axis=0)
        
        print(f"\nOVERALL SUMMARY (Horizon: {eval_h}):")
        print(f"  Overall Avg LSTM Error: {np.mean(avg_lstm_overall[:eval_h]):.4f} deg")
        print(f"  Overall Avg HMD-center Error: {np.mean(avg_pers_overall[:eval_h]):.4f} deg")

        summary_data["Overall"] = {
            "lstm_err": avg_lstm_overall[:eval_h].tolist(),
            "hmd_center_err": avg_pers_overall[:eval_h].tolist(),
            "mean_past_err": avg_mean_overall[:eval_h].tolist()
        }

        # Use dataset-specific filename inside the model directory
        suffix = "dsav" if "dsav" in args.dataset_path.lower() else "eht"
        model_dir = os.path.dirname(args.lstm_model_path)
        out_json = os.path.join(model_dir, f"optimal_horizon_study_{suffix}.json")

        with open(out_json, "w") as f:
            json.dump(summary_data, f, indent=4)
        print(f"\nStudy results saved to: {out_json}")

    else:
        # Standard Zero-Shot Comparison
        horizon = args.eval_horizon if args.eval_horizon is not None else args.future_timepoints
        avg_lstm_per_step = np.mean(all_lstm_errors, axis=0)
        avg_hmd_center_per_step = np.mean(all_hmd_center_errors, axis=0)
        avg_mean_past_per_step = np.mean(all_mean_hmd_errors, axis=0)

        print("\n" + "="*95)
        print(f"ZERO-SHOT EVALUATION: SR LAG-COMPENSATION STUDY (Horizon: {horizon})")
        print("="*95)
        print(f"Dataset: {args.dataset_path}")
        print(f"Total Test Sequences: {len(all_lstm_errors)}")
        
        print(f"\nTEMPORAL BREAKDOWN (Angular Error in Degrees)")
        print(f"  {'Step':<10} | {'Lag Time':<12} | {'LSTM Error':<18} | {'HMD-center'}")
        print(f"  {'-'*75}")
        for i in range(horizon):
            print(f"  Step {i:02d}    | +{i*33:03d} ms     | {avg_lstm_per_step[i]:<18.4f} | {avg_hmd_center_per_step[i]:.4f}")
        
        print(f"\nSUMMARY ANALYSIS:")
        improvements = avg_hmd_center_per_step[:horizon] - avg_lstm_per_step[:horizon]
        best_step = np.argmax(improvements)
        
        print(f"  - Peak Improvement over HMD-center: {improvements[best_step]:.4f} deg at Step {best_step} (+{best_step*33}ms)")
        print(f"  - Overall Avg Gaze Error: {np.mean(avg_lstm_per_step[:horizon]):.4f} degrees")
        print(f"  - Overall Avg HMD-center Error: {np.mean(avg_hmd_center_per_step[:horizon]):.4f} degrees")
        print("="*95)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm_model_path', type=str, required=True)
    parser.add_argument('--tsmixer_model_path', type=str) # Placeholder
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--sal_path', type=str, default="./models/unisal.pth")
    parser.add_argument('--future_timepoints', type=int, default=20)
    parser.add_argument('--eval_horizon', type=int, default=None, help='Horizon for error calculation (defaults to full horizon)')
    parser.add_argument('--lstm_hidden_states', type=int, default=32)
    parser.add_argument('--study_mode', action='store_true')
    parser.add_argument('--saliency_sampling', type=int, default=1, help='Sample saliency every N frames')
    parser.add_argument('--force', action='store_true', help='Force recalculation')
    args = parser.parse_args()
    
    run_study(args)
