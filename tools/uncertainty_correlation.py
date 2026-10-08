import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import os
import json
from glob import glob
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import sys
import argparse

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import Conv_LSTM
import core.utils as utils
from tools.uncertainty_aware_evaluation import OptimizedTimedDataset, angular_error

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy.cpu().numpy()

def run_correlation_analysis(dataset_path, model_path, sal_path, horizon=10, is_eht=True, force=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup Cache
    dataset_name = "eht" if is_eht else "dsav"
    model_dir = os.path.dirname(model_path)
    output_json = os.path.join(model_dir, f"uncertainty_correlation_{dataset_name}_h{horizon}.json")

    if os.path.exists(output_json) and not force:
        print(f"Skipping analysis. Results found at: {output_json}")
        with open(output_json, 'r') as f:
            data = json.load(f)
            print(f"\n--- Cached Correlation Analysis for {dataset_name} (Horizon: {horizon}) ---")
            print(f"Global Pearson: {data['global_pearson']:.4f}")
            print(f"Global Spearman: {data['global_spearman']:.4f}")
            return

    # Setup Data
    if is_eht:
        split_file = "../processed_dataset/splits/ehtask_splits.json"
    else:
        split_file = "../processed_dataset/splits/dsav360_splits.json"
        
    with open(split_file, 'r') as f:
        test_paths = json.load(f)['test']
    
    # Prepend dataset path
    test_vd_paths = [os.path.join(dataset_path, p) for p in test_paths]
    # Filter for valid existing paths
    test_vd_paths = [p for p in test_vd_paths if os.path.exists(p)]
    
    # Expand to all .mp4 files in each directory
    all_clips = []
    for p in test_vd_paths:
        clips = glob(os.path.join(p, "*.mp4"))
        all_clips.extend(clips)
    test_vd_paths = all_clips
    
    print(f"Analyzing correlation on {len(test_vd_paths)} video clips...")
    dataset = OptimizedTimedDataset(test_vd_paths, future_timepoints=horizon)

    unisal_model = utils.load_unisal(sal_path, device)
    
    # Auto-detect config
    model_dir = os.path.dirname(model_path)
    config_file = None
    for j in glob(os.path.join(model_dir, "*.json")):
        with open(j, 'r') as f:
            tmp_cfg = json.load(f)
            if 'num_time_features' in tmp_cfg:
                config_file = j
                cfg = tmp_cfg
                break

    dual_model = Conv_LSTM(num_time_features=4, conv_output_embeddings=10, 
                          sequence_length=15, prediction_length=20, 
                          prediction_features=2, lstm_hidden_size=cfg.get('lstm_hidden_states', 32), use_task_output=True).to(device)
    dual_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    dual_model.eval()

    entropies = []
    errors = []
    tasks = []

    with torch.no_grad():
        for vid_idx in tqdm(range(len(dataset)), desc=f"Evaluating {os.path.basename(dataset_path)}"):
            sequences = dataset[vid_idx]
            if not sequences: continue
            
            for seq in sequences:
                imgs = seq['images'].to(device)
                ts = seq['timeseries'].unsqueeze(0).to(device)
                gt = seq['gt_gaze'].cpu().numpy()
                tid = seq['task_id']
                
                # Saliency
                sal_list = [unisal_model(img.unsqueeze(0)) for img in imgs]
                stacked_sal = torch.stack(sal_list).unsqueeze(0)
                
                # UA Model Inference
                ua_out, task_logits = dual_model(stacked_sal, ts)
                ua_gaze = ua_out.squeeze().cpu().numpy()
                
                # Entropy
                entropy = compute_entropy(task_logits)[0]
                
                # Error at the given horizon (e.g. Step 10 -> index 9)
                err = angular_error(ua_gaze[horizon-1], gt[horizon-1])
                
                # We use Spherical MSE (squared angular error) for correlation
                spherical_mse = err ** 2
                
                entropies.append(entropy)
                errors.append(spherical_mse)
                tasks.append(tid)

    entropies = np.array(entropies)
    errors = np.array(errors)
    tasks = np.array(tasks)
    
    p_corr, p_pval = pearsonr(entropies, errors)
    s_corr, s_pval = spearmanr(entropies, errors)

    results_data = {
        "global_pearson": float(p_corr),
        "global_pearson_pval": float(p_pval),
        "global_spearman": float(s_corr),
        "global_spearman_pval": float(s_pval),
        "tasks": {}
    }

    print(f"\n--- Correlation Analysis for {os.path.basename(dataset_path)} (Horizon: {horizon}) ---")
    print(f"Global Pearson: {p_corr:.4f} (p={p_pval:.4e})")
    print(f"Global Spearman: {s_corr:.4f} (p={s_pval:.4e})")

    if is_eht:
        task_names = {0: "Free Viewing", 1: "Visual Search", 2: "Saliency", 3: "Track"}
        for t_idx, t_name in task_names.items():
            mask = tasks == t_idx
            if np.sum(mask) > 0:
                p_c, p_p = pearsonr(entropies[mask], errors[mask])
                s_c, s_p = spearmanr(entropies[mask], errors[mask])
                print(f"\nTask: {t_name} (N={np.sum(mask)})")
                print(f"  {t_name} - Pearson: {p_c:.4f} (p={p_p:.4e}), Spearman: {s_c:.4f} (p={s_p:.4e})")
                results_data["tasks"][t_name] = {
                    "pearson": float(p_c),
                    "pearson_pval": float(p_p),
                    "spearman": float(s_c),
                    "spearman_pval": float(s_p),
                    "count": int(np.sum(mask))
                }

    # Save to Cache
    with open(output_json, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"Results saved to: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--eht_path', type=str, help='Path to EHT dataset')
    parser.add_argument('--dsav_path', type=str, help='Path to DSAV dataset')
    parser.add_argument('--sal_path', type=str, default="./models/unisal.pth")
    parser.add_argument('--horizon', type=int, default=10, help='Horizon step for error calculation')
    parser.add_argument('--force', action='store_true', help='Force recalculation')
    args = parser.parse_args()
    
    if args.eht_path:
        run_correlation_analysis(args.eht_path, args.model_path, args.sal_path, horizon=args.horizon, is_eht=True, force=args.force)
    if args.dsav_path:
        run_correlation_analysis(args.dsav_path, args.model_path, args.sal_path, horizon=args.horizon, is_eht=False, force=args.force)
    
    if not args.eht_path and not args.dsav_path:
        print("Error: No dataset path provided. Use --eht_path or --dsav_path.")
