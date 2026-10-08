import torch
import cv2
import json
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import Conv_LSTM
import core.utils as utils

def visualize_inference(video_path, model_path, output_path, alpha=0.4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Metadata
    json_path = os.path.splitext(video_path)[0] + ".json"
    if not os.path.exists(json_path):
        print(f"Error: Metadata not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # 2. Auto-configure and Load Model
    model_dir = os.path.dirname(model_path)
    config_files = glob(os.path.join(model_dir, "*.json"))
    if not config_files:
        print(f"Error: No config JSON found in {model_dir}")
        return
        
    with open(config_files[0], 'r') as f:
        config = json.load(f)
    
    hidden_size = config.get('lstm_hidden_states', 32)
    use_task = config.get('use_task_output', True)
    future_steps = config.get('future_timepoints', 10)
    
    print(f"Config: Hidden={hidden_size}, TaskHead={use_task}, Horizon={future_steps}")

    model = Conv_LSTM(
        num_time_features=4, conv_output_embeddings=10, 
        sequence_length=15, prediction_length=future_steps, 
        prediction_features=2, 
        lstm_hidden_size=hidden_size,
        use_task_output=use_task
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    # Load Saliency Model
    print("Loading UniSal model...")
    sal_path = "./models/unisal.pth"
    if not os.path.exists(sal_path):
        sal_path = "../models/unisal.pth" # Try parent if run from tools/
    unisal_model = utils.load_unisal(sal_path, device)

    # 3. Setup Video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = 384, 288
    
    # Ensure output is in model directory if it's just a filename
    if not os.path.isabs(output_path) and os.sep not in output_path:
        output_path = os.path.join(model_dir, output_path)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Precompute Motion Features
    meta_df = pd.DataFrame(metadata)
    angles = torch.tensor(np.column_stack((
        meta_df["horz_hmd_deg"].astype(np.float32), 
        meta_df["vert_hmd_deg"].astype(np.float32)
    )))
    vel_diff = utils.compute_spherical_motion(angles)
    hmd_features = np.column_stack((
        vel_diff['horz_vel'], vel_diff['vert_vel'], 
        vel_diff['horz_diff'], vel_diff['vert_diff']
    )).astype(np.float32)

    # Constants
    deg_to_pix = width / 110.0 
    cx, cy = width // 2, height // 2

    # Buffer for sequence prediction
    past_sal = []
    
    print(f"Processing: {os.path.basename(video_path)}")
    for i in tqdm(range(len(metadata))):
        ret, frame = cap.read()
        if not ret: break
        
        # Preprocess frame
        pre_frame = utils.preprocess_sal_img(frame).to(device)
        
        with torch.no_grad():
            smap = unisal_model(pre_frame.unsqueeze(0)) # Result is in range [-1, 1]
        
        past_sal.append(smap)
        if len(past_sal) > 15: past_sal.pop(0)

        # Skip rendering/saving for the first 14 frames because we don't have enough history
        if len(past_sal) < 15:
            continue

        # Convert from [-1, 1] back to [0, 1] for visualization
        sal_img = (smap.squeeze().cpu().numpy() + 1.0) / 2.0
        sal_norm = (sal_img - sal_img.min()) / (sal_img.max() - sal_img.min() + 1e-7)
        sal_norm = np.power(sal_norm, 3)
        
        heatmap = cv2.applyColorMap((sal_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (width, height))
        frame_viz = cv2.addWeighted(frame, 1.0, heatmap, alpha, 0)

        # --- 2. LSTM Inference ---
        # Prepare inputs
        ts_seq = torch.tensor(hmd_features[i-14:i+1], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Shape: (1, 15, 1, 1, 288, 384)
            stacked_sal = torch.stack(past_sal).unsqueeze(0)

            model_out = model(stacked_sal, ts_seq)
            if isinstance(model_out, tuple):
                pred_gaze, _ = model_out
            else:
                pred_gaze = model_out
                
            # --- Draw Predicted Trajectory (Red) ---
            # Extract all future points for the horizon
            pred_points = pred_gaze[0].cpu().numpy() # (Horizon, 2)
            
            for step in range(len(pred_points)):
                p_pt = pred_points[step]
                px = int(cx + (p_pt[0] * deg_to_pix))
                py = int(cy - (p_pt[1] * deg_to_pix))
                
                # Fading size/color for future steps
                radius = max(2, 8 - (step // 3))
                alpha_step = max(0.2, 1.0 - (step / len(pred_points)))
                
                cv2.circle(frame_viz, (px, py), radius, (0, 0, 255), -1) # Solid dot
                
                # Draw lines between steps
                if step > 0:
                    prev_pt = pred_points[step-1]
                    ppx = int(cx + (prev_pt[0] * deg_to_pix))
                    ppy = int(cy - (prev_pt[1] * deg_to_pix))
                    cv2.line(frame_viz, (ppx, ppy), (px, py), (0, 0, 200), 1)

        # --- 3. Draw True Gaze Trajectory (Green) ---
        # Fetch ground truth for the same horizon length
        horizon = min(len(pred_points), len(metadata) - i) if 'pred_points' in locals() else 1
        
        gt_points = []
        for h_idx in range(horizon):
            hmd_future = meta_df.loc[i + h_idx, ["horz_hmd_deg", "vert_hmd_deg"]].values
            gaze_future = meta_df.loc[i + h_idx, ["giw_long", "giw_lat"]].values
            
            # Note: We visualize relative to the CURRENT HMD (at step i)
            hmd_ref = meta_df.loc[i, ["horz_hmd_deg", "vert_hmd_deg"]].values
            
            rel_long = gaze_future[0] - hmd_ref[0]
            if rel_long > 180: rel_long -= 360
            if rel_long < -180: rel_long += 360
            rel_lat = gaze_future[1] - hmd_ref[1]
            
            tx = int(cx + (rel_long * deg_to_pix))
            ty = int(cy - (rel_lat * deg_to_pix))
            gt_points.append((tx, ty))
            
            radius = max(2, 6 - (h_idx // 4))
            cv2.circle(frame_viz, (tx, ty), radius, (0, 255, 0), -1)
            
            if h_idx > 0:
                cv2.line(frame_viz, gt_points[h_idx-1], (tx, ty), (0, 200, 0), 1)

        out.write(frame_viz)

    cap.release()
    out.release()
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input FOV .mp4")
    parser.add_argument("--model", required=True, help="Path to trained LSTM .pth")
    parser.add_argument("--output", default="inference_viz.mp4", help="Output filename")
    parser.add_argument("--alpha", type=float, default=0.4, help="Saliency overlay transparency")
    args = parser.parse_args()
    
    visualize_inference(args.video, args.model, args.output, args.alpha)
