import os
import sys
import json
import cv2
import torch
import numpy as np
import argparse
import glob

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import utils, models

def visualize_single_frame(video_path, model_path, frame_idx=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Metadata and Video
    json_path = video_path.replace(".mp4", ".json")
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    cap = cv2.VideoCapture(video_path)
    
    # 2. Load Model and Prepare Input
    # Auto-detect config
    model_dir = os.path.dirname(model_path)
    all_jsons = glob.glob(os.path.join(model_dir, "*.json"))
    config_file = None
    for j in all_jsons:
        with open(j, 'r') as f:
            tmp_cfg = json.load(f)
            if 'num_time_features' in tmp_cfg:
                config_file = j
                cfg = tmp_cfg
                break
    
    if not config_file:
        print(f"Could not find model configuration JSON in {model_dir}")
        return
    
    model = models.Conv_LSTM(
        num_time_features=cfg['num_time_features'],
        conv_output_embeddings=cfg['conv_output_embeddings'],
        sequence_length=cfg['past_timepoints'],
        prediction_length=cfg['future_timepoints'],
        prediction_features=cfg['prediction_features'],
        lstm_hidden_size=cfg.get('lstm_hidden_states', 32),
        lstm_num_layers=cfg.get('lstm_layers', 1),
        use_task_output=cfg.get('use_task_output', False)
    )
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device).eval()

    # Seek to the correct future frame index (Past Context + Step - 1)
    target_video_idx = cfg['past_timepoints'] + frame_idx - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_video_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame at index {target_video_idx}.")
        return
    
    def unisal_model_fn(x):
        if not hasattr(visualize_single_frame, "_unisal"):
            visualize_single_frame._unisal = utils.load_unisal(cfg.get('sal_path', './models/unisal.pth'), device)
        return visualize_single_frame._unisal(x)
    
    # Prepare historical context (past 15 frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    sal_list = []
    for _ in range(cfg['past_timepoints']):
        r, f = cap.read()
        img_tensor = utils.preprocess_sal_img(f).to(device)
        with torch.no_grad():
            sal = unisal_model_fn(img_tensor.unsqueeze(0))
            sal_list.append(sal)

    stacked_sal = torch.stack(sal_list).unsqueeze(0).to(device)
    
    # Prepare head motion features (from metadata)
    hmd_angles = []
    for i in range(cfg['past_timepoints']):
        hmd_angles.append([metadata[i]['horz_hmd_deg'], metadata[i]['vert_hmd_deg']])
    
    motion_info = utils.compute_spherical_motion(torch.tensor(hmd_angles))
    ts_input = torch.tensor(np.column_stack((
        motion_info['horz_vel'], motion_info['vert_vel'],
        motion_info['horz_diff'], motion_info['vert_diff']
    ))).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        out = model(stacked_sal, ts_input)
        if isinstance(out, tuple):
            pred_gaze_rel = out[0].cpu().numpy()[0] # (Future, 2)
        else:
            pred_gaze_rel = out.cpu().numpy()[0]

    # 4. Draw
    pred_rel = pred_gaze_rel[frame_idx-1] # (az, el) relative to HMD at Step 0
    
    # Ground Truth Relative Gaze from metadata
    ref_hmd = np.array([metadata[cfg['past_timepoints']]['horz_hmd_deg'], metadata[cfg['past_timepoints']]['vert_hmd_deg']])
    gt_abs = np.array([metadata[target_video_idx]['giw_long'], 
                       metadata[target_video_idx]['giw_lat']])
    
    # Relative to HMD at start of prediction
    gt_rel = gt_abs - ref_hmd
    
    # --- Resolution Upscaling (2x) ---
    scale = 2.0
    h, w = frame.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Map Relative Degrees to Pixel Coordinates
    def deg_to_pixel(rel_deg, size, fov):
        return int((rel_deg / fov + 0.5) * size)

    gt_px = (deg_to_pixel(gt_rel[0], new_w, 110), deg_to_pixel(gt_rel[1], new_h, 110))
    pr_px = (deg_to_pixel(pred_rel[0], new_w, 110), deg_to_pixel(pred_rel[1], new_h, 110))

    # Scaled markers with center dots
    radius = int(8 * scale)
    thickness = int(1 * scale)
    dot_radius = int(1 * scale)
    
    # GT Green
    cv2.circle(frame, gt_px, radius, (0, 255, 0), thickness)
    cv2.circle(frame, gt_px, dot_radius, (0, 255, 0), -1)
    
    # Pred Red
    cv2.circle(frame, pr_px, radius, (0, 0, 255), thickness)
    cv2.circle(frame, pr_px, dot_radius, (0, 0, 255), -1)
    
    # HMD Center Blue
    cv2.circle(frame, (new_w//2, new_h//2), radius, (255, 0, 0), thickness)
    cv2.circle(frame, (new_w//2, new_h//2), dot_radius, (255, 0, 0), -1)

    output_name = f"gaze_viz_frame{frame_idx}_2x.png"
    cv2.imwrite(output_name, frame)
    print(f"High-res visualization saved to {output_name}")
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--step", type=int, default=10, help="Prediction step to visualize (1-20)")
    args = parser.parse_args()
    visualize_single_frame(args.video, args.model, frame_idx=args.step)
