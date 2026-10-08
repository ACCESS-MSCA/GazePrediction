import os
import glob as gb
import json
# import re
import numpy as np
import pandas as pd
import cv2
# from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import torch
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import SphericalMSELoss
import core.utils as utils

import warnings
warnings.filterwarnings("ignore")
import argparse

def add_inference_log(log_data, filename="args_config.json"):
    # Load existing JSON data if file exists, otherwise create a new dictionary
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # Initialize or update training logs
    if "inference_logs" not in data:
        data["inference_logs"] = []

    data["inference_logs"].append(log_data)

    # Save updated data back to JSON
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for processing video dataset and applying models.")
    
    parser.add_argument(
        '--cut_dataset_path', 
        type=str, 
        default='/mnt/DataLake/AlienBackup/access/processed_dataset/eht_dataset_prossesed_vid',
        help="Path to the processed dataset directory."
    )
    parser.add_argument(
        '--raw_videos_path', 
        type=str, 
        default='/mnt/DataLake/AlienBackup/access/eht_dataset/Videos/*.mp4',
        help="Path to the raw videos."
    )
    parser.add_argument(
        '--no_save_video', 
        action='store_false',
        help="Flag to save the video with the prediction (True/False)."
    )
    parser.add_argument(
        '--trained_model_folder', 
        type=str, 
        default='trained_models_combined',
        help="Path to the folder containing the trained models."
    )
    parser.add_argument(
        '--model_base_name', 
        type=str, 
        default='conv_lstm_past_15_future_10_window_15_20250316_162231',
        help="Base name for the model being used."
    )
    parser.add_argument(
        '--model_epoch_name_path', 
        type=str, 
        default='model_epoch_5_val_loss_49.0553.pth',
        help="Path to the specific model epoch file."
    )
    parser.add_argument(
        '--sal_path', 
        type=str, 
        default='/mnt/DataLake/AlienBackup/access/eht_gaze_prediction/models/unisal.pth',
        help="Path to the SAL model file."
    )
    
    parser.add_argument(
        '--folder_save_path', 
        type=str, 
        default='/mnt/DataLake/AlienBackup/access/results_combined_models',
        help="Path to the folder where outputs will be saved."
    )
    
    
    return parser.parse_args()

args = parse_arguments()

test_parent_folder = (
        os.path.join(args.cut_dataset_path,"Video_8","User_8_Task_1"),
        os.path.join(args.cut_dataset_path,"Video_1","User_26_Task_1")
    )

json_paths_videos = []
for test_path in test_parent_folder:
    json_paths = gb.glob(os.path.join(test_path, "*.json"))
    json_paths.sort(key=utils.natural_keys)
    json_paths_videos.append(json_paths)

raw_video_paths = gb.glob(args.raw_videos_path)

raw_video_path = []
metadata_videos = []
for json_paths in tqdm(json_paths_videos):
    metadata = []
    for json_path in json_paths:
        with open(json_path, "r") as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            metadata.append(df)
    
    metadata = pd.concat(metadata, ignore_index=True)
    video_id = metadata.iloc[0]['video_id']
    raw_video_path.extend([path for path in raw_video_paths if f"Video {video_id} " in path])
    metadata_videos.append(metadata)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unisal_model = utils.load_unisal(args.sal_path, device)

time_model, timepoints, future_steps = utils.load_time_model(args.trained_model_folder, args.model_base_name, args.model_epoch_name_path, device)

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Specify the codec

# Initliaze raw video capture
for idx_video, raw_video in enumerate(raw_video_path):
    preds = []
    preds_giw_diff = []
    ground_truth = []
    gt_giw_diff = []
    all_previous_giw = []
    all_previous_hmd = []
    centers = []
    cap = cv2.VideoCapture(raw_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {raw_video}.")
        continue
    
    metadata = metadata_videos[idx_video]
    os.makedirs(f"{args.folder_save_path}/{args.model_base_name}", exist_ok=True)
    if args.no_save_video:
        output_video = cv2.VideoWriter(
        f"{args.folder_save_path}/{args.model_base_name}/pred{future_steps}_{metadata.iloc[0]['video_id']}_{metadata.iloc[0]['user_id']}_{metadata.iloc[0]['task_id']}.mp4",
        fourcc,
        30.0,
        (1280, 720),
        )
    
    sal_frames = []
    # Create salency maps for timepoints-1 frames and save to sal_frames list
    for i in tqdm(range(timepoints-1)):
        fnumber = metadata.iloc[i]["frame_number"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnumber)
        frame = cap.read()[1]
        if frame is None:
            continue
        frame_tensor = torch.tensor(np.array([utils.preprocess_sal_img(frame)]), dtype=torch.float32)
        with torch.no_grad():
            sal_frame = unisal_model(frame_tensor.to(device))
        sal_frames.append(sal_frame.squeeze().squeeze())

    for i in tqdm(range(timepoints, len(metadata) - future_steps - 1)):#len(metadata)
        # Get the last {current} frame
        fnumber = metadata.iloc[i]["frame_number"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnumber)
        original_frame = cap.read()[1]
        if original_frame is None:
            continue
        frame = utils.preprocess_sal_img(original_frame)
        
        frame_tensor = torch.tensor(np.array([frame]), dtype=torch.float32)
        # Load timepoint metadata
        meta_timepoints = metadata.iloc[i-timepoints: i]
        current_frame_data = meta_timepoints.iloc[-1]

        angles_horz = meta_timepoints["horz_hmd_deg"].to_numpy()
        angles_vert = meta_timepoints["vert_hmd_deg"].to_numpy()
        angles = torch.tensor(np.column_stack((angles_horz, angles_vert)))
        vel_diff = utils.compute_spherical_motion(angles)
        hmd_ang_vel_horz = vel_diff['horz_vel']
        hmd_ang_vel_vert = vel_diff['vert_vel']
        hmd_ang_diff_horz = vel_diff['horz_diff']
        hmd_ang_diff_vert = vel_diff['vert_diff']
        # Load current gaze and future gaze
        future_metadata_timepoints = metadata.iloc[i : i+future_steps]
        giw_angdiff_long = (
            future_metadata_timepoints["giw_long"]
            - current_frame_data["horz_hmd_deg"]
        )
        giw_angdiff_lat = (
            future_metadata_timepoints["giw_lat"]
            - current_frame_data["vert_hmd_deg"]
        )

        # Load saliency maps
        if args.saliency_sampling > 1:
            # We only calculate a new saliency if (i % sampling == 0)
            # and then we keep using it for the next sampling-1 steps.
            # But the 'sal_frames' list needs to represent the sliding window correctly.
            if i % args.saliency_sampling == 0:
                with torch.no_grad():
                    sal_frame = unisal_model(frame_tensor.to(device))
                current_sal = sal_frame.squeeze().squeeze()
            else:
                # current_sal remains the same from the last valid sampling point
                pass
            sal_frames.append(current_sal)
        else:
            with torch.no_grad():
                sal_frame = unisal_model(frame_tensor.to(device))
            sal_frames.append(sal_frame.squeeze().squeeze())

        if len(sal_frames)>timepoints:
            sal_frames.pop(0)

        sal_frames_processed = torch.stack(sal_frames, dim=0).cpu().detach()
        
        timeseries = list(
            zip(hmd_ang_vel_horz, hmd_ang_vel_vert, 
                hmd_ang_diff_horz, hmd_ang_diff_vert)
        )
        with torch.no_grad():
            pred = time_model(
                torch.tensor(sal_frames_processed, dtype=torch.float32).unsqueeze(0).to(device),
                torch.tensor(timeseries, dtype=torch.float32).unsqueeze(0).to(device),
            )
        
        pred_np = pred[0].detach().cpu().numpy()
        # Visualize prediction
        p_long = np.where(pred_np[:, 0] + current_frame_data["horz_hmd_deg"]>frame.shape[1], pred_np[:, 0] + current_frame_data["horz_hmd_deg"]-frame.shape[1], pred_np[:, 0] + current_frame_data["horz_hmd_deg"])
        p_lat = pred_np[:, 1] + current_frame_data["vert_hmd_deg"]
        preds.append([p_long, p_lat])
        preds_giw_diff.append([pred_np[:,0], pred_np[:, 1]])

        g_long = future_metadata_timepoints["giw_long"].to_numpy()
        g_lat = future_metadata_timepoints["giw_lat"].to_numpy()
        ground_truth.append([g_long, g_lat])
        gt_giw_diff.append([giw_angdiff_long, giw_angdiff_lat])
        
        previous_future_metadata = metadata.iloc[i-timepoints: i+future_steps]
        # input degrees
        i_long = previous_future_metadata["giw_long"].to_numpy()
        i_lat = previous_future_metadata["giw_lat"].to_numpy()
        all_previous_giw.append([i_long, i_lat])
        
        i_long = previous_future_metadata["horz_hmd_deg"].to_numpy()
        i_lat = previous_future_metadata["vert_hmd_deg"].to_numpy()
        all_previous_hmd.append([i_long, i_lat])
        # Visualize frame
        # Video dimensions
        width, height = original_frame.shape[1], original_frame.shape[0]
        fov_bottom, fov_top, fov_left, fov_right = current_frame_data[
            ["fov_bottom", "fov_top", "fov_left", "fov_right"]
        ].values
        if args.no_save_video:
            # Draw FOV
            vis_frame = utils.draw_fov_frame(
                original_frame, fov_top, fov_bottom, fov_left, fov_right, width, height
            )
            if vis_frame is None:
                pass
            
            # Draw gaze circles ground truth and prediction # Green True, Red Prediction
            vis_frame = utils.draw_gaze(
                vis_frame, g_long[-1], g_lat[-1], width, height, color=(0, 255, 0)
            )
            vis_frame = utils.draw_gaze(
                vis_frame, p_long[-1], p_lat[-1], width, height, color=(0, 0, 255)
            )
            vis_frame = cv2.resize(vis_frame, (1280, 720))
            
            # Save the current frame to the video file
            if vis_frame is not None:
                output_video.write(vis_frame)
    cap.release()
    if args.no_save_video:
        output_video.release()
    for i in random.sample(range(0, len(preds)), 10):
        plt.figure()
        plt.plot(range(i, i+timepoints+future_steps), all_previous_giw[i][0], '-o', label='giw')
        plt.plot(range(i, i+timepoints+future_steps), all_previous_hmd[i][0], '-o', label='hmd', color='orange')
        plt.plot(range(i+timepoints, i+timepoints+future_steps), preds[i][0], '-o', color='r', label='prediction')
        plt.axvline(x = i+timepoints, color = 'r', linestyle = '--') 
        plt.title(f'Horizontal axis (current frame {i+timepoints})')
        plt.legend()
        mini = min(min(all_previous_giw[i][0]), min(all_previous_hmd[i][0]), min(preds[i][0]))
        maxi = max(max(all_previous_giw[i][0]), max(all_previous_hmd[i][0]), max(preds[i][0]))
        plt.ylim(mini - max((90 - (maxi - mini)/2), 90), maxi + max((90 - (maxi - mini)/2), 90))
        plt.savefig(f"{args.folder_save_path}/{args.model_base_name}/{metadata.iloc[0]['video_id']}_{metadata.iloc[0]['user_id']}_{metadata.iloc[0]['task_id']}_frame{i+timepoints}.png", format='png')
    # print(torch.tensor(ground_truth).permute(0, 2, 1).shape)
    # print(torch.tensor(all_previous_hmd).shape)
    # print(torch.tensor(preds).permute(0, 2, 1).shape)
    # print(torch.tensor(all_previous_hmd)[:, :, timepoints].unsqueeze(-1).repeat(1, 1, future_steps).shape)
    # print(torch.mean(torch.tensor(all_previous_hmd)[:, :, timepoints], dim=0).view(2, 1, 1).repeat(1,torch.tensor(ground_truth).shape[0], future_steps).permute(1, 2, 0).shape)
    # print(torch.tensor(ground_truth).shape[0])
    mse_func = SphericalMSELoss('mean_axis')
    mse = mse_func(torch.tensor(preds).permute(0, 2, 1) , torch.tensor(ground_truth).permute(0, 2, 1))
    mse_center = mse_func(torch.tensor(all_previous_hmd)[:, :, timepoints].unsqueeze(-1).repeat(1, 1, future_steps).permute(0, 2, 1), torch.tensor(ground_truth).permute(0, 2, 1))
    mse_mean_value = mse_func(torch.mean(torch.tensor(all_previous_hmd)[:, :, timepoints], dim=0).view(2, 1, 1).repeat(1,torch.tensor(ground_truth).shape[0], future_steps).permute(1, 2, 0), torch.tensor(ground_truth).permute(0, 2, 1))
    
    logs = {}
    logs['video_path'] = raw_video
    logs['MSE_each_forecast_steps_each_axis'] = mse.tolist()
    logs['RMSE_each_forecast_steps_each_axis'] = torch.sqrt(mse).tolist()
    logs['MSE_all_each_axis'] = mse.mean(dim=0).tolist()
    logs['RMSE_all_each_axis'] = torch.sqrt(mse.mean(dim=0)).tolist()
    logs['MSE_center_hmd_each_forecast_steps_each_axis'] = mse_center.tolist()
    logs['RMSE_center_hmd_each_forecast_steps_each_axis'] = torch.sqrt(mse_center).tolist()
    logs['MSE_center_hmd_all_each_axis'] = mse_center.mean(dim=0).tolist()
    logs['RMSE_center_hmd_all_each_axis'] = torch.sqrt(mse_center.mean(dim=0)).tolist()
    logs['MSE_mean_hmd_each_forecast_steps_each_axis'] = mse_mean_value.tolist()
    logs['RMSE_mean_hmd_each_forecast_steps_each_axis'] = torch.sqrt(mse_mean_value).tolist()
    logs['MSE_mean_hmd_all_each_axis'] = mse_mean_value.mean(dim=0).tolist()
    logs['RMSE_mean_hmd_all_each_axis'] = torch.sqrt(mse_mean_value.mean(dim=0)).tolist()
    add_inference_log(logs, f"{args.folder_save_path}/{args.model_base_name}/{metadata.iloc[0]['video_id']}_{metadata.iloc[0]['user_id']}_{metadata.iloc[0]['task_id']}_inference_metrics.json")
