import argparse
import os
import re
import glob
from glob import glob
import json
import numpy as np
import pandas as pd
import cv2
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from scipy.interpolate import interp1d

# Custom encoder class for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# FOV in horizontal and vertical axes (HTC Vive Pro Eye)
fov_horizontal, fov_vertical = 110.0, 110.0 

def uv_to_deg(u, v):
    long_deg = (u - 0.5) * 360
    lat_deg = (0.5 - v) * 180
    return long_deg, lat_deg

def uv_to_xyz(u, v):
    # u in [0, 1] -> longitude in [-pi, pi]
    lon = (u - 0.5) * 2 * np.pi
    # v in [0, 1] -> latitude in [pi/2, -pi/2]
    lat = (0.5 - v) * np.pi
    
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z

def xyz_to_uv(x, y, z):
    # Ensure it's on the unit sphere
    mag = np.sqrt(x**2 + y**2 + z**2)
    # Handle possible zero division
    mag = np.where(mag == 0, 1, mag)
    x, y, z = x/mag, y/mag, z/mag
    
    lat = np.arcsin(np.clip(z, -1.0, 1.0))
    lon = np.arctan2(y, x)
    
    u = lon / (2 * np.pi) + 0.5
    v = 0.5 - lat / np.pi
    return u, v

def convert_hor_degre(degrees):
    return (degrees + 180 + 360) % 360

def convert_ver_degree(degrees):
    return ((-1) * degrees + 90 + 360) % 360

def generate_videos(video_path, df_merged, save_folder, seconds_per_split, target_fps=30, skip_existing=False):
    vd_id = df_merged["video_id"].iloc[0]
    sbj_id = df_merged["user_id"].iloc[0]
    task_id = 0 
    display_task_id = task_id + 1
    
    save_path = os.path.join(save_folder, f"Video_{vd_id}", f"User_{sbj_id}_Task_{display_task_id}")
    
    if skip_existing and os.path.exists(save_path):
        if glob(os.path.join(save_path, "*.mp4")) and glob(os.path.join(save_path, "*.json")):
            return

    os.makedirs(save_path, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    frame_interval = int(target_fps * seconds_per_split)
    clip_number = 1
    save_filename = f"Video_{vd_id}_User_{sbj_id}_Task_{display_task_id}_{clip_number}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    clip_writer = cv2.VideoWriter(f"{save_path}/{save_filename}.mp4", fourcc, target_fps, (384, 288))
    
    metadata_list = []
    
    for i, (_, row) in enumerate(df_merged.iterrows()):
        fnumber = int(row['orig_frame'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnumber)
        res, frame = cap.read()
        if not res:
            continue
            
        horz_hmd_deg = row['hmd_long_deg']
        vert_hmd_deg = row['hmd_lat_deg']
        giw_long_deg = row['gaze_long_deg']
        giw_lat_deg = row['gaze_lat_deg']
        
        width, height = frame.shape[1], frame.shape[0]
        yaw_hmd = convert_hor_degre(horz_hmd_deg)
        pitch_hmd = convert_ver_degree(vert_hmd_deg)
        
        fov_left = int((yaw_hmd - fov_horizontal / 2) * width / 360)
        fov_right = int((yaw_hmd + fov_horizontal / 2) * width / 360)
        fov_top = int((pitch_hmd - fov_vertical / 2) * height / 180)
        fov_bottom = int((pitch_hmd + fov_vertical / 2) * height / 180)

        if fov_right > width:
            cropped_frame_left = frame[fov_top:fov_bottom, fov_left:].copy()
            cropped_frame_right = frame[fov_top:fov_bottom, : fov_right - width].copy()
            cropped_frame = np.hstack((cropped_frame_left, cropped_frame_right))
        elif fov_left < 0:
            cropped_frame_left = frame[fov_top:fov_bottom, fov_left:].copy()
            cropped_frame_right = frame[fov_top:fov_bottom, :fov_right].copy()
            cropped_frame = np.hstack((cropped_frame_left, cropped_frame_right))
        else:
            cropped_frame = frame[fov_top:fov_bottom, fov_left:fov_right].copy()
            
        if fov_top < 0:
            top_padding = np.zeros((-fov_top, cropped_frame.shape[1], 3), dtype=np.uint8)
            cropped_frame = np.vstack((top_padding, cropped_frame))
        if fov_bottom > height:
            bottom_padding = np.zeros((fov_bottom - height, cropped_frame.shape[1], 3), dtype=np.uint8)
            cropped_frame = np.vstack((cropped_frame, bottom_padding))

        cropped_frame = cv2.resize(cropped_frame, (384, 288))
        clip_writer.write(cropped_frame)

        json_data = {
            "frame_number": i + 1,
            "orig_frame_number": fnumber,
            "horz_hmd_deg": horz_hmd_deg,
            "vert_hmd_deg": vert_hmd_deg,
            "giw_long": giw_long_deg,
            "giw_lat": giw_lat_deg,
            "eih_x": 0, "eih_y": 0,
            "cropped_frame_filename": save_filename,
            "video_id": vd_id, "user_id": sbj_id, "task_id": display_task_id,
        }
        metadata_list.append(json_data)
        
        if (i + 1) % frame_interval == 0:
            clip_writer.release()
            with open(os.path.join(save_path, save_filename + ".json"), "w") as json_file:
                json.dump(metadata_list, json_file, cls=NumpyEncoder)
            clip_number += 1
            metadata_list = []
            save_filename = f"Video_{vd_id}_User_{sbj_id}_Task_{display_task_id}_{clip_number}"
            clip_writer = cv2.VideoWriter(f"{save_path}/{save_filename}.mp4", fourcc, target_fps, (384, 288))

    if metadata_list:
        clip_writer.release()
        with open(os.path.join(save_path, save_filename + ".json"), "w") as json_file:
            json.dump(metadata_list, json_file, cls=NumpyEncoder)
    else:
        clip_writer.release()
        if os.path.exists(f"{save_path}/{save_filename}.mp4"):
            os.remove(f"{save_path}/{save_filename}.mp4")
    cap.release()

def sync_and_smooth(df_head, df_gaze, video_path, target_fps=30):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # 1. Align time units and Zero-Base
    # Head is seconds, Gaze is milliseconds
    df_head = df_head.copy()
    df_gaze = df_gaze.copy()
    
    df_gaze['t'] = df_gaze['t'] / 1000.0
    
    # Critical: Use a common start time to preserve relative lag/sync
    # Assumption: Both trackers share a common clock (system time)
    t_global_start = min(df_head['t'].min(), df_gaze['t'].min())
    
    df_head['t'] = df_head['t'] - t_global_start
    df_gaze['t'] = df_gaze['t'] - t_global_start
    
    duration = (total_frames - 1) / orig_fps
    target_times = np.arange(0, duration, 1.0 / target_fps)
    
    def interpolate_and_smooth_spherical(df, times):
        # 1. Convert to XYZ
        x, y, z = uv_to_xyz(df['u'].values, df['v'].values)
        df_xyz = pd.DataFrame({'t': df['t'], 'x': x, 'y': y, 'z': z})
        
        # 2. Group by time to remove duplicates
        df_clean = df_xyz.groupby('t')[['x', 'y', 'z']].mean().reset_index()
        
        # 3. Interpolate in 3D Cartesian space using Cubic Splines for natural motion curves
        # Fallback to linear if not enough points are available
        interp_kind = 'cubic' if len(df_clean) > 3 else 'linear'
        f_x = interp1d(df_clean['t'], df_clean['x'], kind=interp_kind, bounds_error=False, fill_value=(df_clean['x'].iloc[0], df_clean['x'].iloc[-1]))
        f_y = interp1d(df_clean['t'], df_clean['y'], kind=interp_kind, bounds_error=False, fill_value=(df_clean['y'].iloc[0], df_clean['y'].iloc[-1]))
        f_z = interp1d(df_clean['t'], df_clean['z'], kind=interp_kind, bounds_error=False, fill_value=(df_clean['z'].iloc[0], df_clean['z'].iloc[-1]))
        
        x_interp = f_x(times)
        y_interp = f_y(times)
        z_interp = f_z(times)
        
        # 4. Apply rolling mean smoothing in 3D (Reduced window to preserve natural dynamics)
        df_interp = pd.DataFrame({'x': x_interp, 'y': y_interp, 'z': z_interp})
        df_smooth = df_interp.rolling(window=5, center=True, min_periods=1).mean()
        
        # 5. Convert back to normalized UV
        u_final, v_final = xyz_to_uv(df_smooth['x'].values, df_smooth['y'].values, df_smooth['z'].values)
        return u_final, v_final

    u_head, v_head = interpolate_and_smooth_spherical(df_head, target_times)
    u_gaze, v_gaze = interpolate_and_smooth_spherical(df_gaze, target_times)
    
    df_sync = pd.DataFrame({
        't': target_times,
        'u_head': u_head, 'v_head': v_head,
        'u_gaze': u_gaze, 'v_gaze': v_gaze,
        'orig_frame': (target_times * orig_fps).round().astype(int)
    })
    
    df_sync['hmd_long_deg'], df_sync['hmd_lat_deg'] = uv_to_deg(df_sync['u_head'], df_sync['v_head'])
    df_sync['gaze_long_deg'], df_sync['gaze_lat_deg'] = uv_to_deg(df_sync['u_gaze'], df_sync['v_gaze'])
    
    return df_sync

def process_single_task(task_data, extracted_path, save_folder, seconds_per_split, target_fps, skip_existing):
    video_id, user_id = task_data
    video_id_str = f"{video_id:04d}"
    
    # Quick check for skip_existing before heavy processing
    save_path = os.path.join(save_folder, f"Video_{video_id}", f"User_{user_id}_Task_1")
    if skip_existing and os.path.exists(save_path):
        if glob(os.path.join(save_path, "*.mp4")) and glob(os.path.join(save_path, "*.json")):
            return

    video_path_glob = os.path.join(extracted_path, "videos", video_id_str, f"{video_id_str}_mono_60fps.mp4")
    video_paths = glob(video_path_glob)
    if not video_paths: return
    video_path = video_paths[0]
    
    head_csv = os.path.join(extracted_path, "head_data", "head_data", f"head_video_{video_id_str}.csv")
    gaze_csv = os.path.join(extracted_path, "gaze_data", f"gaze_video_{video_id_str}.csv")
    
    # Only load data for specific user to save memory and time
    df_head = pd.read_csv(head_csv)
    df_head = df_head[df_head['id'] == user_id].copy()
    df_gaze = pd.read_csv(gaze_csv)
    df_gaze = df_gaze[df_gaze['id'] == user_id].copy()
    
    df_sync = sync_and_smooth(df_head, df_gaze, video_path, target_fps)
    df_sync['video_id'] = video_id
    df_sync['user_id'] = user_id
    
    generate_videos(video_path, df_sync, save_folder, seconds_per_split, target_fps, skip_existing)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DSAV360 dataset with High-Level Parallelism")
    parser.add_argument("--extracted_path", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--seconds_per_split", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=cpu_count())
    parser.add_argument("--video_range", type=str, default="1-85")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--test_only", action="store_true", help="Process only the test set sequences")
    parser.add_argument("--split_file", type=str, default="../processed_dataset/splits/dsav360_splits.json", help="Path to the JSON split file")
    
    args = parser.parse_args()
    
    start_vid, end_vid = map(int, args.video_range.split('-'))
    video_ids = list(range(start_vid, end_vid + 1))
    
    # Load splits if test_only is requested
    test_filter = None
    if args.test_only:
        if not os.path.exists(args.split_file):
            print(f"Error: Split file {args.split_file} not found.")
            exit(1)
        with open(args.split_file, 'r') as f:
            splits = json.load(f)
            test_filter = set(splits.get('test', []))
            print(f"Loaded {len(test_filter)} test sequences from split file.")

    print("Building task list...")
    all_tasks = []
    for vid in video_ids:
        vid_str = f"{vid:04d}"
        head_csv = os.path.join(args.extracted_path, "head_data", "head_data", f"head_video_{vid_str}.csv")
        gaze_csv = os.path.join(args.extracted_path, "gaze_data", f"gaze_video_{vid_str}.csv")
        if os.path.exists(head_csv) and os.path.exists(gaze_csv):
            df_head = pd.read_csv(head_csv, usecols=['id'])
            df_gaze = pd.read_csv(gaze_csv, usecols=['id'])
            users = list(set(df_head['id'].unique()) & set(df_gaze['id'].unique()))
            for user_id in users:
                # Format matching the split file: "Video_X/User_Y_Task_1"
                task_key = f"Video_{vid}/User_{user_id}_Task_1"
                if test_filter is None or task_key in test_filter:
                    all_tasks.append((vid, user_id))
    
    print(f"Total tasks to process: {len(all_tasks)}")
    
    process_func = partial(process_single_task, 
                           extracted_path=args.extracted_path, 
                           save_folder=args.save_folder, 
                           seconds_per_split=args.seconds_per_split,
                           target_fps=30,
                           skip_existing=args.skip_existing)
    
    with Pool(processes=args.num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_func, all_tasks), total=len(all_tasks), desc="Total Progress"))
