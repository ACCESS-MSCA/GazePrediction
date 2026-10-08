import cv2
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def uv_to_deg(u, v):
    long_deg = (u - 0.5) * 360
    lat_deg = (0.5 - v) * 180
    return long_deg, lat_deg

def convert_hor_degree(degrees):
    return (degrees + 180 + 360) % 360

def convert_ver_degree(degrees):
    return ((-1) * degrees + 90 + 360) % 360

def test_mapping(video_path, head_csv, gaze_csv, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    df_gaze = pd.read_csv(gaze_csv)
    # Filter for first user
    uid = df_gaze['id'].unique()[0]
    df_gaze = df_gaze[df_gaze['id'] == uid].head(300) # 5 seconds at 60fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Testing mapping for user {uid}...")
    orig_fps = 60
    for i, row in tqdm(df_gaze.iterrows(), total=len(df_gaze)):
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Compare synchronization
        calc_frame = int(round(row['t'] / 1000.0 * orig_fps))
        csv_frame = int(row['frame'])
        
        if i < 10:
            print(f"Frame {i}: Calc={calc_frame}, CSV={csv_frame}, Diff={calc_frame - csv_frame}")
        cv2.circle(frame, (px_f, py), 10, (0, 0, 255), -1) # Red = Flipped Long
        
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"Saved test video to {output_path}")

if __name__ == "__main__":
    v_path = "../dsav360_extracted/videos/0001/0001_mono_60fps.mp4"
    h_csv = "../dsav360_extracted/head_data/head_data/head_video_0001.csv"
    g_csv = "../dsav360_extracted/gaze_data/gaze_video_0001.csv"
    test_mapping(v_path, h_csv, g_csv, "test_dsav_mapping.mp4")
