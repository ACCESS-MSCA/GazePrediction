import cv2
import json
import os
import argparse
import numpy as np
from tqdm import tqdm

def visualize_video(video_path, output_path):
    # 1. Setup paths
    json_path = os.path.splitext(video_path)[0] + ".json"
    if not os.path.exists(json_path):
        print(f"Error: Metadata not found at {json_path}")
        return

    # 2. Load Metadata
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # 3. Open Video
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Should be 384
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Should be 288
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    # 4. Setup Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing: {os.path.basename(video_path)}")
    
    # Scaling factor: The FOV width is 110 degrees, mapped to the video width (384px)
    # So 1 degree = width / 110 pixels
    deg_to_pix = width / 110.0
    center_x, center_y = width // 2, height // 2

    trail = [] # Store last 10 points for a "tail" effect

    for i, frame_meta in enumerate(tqdm(metadata, desc="Drawing Gaze")):
        ret, frame = cap.read()
        if not ret:
            break

        # Gaze and HMD positions
        gaze_long = frame_meta['giw_long']
        gaze_lat  = frame_meta['giw_lat']
        hmd_long  = frame_meta['horz_hmd_deg']
        hmd_lat   = frame_meta['vert_hmd_deg']

        # Calculate relative offset from the CENTER of the crop (which is HMD)
        rel_long = gaze_long - hmd_long
        if rel_long > 180: rel_long -= 360
        if rel_long < -180: rel_long += 360
        
        rel_lat = gaze_lat - hmd_lat

        # Current Gaze Pixel Position
        gx = int(center_x + (rel_long * deg_to_pix))
        gy = int(center_y - (rel_lat * deg_to_pix)) # Invert Y

        # 1. Draw HMD Center (Small crosshair)
        cv2.drawMarker(frame, (center_x, center_y), (255, 255, 255), cv2.MARKER_CROSS, 15, 1)

        # 2. Draw Trail (Smoothness visualization)
        trail.append((gx, gy))
        if len(trail) > 10: trail.pop(0)
        for t_idx, pos in enumerate(trail):
            alpha = (t_idx + 1) / len(trail)
            cv2.circle(frame, pos, 2, (0, 255, 0), -1)

        # 3. Draw Gaze Circle (Main marker)
        cv2.circle(frame, (gx, gy), 8, (0, 255, 0), 2)
        
        # 4. Add Stats
        cv2.putText(frame, f"Frame: {i}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Gaze Offset: {rel_long:.1f}, {rel_lat:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved visualization to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize true gaze on a 5s clip")
    parser.add_argument("--input", type=str, required=True, help="Path to input .mp4 file")
    parser.add_argument("--output", type=str, default="gaze_viz.mp4", help="Output path")
    
    args = parser.parse_args()
    visualize_video(args.input, args.output)
