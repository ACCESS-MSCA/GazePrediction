import argparse
import os
import re
import glob
from glob import glob
import json
import numpy as np
import pandas as pd
import cv2
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.utils as utils

from multiprocessing import Process


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


# FOV in horizontal and vertical axes (replace with the HMD values)
fov_horizontal, fov_vertical = 108.77, 111.53


def convert_hor_degre(degrees):
    return (degrees + 180 + 360) % 360


def convert_ver_degree(degrees):
    # -1 because the y-axis is inverted for opencv
    return ((-1) * degrees + 90 + 360) % 360


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def generate_videos(video_path, df_gaze, save_folder, seconds_per_split):
    # Open video cap
    cap = cv2.VideoCapture(video_path)
    # Check if video cap is opened successfully
    if not cap.isOpened():
        raise Exception(f"Failed to open video {vp}")
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps
    # Calculate frame interval
    frame_interval = int(fps * seconds_per_split)

    metadata_list = []
    clip_number = 1
    vd_id = df_gaze["video_id"].iloc[0]
    sbj_id = df_gaze["user_id"].iloc[0]
    task_id = df_gaze["task_id"].iloc[0]
    # Init video format and filename for exporting cropped saliency videos
    save_path = os.path.join(
        save_folder, f"Video_{vd_id}", f"User_{sbj_id}_Task_{task_id}"
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_filename = f"Video_{vd_id}_User_{sbj_id}_Task_{task_id}_{clip_number}"
    # Create video writer for the clip
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    clip_writer = cv2.VideoWriter(
        f"{save_path}/{save_filename}.mp4", fourcc, fps, (384, 288)
    )
    for i, (fnumber, frame_gaze) in enumerate(df_gaze.groupby("frame")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnumber - 1)
        # grab frame
        res, frame = cap.read()
        if res:
            horz_hmd_deg = frame_gaze["hmd_long"].iloc[0]
            vert_hmd_deg = frame_gaze["hmd_lat"].iloc[0]
            giw_long_deg = frame_gaze["giw_long"].iloc[0]
            giw_lat_deg = frame_gaze["giw_lat"].iloc[0]
            eih_x = frame_gaze["eih_x"].iloc[0]
            eih_y = frame_gaze["eih_y"].iloc[0]
            vd_id = frame_gaze["video_id"].iloc[0]
            sbj_id = frame_gaze["user_id"].iloc[0]
            task_id = frame_gaze["task_id"].iloc[0]

            # Video dimensions
            width, height = frame.shape[1], frame.shape[0]
            # Conersion of deegre ranges (-180,180) to (0, 360) and (-90, 90) to (0, 180)
            # hmd
            yaw_hmd = convert_hor_degre(horz_hmd_deg)
            pitch_hmd = convert_ver_degree(vert_hmd_deg)
            # Gaze
            giw_long = convert_hor_degre(giw_long_deg)
            giw_lat = convert_ver_degree(giw_lat_deg)

            # Calculate FOV boundaries in equirectangular coordinates
            # Hmd FOV
            fov_left = int((yaw_hmd - fov_horizontal / 2) * width / 360)
            fov_right = int((yaw_hmd + fov_horizontal / 2) * width / 360)
            fov_top = int((pitch_hmd - fov_vertical / 2) * height / 180)
            fov_bottom = int((pitch_hmd + fov_vertical / 2) * height / 180)

            # Crop the area of interest from the original frame,
            # handle horizontal wrapping
            if fov_right > width:
                cropped_frame_left = frame[fov_top:fov_bottom, fov_left:].copy()
                cropped_frame_right = frame[
                    fov_top:fov_bottom, : fov_right - width
                ].copy()
                cropped_frame = np.hstack((cropped_frame_left, cropped_frame_right))
            elif fov_left < 0:
                cropped_frame_left = frame[fov_top:fov_bottom, fov_left:].copy()
                cropped_frame_right = frame[fov_top:fov_bottom, :fov_right].copy()
                cropped_frame = np.hstack((cropped_frame_left, cropped_frame_right))
            else:
                cropped_frame = frame[fov_top:fov_bottom, fov_left:fov_right].copy()
            # Handle vertical wrapping
            if fov_top < 0:
                top_padding = np.zeros(
                    (-fov_top, cropped_frame.shape[1], 3), dtype=np.uint8
                )
                cropped_frame = np.vstack((top_padding, cropped_frame))
            if fov_bottom > height:
                bottom_padding = np.zeros(
                    (fov_bottom - height, cropped_frame.shape[1], 3), dtype=np.uint8
                )
                cropped_frame = np.vstack((cropped_frame, bottom_padding))

            # Save the cropped frame
            cropped_frame = cv2.resize(cropped_frame, (384, 288))
            clip_writer.write(cropped_frame)

            # metadata save in json
            json_data = {
                "frame_number": fnumber,
                "horz_hmd_deg": horz_hmd_deg,
                "vert_hmd_deg": vert_hmd_deg,
                "giw_long": giw_long_deg,
                "giw_lat": giw_lat_deg,
                "eih_x": eih_x,
                "eih_y": eih_y,
                "fov_left": fov_left,
                "fov_right": fov_right,
                "fov_top": fov_top,
                "fov_bottom": fov_bottom,
                "cropped_frame_filename": save_filename,
                "video_id": vd_id,
                "user_id": sbj_id,
                "task_id": task_id,
            }
            metadata_list.append(json_data)
            # Check if end of clip
            if (i + 1) % frame_interval == 0:
                clip_writer.release()
                # Save the gaze data to a JSON file
                with open(
                    os.path.join(save_path, save_filename + ".json"), "w"
                ) as json_file:
                    json.dump(metadata_list, json_file, cls=NumpyEncoder)
                clip_number += 1
                metadata_list = []
                save_filename = (
                    f"Video_{vd_id}_User_{sbj_id}_Task_{task_id}_{clip_number}"
                )
                clip_writer = cv2.VideoWriter(
                    f"{save_path}/{save_filename}.mp4", fourcc, fps, (384, 288)
                )
    clip_writer.release()
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from ETH video files and gaze data and save them into smaller videos "
    )
    parser.add_argument(
        "-inv", "--input_videos", help="folder with the videos ", required=True
    )
    parser.add_argument(
        "-int",
        "--input_txts",
        help="folder with the txts gaze metadata ",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--saveOutput",
        help="path and name of folder to save the exported videos and .json generated file",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--numberOfThreads",
        help="Number of threads to run the conversion",
        nargs="?",
        const=1,
        type=int,
        default=1,
    )
    parser.add_argument(
        "-sps",
        "--seconds_per_split",
        type=int,
        default=5,
        help="Number of seconds to split the video. Default is 5 seconds.",
    )

    opts = parser.parse_args()

    video_folder = opts.input_videos
    txt_folder = opts.input_txts
    save_folder = opts.saveOutput
    num_threads = opts.numberOfThreads
    seconds_per_split = opts.seconds_per_split

    # Check input
    video_paths = glob(f"{video_folder}/*.[mM][pP][4]")

    if not video_paths:
        raise FileNotFoundError(f"No video files found in path {video_folder}")
    if not os.path.exists(txt_folder):
        raise FileNotFoundError(f"Gaze data folder path {txt_folder} does not exist")
    # Check if save path exists
    if not os.path.exists(save_folder):
        print(f"Creating save folder {save_folder}")
        os.makedirs(save_folder)
    # Open filelist
    with open(os.path.join(txt_folder, "filelist.txt"), "r") as f:
        file_list = [line.strip() for line in f.readlines()]

    # Define a regular expression pattern to match the video number
    pattern_vn = r"Video_(\d+)"
    # pattern_fn = r'User_(\d+)_Video_(\d+)_Task_(\d+)\.txt'

    for vp in video_paths:
        vp_number = int(os.path.basename(vp).split("-")[0].split("Video ")[1])
        valid_files = []
        for file_name in file_list:
            # Use re.search to find the match in the file name
            match = re.search(pattern_vn, file_name)
            # Check if a match is found
            if match:
                # Extract the video number from the match
                video_number = int(match.group(1))
                # Check if the video number matches
                if vp_number == video_number:
                    valid_files.append(file_name)
            else:
                print("A video does not follow the format")
                pass
        print(f"Generating videos for video number {vp_number}")
        print(f"Total Valid files: {len(valid_files)}")
        print(f"Extracting frames from videos with {num_threads} threads")
        # Calculate the step size for every 20% progress
        progress_step = 1
        for i in range(0, len(valid_files), num_threads):
            # Calculate progress percentage
            progress_percent = (i / len(valid_files)) * 100
            if i % progress_step == 0:
                print(f"Video {vp_number} Progress: {int(progress_percent)}%")
            Pros = []
            tmp_valid_files = valid_files[i : i + num_threads]
            for vf in tmp_valid_files:
                gaze_txt_path = os.path.join(txt_folder, vf)
                if not os.path.exists(gaze_txt_path):
                    raise FileNotFoundError(
                        f"Gaze data file {gaze_txt_path} does not exist"
                    )
                # Read the gaze data from the text file corresponding to the video
                gaze_data = pd.read_csv(
                    gaze_txt_path,
                    sep="\t",
                    names=[
                        "timestamp",
                        "frame",
                        "hmd_long",
                        "hmd_lat",
                        "eih_x",
                        "eih_y",
                        "giw_long",
                        "giw_lat",
                    ],
                )
                # Drop duplicate rows based on the 'frame' column
                gaze_data = gaze_data.astype({"frame": "int"})
                df_gaze = gaze_data.drop_duplicates(
                    subset=["frame"], keep="last"
                ).sort_values(by="frame")
                df_gaze["file"] = vf
                df_gaze["video_id"] = vp_number
                df_gaze["user_id"] = int(vf.split("_")[1])
                df_gaze["task_id"] = int(vf.split("_")[-1].split(".")[0])
                # generate_saliency_video(vp, df_gaze, save_folder, sal_model, seconds_per_split)
                # Generate videos with multiple threads
                p = Process(
                    target=generate_videos,
                    args=(vp, df_gaze, save_folder, seconds_per_split),
                )
                Pros.append(p)
                p.start()

            # block until all the threads finish (i.e. block until all function_x calls finish)
            for t in Pros:
                t.join()
