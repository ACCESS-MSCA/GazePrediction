import cv2
import os
import json
import numpy as np
import pandas as pd
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
import random
from . import utils

class GazePredDataset(Dataset):
    def __init__(
        self,
        video_paths,
        past_timepoints=15,
        future_timepoints=6,
        frame_sample_rate=5,
        sliding_window=1,
        original_fps=30,
    ):
        self.video_paths = video_paths
        self.past_timepoints = past_timepoints
        self.future_timepoints = future_timepoints
        self.sliding_window = sliding_window
        self.frame_sample_rate = frame_sample_rate
        self.original_fps = original_fps

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = []
        org_frames = []
        head_positions = []
        prediction_sequences = []
        filenames = []
        tasks = []
        # load json metadata
        json_meta_path = os.path.splitext(video_path)[0] + ".json"
        if os.path.exists(video_path) and os.path.exists(json_meta_path):
            with open(json_meta_path, "r") as file:
                json_data = json.load(file)
            metadata = pd.DataFrame(json_data)

            # calculate velocities and angle diffs in all 5 sec videos
            angles_horz = metadata["horz_hmd_deg"].to_numpy()
            angles_vert = metadata["vert_hmd_deg"].to_numpy()
            angles = torch.tensor(np.column_stack((angles_horz, angles_vert)))
            vel_diff = utils.compute_spherical_motion(angles)
            hmd_ang_vel_horz = vel_diff['horz_vel']
            hmd_ang_vel_vert = vel_diff['vert_vel']
            hmd_ang_diff_horz = vel_diff['horz_diff']
            hmd_ang_diff_vert = vel_diff['vert_diff']
            giw_horz = metadata["giw_long"].to_numpy()
            giw_vert = metadata["giw_lat"].to_numpy()
            filename = metadata["cropped_frame_filename"].to_numpy()[0]
            task_id = metadata["task_id"].to_numpy()[0]
            # start frame extraction - Read once into memory
            cap = cv2.VideoCapture(video_path)
            all_frames_in_vid = []
            while True:
                ret, frame_org = cap.read()
                if not ret:
                    break
                # Pre-process saliency input immediately to save memory and time later
                all_frames_in_vid.append(utils.preprocess_sal_img(frame_org))
            cap.release()
            
            if not all_frames_in_vid:
                return None

            # fnmber is the present frame
            for fnumber in range(
                self.past_timepoints,
                len(metadata) - self.future_timepoints,
                self.sliding_window,
            ):
                # Gather frames from pre-loaded list and stack into a tensor
                seq_tensors = all_frames_in_vid[fnumber - self.past_timepoints : fnumber]
                
                # Check if we have enough frames
                if len(seq_tensors) < self.past_timepoints:
                    continue
                
                images_seq = torch.stack(seq_tensors)
                
                # Gather hmd potional metadata
                hmd_seq = list(
                    zip(
                        hmd_ang_vel_horz[fnumber - self.past_timepoints : fnumber ],
                        hmd_ang_vel_vert[fnumber - self.past_timepoints : fnumber ],
                        hmd_ang_diff_horz[fnumber - self.past_timepoints : fnumber ],
                        hmd_ang_diff_vert[fnumber - self.past_timepoints : fnumber ],
                        
                    )
                )
                pred_horz = giw_horz[fnumber : fnumber + self.future_timepoints] - angles_horz[fnumber]
                pred_horz = np.where(pred_horz > 180, pred_horz - 360, pred_horz)
                pred_horz = np.where(pred_horz < -180, pred_horz + 360, pred_horz)

                pred_seq = list(
                    zip(
                        (
                            pred_horz
                        ),
                        (
                            giw_vert[fnumber : fnumber + self.future_timepoints]
                            - angles_vert[fnumber]
                        ),
                    )
                )
                # Gather all inputs
                frames.append(images_seq)
                head_positions.append(hmd_seq)
                prediction_sequences.append(pred_seq)
                filenames.extend([filename])
                tasks.append(task_id - 1)
            
            return (
                torch.stack(frames) if frames else None,
                torch.tensor(np.array(head_positions), dtype=torch.float32) if head_positions else None,
                torch.tensor(np.array(prediction_sequences), dtype=torch.float32) if prediction_sequences else None,
                np.array(filenames, dtype=str) if filenames else None,
                torch.tensor(np.array(tasks), dtype=torch.long) if tasks else None,
            )


def collate_fn(batch, minibatch=16, shuffle=False):
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None and sample[0] is not None]
    if len(batch) == 0:
        return None  # Return None if all samples are None

    frames_batch, head_positions_batch, predictions, filename, tasks_batch = zip(*batch)
    # Flatten the first dimension
    frames_flat = torch.cat(frames_batch, dim=0)
    head_positions_flat = torch.cat(head_positions_batch, dim=0)
    predictions_flat = torch.cat(predictions, dim=0)
    filenames_flat = np.concatenate(filename)
    tasks_flat = torch.cat(tasks_batch, dim=0)
    
    if shuffle:
        # Shuffle the flattened data
        indices = torch.randperm(frames_flat.size(0))
        frames_flat = frames_flat[indices]
        head_positions_flat = head_positions_flat[indices]
        predictions_flat = predictions_flat[indices]
        filenames_flat = filenames_flat[indices]
        tasks_flat = tasks_flat[indices]
        
    # Split batch into chunks of batch_size
    batches = [
        [
            frames_flat[i : i + minibatch],
            head_positions_flat[i : i + minibatch],
            predictions_flat[i : i + minibatch],
            filenames_flat[i : i + minibatch],
            tasks_flat[i : i + minibatch],
        ]
        for i in range(0, len(frames_flat), minibatch)
    ]
    # Filter out any smaller batches
    batches = [
        b for b in batches if len(b[0]) == minibatch
    ]
    return batches


# Function to create the dataloader
def get_dataloader(video_paths, batch_size=5, minibatch=32, 
                   past_timepoints=15, future_timepoints=6,
                   frame_sample_rate=5, sliding_window=5,
                   original_fps=30, shuffle=False, num_workers=8):
    ## TODO: frame sample rate implementation
    dataset = GazePredDataset(video_paths,
                                past_timepoints=past_timepoints,
                                future_timepoints=future_timepoints,
                                sliding_window=sliding_window,
                                frame_sample_rate=frame_sample_rate,
                                original_fps=original_fps)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, minibatch, shuffle),
    )
    return dataloader


if __name__ == "__main__":
    # video and metadata folder input
    vd_paths = glob("../processed_dataset/eht_dataset_prossesed_vid/**/**/*.mp4")
    # random.shuffle(vd_paths)
    print("Total videos in path:", len(vd_paths))
    dataloader = get_dataloader(vd_paths[0:5])

    for batch in dataloader:
        for minibatch in batch:
            print(len(minibatch))
            print(minibatch[0].shape)
            print(minibatch[1].shape)
            print(minibatch[2].shape)
            print(len(minibatch[3]))
            # Convert the tensor to decimal format
            decimal_tensor = minibatch[1][0].tolist()
            # Print the decimal tensor
            for row in decimal_tensor:
                print([float(format(x, ".3f")) for x in row])
            print(minibatch[2].shape)
            pred_decimal_tensor = minibatch[2][0].tolist()
            for row in pred_decimal_tensor:
                print([float(format(x, ".4f")) for x in row])
