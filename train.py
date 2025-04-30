import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from natsort import natsorted
from glob import glob
from tqdm import tqdm
import random
random.seed(42)
import os
from datetime import datetime
from utils import load_unisal

from dataloader import get_dataloader
from models import Conv_tsmixer, Conv_LSTM

import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import json
import argparse

import math
torch.manual_seed(42)

class SphericalAngularLoss(nn.Module):
    """
    PyTorch module to compute angular distance between predicted and true angles
    in spherical coordinates (azimuth and elevation).
    - Azimuth: horizontal angle in range [-180, 180]
    - Elevation: vertical angle in range [-90, 90]
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def compute_azimuth_diff(self, pred, true):
        # Convert degrees to radians
        pred_rad = pred * math.pi / 180.0
        true_rad = true * math.pi / 180.0
        
        # Calculate the absolute angular difference
        diff = torch.abs(true_rad - pred_rad)
        
        # Handle wraparound for azimuth: ensure we get the smaller angle
        diff = torch.min(diff, 2 * math.pi - diff)
        
        # Convert back to degrees
        return diff * 180.0 / math.pi
    
    def compute_elevation_diff(self, pred, true):
        # For elevation, we just need the absolute difference
        # No wraparound needed since it's bounded at ±90°
        return torch.abs(pred - true)
    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: tensor of shape [..., 2] where [..., 0] is azimuth and [..., 1] is elevation
            y_true: tensor of shape [..., 2] where [..., 0] is azimuth and [..., 1] is elevation
        """
        # Split predictions and true values into azimuth and elevation
        pred_azimuth, pred_elevation = y_pred[..., 0], y_pred[..., 1]
        true_azimuth, true_elevation = y_true[..., 0], y_true[..., 1]
        
        # Compute differences
        azimuth_diff = self.compute_azimuth_diff(pred_azimuth, true_azimuth)
        elevation_diff = self.compute_elevation_diff(pred_elevation, true_elevation)
        
        # Combine differences
        combined_diff = torch.stack([azimuth_diff, elevation_diff], dim=-1)
        
        if self.reduction == 'mean':
            return torch.mean(combined_diff)
        elif self.reduction == 'sum':
            return torch.sum(combined_diff)
        else:  # 'none'
            return combined_diff

class SphericalMSELoss(nn.Module):
    """
    Mean Squared Error loss that accounts for spherical coordinates.
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.angular_loss = SphericalAngularLoss(reduction='none')
        self.reduction = reduction
    
    def forward(self, y_pred, y_true):
        angular_diff = self.angular_loss(y_pred, y_true)
        squared_diff = torch.square(angular_diff)
        
        if self.reduction == 'mean':
            return torch.mean(squared_diff)
        elif self.reduction == 'mean_axis':
            return squared_diff.mean(dim=0)
        elif self.reduction == 'sum':
            return torch.sum(squared_diff)
        else:  # 'none'
            return squared_diff

def get_args():
    parser = argparse.ArgumentParser(description="Set hyperparameters for model training.")

    # Time and sampling parameters
    parser.add_argument('--past_timepoints', type=int, default=15, help='Number of past timepoints to consider')
    parser.add_argument('--future_timepoints', type=int, default=10, help='Number of future timepoints to predict')
    parser.add_argument('--frame_sample_rate', type=int, default=5, help='Rate at which frames are sampled')
    parser.add_argument('--sliding_window', type=int, default=15, help='Sliding window size for temporal data')
    parser.add_argument('--original_fps', type=int, default=30, help='Original frames per second of the data')

    # Feature and embedding parameters
    parser.add_argument('--num_time_features', type=int, default=4, help='Number of time-related features')
    parser.add_argument('--conv_output_embeddings', type=int, default=10, help='Output embeddings from convolution layer')
    parser.add_argument('--prediction_features', type=int, default=2, help='Number of features to predict')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--minibatch', type=int, default=32, help='Size of each minibatch')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker threads for data loading')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--reduce_lr_patience', type=int, default=5, help='Patience for reducing learning rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=8, help='Patience for early stopping rate')

    # File paths
    parser.add_argument('--sal_path', type=str, default="./models/unisal.pth", help='Path to the saliency model file')
    parser.add_argument('--dataset_path', type=str, default="../processed_dataset/eht_dataset_prossesed_vid", help='Path to the processed dataset')

    parser.add_argument('--select_time_model', type=str, choices=["tsmixer", "lstm"], required=True, help='Select time model. Proper arguments {tsmixer, lstm}')
    parser.add_argument('--lstm_hidden_states', type=int, default=32, help='if use lstm set up the lstm hidden states')
    parser.add_argument('--lstm_layers', type=int, default=1, help='if use lstm set up the lstm layers')
    parser.add_argument('--previous_run_weights', type=str, default=None, help='if you want to resume a previous run, put the weights path.')
    return parser.parse_args()


def load_dataloaders(dataset_path, batch_size, minibatch, num_workers, past_timepoints, 
                    future_timepoints, sliding_window,frame_sample_rate, original_fps):
    vd_paths_srt = natsorted(
        glob(os.path.join(dataset_path,"**","**","*.mp4"))
    )
    test_parent_folder = (
        os.path.join(dataset_path,"Video_8","User_8_Task_1"),
        os.path.join(dataset_path,"Video_1","User_26_Task_1")
    )

    # Split the video paths into training and validation sets
    test_vd_path = [
        path for path in vd_paths_srt if path.startswith(test_parent_folder)
    ]
    vd_paths = [
        path for path in vd_paths_srt if not path.startswith(test_parent_folder)
    ]
    
    train_vd_paths = vd_paths[: int(0.8 * len(vd_paths))]
    random.shuffle(train_vd_paths)
    val_vd_paths = vd_paths[int(0.8 * len(vd_paths)) :]

    train_dataloader = get_dataloader(
        train_vd_paths, 
        batch_size=batch_size, 
        minibatch=minibatch, 
        shuffle=True, 
        num_workers=num_workers, 
        past_timepoints=past_timepoints,
        future_timepoints=future_timepoints,
        sliding_window=sliding_window,
        frame_sample_rate=frame_sample_rate,
        original_fps=original_fps
    )
    val_dataloader = get_dataloader(
        val_vd_paths, 
        batch_size=batch_size, 
        minibatch=minibatch, 
        shuffle=False, 
        num_workers=num_workers, 
        past_timepoints=past_timepoints,
        future_timepoints=future_timepoints,
        sliding_window=sliding_window,
        frame_sample_rate=frame_sample_rate,
        original_fps=original_fps
    )
    
    test_dataloader = get_dataloader(
        test_vd_path, 
        batch_size=batch_size, 
        minibatch=minibatch, 
        shuffle=False, 
        num_workers=num_workers, 
        past_timepoints=past_timepoints,
        future_timepoints=future_timepoints,
        sliding_window=sliding_window,
        frame_sample_rate=frame_sample_rate,
        original_fps=original_fps
    )
    
    return train_dataloader, val_dataloader, test_dataloader

def save_args_to_json(args, filename="args_config.json"):
    # Convert args to dictionary and save as JSON
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4)


def add_training_log(log_data, filename="args_config.json"):
    # Load existing JSON data if file exists, otherwise create a new dictionary
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # Initialize or update training logs
    if "training_logs" not in data:
        data["training_logs"] = []

    data["training_logs"].append(log_data)

    # Save updated data back to JSON
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    
    args = get_args()
    
    # Create directory for saving models
    model_dir = 'trained_models_combined'
    os.makedirs(model_dir, exist_ok=True)
    
    if args.previous_run_weights is None:
        exp_name = f'conv_{args.select_time_model}_past_{args.past_timepoints}_future_{args.future_timepoints}_window_{args.sliding_window}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(os.path.join(model_dir, exp_name), exist_ok=True)
        save_args_to_json(args, os.path.join(model_dir,exp_name,exp_name+'.json'))
    else:
        exp_name = args.previous_run_weights.split(os.sep)[-2]
        os.makedirs(os.path.join(model_dir, exp_name), exist_ok=True)
    
    # Initialize unisal
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    unisal_model = load_unisal(args.sal_path, device)
    
    # Initialize the model
    if args.select_time_model == 'tsmixer':
        time_model = Conv_tsmixer(
            args.num_time_features,
            args.conv_output_embeddings,
            args.past_timepoints,
            args.future_timepoints,
            args.prediction_features,
        ).to(device)
    elif args.select_time_model =='lstm':
        time_model = Conv_LSTM(
            args.num_time_features,
            args.conv_output_embeddings,
            args.past_timepoints,
            args.future_timepoints,
            args.prediction_features,
            lstm_hidden_size=args.lstm_hidden_states,
            lstm_num_layers=args.lstm_layers
        ).to(device)
    
    if args.previous_run_weights is not None:
        time_model.load_state_dict(torch.load(args.previous_run_weights))
    
    # Define your vd_paths

    train_dataloader, val_dataloader, _ = load_dataloaders(args.dataset_path, args.batch_size, args.minibatch,
                                                           args.num_workers, args.past_timepoints, 
                                                           args.future_timepoints, args.sliding_window,
                                                           args.frame_sample_rate, args.original_fps)
    
    # TensorBoard writer
    writer = SummaryWriter()
    # Loss function and optimizer
    criterion = SphericalMSELoss('mean')
    optimizer = optim.Adam(time_model.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=args.reduce_lr_patience, verbose=True
    )
    current_patience = 0
    
    if args.previous_run_weights is not None:
        starting_epoch = int(args.previous_run_weights.split('_')[-4])
        best_val_loss = float(args.previous_run_weights.split('_')[-1].split('.pth')[0])
    else:
        starting_epoch = 0
    for epoch in tqdm(range(starting_epoch, args.num_epochs)):
        epoch_loss = 0.0
        len_train_dataloader = 0
        train_progress_bar =  tqdm(train_dataloader)
        time_model.train()
        for batch in train_progress_bar:
            if batch is None: continue
            for images, timeseries, gt_gaze, _ in batch:
                
                images, timeseries, gt_gaze = (
                    images.to(device),
                    timeseries.to(device),
                    gt_gaze.to(device),
                )
                
                output_list = []
                for batch_image in images:
                    batch_list = []
                    for image in batch_image:
                        with torch.no_grad():
                            output = unisal_model(image.unsqueeze(0))
                        batch_list.append(output)
                    output_list.append(torch.stack(batch_list))

                stacked_sal_preds = torch.stack(output_list)

                # Zero the parameter gradients
                optimizer.zero_grad()
                outputs = time_model(stacked_sal_preds, timeseries)
                
                # Forward pass
                loss = criterion(outputs, gt_gaze)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                len_train_dataloader += 1
                train_progress_bar.set_postfix(loss=epoch_loss / len_train_dataloader)
        avg_epoch_loss = epoch_loss / len_train_dataloader
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Training Loss: {avg_epoch_loss:.4f}")

        # Log the training loss to TensorBoard
        writer.add_scalar("Training Loss", avg_epoch_loss, epoch)

        # Validation step
        time_model.eval()
        val_loss = 0.0
        len_val_dataloader = 0
        val_progress_bar =  tqdm(val_dataloader)
        with torch.no_grad():
            for batch in val_progress_bar:
                if batch is None: continue
                for images, timeseries, gt_gaze, _ in batch:
                    
                    images, timeseries, gt_gaze = (
                        images.to(device),
                        timeseries.to(device),
                        gt_gaze.to(device),
                    )
                    
                    output_list = []
                    for batch_image in images:
                        batch_list = []
                        for image in batch_image:
                            output = unisal_model(image.unsqueeze(0))
                            batch_list.append(output)
                        output_list.append(torch.stack(batch_list))

                    # Stack along batch dimension (32)
                    stacked_sal_preds = torch.stack(output_list)
                    outputs = time_model(stacked_sal_preds, timeseries)
                
                    # Forward pass
                    loss = criterion(outputs, gt_gaze)
                    val_loss += loss.item()
                    len_val_dataloader += 1
                    val_progress_bar.set_postfix(loss=val_loss / len_val_dataloader)
        avg_val_loss = val_loss / len_val_dataloader
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Validation Loss: {avg_val_loss:.4f}")
        
        # Log the validation loss to TensorBoard
        writer.add_scalar("Validation Loss", avg_val_loss, epoch)

        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        training_log = {
            "epoch": epoch+1,
            "train_loss": avg_epoch_loss,
            "val_loss": avg_val_loss,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        add_training_log(training_log, os.path.join(model_dir,exp_name,exp_name+'.json'))
        
        # Save the model if the validation loss is the best we've seen so far.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(
                model_dir,
                exp_name,
                f"model_epoch_{epoch+1}_val_loss_{avg_val_loss:.4f}.pth",
            )
            torch.save(time_model.state_dict(), model_path)
            print(f"Model saved with validation loss: {avg_val_loss:.4f}")
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= args.early_stopping_patience:
                break
        
    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
