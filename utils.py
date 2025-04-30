import os, json
import cv2
import PIL
import re
import numpy as np


import torch
import unisal.unisal
from torchvision import transforms
from models import Conv_tsmixer, WrappedUNISAL, Conv_LSTM

# Convert the degrees to the range [0, 360]
def convert_hor_degree(degrees):
    return (degrees + 180 + 360) % 360


def convert_ver_degree(degrees):
    # -1 because the y-axis is inverted for opencv
    return ((-1) * degrees + 90 + 360) % 360


def calculate_angular_velocity(angles, sample_rate=30, vel_range=700):
    # Convert angles from degrees to radians
    angles_rad = np.deg2rad(angles)

    # Calculate the differences between consecutive angles
    angle_differences = np.diff(angles_rad)

    # Compute angular velocity in radians per second
    angular_velocity = angle_differences * sample_rate

    # Convert angular velocity to degrees per second
    angular_velocity = np.rad2deg(angular_velocity)

    # Clamp the angular velocity to the specified range
    angular_velocity = np.clip(angular_velocity, -vel_range, vel_range)

    # Add the first velocity again at the beginning to match the length of input angles
    angular_velocity = np.concatenate(([angular_velocity[0]], angular_velocity))

    return angular_velocity


def calculate_angle_diff(angles):
    angles_diff = np.diff(angles)
    angles_diff = np.concatenate(([0], angles_diff))
    return angles_diff

def preprocess_sal_img(img):
    img = np.ascontiguousarray(img[:, :, ::-1])

    preproc_cfg = {
        "out_size": [288, 384],
        "rgb_mean": (0.485, 0.456, 0.406),
        "rgb_std": (0.229, 0.224, 0.225),
    }
    transformations = [
        transforms.ToPILImage(),
        transforms.Resize(preproc_cfg["out_size"], interpolation=PIL.Image.LANCZOS),
        transforms.ToTensor(),
    ]
    if "rgb_mean" in preproc_cfg:
        transformations.append(
            transforms.Normalize(preproc_cfg["rgb_mean"], preproc_cfg["rgb_std"])
        )
    processing = transforms.Compose(transformations)
    tensor = processing(img)
    return tensor

def sal_process(smap):
    smap = (smap - torch.min(smap)) / (torch.max(smap) - torch.min(smap))
    smap = (smap - 0.5) / 0.5  # Scale to [-1,1]
    return smap

def load_unisal(sal_path, device):
    
    unisal_model = WrappedUNISAL(sal_path)
    unisal_model.to(device)
    unisal_model.eval()
    # Freeze the weights of the UNISAL model
    for param in unisal_model.parameters():
        param.requires_grad = False
    
    return unisal_model

def load_time_model(trained_model_folder:str, model_base_name:str, model_epoch_name_path:str, device:str) -> Conv_tsmixer:
    
    model_path = os.path.join(os.path.dirname(__file__), trained_model_folder, model_base_name, model_epoch_name_path)

    model_json_path = os.path.join(os.path.dirname(__file__), trained_model_folder, model_base_name, model_base_name + '.json') 
    with open(model_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    timepoints = data['past_timepoints']
    future_steps = data['future_timepoints'] 
    num_time_features = data['num_time_features']
    conv_output_embeddings = data['conv_output_embeddings']
    prediction_features = data['prediction_features']
    if 'select_time_model' not in data or data['select_time_model'] == 'tsmixer': 
        time_model = Conv_tsmixer(
            num_time_features,
            conv_output_embeddings,
            timepoints,
            future_steps,
            prediction_features,
        )
    elif data['select_time_model'] == 'lstm':
        time_model = Conv_LSTM(
            num_time_features,
            conv_output_embeddings,
            timepoints,
            future_steps,
            prediction_features,
            lstm_hidden_size=data['lstm_hidden_states'],
            lstm_num_layers=data['lstm_layers']
        )
    time_model.load_state_dict(torch.load(model_path))
    time_model.to(device)
    time_model.eval()
    return time_model, timepoints, future_steps

def draw_fov_frame(frame, fov_top, fov_botom, fov_left, fov_right, width, height):
    visualization_frame = frame.copy()
    
    # Handle horizontal wrapping
    if fov_right > width:
        cv2.rectangle(
            visualization_frame,
            (fov_left, fov_top),
            (width, fov_botom),
            (255, 255, 0),
            4,
        )
        cv2.rectangle(
            visualization_frame,
            (0, fov_top),
            (fov_right - width, fov_botom),
            (255, 255, 0),
            4,
        )
    elif fov_left < 0:
        cv2.rectangle(
            visualization_frame,
            (width + fov_left, fov_top),
            (width, fov_botom),
            (255, 255, 0),
            4,
        )
        cv2.rectangle(
            visualization_frame,
            (0, fov_top),
            (fov_right, fov_botom),
            (255, 255, 0),
            4,
        )
    else:
        try:
            cv2.rectangle(
                visualization_frame,
                (fov_left, fov_top),
                (fov_right, fov_botom),
                (255, 255, 0),
                4,
            )
        except Exception as e:
            print('Exception:', e)
            pass
    return visualization_frame


def draw_gaze(frame, g_long, g_lat, width, height, color=(0, 255, 0)):
    visualization_frame = frame.copy()
    # Handle wrapping for circle (giw_long, giw_lat)
    center = (
        int(convert_hor_degree(g_long) * width / 360),
        int(convert_ver_degree(g_lat) * height / 180),
    )
    if center[0] < 0:
        cv2.circle(
            visualization_frame,
            (width + center[0], center[1]),
            100,
            (0, 255, 0),
            4,
        )
    else:
        cv2.circle(visualization_frame, center, 100, color, 4)
    return visualization_frame

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]

def compute_spherical_motion(angles, time_delta=1/30.0):
    """
    Compute differences and velocities for both horizontal and vertical angles.
    
    Args:
        angles: tensor of shape (T, 2) where:
               [:, 0] is horizontal angle in range [-180, 180]
               [:, 1] is vertical angle in range [-90, 90]
        time_delta: float, time between consecutive samples in seconds
    
    Returns:
        Dictionary containing:
        - differences: angle differences (wrapped appropriately) shape (T, 2)
        - velocities: angular velocities in degrees per second shape (T, 2)
    """
    # Extract horizontal and vertical angles
    horz = angles[:, 0]  # Shape: (T,)
    vert = angles[:, 1]  # Shape: (T,)
    
    # Compute differences
    horz_diff = horz[1:] - horz[:-1]  # Shape: (T-1,)
    vert_diff = vert[1:] - vert[:-1]  # Shape: (T-1,)
    
    # Handle wrapping for horizontal angles
    horz_diff = torch.where(horz_diff > 180, horz_diff - 360, horz_diff)
    horz_diff = torch.where(horz_diff < -180, horz_diff + 360, horz_diff)
    
    # Pad to maintain original length
    horz_diff = torch.cat([horz_diff[:1], horz_diff])  # Shape: (T,)
    vert_diff = torch.cat([vert_diff[:1], vert_diff])  # Shape: (T,)
    
    # Stack differences back to original shape
    differences = torch.stack([horz_diff, vert_diff], dim=1)  # Shape: (T, 2)
    
    # Compute velocities (degrees per second)
    velocities = differences / time_delta
    
    return {
        'differences': differences,  # Shape: (T, 2)
        'velocities': velocities,    # Shape: (T, 2)
        'horz_diff': horz_diff,      # Shape: (T,)
        'vert_diff': vert_diff,      # Shape: (T,)
        'horz_vel': horz_diff / time_delta,  # Shape: (T,)
        'vert_vel': vert_diff / time_delta   # Shape: (T,)
    }

def CalAngularDist(gth, prd):
    # Check input dimensions
    if len(gth) != 2:
        raise ValueError('Ground truth gaze data must contain 2 elements.')
    if len(prd) != 2:
        raise ValueError('Predicted gaze data must contain 2 elements.')

    # Parameters of the HMD (HTC Vive)
    VerticalFov = np.pi * 110 / 180  # Vertical FOV in radians
    ScreenCenter = np.array([0.5, 0.5])  # Normalized screen center
    ScreenDist = 0.5 / np.tan(VerticalFov / 2)  # Distance from eye to screen center

    # Normalize the angular coordinates to screen coordinates [0, 1]
    gth = np.array([(gth[0] + 180) / 360, (gth[1] + 90) / 180])  # Normalize ground truth
    prd = np.array([(prd[0] + 180) / 360, (prd[1] + 90) / 180])  # Normalize predicted

    # Calculate the distance between the eye and ground truth
    eye2gth = np.sqrt(ScreenDist**2 + (gth[0] - ScreenCenter[0])**2 + (gth[1] - ScreenCenter[1])**2)
    
    # Calculate the distance between the eye and predicted gaze
    eye2prd = np.sqrt(ScreenDist**2 + (prd[0] - ScreenCenter[0])**2 + (prd[1] - ScreenCenter[1])**2)
    
    # Calculate the distance between ground truth and predicted gaze
    gth2prd = np.sqrt((prd[0] - gth[0])**2 + (prd[1] - gth[1])**2)

    # Calculate the angular distance between ground truth and predicted gaze
    angular_dist = 180 / np.pi * np.arccos((eye2gth**2 + eye2prd**2 - gth2prd**2) / (2 * eye2gth * eye2prd))

    return angular_dist
