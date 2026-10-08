import os
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import coremltools as ct
import core.utils as utils
import torchinfo

import warnings
warnings.filterwarnings("ignore")
import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="Script that transform gaze model to coreml")
    
    parser.add_argument(
        '--trained_model_folder', 
        type=str, 
        default='/mnt/DataLake/AlienBackup/access/eht_gaze_prediction/trained_models_combined',
        help="Path to the folder containing the trained models."
    )
    parser.add_argument(
        '--model_base_name', 
        type=str, 
        default='conv_lstm_past_15_future_10_window_15_20250113_182736',
        help="Base name for the model being used."
    )
    parser.add_argument(
        '--model_epoch_name_path', 
        type=str, 
        default='model_epoch_5_val_loss_49.1556.pth',
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
        default='/mnt/DataLake/AlienBackup/access/eht_gaze_prediction/models',
        help="Path to the folder where outputs will be saved."
    )

    return parser.parse_args()

if __name__=='__main__':
    args = parse_arguments()
    ## Check save path and create sace folder
    if not os.path.exists(args.folder_save_path):
        os.makedirs(args.folder_save_path)
    if not os.path.exists(f"{args.folder_save_path}/coreml_{args.model_base_name}"):
        os.makedirs(f"{args.folder_save_path}/coreml_{args.model_base_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load saliency model
    unisal_model = utils.load_unisal(args.sal_path, device)
    # Load time series model
    time_model, timepoints, future_steps = utils.load_time_model(args.trained_model_folder, args.model_base_name, args.model_epoch_name_path, device)
    
    # Create dummy input for sal model
    image_sequence = torch.randn(timepoints, 288, 384, 3).cpu().numpy()
    print(image_sequence.shape)
    print(image_sequence[0].shape)
    processed_imgs = []
    for im_idx in range(timepoints):
        processed_imgs.append(utils.preprocess_sal_img(image_sequence[im_idx]))
    processed_imgs = torch.stack(processed_imgs).to(device)
    # print(processed_imgs[0].shape)
    torchinfo.summary(unisal_model, processed_imgs[0].unsqueeze(0).shape)
    # Convert Salincy model to CoreMl and save it to save folder
    unisal_model_traced = torch.jit.trace(unisal_model,[processed_imgs[0].unsqueeze(0)]).to(device)
    sal_mlcore = ct.convert(unisal_model_traced,
                            convert_to="mlprogram",
                            inputs=[ct.TensorType(name="fov_rgb", shape=processed_imgs[0].unsqueeze(0).shape,)],
                            outputs=[ct.TensorType(name="saliecy_maps")])
    try:
        sal_mlcore.save(f"{args.folder_save_path}/coreml_{args.model_base_name}/saliency.mlpackage")
    except:
        raise ValueError('Could not save converted saliency model to path')
    
    # Create dummy input for time series model
    time_series_data = torch.randn(1, timepoints, 4).to(device)  # 4 is the number of original time features

    sal_maps = []
    with torch.no_grad():
        for img in processed_imgs:
            sal_maps.append(unisal_model(img.unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0))
    print(unisal_model(img.unsqueeze(0)).shape)
    
    processed_maps = torch.stack(sal_maps).unsqueeze(0).to(device)
    print(processed_maps.shape)
    print(time_series_data.shape)
    # Convert time series model to CoreML and save it to folder
    time_model_traced = torch.jit.trace(time_model,[processed_maps,time_series_data]).to(device)

    time_mlcore = ct.convert(time_model_traced,
                             convert_to="mlprogram",
                             inputs=[ct.TensorType(shape=processed_maps.shape, name="processed_saliency_maps"),ct.TensorType(shape=time_series_data.shape, name="hmd_time_series")],
                             outputs=[ct.TensorType(name="gaze_predictions")])
    try:
        time_mlcore.save(f"{args.folder_save_path}/coreml_{args.model_base_name}/time_series.mlpackage")
    except:
        raise ValueError('Could not save converted time series model to path')