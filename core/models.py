import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtsmixer import TSMixer
import unisal.unisal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the TSMixer module
class Conv_tsmixer(nn.Module):
    def __init__(
        self,
        num_time_features,
        conv_output_embeddings,
        sequence_length,
        prediction_length,
        prediction_features,
        num_classes=4,
        image_kernel_size=5,  # Default kernel size for convolutions
        image_padding=0,
        use_task_output=False,
    ):
        super(Conv_tsmixer, self).__init__()

        # Convolutional layers for each image (separate branches)
        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=image_kernel_size, padding=image_padding
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
            16,
            8,
            kernel_size=image_kernel_size,
            padding=image_padding,
        )
        self.bn2 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(8 * 3 * 3, conv_output_embeddings)
        self.Dropout = nn.Dropout(0.5)
        # Input channel size adjusted to consider individual image features
        self.tsmixer = TSMixer(
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            input_channels=conv_output_embeddings
            + num_time_features,  # Adjusted input channels
            output_channels=prediction_features,
        )
        self.sal_emb_length = conv_output_embeddings
        
        self.use_task_output = use_task_output
        if self.use_task_output:
            # ORIGINAL HEAD (Commented out)
            # self.fc_task = nn.Linear(prediction_features*prediction_length, num_classes)
            
            # IMPROVED HEAD: Deep MLP with BatchNorm and Dropout
            self.task_head = nn.Sequential(
                nn.Linear(conv_output_embeddings + num_time_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )

    def forward(self, image, time_series):
        batch_size, num_images, _, _, H, W = image.shape
        
        # Reshape to remove the extra dimensions and process all images at once
        image_reshaped = image.view(batch_size * num_images, 1, H, W)
        
        # Process all images in batch
        x = F.interpolate(image_reshaped, size=(25, 25))
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(batch_size * num_images, -1)  # Flatten
        x = self.Dropout(F.relu(self.fc1(x)))
        
        # Reshape back to separate images per batch
        img_features = x.view(batch_size, num_images, -1)
        # Concatenate time series data with features from all images
        combined_features = torch.cat((time_series, img_features), dim=2)
        # print(combined_features.shape)
        # # Pass the concatenated features through the TSMixer
        gaze_output = self.tsmixer(combined_features)

        if self.use_task_output:
            # ORIGINAL LOGIC (Commented out)
            # task_output = self.fc_task(gaze_output.view(batch_size, -1))
            
            # IMPROVED LOGIC: Temporal average pooling across the window
            task_features = torch.mean(combined_features, dim=1) 
            task_output = self.task_head(task_features)
            return gaze_output, task_output
        
        return gaze_output
    
# Define the LSTM module
class Conv_LSTM(nn.Module):
    def __init__(
        self,
        num_time_features,
        conv_output_embeddings,
        sequence_length,
        prediction_length,
        prediction_features,
        num_classes=4,
        image_kernel_size=5,  # Default kernel size for convolutions
        image_padding=0,
        lstm_hidden_size=128,  # Hidden size for LSTM
        lstm_num_layers=1,     # Number of LSTM layers
        use_task_output=False,
    ):
        super(Conv_LSTM, self).__init__()

        # Convolutional layers for each image (separate branches)
        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=image_kernel_size, padding=image_padding
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
            16,
            8,
            kernel_size=image_kernel_size,
            padding=image_padding,
        )
        self.bn2 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(8 * 3 * 3, conv_output_embeddings)
        self.Dropout = nn.Dropout(0.5)
        # Input channel size adjusted to consider individual image features
        self.lstm = nn.LSTM(
            input_size=conv_output_embeddings + num_time_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Fully connected layer to map LSTM outputs to prediction features
        self.fc_output = nn.Linear(lstm_hidden_size, prediction_features*prediction_length)

        # Sequence length and future steps
        self.future_steps = prediction_length
        self.prediction_features = prediction_features

        self.use_task_output = use_task_output
        if self.use_task_output:
            # ORIGINAL HEAD (Commented out)
            # self.fc_task = nn.Linear(lstm_hidden_size, num_classes)
            
            # IMPROVED HEAD: Deep MLP with BatchNorm and Dropout
            self.task_head = nn.Sequential(
                nn.Linear(lstm_hidden_size, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        
    def forward(self, image, time_series):
        batch_size, num_images, _, _, H, W = image.shape
        
        # Reshape to remove the extra dimensions and process all images at once
        image_reshaped = image.view(batch_size * num_images, 1, H, W)
        
        # Process all images in batch
        x = F.interpolate(image_reshaped, size=(25, 25))
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(batch_size * num_images, -1)  # Flatten
        x = self.Dropout(F.relu(self.fc1(x)))
        
        # Reshape back to separate images per batch
        img_features = x.view(batch_size, num_images, -1)
        
        # Concatenate time series data with features from all images
        combined_features = torch.cat((time_series, img_features), dim=2)
        
        lstm_out, (h_n, _) = self.lstm(combined_features)
        last_hidden = h_n[-1]  # Use the hidden state of the last LSTM layer
        gaze_output = self.fc_output(last_hidden)  # Predict all future steps at once
        
        if self.use_task_output:
            # ORIGINAL LOGIC (Commented out)
            # task_output = self.fc_task(last_hidden)
            
            # IMPROVED LOGIC: Temporal average pooling across all LSTM outputs
            task_features = torch.mean(lstm_out, dim=1)
            task_output = self.task_head(task_features)
            return gaze_output.view(-1, self.future_steps, self.prediction_features), task_output
        
        return gaze_output.view(-1, self.future_steps, self.prediction_features)

class InputWrapper(nn.Module):
    def forward(self, x):
        if x.shape != (1, 3, 288, 384):
            raise ValueError(f"Expected input shape (1, 3, 288, 384), but got {x.shape}")
        return x.unsqueeze(0)

class WrappedUNISAL(nn.Module):
    def __init__(self, sal_path):
        super(WrappedUNISAL, self).__init__()
        self.input_wrapper = InputWrapper()
        self.unisal_model = unisal.unisal.model.UNISAL()
        self.unisal_model.load_state_dict(torch.load(sal_path))
        self.unisal_model.eval()  # Set to evaluation mode
         # Fixed normalization using BatchNorm2d without learnable parameters
        self.norm = nn.BatchNorm2d(1, affine=False, track_running_stats=False)
    
    def sal_process(self, smap):
        smap = (smap - torch.min(smap)) / (torch.max(smap) - torch.min(smap))
        smap = smap*2 - 1  # Scale to [-1,1]
        return smap
    
    def forward(self, x):
        x = self.input_wrapper(x)
        sal_pred = self.unisal_model(x)
        sal_pred = sal_pred.squeeze(dim=1)  # Assuming the saliency model adds a singleton channel dimension
        
        # Ensure the output is 4D: [batch_size, channels, height, width]
        if len(sal_pred.shape) != 4:
            raise ValueError(f"Expected 4D output from saliency model, but got {sal_pred.shape}")

        # Normalize using BatchNorm (fixed mean/std)
        sal_pred = self.norm(sal_pred)
        # Scale from [0,1] to [-1,1] in a CoreML-friendly way
        sal_pred = sal_pred * 2 - 1  
        # sal_pred = self.sal_process(sal_pred).squeeze()
        return sal_pred