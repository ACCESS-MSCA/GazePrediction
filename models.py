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
        image_kernel_size=5,  # Default kernel size for convolutions
        image_padding=0,
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

    def forward(self, image, time_series):
        # Image processing for each image (separate branches)
        batch_size = image.size(0)
        num_images = image.size(1)
        num_time_steps = time_series.size(1)

        image_features = []
        for i in range(num_images):  # Process each image in the batch
            img = image[:, i, :, :]  # Select the ith image
            img = img.unsqueeze(1)  # Add channel dimension
            x = F.interpolate(img, size=(25, 25))  # Resize
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(batch_size, -1)  # Flatten
            x = self.Dropout(F.relu(self.fc1(x)))
            # Expand features to match time series length and concatenate with time series data
            image_features.append(x.unsqueeze(1))
        img_features = torch.cat(image_features, dim=1)
        # print(img_features.shape)

        # linspace_imgs_to_time_series = torch.linspace(
        #     0, num_time_steps, num_images + 1, dtype=torch.int
        # )
        # features = torch.zeros(
        #     batch_size, num_time_steps, self.sal_emb_length, device=device
        # )

        # # Copy the slices from A to the appropriate ranges in B
        # for i in range(num_images):
        #     start_idx = linspace_imgs_to_time_series[i]
        #     end_idx = linspace_imgs_to_time_series[i + 1]
        #     features[:, start_idx:end_idx, :] = (
        #         img_features[:, i, :].unsqueeze(1).repeat(1, end_idx - start_idx, 1)
        #     )

        # print(features.shape)  # Should print torch.Size([8, 15, 10])
        # Concatenate time series data with features from all images
        combined_features = torch.cat((time_series, img_features), dim=2)
        # print(combined_features.shape)
        # # Pass the concatenated features through the TSMixer
        output = self.tsmixer(combined_features)
        return output
    
# Define the LSTM module
class Conv_LSTM(nn.Module):
    def __init__(
        self,
        num_time_features,
        conv_output_embeddings,
        sequence_length,
        prediction_length,
        prediction_features,
        image_kernel_size=5,  # Default kernel size for convolutions
        image_padding=0,
        lstm_hidden_size=128,  # Hidden size for LSTM
        lstm_num_layers=1,     # Number of LSTM layers
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

    def forward(self, image, time_series):
        # Image processing for each image (separate branches)
        batch_size = image.size(0)
        num_images = image.size(1)
        num_time_steps = time_series.size(1)

        image_features = []
        for i in range(num_images):  # Process each image in the batch
            img = image[:, i, :, :]  # Select the ith image
            img = img.unsqueeze(1)  # Add channel dimension
            x = F.interpolate(img, size=(25, 25))  # Resize
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(batch_size, -1)  # Flatten
            x = self.Dropout(F.relu(self.fc1(x)))
            # Expand features to match time series length and concatenate with time series data
            image_features.append(x.unsqueeze(1))
        img_features = torch.cat(image_features, dim=1)
        # Concatenate time series data with features from all images
        combined_features = torch.cat((time_series, img_features), dim=2)

        _, (h_n, _) = self.lstm(combined_features)
        last_hidden = h_n[-1]  # Use the hidden state of the last LSTM layer
        output = self.fc_output(last_hidden)  # Predict all future steps at once
        return output.view(-1, self.future_steps, self.prediction_features)

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