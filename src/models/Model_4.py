import torch
import torch.nn as nn
import torch.nn.functional as F

# Target: Reduce the number of parameters to less than 8k still maintaining the gap between training and test accuracy
# Results:
## Parameters: 7k
## Best Training Accuracy: 97.73%
## Best Test Accuracy: 97.39%
# Analysys:
## Parameters are less than 8k
## Good model with slowly increasing accuracy each epoch with a decent gap between training and test accuracy
## Can be trained further to get better results
class Model_4(nn.Module):
    def __init__(self):
        super(Model_4, self).__init__()
        
        dropout_rate = 0.1 # Dropout


        # Convolution Layers ()
        self.features = nn.Sequential(
            # Convolution Layer 1
            # (ImageSize 28x28)
            nn.Conv2d(1, 6, kernel_size=3, padding=1), # Input: 1, Output: 6, RF: 3 
            nn.BatchNorm2d(6),  # Batch normalization
            nn.Dropout(dropout_rate),  # Dropout
            nn.ReLU(), # Activation function

            # Convolution Layer 2
            nn.Conv2d(6, 12, kernel_size=3, padding=1), # Input: 6, Output: 12, RF: 5
            nn.BatchNorm2d(12),  # Batch normalization
            nn.Dropout(dropout_rate),  # Dropout
            nn.ReLU(),

            # Transition Layer
            nn.MaxPool2d(2), # Input: 24, Output: 24, RF: 6 (Max pool reduces the image size by half, not the output channels)
            # ImageSize 14 x 14
            nn.Conv2d(12, 6, kernel_size=1, padding=0), # 1x1 MixerInput: 12, Output: 6

            # ImageSize 14 x 14
            # Convolution Layer 3
            nn.Conv2d(6, 12, kernel_size=3, padding=1), # Input: 6, Output: 12, RF: 10
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate),  # Dropout
            nn.ReLU(),

            # Convolution Layer 4
            nn.Conv2d(12, 24, kernel_size=3, padding=1), # Input: 12, Output: 24, RF: 14
            nn.BatchNorm2d(24),
            nn.Dropout(dropout_rate),  # Dropout
            nn.ReLU(),

            # Transit Layer
            nn.MaxPool2d(2), # Input: 24, Output: 24, RF: 20 
            # (ImageSize 7 x 7)
            nn.Conv2d(24, 6, kernel_size=1, padding=0), # 1x1 Mixer Input: 24, Output: 6

            # ImageSize 7 x 7
            # Convolution Layer 6
            nn.Conv2d(6, 12, kernel_size=3, padding=1), # Input: 6, Output: 12, RF: 28
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate),  # Dropout
            nn.ReLU(),

            # Convolution Layer 7
            nn.Conv2d(12, 16, kernel_size=3, padding=1), # Input: 12, Output: 16, RF: 36
            # nn.BatchNorm2d(16), No Batch norm in the last layer
            # nn.Dropout(dropout_rate), No Dropout in the last layer
            # nn.ReLU(), No Activation func in the last layer

            # Output Layer
            # ImageSize 7 x 7
            nn.Conv2d(16, 10, kernel_size=1, padding=0), # 1x1 Mixer Input: 16, Output: 10 (7 x 7 x 10)
            nn.AvgPool2d(kernel_size=7),  #  Output: 1 x 1 x 10   # nn.AdaptiveAvgPool2d(1)
            # Image Size 1 x 1 (x 10)
            nn.Conv2d(10, 10, kernel_size=1, padding=0),

        )

        
    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x 