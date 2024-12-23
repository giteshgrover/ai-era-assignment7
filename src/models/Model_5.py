import torch
import torch.nn as nn
import torch.nn.functional as F

# Target: Increase the accuracy while still keeping the parameters under 8k. Keeping the gap between training and test accuracy low
# Results:
## Parameters: 7,778
## Best Training Accuracy: 98.75%
## Best Test Accuracy: 99.44% (validation accuracy: 99.50%)
# Analysys:
## Parameters are less than 8k
## Good model with high accuracy and low gap between training and test accuracy
class Model_5(nn.Module):
    def __init__(self):
        super(Model_5, self).__init__()
        
        dropout_rate = 0.1 # Dropout

        # Convolution Layers ()
        self.features = nn.Sequential(
            # Convolution Layer 1
            # (ImageSize 28x28)
            nn.Conv2d(1, 8, kernel_size=3, padding=1), # Input: 1, Output: 8, RF: 3 
            nn.BatchNorm2d(8),  # Batch normalization
            nn.Dropout(dropout_rate),  # Dropout
            nn.ReLU(), # Activation function

            # Convolution Layer 2
            nn.Conv2d(8, 12, kernel_size=3, padding=1), # Input: 8, Output: 12, RF: 5
            nn.BatchNorm2d(12),  # Batch normalization
            nn.Dropout(dropout_rate),  # Dropout
            nn.ReLU(),

            # Transition Layer
            nn.MaxPool2d(2), # Input: 12, Output: 12, RF: 6 (Max pool reduces the image size by half, not the output channels)
            # ImageSize 14 x 14
            nn.Conv2d(12, 8, kernel_size=1, padding=1), # 1x1 MixerInput: 12, Output: 8, RF: 6

            # Convolution Layer 3
            nn.Conv2d(8, 8, kernel_size=3, padding=0), # Input: 8, Output: 8, RF: 10
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_rate),  # Dropout
            nn.ReLU(),

            # Convolution Layer 4
            nn.Conv2d(8, 8, kernel_size=3, padding=0), # Input: 8, Output: 8, RF: 14
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_rate),  # Dropout
            nn.ReLU(),
            
            # Convolution Layer 5
            nn.Conv2d(8, 12, kernel_size=3, padding=0), # Input: 8, Output: 12, RF: 18
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate),  # Dropout
            nn.ReLU(),

            # Convolution Layer 6
            nn.Conv2d(12, 12, kernel_size=3, padding=0), # Input: 12, Output: 12, RF: 22
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate),  # Dropout
            nn.ReLU(),

            # Convolution Layer 7
            nn.Conv2d(12, 12, kernel_size=3, padding=0), # Input: 12, Output: 12, RF: 26
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_rate),  # Dropout
            nn.ReLU(),

            # Convolution Layer 8
            nn.Conv2d(12, 16, kernel_size=3, padding=0), # Input: 12, Output: 16, RF: 30
            # nn.BatchNorm2d(16),
            # nn.Dropout(dropout_rate),  # Dropout
            # nn.ReLU(),


            # Output Layer
            # ImageSize 4 x 4
            nn.AdaptiveAvgPool2d(1),  #  Output: 1 x 1 x 10   # nn.AvgPool2d(kernel_size=7)
            # Image Size 1 x 1 (x 10)
            nn.Conv2d(16, 10, kernel_size=1, padding=0),

        )

        
    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x 