import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolution function: nn.Conv2d(inputChannel, NumOfKernels or Output Channels, kernel_size=3, padding=1)
# Batch Normalization: nn.BatchNorm2d(inputChannel) - not to be done at the last layer
# DropOut: nn.Dropout(dropout_rate) - not to be done at the last layer
# Activation Function: Relu - nn.ReLU() - not to be done at the last layer
# MaxPool: nn.MaxPool2d(n) (where, n will be the nxn kernel size and stride of n) - MaxPool reduces the imageSize by n and add a stride of n. It doesn't change the number of Output channels
# 1x1 Mixer: nn.Conv2d(inputChannel, outputChannel, 1, 0) - It is a convolution of kernel size '1' with 0 padding
# AveragePooling:  nn.AvgPool2d(kernel_size=n) Similar to maxpool, a nxn kernel calculates the average pool. If operated on nxn image siz, produces 1 x 1 ouput image. Number of channels remain same
# AdaptivePool: nn.AdaptiveAvgPool2d(m) Same as average pool but don't need to supply the kernel size. It auto calculates based on the input image size to produce output of size mxm. Number of channels remain same 


# Target: Base model with base skelton using correct transforms, data sets and correct architecture
# Results:
## Parameters: 29k
## Best Training Accuracy: 99.69%
## Best Test Accuracy: 99.05%
# Analysys:
## Large number of parameters
## Gap between training and test accuracy (Over Fitting)
class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        
        dropout_rate = 0.1 # Dropout


        # Convolution Layers ()
        self.features = nn.Sequential(
            # Convolution Layer 1
            # (ImageSize 28x28)
            nn.Conv2d(1, 12, kernel_size=3, padding=1), # Input: 1, Output: 12, RF: 3 
            nn.ReLU(), # Activation function

            # Convolution Layer 2
            nn.Conv2d(12, 24, kernel_size=3, padding=1), # Input: 12, Output: 24, RF: 5
            nn.ReLU(),

            # Transition Layer
            nn.MaxPool2d(2), # Input: 24, Output: 24, RF: 6 (Max pool reduces the image size by half, not the output channels)
            # ImageSize 14 x 14
            nn.Conv2d(24, 12, kernel_size=1, padding=0), # 1x1 MixerInput: 24, Output: 12

            # ImageSize 14 x 14
            # Convolution Layer 3
            nn.Conv2d(12, 16, kernel_size=3, padding=1), # Input: 12, Output: 16, RF: 10
            nn.ReLU(),

            # Convolution Layer 4
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # Input: 16, Output: 32, RF: 14
            nn.ReLU(),

            # Convolution Layer 5
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # Input: 32, Output: 32, RF: 18
            nn.ReLU(),

            # Transit Layer
            nn.MaxPool2d(2), # Input: 24, Output: 24, RF: 20 
            # (ImageSize 7 x 7)
            nn.Conv2d(32, 12, kernel_size=1, padding=0), # 1x1 Mixer Input: 32, Output: 12

            # ImageSize 7 x 7
            # Convolution Layer 6
            nn.Conv2d(12, 24, kernel_size=3, padding=1), # Input: 12, Output: 24, RF: 28
            nn.ReLU(),

            # Convolution Layer 7
            nn.Conv2d(24, 32, kernel_size=3, padding=1), # Input: 24, Output: 32, RF: 36
            nn.ReLU(),

            nn.Conv2d(32, 10, kernel_size=7, padding=0)

        )

        
        # self.classifier = nn.Sequential(
        #     # nn.Dropout(0.5),
        #     nn.Linear(24 * 5 * 5, 12),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     nn.Linear(12, 10)
        # )
        
    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x 