#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

#  -------------------------------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """A simple Convolutional Neural Network for image classification. 
    Assumes the input image size is (224 , 224)

    This network consists of several convolutional layers followed by
    fully connected layers. It uses ReLU activations and max-pooling.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1  = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2  = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 4)
        self.bn3  = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 25 * 25, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#  -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    model = SimpleCNN()
    print(model)
