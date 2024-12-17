#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F

#  -------------------------------------------------------------------------------------------------


class BirdCallCNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) for classifying bird calls based on their audio spectrograms.
    The model is designed for two classes (binary classification).
    """

    def __init__(self, num_classes=2):
        super(BirdCallCNN, self).__init__()

        # Define the CNN architecture
        # First convolutional layer: Input (1, 128, time_steps) -> Output (16, 128, time_steps)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization for regularization

        # Second convolutional layer: Output (32, 64, time_steps / 2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Max pooling: Reduces the size of the feature maps by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional layer: Output (64, 32, time_steps / 4)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layer: Output (128)
        self.fc1 = nn.Linear(64 * 32 * 107, 64 * 32)  # Adjust based on input size after convolution/pooling
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

        # Final output layer for 2 classes (binary classification)
        self.fc2 = nn.Linear(64 * 32, 128)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass of the CNN model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 128, time_steps).

        Returns:
            torch.Tensor: Output tensor with class probabilities.
        """
        # Apply first convolutional layer, followed by batch normalization, ReLU, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (16 , 1, 128 , 431)

        # Apply second convolutional layer, batch normalization, ReLU, and pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (16 , 16 , 64 , 215)

        # Apply third convolutional layer, batch normalization, and ReLU
        x = F.relu(self.bn3(self.conv3(x)))  # (16, 64, 32, 107)

        # Flatten the tensor for the fully connected layers
        x = nn.Flatten()(x)

        # Apply the first fully connected layer with dropout
        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        # Final output layer (logits for binary classification)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    # Model initialization
    model = BirdCallCNN(num_classes=2)
    print(model)
