#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

# IMPORTS
#  -------------------------------------------------------------------------------------------------

import torch.nn as nn

#  -------------------------------------------------------------------------------------------------


class StdRegressor(nn.Module):
    """
    A PyTorch model that estimates the standard deviation of the input features.

    Args:
        input_dim (int): The size of the input dimension.

    Attributes:
        linear (nn.Linear): A linear layer that converts the output to a single dimension.
    """

    def __init__(self, input_dim: int):
        """
        Initializes the SumRegressor model.

        Args:
            input_dim (int): The size of the input dimension.
        """
        super(StdRegressor, self).__init__()

        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The model's approximation of the sum of the input features.
        """

        # Convert to single dimension
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)

        return x
