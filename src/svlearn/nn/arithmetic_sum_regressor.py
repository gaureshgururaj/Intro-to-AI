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


class SumRegressor(nn.Module):
    """
    A PyTorch model that finds the sum of the input features.

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
        super(SumRegressor, self).__init__()

        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The model's approximation of the sum of the input features.
        """

        # Convert to single dimension
        x = self.linear(x)

        return x
