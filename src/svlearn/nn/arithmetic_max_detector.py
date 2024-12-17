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


class MaxDetector(nn.Module):
    """
    A PyTorch model that detects the the maximum among the list of numbers.

    Args:
        input_dim (int): The size of the input dimension.

    Attributes:
        linear (nn.Linear): A linear layer that returns a tensor of the same dimension as the input.
    """

    def __init__(self, input_dim: int):
        """
        Initializes the MaxDetector model.

        Args:
            input_dim (int): The size of the input dimension.
        """
        super(MaxDetector, self).__init__()

        self.linear = nn.Linear(input_dim, input_dim)
        self.activation = nn.Softmax()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output probability tensor indicating the highest number among the rest.
        """
        x = self.linear(x)

        # apply softmax to convert numbers to probabilities
        x = self.activation(x)
        return x


#  -------------------------------------------------------------------------------------------------
