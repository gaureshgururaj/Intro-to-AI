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


class NegativeNumberDetector(nn.Module):
    """
    A PyTorch model that detects the presence of negative numbers in the input.

    Args:
        input_dim (int): The size of the input dimension.

    Attributes:
        activation (nn.ReLU): A ReLU activation function that passes only positive values.
        linear (nn.Linear): A linear layer that converts the output to a single dimension.
        sigmoid (nn.Sigmoid): A sigmoid activation function to output a probability.
    """

    def __init__(self, input_dim: int):
        """
        Initializes the NegativeNumberDetector model.

        Args:
            input_dim (int): The size of the input dimension.
        """
        super(NegativeNumberDetector, self).__init__()

        # Define layers
        self.activation = nn.ReLU()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output probability tensor indicating the presence of negative numbers.
        """
        # Apply negative ReLU: pass only negative values
        x = self.activation(-x)

        # Convert to single dimension
        x = self.linear(x)

        # Apply sigmoid activation
        x = self.sigmoid(x)

        return x
