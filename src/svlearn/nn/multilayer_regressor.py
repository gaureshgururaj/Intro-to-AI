# ------------------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
# ------------------------------------------------------------------------------------------------------------

from torch import nn
from torch.nn import Linear, ReLU

# ------------------------------------------------------------------------------------------------------------

class MultilayerRegressor(nn.Module):
    """
    A simple regression model to predict the standard deviation of the input data.
    """

    def __init__(self, input_dimension: int = 10, output_dimension: int = 1):
        super(MultilayerRegressor, self).__init__()
        # Define a single layer with 1 neuron, taking a 10-dimensional input
        self.layers = nn.Sequential(
            Linear(in_features=input_dimension, out_features=64),
            ReLU(),
            Linear(64, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 1),
        )

    # --------------------------------------------------------------------------------------------------------
    def forward(self, x):
        """
        Forward pass that computes the standard deviation of the input data.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing the standard deviation of the input data.
        """
        return self.layers(x)


# ------------------------------------------------------------------------------------------------------------
