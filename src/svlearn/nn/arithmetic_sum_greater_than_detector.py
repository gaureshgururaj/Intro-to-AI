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

import torch
import torch.nn as nn
import torch.nn.init as init

#  -------------------------------------------------------------------------------------------------


class SumGreaterThan100Detector(nn.Module):
    """
    A PyTorch model that detects if the sum of numbers in the input is greater than 100.

    Args:
        input_dim (int): The size of the input dimension.

    Attributes:
        activation (nn.ReLU): A ReLU activation function that passes only positive values.
        linear (nn.Linear): A linear layer that converts the output to a single dimension.
        sigmoid (nn.Sigmoid): A sigmoid activation function to output a probability.
    """

    def __init__(self, input_dim: int):
        """
        Initializes the SumGreaterThan100Detector model.

        Args:
            input_dim (int): The size of the input dimension.
        """
        super(SumGreaterThan100Detector, self).__init__()

        # Define layers
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output probability tensor indicating whether the sum of numbers
                          in the input is greater than 100.
        """
        # Convert to single dimension - essentially we want to sum the numbers (weights = 1) and subtract 100 (bias = -100)
        x = self.linear(x)

        # Apply sigmoid activation to convert to probability
        x = self.sigmoid(x)

        return x


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, -100)


# Example usage:
if __name__ == "__main__":
    # Initialize the model
    input_dim = 10
    model = SumGreaterThan100Detector(input_dim=input_dim)

    # Sample input (batch_size=5, input_dim=10)
    sample_input = torch.tensor(
        [
            [0.5, -0.2, 0.1, -1.0, 0.7, 0.4, -0.3, -0.6, 0.9, -0.1],
            [1.2, 0.3, 0.4, -0.7, -0.8, 1.5, 0.2, -0.5, -1.1, 0.2],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [-1.2, -0.4, -0.6, -0.8, -0.1, -0.3, -0.5, -0.7, -0.9, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    # Forward pass
    output = model(sample_input)
    print(f"Model output:\n{output}")
