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
from torch.utils.data import Dataset
import numpy as np

#  -------------------------------------------------------------------------------------------------


class SimpleNumpyDataset(Dataset):
    """
    A PyTorch Dataset class for handling numpy array data.

    Args:
        X (np.array): The input features array.
        y (np.array): The target labels array.

    Attributes:
        X (torch.Tensor): The input features as a PyTorch tensor.
        y (torch.Tensor): The target labels as a PyTorch tensor.
        length (int): The number of samples in the dataset.
    """

    def __init__(self, X: np.array, y: np.array):
        super(SimpleNumpyDataset, self).__init__()
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = len(X)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return self.length

    def __getitem__(self, index):
        """
        Retrieve the sample and label at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (input_features, label) corresponding to the sample at the given index.
        """
        return self.X[index], self.y[index]
