#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

from typing import Tuple
from torch.utils.data import random_split , Dataset

#  -------------------------------------------------------------------------------------------------

def split_dataset(dataset: Dataset, eval_split: float = 0.2) -> Tuple[Dataset, Dataset]:
    """Shuffles and splits the dataset into training and evaluation datasets.

    Args:
        dataset (Dataset): The full dataset to split.
        eval_split (float): The proportion of the dataset to use for evaluation. Default is 0.2.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: The training and evaluation datasets.
    """
    # Calculate the number of samples for training and evaluation
    dataset_size = len(dataset)
    eval_size = int(eval_split * dataset_size)
    train_size = dataset_size - eval_size

    # Use random_split to create training and evaluation datasets
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    return train_dataset, eval_dataset