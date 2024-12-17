#  Copyright (c) 2020.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: Asif Qamar
#

from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans' 

# There are three hidden layers, with the below number of nodes.
HIDDEN_LAYERS = [128, 64, 16]

# noinspection PyAbstractClass
class SimpleFeedForwardNet(nn.Module):
    """
    A simple feed forward network, which we will use for various tasks.
    """
    def __init__(self, input_dimension: int = 1,
                 output_dimension: int = 1):
        """
        Let us build the network, and decide on an activation function.
        :param input_dimension: the number of input features
        :param output_dimension: the response dimension (usually 1)
        """
        super(SimpleFeedForwardNet, self).__init__()
        if input_dimension < 1 or output_dimension < 1:
            raise ValueError(f'Invalid inputs: '
                             f'[input_dimension, layers, output_dimension]'
                             f' = [{input_dimension}, {output_dimension}]'
                             f' all scalar values must be greater than zero')

        self.fc0 = nn.Linear(input_dimension, HIDDEN_LAYERS[0])
        self.fc1 = nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1])
        self.fc2 = nn.Linear(HIDDEN_LAYERS[1], HIDDEN_LAYERS[2])
        self.fc3 = nn.Linear(HIDDEN_LAYERS[2], output_dimension)
        # By default, assume ReLU.
        self.activation = torch.tanh

    def forward(self, inputs, p: float = 0.0):
        """
        The forward prediction path as a feed-forward
        :return:
        :param p: dropout rate
        :param inputs: the regression inputs
        :return: the output prediction
        """
        x = self.activation(self.fc0(inputs))
        x = F.dropout(x, p=p, training=True)
        x = self.activation(self.fc1(x))
        x = F.dropout(x, p=p, training=True)
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def pretty_table(self) -> str:
        """
        A helper function to print a more readable and detailed description
        of the network.
        :return: a string that provides a clean table describing the network.
        """
        line = '--------------------------------------------------------------------------------------'
        rows = [line]
        template = '| {:^6} | {:^14} | {:^14} | {:^10} | {:^10} | {:^12} |'
        header = template.format('LAYER', 'IN FEATURES', 'OUT FEATURES', 'WEIGHTS', 'BIASES', 'TOTAL PARAMS')
        rows.append(header)
        rows.append(line)

        i = 0
        total = 0
        for fc in [self.fc0, self.fc1, self.fc2, self.fc3]:
            layer_param_count = fc.in_features * fc.out_features + fc.out_features
            total += layer_param_count
            row = f'fc{i}', \
                  fc.in_features, \
                  fc.out_features, \
                  fc.in_features * fc.out_features, \
                  fc.out_features, \
                  layer_param_count
            i += 1
            rows.append(template.format(*row))
        rows.append(line)
        rows.append(f'                          TOTAL MODEL PARAMETERS: {total}')
        rows.append(line)
        return '\n'.join([row for row in rows])

    def __repr__(self):
        return super().__repr__() + '\n' + self.pretty_table()


class SimpleNumpyDataset(Dataset):
    """
    A simple dataset class to hold the data in memory as numpy
    """

    def __init__(self, x, y) -> None:
        super().__init__()
        if len(x) != len(y):
            raise ValueError('Length of X and y have to agree.')

        dim_x = x.shape[1]
        x = np.array(x, dtype=np.float32).reshape(-1, dim_x)        
        y = np.array(y, dtype=np.float32)
        self.data = torch.from_numpy(x)
        self.labels = torch.from_numpy(y)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)

def create_plots(epochs: int,  # the number of epochs for the training
                 losses: List[float]  # the list of losses
                 ):
    _, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
    ax1.plot(range(epochs), losses, c='blue', linewidth=2)
    ax1.set_xlabel("Number of Iterations")
    ax1.set_ylabel("Loss")
    plt.show()
