#  Copyright (c) 2020.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: Asif Qamar
#
from collections import OrderedDict
from typing import Callable, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import ReLU, Linear, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans' 

from svlearn.common.svexception import SVError


class UnivariatePrediction:
    r"""
    A simple datum to hold a prediction for a given predictor (x),
    a ground-truth value (y), and the associated prediction (y_hat).
    """
    __slots__ = ['x', 'y', 'y_hat']

    def __init__(self, x: float, y: float, y_hat: float) -> None:
        """
        Constructor of the prediction datum.
        :param x: the predictor value
        :type x: float
        :param y: the ground-truth response
        :type y: float
        :param y_hat: the predicted value
        :type y_hat: float
        """
        super().__init__()
        self.x = x
        self.y = y
        self.y_hat = y_hat

    def __repr__(self) -> str:
        return f'UnivariatePrediction [x:{self.x:<.4f}, y:{self.y:<.4f}, y_hat:{self.y_hat:<.4f}]'


# -----------------------------------------------------------------------------

class UnivariateApproximator:
    def __init__(self, func: Callable[[float], float], start: float = 0, end: float = 1, scale: bool = True) -> None:
        """Creates an instance of the approximator of the given univariate function.
        :param func: the univariate function
        :param start: the start of the domain of input, as float
        :param end: the end of the domain of input, as float
        :param scale: whether to scale the x values to range of 0 to 1 or not.  Default is True
        """
        super().__init__()
        self.func = func
        self.activation =ReLU()
        if start >= end:
            raise SVError(f' Argument min must be less than the argument max! Values supplied min: {min}, max:{max}')
        # Create the data and the data loaders
        xx_raw = np.linspace(start=start, stop=end, num=100_000)
        # Scale the input, since neural-nets work best with [0,1] interval
        # So we min-max scale it.
        if scale:
            xmax = np.max(xx_raw)
            xmin = np.min(xx_raw)
            xx = (xx_raw - xmin) / (xmax - xmin)
        else:
            xx = xx_raw
        # compute the ground-truth response from the function
        yy = self.func(xx)
        data = np.column_stack((xx, yy))
        np.random.shuffle(data)
        train, test = data[:20_000, ], data[20_000:, ]
        x_train, y_train = Tensor(train[:, 0]), Tensor(train[:, 1])
        x_test, y_test = Tensor(test[:, 0]), Tensor(test[:, 1])
        self.train_dataloader = DataLoader(TensorDataset(x_train, y_train))
        self.test_dataloader = DataLoader(TensorDataset(x_test, y_test))

        # Now, create the network
        self.network = self.create_network()

        # define the loss-function, and select and optimizer
        self.loss_function = MSELoss()
        self.optimizer = Adam(self.network.parameters(), lr=0.001)

    def create_network(self):
        return torch.nn.Sequential(OrderedDict([
            ('input-layer', Linear(1, 128)),  # first layer
            ('first-activation', self.activation),
            ('first-hidden-layer', Linear(128, 64)),  # second layer
            ('second-activation', self.activation),
            ('third-hidden-layer', Linear(64, 64)),  # third layer
            ('third-activation', self.activation),
            ('fourth-hidden-layer', Linear(64, 16)),  # third layer
            ('fourth-activation', self.activation),
            ('final-layer', Linear(16, 1))  # output layer
        ]))

    def train(self, epochs: int = 10) -> None:
        """
        Train the neural network for the given number of epochs.
        :param epochs: int, the number of epochs to run over the training data
        :return: None
        """
        if epochs < 1:
            raise ValueError("The argument 'epoch' must be given a positive value")

        for_depiction = OrderedDict()

        for epoch in range(epochs):
            running_loss = 0.0

            for i, (inputs, responses) in enumerate(self.train_dataloader, 0):

                # recall that we will receive a mini-batch of the training-data

                # reset the parameter gradients
                self.optimizer.zero_grad()
                # Now, for each step:
                #   perform the forward pass to compute the prediction
                #   compute the loss comparing predictions_list to actual responses
                #   perform the backward propagation of loss gradients to update params
                outputs = self.network(inputs)

                loss = self.loss_function(outputs, responses)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                # Print the progress statistics to the console
                if i % 1000 == 999:  # print every 1000 mini-batches
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, i + 1, running_loss / 1000))
                    running_loss = 0.0

        print('Finished training')


    def evaluate_model(self) -> List[UnivariatePrediction]:
        """
        Run the model on the test data, and return tuples of (y, y_hat), i.e. actual label
        for an input, and the model-prediction.
        :return: a list of all predictions_list for the test data-set.
        """
        all_predictions = []

        # Now run inference on all the test data, and accumulate
        # the results in the two lists above.
        with torch.no_grad():  # since we are in inference-model...
            for inputs, responses in self.test_dataloader:
                batch_yhats = self.network(inputs)
                batch_predictions = [UnivariatePrediction(xx.item(), yy.item(), y_hat.item())
                                     for xx, yy, y_hat in zip(inputs, responses, batch_yhats)]
                all_predictions.extend(batch_predictions)
        self.predictions_ = all_predictions
        return all_predictions


    def to_pandas(self) -> pd.DataFrame:
        x = [datum.x for datum in self.predictions_]
        y = [datum.y for datum in self.predictions_]
        y_hat = [datum.y_hat for datum in self.predictions_]
        df = pd.DataFrame(data={'x': x, 'y': y, 'y_hat': y_hat})
        df.sort_values(by=['x'], inplace=True)
        self.df_ = df
        return df

    def correlation(self) -> float:
        y = [datum.y for datum in self.predictions_]
        y_hat = [datum.y_hat for datum in self.predictions_]
        corr_matrix = np.corrcoef(y, y_hat)
        return corr_matrix[0, 1]  # return the off-diagonal element!

    def create_plots(self):
        data = self.to_pandas()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        fig.suptitle('Actual Function vs Neural Approximation')
        ax1.plot(data.x, data.y)
        ax1.set_title('Actual function')
        ax2.plot(data.x, data.y_hat)
        ax2.set_title('Neural Approximation')
        return fig
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # noinspection SpellCheckingInspection

    x_sinx = lambda val: val * np.sin(val)  # an example function
    wierd_x = lambda x: np.sin(x * x) * (7 - 5 * x + x * x - 1.5 * x ** 3)
    poly_x = lambda x: 7 - 5 * x + x * x - 1.5 * x ** 3
    approximator = UnivariateApproximator(wierd_x, 0, 5)
    approximator.train(1)
    predictions: List[UnivariatePrediction] = approximator.evaluate_model()
    labels = [datum.y for datum in predictions]
    y_hats = [datum.y_hat for datum in predictions]
    correlation = np.corrcoef(labels, y_hats)
    print(f'The Pearson correlation between ground truth and prediction is {correlation}')

    # xplot = np.linspace(0, 5, 1000)
    # pred = approximator.predict(xplot)
    # print (pred)
