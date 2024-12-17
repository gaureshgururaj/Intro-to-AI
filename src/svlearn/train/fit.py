# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------------------------------------
#  Copyright (c) 2024.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: Asif Qamar
# ------------------------------------------------------------------------------------------------------------


import numpy as np
from rich import print as rprint
from sklearn.metrics import confusion_matrix


class ModelFitter:
    """
    A class that provides methods for fitting and evaluating regression and classification models,
    as well as plotting the loss history.

    Methods:
        regression_fit_evaluate_plot: Fits the regression model, evaluates its performance, and plots the loss history.
        classification_fit_eval_plot: Fits the model to the training data, evaluates it on the test data, and plots the loss history.
        fit_predict: Fits the model to the training data and predicts the output for the test data.
        plot_loss: Plots the training and validation loss over epochs.
    """

    # --------------------------------------------------------------------------------------------------------
    def regression_fit_evaluate_plot(self, model, x_train, y_train, x_test, y_test, plot_title: str = "Loss History"):
        """
        Fits the regression model, evaluates its performance, and plots the loss history.

        Args:
            model: The regression model to be trained.
            x_train: The input training data.
            y_train: The target training data.
            x_test: The input test data.
            y_test: The target test data.

        Returns:
            None
        """
        y_pred = self.fit_predict(model, x_train, y_train, x_test)

        # Let us check the correlation between the predicted and the actual values
        rprint(np.corrcoef(y_test.flatten(), y_pred.flatten()))
        self.plot_loss(model.history, plot_title)

    # --------------------------------------------------------------------------------------------------------
    def classification_fit_eval_plot(self, model, x_train, y_train, x_test, y_test):
        """
        Fits the model to the training data, evaluates it on the test data, and plots the loss history.

        Args:
            model: The model to be trained and evaluated.
            x_train: The input features of the training data.
            y_train: The target labels of the training data.
            x_test: The input features of the test data.
            y_test: The target labels of the test data.

        Returns:
            None
        """

        y_pred = self.fit_predict(model, x_train, y_train, x_test)
        rprint("The Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
        self.plot_loss(model.history)

    # --------------------------------------------------------------------------------------------------------
    def fit_predict(self, model, x_train, y_train, x_test):
        """
        Fits the model to the training data and predicts the output for the test data.

        Parameters:
            model (object): The regression model to be fitted.
            x_train (array-like): The input features of the training data.
            y_train (array-like): The target values of the training data.
            x_test (array-like): The input features of the test data.

        Returns:
            array-like: The predicted output values for the test data.
        """
        model.fit(x_train, y_train)

        # Evaluate the regressor with test data
        y_pred = model.predict(x_test)
        return y_pred

    # --------------------------------------------------------------------------------------------------------
    def plot_loss(self, history, plot_title: str = "Loss History"):
        """
        Plots the training and validation loss over epochs.

        Parameters:
        - history: A dictionary containing the training history, typically obtained from model.fit().

        Returns:
        - None
        """
        training_loss = history[:, "train_loss"]
        validation_loss = history[:, "valid_loss"]

        import matplotlib.pyplot as plt

        plt.plot(training_loss, label="Training Loss")
        plt.plot(validation_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(plot_title)
        plt.legend()
        plt.show()


# ------------------------------------------------------------------------------------------------------------
