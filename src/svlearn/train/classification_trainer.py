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
from typing import List

import torch
from torch import Tensor
from torch.utils.data import DataLoader

#  -------------------------------------------------------------------------------------------------


class ClassificationTrainer:
    """
    A class to train and evaluate a PyTorch classification model.

    Args:
        train_dataset (torch.utils.data.Dataset): The dataset used for training.
        eval_dataset (torch.utils.data.Dataset): The dataset used for evaluation.
        model (torch.nn.Module): The PyTorch model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler. Defaults to None.
        batch_size (int): The batch size used for training and evaluation. Defaults to 32.
        num_epochs (int): The number of epochs to train the model. Defaults to 10.
        device (str): The device on which to perform training (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.
    """

    def __init__(
        self,
        train_dataset,
        eval_dataset,
        model,
        loss_func,
        optimizer,
        scheduler=None,
        batch_size=32,
        num_epochs=10,
        device="cpu",
        show_every=10,
        multi_class: bool = False,
    ):
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        self.criterion = loss_func
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.metrics = {"epoch": [], "train_loss": [], "eval_loss": [], "train_acc": [], "eval_acc": []}
        self.show_every = show_every
        self.multi_class = multi_class

        # Move model to the specified device
        self.model.to(self.device)

    def train(self):
        """
        Trains the model and evaluates it after each epoch.
        Logs training and evaluation loss and accuracy after each epoch.
        """
        for epoch in range(self.num_epochs):
            # Set model to training mode
            self.model.train()

            # Initialize variables to track training loss and correct predictions
            train_loss = 0.0
            train_correct = 0
            total_train_samples = 0

            # Training loop
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass

                outputs: Tensor = self.model(inputs)
                targets_unsqueezed = torch.unsqueeze(targets , dim=1).float()
                loss: Tensor = self.criterion(outputs, targets_unsqueezed)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Update training loss and correct predictions
                train_loss += loss.item()
                predictions = (outputs >= 0.5).int().squeeze()
                train_correct += (predictions == targets.int()).sum().item()
                total_train_samples += targets.size(0)

            # Scheduler step (if applicable)
            if self.scheduler:
                self.scheduler.step()

            # Calculate average training loss and accuracy
            avg_train_loss = train_loss / total_train_samples
            train_accuracy = train_correct / total_train_samples

            # Evaluate the model on the evaluation set
            eval_loss, eval_accuracy = self.evaluate()

            # Log metrics

            self.update_metrics(avg_train_loss, train_accuracy, eval_loss, eval_accuracy, epoch)

    def evaluate(self):
        """
        Evaluates the model on the evaluation dataset.

        Returns:
            tuple: (eval_loss, eval_accuracy) - The evaluation loss and accuracy.
        """
        # Set model to evaluation mode
        self.model.eval()

        # Initialize variables to track evaluation loss and correct predictions
        eval_loss = 0.0
        eval_correct = 0
        total_eval_samples = 0

        # Disable gradient calculations for evaluation
        with torch.no_grad():
            for inputs, targets in self.eval_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs: Tensor = self.model(inputs)
                loss: Tensor = self.criterion(outputs, torch.unsqueeze(targets , dim=1).float())

                # Update evaluation loss and correct predictions
                eval_loss += loss.item()
                predictions = (outputs >= 0.5).int()
                eval_correct += (predictions.squeeze() == targets.int()).sum().item()
                total_eval_samples += targets.size(0)

        # Calculate average evaluation loss and accuracy
        avg_eval_loss = eval_loss / total_eval_samples
        eval_accuracy = eval_correct / total_eval_samples

        return avg_eval_loss, eval_accuracy

    def update_metrics(
        self, train_loss: List[float], train_acc: List[float], eval_loss: List[float], eval_acc: List[float], epoch: int
    ):
        """updates the training dictionary with the current epoch's metrics

        Args:
            train_loss (List[float]): List of training losses recorded after each epoch.
            train_acc (List[float]): List of training accuracies recorded after each epoch.
            eval_loss (List[float]): List of evaluation losses recorded after each epoch.
            eval_acc (List[float]): List of evaluation accuracies recorded after each epoch.
            epoch (int): The current epoch index.
        """

        if epoch % self.show_every == 0:
            print(f"Epoch: {epoch} / {self.num_epochs}")
            print(
                f"Training Loss: {train_loss:.4f}, Evaluation Loss: {eval_loss:.4f} , Training Accuracy: {train_acc:.4f}, Evaluation Accuracy: {eval_acc:.4f}"
            )

        self.metrics["epoch"].append(epoch)
        self.metrics["train_loss"].append(train_loss)
        self.metrics["train_acc"].append(train_acc)
        self.metrics["eval_loss"].append(eval_loss)
        self.metrics["eval_acc"].append(eval_acc)
