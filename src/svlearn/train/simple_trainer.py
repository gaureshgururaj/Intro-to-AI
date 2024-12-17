#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader
from tqdm.notebook import tqdm_notebook
from typing import Dict
#  -------------------------------------------------------------------------------------------------

# source: 
# https://github.com/EdwardRaff/Inside-Deep-Learning/blob/main/idlmam.py

#  -------------------------------------------------------------------------------------------------

def run_epoch(model: nn.Module, optimizer, data_loader: dataloader, loss_func, device,
              results: Dict, score_funcs: Dict, prefix=" ", desc: str =None, classify: bool =False , lr_scheduler=None) -> float:
    """run an epoch of training / validation and returns the time taken to run the epoch

    Args:
        model (nn.Module): the neural network 
        optimizer (_type_): optimizer
        data_loader (dataloader): dataloader
        loss_func (_type_): loss function
        device (_type_): device (cuda or cpu)
        results (Dict): dictionary to store results
        score_funcs (Dict): evaluation metrics
        prefix (str, optional): specify the prefix in the  results dictionary keys. train or test
        desc (str, optional): description to display on progress bar. Defaults to None.
        classify (bool, optional): True if the task is a classification problem. Defaults to False.

    Returns:
        float: execution time in seconds
    """
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()

    for inputs, labels in tqdm_notebook(data_loader, desc=desc, leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if classify:
            labels = labels.type(torch.LongTensor)

        y_hat = model(inputs)
        loss = loss_func(y_hat, labels)

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if len(score_funcs) > 0:
            labels = labels.detach().cpu().numpy()
            # if the problem is a classification problem
            if classify:
                # find the argmax of the softmax function for each observation in the batch
                y_hat = np.argmax(F.softmax(y_hat, dim=1).detach().cpu().numpy(), axis=1)
            else:
                y_hat = y_hat.detach().cpu().numpy()

            y_pred.extend(y_hat.tolist())
            y_true.extend(labels.tolist())

        # to get a Python number from a tensor containing a single value:
        running_loss.append(loss.item())

    results[prefix + " loss"].append(np.mean(running_loss))
    for name, score_func in score_funcs.items():
        results[prefix + " " + name].append(score_func(y_true, y_pred))

    if lr_scheduler:
        lr_scheduler.step()

    end = time.time()

    # return training time
    return end - start

#  -------------------------------------------------------------------------------------------------

def train_simple_network(model: nn.Module,
                        loss_func,
                        train_loader: dataloader,
                        optimizer = None,
                        test_loader: dataloader=None,
                        score_funcs: Dict =None,
                        epochs: int=20,
                        device=torch.device("cpu"),
                        lr: float =0.0001,
                        classify: bool =False,
                        checkpoint_file: str =None,
                        lr_scheduler = None
                        ) -> pd.DataFrame:
    """method to train and evaluate a model

    Args:
        model (nn.Module): the neural network 
        optimizer (_type_): optimizer
        train_loader (dataloader): dataloader for training
        test_loader (dataloader, optional): dataloader for testing. Defaults to None.
        score_funcs (Dict, optional): evaluation metrics. Defaults to None.
        epochs (int, optional): number of epochs. Defaults to 20.
        device (_type_, optional): device (cuda or cpu). Defaults to torch.device("cpu").
        lr (float, optional): learning rate. Defaults to 0.0001.
        classify (bool, optional): True if the task is a classification problem. Defaults to False.
        checkpoint_file (str, optional): path to save the model and results after each epoch. Defaults to None.

    Returns:
        pd.DataFrame: Results dictionary
    """
    if not optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    model.to(device)
    best_val_loss = float('inf')

    # initialize results dictionary
    results = {"epoch": [],
               "train time": [],
               "train loss": [],
               "test loss": [],
               }
    # add keys for each scoring metric
    if score_funcs is not None:
        if test_loader is not None:
            for name in score_funcs:
                results[f"train {name}"] = []
                results[f"test {name}"] = []
        else:
            for name in score_funcs:
                results[f"train {name}"] = []

    total_train_time = 0.0

    for epoch in tqdm_notebook(range(epochs), desc="Epoch"):
        model = model.train()
        # train
        total_train_time += run_epoch(model=model,
                                      optimizer=optimizer,
                                      data_loader=train_loader,
                                      loss_func=loss_func,
                                      device=device,
                                      results=results,
                                      score_funcs=score_funcs,
                                      prefix="train",
                                      desc="Training",
                                      classify=classify,
                                      lr_scheduler=lr_scheduler
                                      )
        results["train time"].append(total_train_time)
        results["epoch"].append(epoch)

        # validate
        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model=model,
                          optimizer=optimizer,
                          data_loader=test_loader,
                          loss_func=loss_func,
                          device=device,
                          results=results,
                          score_funcs=score_funcs,
                          prefix="test",
                          desc="Testing",
                          classify=classify
                          )
            # save
            if checkpoint_file is not None:
                
                current_val_loss = results["test loss"][-1]
                if best_val_loss > current_val_loss:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'results' : results
                        }, checkpoint_file)
                    print(f'best results thus far: {results}')
                    best_val_loss = current_val_loss

    return pd.DataFrame.from_dict(results)

#  -------------------------------------------------------------------------------------------------