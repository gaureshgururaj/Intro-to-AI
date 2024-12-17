#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from svlearn.gans import config, current_task
from svlearn.gans.dcgan_datasets import load_task_specific_tain_test_split
from svlearn.gans.classifier import get_task_specific_classifier
from svlearn.common.utils import directory_writable, ensure_directory
from svlearn.train.simple_trainer import train_simple_network

from sklearn.metrics import accuracy_score

#  -------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    data_dir = config[current_task]['data']
    directory_writable(data_dir)
    results_dir = config[current_task]['results']
    ensure_directory(results_dir)

    checkpoint_file = f"{results_dir}/control-classifier.pt"

    train_dataset , test_dataset = load_task_specific_tain_test_split()
       
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    model = get_task_specific_classifier()

    optimizer = AdamW(model.parameters(), lr = 0.001)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    result = train_simple_network(
                            model=model,
                            optimizer=optimizer,
                            lr_scheduler=scheduler,
                            loss_func=nn.CrossEntropyLoss(),
                            train_loader=train_loader,
                            test_loader=val_loader,
                            epochs=10,
                            score_funcs={'accuracy': accuracy_score},
                            classify=True,
                            checkpoint_file=checkpoint_file)
    
    print(result)
    
#  -------------------------------------------------------------------------------------------------