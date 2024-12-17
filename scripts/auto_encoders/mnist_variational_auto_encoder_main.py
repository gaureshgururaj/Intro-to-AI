#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#
import os
import torch
from torch.utils.data import DataLoader
from svlearn.config.configuration import ConfigurationMixin

from svlearn.auto_encoders.variational_auto_encoder_mnist import VariationalAutoencoderMnist
from svlearn.auto_encoders.auto_encoder_single_channel_util import train_autoencoder

import torchvision.datasets as datasets

if __name__ == "__main__":
    config = ConfigurationMixin().load_config()
    mnist_data_path = config['paths']['data_dir']
    model_path = config['paths']['results_dir']
    
    mnist_trainset = datasets.MNIST(root=mnist_data_path, train=True, download=True, transform=None)

    train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
    eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VariationalAutoencoderMnist().to(device)
    
    checkpoint_file = f'{model_path}/mnist/mnist_variational_autoencoder.pt'
    
    starting_learning_rate = 1e-3
    
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        results = checkpoint['results']
        if 'lr' in results:
            starting_learning_rate = results['lr'][-1]
            
    train_autoencoder(model=model, 
                      train_loader=train_loader, 
                      val_loader=val_loader, 
                      num_epochs=50, 
                      device=device,  
                      checkpoint_file=checkpoint_file,
                      learning_rate=starting_learning_rate)