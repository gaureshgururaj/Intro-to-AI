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
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from svlearn.gans import config, current_task
from svlearn.gans.dcgan_datasets import load_task_specific_dataset
from svlearn.gans.conditional_dcgan import load_task_specific_conditional_networks
from svlearn.common.utils import ensure_directory, directory_writable
from svlearn.gans.gan_trainer import train_func, load_checkpoint

#  -------------------------------------------------------------------------------------------------
    
if __name__ == "__main__":
    # Root directory for dataset
    dataroot = config[current_task]['data']
    directory_writable(dataroot)
    results_dir = config[current_task]['results']
    ensure_directory(results_dir)

    checkpoint_generator_file = f'{results_dir}/conditional_generator.pt'
    checkpoint_discriminator_file = f'{results_dir}/conditional_discriminator.pt'

    # Number of workers for dataloader
    workers = 2

    # Batch size during training (if -1 use full data for batch)
    batch_size = 256

    # Number of training epochs
    num_epochs = 500

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Global variables
    # Number of channels in the training images. For color images this is 3
    nc = config[current_task]['num_channels']

    # Size of z latent vector (i.e. size of generator input)
    nz = config[current_task]['nz']

    # Size of feature maps in generator
    ngf = config[current_task]['ngf']

    # Size of feature maps in discriminator
    ndf = config[current_task]['ndf']

    # number of classes of training images
    num_classes = config[current_task]['num_classes']

    # Decide which device we want to run on
    device =  torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")    

    dataset = load_task_specific_dataset(balanced=True)
    
    # Create the dataloader
    if batch_size == -1:
        batch_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=workers)

    # Create the Generator and Discriminator
    netG, netD = load_task_specific_conditional_networks(ngpu, device, nc, nz, ngf, ndf, num_classes)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    # load model weights from checkpoint if available else initialize weights randomly
    load_checkpoint(netG , checkpoint_generator_file)
    load_checkpoint(netD , checkpoint_discriminator_file)
     
    train_func(num_epochs=500, 
               dataloader=dataloader, 
               netD=netD, netG=netG, 
               device=device, 
               checkpoint_discriminator_file=checkpoint_discriminator_file, 
               checkpoint_generator_file=checkpoint_generator_file, 
               nz=nz,
               num_classes=num_classes,
               learning_rate = 1e-5)

#  -------------------------------------------------------------------------------------------------