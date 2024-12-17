#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

'''
This code has been borrowed significantly from the [pytorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
The dataset of images has been changed however to a custom dataset of tree images of two kinds (Oak and Weeping Willow)
'''

#  -------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from svlearn.config.configuration import ConfigurationMixin
from svlearn.gans import current_task
from svlearn.gans.dcgan_datasets import load_task_specific_dataset
from svlearn.gans.dcgan_models import load_task_specific_networks
from svlearn.gans.gan_trainer import train_func, load_checkpoint

#  -------------------------------------------------------------------------------------------------


if __name__ == "__main__":

#  -------------------------------------------------------------------------------------------------
    # Configurations
    config = ConfigurationMixin().load_config()
    dataroot = config[current_task]['data']
    results_dir = config[current_task]['results']

    checkpoint_generator_file = f'{results_dir}/generator.pt'
    checkpoint_discriminator_file = f'{results_dir}/discriminator.pt'


    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

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
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")   

#  -------------------------------------------------------------------------------------------------
    # Create the dataset
    dataset = load_task_specific_dataset()
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
#  -------------------------------------------------------------------------------------------------
 
    # Create the generator and discriminator models
    netG , netD = load_task_specific_networks(ngpu, device, nc, nz, ngf, ndf)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    # load model weights from checkpoint if available else initialize weights randomly
    load_checkpoint(netG , checkpoint_generator_file)
    load_checkpoint(netD , checkpoint_discriminator_file)
    
    # train model
    train_func(num_epochs=500, 
               dataloader=dataloader, 
               netD=netD, netG=netG, 
               device=device, 
               checkpoint_discriminator_file=checkpoint_discriminator_file, 
               checkpoint_generator_file=checkpoint_generator_file, 
               nz=nz)

#  -------------------------------------------------------------------------------------------------