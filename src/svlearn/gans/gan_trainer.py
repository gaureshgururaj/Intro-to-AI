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

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from svlearn.gans.dcgan_models import weights_init
from svlearn.gans.conditional_dcgan import ConditionalDiscriminator, ConditionalGenerator

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

def load_checkpoint(model: nn.Module, checkpoint_file: str):
    if os.path.isfile(checkpoint_file):
        checkpoint_dis = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint_dis['model_state_dict'])
    else:
        # Apply the ``weights_init`` function to randomly initialize all weights
        # like this: ``to mean=0, stdev=0.2``.
        model.apply(weights_init) 
    
    return model

def train_func(num_epochs: int, 
               dataloader: torch.utils.data.DataLoader, 
               netD: nn.Module, 
               netG: nn.Module, 
               device: str, 
               checkpoint_discriminator_file: str, 
               checkpoint_generator_file: str,
               nz: int,
               num_classes: int = None,
               learning_rate:float = None) -> None:
    """The training loop of the GAN

    Args:
        num_epochs (int): epochs to train for
        dataloader (torch.utils.data.DataLoader): training dataloader consisting of the tree images
        netD (Discriminator): Discriminator network
        netG (Generator): Generator network
        device (str): cuda or cpu
        checkpoint_discriminator_file (str): discriminator model file checkpoint to save to
        checkpoint_generator_file (str): generator model file checkpoint to save to
        nz (int): noise hidden vector dimensionality
        num_classes (int, Optional): Defaults to None
        learning_rate (float, Optional): If None, uses the lr defined in global variables
    """
    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    if learning_rate is None:
        learning_rate = lr
        
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    # Training Loop
    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    iters = 0
    print("Starting Training Loop...")
    # initialize results dictionary
    results = {"epoch": [],
                "gen loss": G_losses,
                "dis loss": D_losses,
                }
    
    # Loop through each epoch
    for epoch in range(num_epochs):
        current_gen_loss = 0  # Initialize generator loss for the current epoch
        current_dis_loss = 0  # Initialize discriminator loss for the current epoch
        batch_iter = 0  # Counter for number of batches per epoch

        # Loop through each batch of data provided by the dataloader
        for i, data in enumerate(dataloader, 0):
            
            #--------------------------------------------------------------------
            # (1) Update Discriminator (D): Maximize log(D(x)) + log(1 - D(G(z)))
            #--------------------------------------------------------------------
            # Set discriminator gradients to zero before starting backpropagation
            netD.zero_grad()

            ## Train with all-real batch
            # Format real data batch for the device (e.g., GPU)
            real_images = data[0].to(device)
            b_size = real_images.size(0)  # Batch size
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)  # Set label for real data
            
            # Forward pass real batch through discriminator
            if isinstance(netD, ConditionalDiscriminator):  # Check if discriminator is conditional
                # Get labels for conditional discriminator
                image_labels = data[1].to(device)
                output = netD(real_images, image_labels).view(-1)  # Discriminator's prediction on real data
            else:
                output = netD(real_images).view(-1)  # Non-conditional discriminator prediction
            
            # Calculate loss for discriminator on real data
            errD_real = criterion(output, label)
            # Backward pass to calculate gradients for discriminator (D) on real data
            errD_real.backward()

            ## Train with all-fake batch
            # Generate a batch of random latent vectors for the generator input
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            
            # Generate fake images from latent vectors using the generator (G)
            if isinstance(netG, ConditionalGenerator):  # Check if generator is conditional
                # Generate random labels for conditional generator
                image_labels = torch.LongTensor(np.random.randint(0, num_classes, b_size)).to(device)
                fake = netG(noise, image_labels)  # Generate fake images conditioned on labels
                # Get discriminator's output for fake images
                output = netD(fake.detach(), image_labels).view(-1)  
            else:
                fake = netG(noise)  # Non-conditional generator
                output = netD(fake.detach()).view(-1)  # Discriminator prediction on fake images
            
            # Set label to fake for calculating loss
            label.fill_(fake_label)
            # Calculate discriminator's loss on fake images
            errD_fake = criterion(output, label)
            # Backward pass to calculate gradients for discriminator on fake data
            errD_fake.backward()
            # Total loss for discriminator is the sum of losses on real and fake data
            errD = errD_real + errD_fake
            # Update discriminator's weights
            optimizerD.step()

            #--------------------------------------------------------------------
            # (2) Update Generator (G): Maximize log(D(G(z)))
            #--------------------------------------------------------------------
            netG.zero_grad()  # Reset gradients for generator
            label.fill_(real_label)  # The goal is for the generator to "fool" the discriminator
            
            # Forward pass fake data through discriminator to check if discriminator recognizes it as real
            if isinstance(netD, ConditionalDiscriminator):  # Conditional discriminator
                output = netD(fake, image_labels).view(-1)  # Discriminator output on fake images
            else:
                output = netD(fake).view(-1)  # Non-conditional discriminator
            
            # Calculate generator's loss based on discriminator's response
            errG = criterion(output, label)
            # Backward pass to calculate gradients for generator
            errG.backward()
            # Update generator's weights
            optimizerG.step()

            # Record losses for plotting/tracking purposes
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1
            batch_iter += 1  # Increment batch count
            results["epoch"].append(epoch + 1)  # Track epoch in results
            current_gen_loss += errG.item()  # Sum generator loss for current epoch
            current_dis_loss += errD.item()  # Sum discriminator loss for current epoch
            
        # Calculate average losses for generator and discriminator over the epoch
        current_gen_loss /= batch_iter
        current_dis_loss /= batch_iter

        # Print the epoch's average losses for both generator and discriminator
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (epoch, num_epochs, current_dis_loss, current_gen_loss)) 

        # Save generator's state to checkpoint file after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': netG.state_dict(),
            'optimizer_state_dict': optimizerG.state_dict(),
            'results' : results
        }, checkpoint_generator_file)

        # Save discriminator's state to checkpoint file after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': netD.state_dict(),
            'optimizer_state_dict': optimizerD.state_dict(),
            'results' : results
        }, checkpoint_discriminator_file)