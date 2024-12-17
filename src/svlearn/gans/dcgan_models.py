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
from svlearn.gans import Task, current_task
import torch.nn as nn

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#  -------------------------------------------------------------------------------------------------

# GAN models for Trees Classification

#  -------------------------------------------------------------------------------------------------
# GENERATOR

class TreeGenerator(nn.Module):
    def __init__(self, nz: int, ngf: int ,nc: int ,ngpu=1, ):
        super(TreeGenerator, self).__init__()
        self.ngpu = ngpu
        # input is Z, going into a convolution
        self.first_layer = nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False)
        self.subsequent_layers = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )
        self.main = nn.Sequential(
            self.first_layer,
            self.subsequent_layers
        )

    def forward(self, input):
        return self.main(input)

#  -------------------------------------------------------------------------------------------------
# DISCRIMINATOR

class TreeDiscriminator(nn.Module):
    def __init__(self, ndf: int, nc: int, ngpu=1, ):
        super(TreeDiscriminator, self).__init__()
        self.ngpu = ngpu
        # input is ``(nc) x 64 x 64`` 
        self.first_layer = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.subsequent_layers = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.main = nn.Sequential(
            self.first_layer,
            self.subsequent_layers
        )

    def forward(self, input):
        return self.main(input)

#  -------------------------------------------------------------------------------------------------

# GAN models for MNIST Classification

#  -------------------------------------------------------------------------------------------------
# GENERATOR

class DigitGenerator(nn.Module):
    def __init__(self, nz, ngf ,nc ,ngpu=1, ):
        super(DigitGenerator, self).__init__()
        self.ngpu = ngpu
        self.first_layer = nn.ConvTranspose2d(nz, ngf * 4, kernel_size=7, stride=1, padding=0, bias=False)
        # Output size: (ngf * 4, 7, 7)

        self.subsequent_layers = nn.Sequential(
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Output size: (ngf * 4, 7, 7)

            # Second layer: Upsample to (ngf * 2, 14, 14)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Output size: (ngf * 2, 14, 14)

            # Third layer: Upsample to (ngf, 28, 28)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Output size: (ngf, 28, 28)

            # Final layer: Output the image with size (nc, 28, 28)
            nn.ConvTranspose2d(ngf, nc, 1, 1, 0, bias=False),
            nn.Tanh()
            # Output size: (nc, 28, 28)
        )


        self.main = nn.Sequential(
            self.first_layer,
            self.subsequent_layers
        )

    def forward(self, input):
        return self.main(input)

#  -------------------------------------------------------------------------------------------------
# DISCRIMINATOR

class DigitDiscriminator(nn.Module):
    def __init__(self, ndf, nc, ngpu=1, ):  # Adjusted ndf and nc as per typical settings
        super(DigitDiscriminator, self).__init__()
        self.ngpu = ngpu
        
        # Input is (nc) x 28 x 28
        self.first_layer = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)  # Reduces to (ndf) x 14 x 14
        self.subsequent_layers = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),           # Reduces to (ndf*2) x 7 x 7
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),       # Reduces to (ndf*4) x 4 x 4
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),             # Reduces to (1) x 1 x 1
            nn.Sigmoid()
        )
        
        self.main = nn.Sequential(
            self.first_layer,
            self.subsequent_layers
        )

    def forward(self, input):
        return self.main(input)

#  -------------------------------------------------------------------------------------------------

def load_task_specific_networks(ngpu, device, nc, nz, ngf, ndf):

    if current_task == Task.TREE.value:
        generator = TreeGenerator(nz, ngf, nc, ngpu=ngpu )
        discriminator = TreeDiscriminator(ndf, nc, ngpu=ngpu)
    elif current_task == Task.MNIST.value:
        generator = DigitGenerator(nz, ngf, nc, ngpu=ngpu )
        discriminator = DigitDiscriminator(ndf, nc, ngpu=ngpu)
    else:
        raise ValueError("Networks for current task are not available")
    
    return generator.to(device) , discriminator.to(device)

#  -------------------------------------------------------------------------------------------------