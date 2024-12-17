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

from svlearn.gans import config, current_task
from svlearn.gans.dcgan_models import TreeDiscriminator, TreeGenerator, DigitGenerator, DigitDiscriminator, load_task_specific_networks

#  -------------------------------------------------------------------------------------------------
# GENERATOR

class ConditionalGenerator(nn.Module):
    def __init__(self, generator: DigitGenerator | TreeGenerator , ngpu=1, class_num=2):
        super(ConditionalGenerator, self).__init__()

        self.ngpu = ngpu 
        self.generator = generator
        self.additional_channels = class_num

        out_channels = self.generator.first_layer.out_channels
        in_channels = self.generator.first_layer.in_channels
        kernel_size = self.generator.first_layer.kernel_size

        # update the first layer of generator as below to concat one-hot of classes in input vector
        self.generator.first_layer = nn.ConvTranspose2d( in_channels + class_num, 
                                                        out_channels, 
                                                        kernel_size, 1, 0, bias=False)

        self.generator.main = nn.Sequential(
            self.generator.first_layer,
            self.generator.subsequent_layers
        )


    def forward(self, z, labels):
        # label to one-hot vector of dimensions (batch_size x additional_channels)
        # ----------------------------------------------------------------------------
        # e.g. 
        # if there are a total of three classes 
        # and we have the labels of 2 samples like [2, 0] 
        # then c = [[0, 0, 1], [1, 0, 0]]
        # ----------------------------------------------------------------------------
        # 64x10
        c = nn.functional.one_hot(labels, num_classes=self.additional_channels).float()
        
        # This reshapes the tensor c to have the shape (batch_size, num_classes, 1, 1)
        c = c.view(c.size(0), c.size(1), 1, 1)
        
        # Concat image & label
        x = torch.cat([z, c], 1)
        # 64 x 110)
        # Generator out
        return self.generator(x)  

#  -------------------------------------------------------------------------------------------------
# DISCRIMINATOR

class ConditionalDiscriminator(nn.Module):
    def __init__(self, discriminator: TreeDiscriminator | DigitDiscriminator, ngpu=1, class_num: int=2):
        super(ConditionalDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.discriminator = discriminator

        out_channels = self.discriminator.first_layer.out_channels
        in_channels = self.discriminator.first_layer.in_channels
        kernel_size = self.discriminator.first_layer.kernel_size

        # update the first layer of discriminator to include the additional label channels
        self.discriminator.first_layer = nn.Conv2d(in_channels + class_num, out_channels, kernel_size, 2, 1, bias=False)

        self.discriminator.main = nn.Sequential(
            self.discriminator.first_layer,
            self.discriminator.subsequent_layers
        )
        self.additional_channels = class_num
    
    def forward(self, input, labels):
        B = input.size(0) # batch size
        H = input.size(2) # height of image in pixels
        W = input.size(3) # width of image in pixels

        # label to one-hot vector of dimensions (B x ADD_C)
        c = nn.functional.one_hot(labels, num_classes=self.additional_channels).float()  
         
        # This reshapes the tensor c to have the shape (batch_size, num_classes, 1, 1)
        c = c.view(B, self.additional_channels, 1, 1)  

        # Thus, c is expanded to match the spatial dimensions of input
        c = c.expand(-1, -1, H, W)   
        
        # Concatenate label embedding with image as additional channels
        # (B, INPUT_C + additional_channels, H, W)
        # tree-classification -> (B, 3+2, 64, 64)
        # mnist-classification -> (B, 1+10 , 28 , 28)
        extended_input = torch.cat([input, c], dim=1)

        # Discriminator output
        return self.discriminator(extended_input)   

#  -------------------------------------------------------------------------------------------------

def load_task_specific_conditional_networks(ngpu, device, nc, nz, ngf, ndf, num_classes):
    # number of classes of training images
    num_classes = config[current_task]['num_classes']

    generator, discriminator = load_task_specific_networks(ngpu, device, nc, nz, ngf, ndf)
    
    netG = ConditionalGenerator(generator, ngpu, num_classes).to(device)
    netD = ConditionalDiscriminator(discriminator,ngpu, num_classes).to(device)

    return netG , netD

#  -------------------------------------------------------------------------------------------------