#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#

import torch.nn as nn
from svlearn.auto_encoders.vanilla_auto_encoder import Autoencoder
from pythae.models.nn.benchmarks.utils import ResBlock

'''
The Vanilla Antoencoder class that does not do any variational inference but uses Resnet with more filters in addition to the
regular Conv2d layers.  It expects an input dimension of 3 X 224 X 224 images and extends the shallow Autoencoder class.
'''
class AutoencoderDeepResnet(Autoencoder):
    def __init__(self):
        super(AutoencoderDeepResnet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x 112 x 112
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: 128 x 56 x 56
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Output: 256 x 28 x 28
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # Output: 512 x 14 x 14
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), # Output: 1024 x 7 x 7
            nn.Sequential(
                ResBlock(in_channels=1024, out_channels=128),
                ResBlock(in_channels=1024, out_channels=128),
            ),
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 2048)  # Latent space size is 2048
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2048, 1024 * 7 * 7),
            nn.Unflatten(1, (1024, 7, 7)),
            nn.Sequential(
                ResBlock(in_channels=1024, out_channels=128),
                ResBlock(in_channels=1024, out_channels=128),
            ),            
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: 512 x 14 x 14
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 256 x 28 x 28
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),   # Output: 128 x 56 x 56
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # Output: 64 x 112 x 112
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # Output: 3 x 224 x 224
        )
