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
The Vanilla Antoencoder trial 4 class that does not do any variational inference and extends the shallow Autoencoder class.
'''
class AutoencoderTrial4(Autoencoder):
    def __init__(self):
        super(AutoencoderTrial4, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, stride=2, padding=1),  # Output: 12 x 112 x 112
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1), # Output: 24 x 56 x 56
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1), # Output: 48 x 28 x 28
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1), # Output: 96 x 14 x 14
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1), # Output: 192 x 7 x 7
            nn.Sequential(
                ResBlock(in_channels=192, out_channels=48),
                ResBlock(in_channels=192, out_channels=48),
            ),
            nn.Flatten(),
            nn.Linear(192 * 7 * 7, 1024)  # Latent space size is 1024
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(1024, 192 * 7 * 7),
            nn.Unflatten(1, (192, 7, 7)),
            nn.Sequential(
                ResBlock(in_channels=192, out_channels=48),
                ResBlock(in_channels=192, out_channels=48),
            ),            
            nn.ConvTranspose2d(192, 96, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: 96 x 14 x 14
            nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 48 x 28 x 28
            nn.ConvTranspose2d(48, 24, kernel_size=3, stride=2, padding=1, output_padding=1),   # Output: 24 x 56 x 56
            nn.ConvTranspose2d(24, 12, kernel_size=3, stride=2, padding=1, output_padding=1),   # Output: 12 x 112 x 112
            nn.ConvTranspose2d(12, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # Output: 3 x 224 x 224
        )
