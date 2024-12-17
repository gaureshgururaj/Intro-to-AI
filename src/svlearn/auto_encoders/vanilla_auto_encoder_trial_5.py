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

'''
The Vanilla Antoencoder trial 5 class that does not do any variational inference and extends the shallow Autoencoder class.
'''
class AutoencoderTrial5(Autoencoder):
    def __init__(self):
        super(AutoencoderTrial5, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x 112 x 112
            nn.ReLU(),
            nn.BatchNorm2d(32),            
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: 64 x 56 x 56
            nn.ReLU(),
            nn.BatchNorm2d(64),
                        
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: 128 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(128),
                        
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Output: 256 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(256),            
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # Output: 512 x 7 x 7,
            nn.ReLU(),
            nn.BatchNorm2d(512),
                        
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024)  # Latent space vector
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512 * 7 * 7),
            nn.Unflatten(1, (512, 7, 7)),
            
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 256 x 14 x 14
            nn.ReLU(),            
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 128 x 28 x 28
            nn.ReLU(),            
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 64 x 56 x 56
            nn.ReLU(),
            nn.BatchNorm2d(64),
                        
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # Output: 32 x 112 x 112
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # Output: 3 x 224 x 224
        )
