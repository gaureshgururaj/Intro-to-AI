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
from svlearn.auto_encoders.resnet_auto_encoder_modified import ResNetAutoEncoderModified
from svlearn.auto_encoders.variational_auto_encoder import VariationalAutoencoder

'''
The configs and bottleneck specific to Resnet50
'''
configs, bottleneck = [3, 4, 6, 3], True
class ResNetVariationalAutoEncoderModified(VariationalAutoencoder):

    def __init__(self):
        """ResNetVariationalAutoEncoderModified init 
        """

        super(ResNetVariationalAutoEncoderModified, self).__init__()
        self.resnet_auto_encoder = ResNetAutoEncoderModified(configs=configs, bottleneck=bottleneck)
        self.encoder = nn.Sequential(
            self.resnet_auto_encoder.encoder,
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 1024),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(1024, 128)  
        self.logvar_layer = nn.Linear(1024, 128) 
        self.decoder = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * 28 * 28),
            nn.ReLU(),
            nn.Unflatten(1, (64, 28, 28)),
            nn.ConvTranspose2d(in_channels=64, out_channels=512, kernel_size=1, stride=1),
            nn.ReLU(),
            self.resnet_auto_encoder.decoder,
        )      