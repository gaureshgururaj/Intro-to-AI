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
from svlearn.auto_encoders.resnet_auto_encoder import ResNetAutoEncoder
from svlearn.auto_encoders.variational_auto_encoder import VariationalAutoencoder

'''
The configs and bottleneck specific to Resnet50
'''
configs, bottleneck = [3, 4, 6, 3], True
class ResNetVariationalAutoEncoder(VariationalAutoencoder):

    def __init__(self):
        """ResNetVariationalAutoEncoder init 
        """

        super(ResNetVariationalAutoEncoder, self).__init__()
        self.resnet_auto_encoder = ResNetAutoEncoder(configs=configs, bottleneck=bottleneck)
        self.encoder = nn.Sequential(
            self.resnet_auto_encoder.encoder,
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 2048),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(2048, 128)  
        self.logvar_layer = nn.Linear(2048, 128) 
        self.decoder = nn.Sequential(
            nn.Linear(128, 2048 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (2048, 7, 7)),
            self.resnet_auto_encoder.decoder,
        )      