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
The Vanilla Antoencoder trial 6 class that does not do any variational inference and extends the shallow Autoencoder class.
'''
class AutoencoderTrial6(Autoencoder):
    def __init__(self):
        super(AutoencoderTrial6, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0),     # ( B , 3 , 224 , 224 ) ->  ( B , 6 , 220 , 220 )
                nn.BatchNorm2d(num_features=6),                                     
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size=2, stride=2),                                  # ( B , 6 , 220 , 220 ) ->  ( B , 6 , 110 , 110 )
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),    # ( B , 6 , 110 , 110 ) ->  ( B , 16 , 106 , 106 )
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),                                   # ( B , 16 , 106 , 106 ) ->  ( B , 16 , 53 , 53 )
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4),              # ( B , 16 , 53 , 53 ) ->  ( B , 32 , 50 , 50 )                           
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),                                   # ( B , 32 , 50 , 50 )   ->  ( B , 32 , 25 , 25 ) 
                nn.Flatten(), # Change from 2D image to 1D tensor to be able to pass inputs to linear layer
                nn.Linear(in_features=32 * 25 * 25, out_features=180),
            )
        # Decoder (reversing the above)
        self.decoder = nn.Sequential(
            nn.Linear(180, 32 * 25 * 25),
            nn.Unflatten(1, (32, 25, 25)),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4),  
            nn.UpsamplingNearest2d(scale_factor=2),          
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),             
            nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5),  
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),                        
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=5),  
            nn.BatchNorm2d(num_features=3),
        )
