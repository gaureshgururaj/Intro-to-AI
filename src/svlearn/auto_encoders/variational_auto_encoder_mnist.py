#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#

import torch
import torch.nn as nn

from svlearn.auto_encoders.vanilla_auto_encoder_mnist import AutoencoderMnist

'''
The VariationalAutoencoderMnist class that does variational inference for single channel images.  
It expects an input dimension of 1 X 28 X 28 images and extends the AutoencoderMnist class.
'''
class VariationalAutoencoderMnist(AutoencoderMnist):
    def __init__(self):
        super(VariationalAutoencoderMnist, self).__init__()
        
        self.mu_layer = nn.Linear(128, 128)  
        self.logvar_layer = nn.Linear(128, 128) 
    
        
    def reparameterize(self, mu, logvar):
        """Reparameterize function that sits between the encoder and decoder for a variational autoencoder

        Args:
            mu (float): mean value from the encoder
            logvar (float): log variance from the encoder

        Returns:
            Tuple[torch.tensor, torch.tensor, float, float]: Tuple consisting of the reconstructed image, hidden vector, mean, log-variance
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std       
    
    def forward(self, x):
        z = self.encoder(x)
        mu = self.mu_layer(z)
        logvar = self.logvar_layer(z)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, z, mu, logvar       
    

