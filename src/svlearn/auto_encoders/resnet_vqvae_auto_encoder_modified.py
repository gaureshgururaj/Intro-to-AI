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
from svlearn.auto_encoders.vanilla_auto_encoder import Autoencoder
from svlearn.auto_encoders.vqvae_embedding import VQEmbedding

'''
The configs and bottleneck specific to Resnet50
'''
configs, bottleneck = [3, 4, 6, 3], True
class ResNetVQVAEAutoEncoderModified(Autoencoder):

    def __init__(self):
        """ResNetVariationalAutoEncoder init 
        """

        super(ResNetVQVAEAutoEncoderModified, self).__init__()
        self.resnet_auto_encoder = ResNetAutoEncoderModified(configs=configs, bottleneck=bottleneck)
        self.encoder = nn.Sequential( 
            self.resnet_auto_encoder.encoder,
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=512, kernel_size=1, stride=1),
            nn.ReLU(),
            self.resnet_auto_encoder.decoder,          
        )
        self.vq_embedding = VQEmbedding(512, 64)

    def forward(self, x):
        z = self.encoder(x)
        # Quantize the latent vectors using VQEmbedding
        z_q, encoding_indices, codebook_loss = self.vq_embedding(z)        
        reconstructed = self.decoder(z_q)
        return reconstructed, z, z_q, encoding_indices, codebook_loss  
    