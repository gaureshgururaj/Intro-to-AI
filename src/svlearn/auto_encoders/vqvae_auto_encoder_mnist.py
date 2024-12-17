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

from svlearn.auto_encoders.vanilla_auto_encoder_mnist import AutoencoderMnist
from svlearn.auto_encoders.vqvae_embedding import VQEmbedding

'''
The VqvaeAutoencoderMnist class that does VQVAE inference for single channel images.  
It expects an input dimension of 1 X 28 X 28 images and extends the AutoencoderMnist class.
'''
class VqvaeAutoencoderMnist(AutoencoderMnist):
    def __init__(self):
        super(VqvaeAutoencoderMnist, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [32, 14, 14]
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [64, 7, 7]
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # [128, 4, 4]
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # [256, 2, 2]
            nn.ReLU(True),     
            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0), # [512, 1, 1]
            nn.ReLU(True),      
            nn.Conv2d(512, 16, kernel_size=1, stride=1)               
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 512, kernel_size=1, stride=1), # [512, 1, 1]
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1),  # [256, 2, 2]
            nn.ReLU(True),       
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # [128, 4, 4]
            nn.ReLU(True),   
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),  # [64, 7, 7]
            nn.ReLU(True),  
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [32, 14, 14]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [1, 28, 28]
            nn.Sigmoid()  # To ensure output pixel values are between 0 and 1
        )        
        self.vq_embedding = VQEmbedding(128, 16)

    def forward(self, x):
        z = self.encoder(x)
        # Quantize the latent vectors using VQEmbedding
        z_q, encoding_indices, loss = self.vq_embedding(z)        
        reconstructed = self.decoder(z_q)
        return reconstructed, z, z_q, encoding_indices, loss     
    

