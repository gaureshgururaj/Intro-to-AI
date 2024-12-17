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
import numpy as np

'''
The Vanilla Antoencoder class that does not do any variational inference.  It expects
an input dimension of 3 X 224 X 224 images
'''
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output: 16 x 112 x 112
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output: 32 x 56 x 56
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: 64 x 28 x 28
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Output: 128 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Output: 256 x 7 x 7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512)  # Latent space size is 512
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 256 * 7 * 7),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: 128 x 14 x 14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 64 x 28 x 28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # Output: 32 x 56 x 56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),   # Output: 16 x 112 x 112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # Output: 3 x 224 x 224
        )
    
    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z  # Return both reconstructed image and latent representation
    
    def add_noise(self, images, noise_factor=0.2):
        """Adds Gaussian noise to images."""
        noisy_images = images + noise_factor * torch.randn_like(images)
        return noisy_images
    
    def generate_mask(self, images, patch_size=16, mask_ratio=0.2):
        """Generates random masks for images by replacing certain patches with zeros."""
        batch_size, _, height, width = images.shape
        mask = torch.ones_like(images)
        num_patches = (height // patch_size) * (width // patch_size)
        num_masked = int(mask_ratio * num_patches)
        
        rng = np.random.default_rng(42)
        for i in range(batch_size):
            mask_indices = rng.choice(num_patches, num_masked, replace=False)
            for idx in mask_indices:
                row = (idx // (width // patch_size)) * patch_size
                col = (idx % (width // patch_size)) * patch_size
                mask[i, :, row:row+patch_size, col:col+patch_size] = 0
                
        return images * mask
    