#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# VQEmbedding layer for vector quantization
'''
The forward pass here has been inspired from a similar logic implemented in https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/vq_vae/vq_vae_utils.py
'''
class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VQEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_loss_factor = 0.25
        self.quantization_loss_factor = 1.0

        # Codebook of embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        
    def initialize_codebook_usage_to_zero(self, latent_grid_size: Tuple = (7, 7)):
        """This method is to be called at the beginning of a epoch training cycle

        Args:
            latent_grid_size (Tuple, optional): _description_. Defaults to (7, 7).
        """
        self.latent_grid_size = latent_grid_size
        self.codebook_usage_per_position = np.zeros((latent_grid_size[0], latent_grid_size[1], self.num_embeddings))

    def update_codebook_usage_count(self, encoding_indices: torch.Tensor):
        """This method will be called during every training forward pass

        Args:
            encoding_indices (torch.Tensor): _description_
        """
        codebook_indices = encoding_indices.view(-1).view(-1, self.latent_grid_size[0], self.latent_grid_size[1])
        for i in range(self.latent_grid_size[0]):
            for j in range(self.latent_grid_size[1]):
                for index in codebook_indices[:, i, j]:  # Select the index for position (i, j)
                    self.codebook_usage_per_position[i, j, index] += 1

    def forward(self, z: torch.Tensor):
        # bring embedding dimension to the end
        z = z.permute(0, 2, 3, 1)
        
        distances = (
            (z.reshape(-1, self.embedding_dim) ** 2).sum(dim=-1, keepdim=True)
            + (self.embedding.weight**2).sum(dim=-1)
            - 2 * z.reshape(-1, self.embedding_dim) @ self.embedding.weight.T
        )

        closest = distances.argmin(-1).unsqueeze(-1)

        quantized_indices = closest.reshape(z.shape[0], z.shape[1], z.shape[2])

        one_hot_encoding = (
            F.one_hot(closest, num_classes=self.num_embeddings)
            .type(torch.float)
            .squeeze(1)
        )

        # quantization
        quantized = one_hot_encoding @ self.embedding.weight
        quantized = quantized.reshape_as(z)
        
        commitment_loss = F.mse_loss(
            quantized.detach().reshape(-1, self.embedding_dim),
            z.reshape(-1, self.embedding_dim),
            reduction="sum",
        )

        embedding_loss = F.mse_loss(
            quantized.reshape(-1, self.embedding_dim),
            z.detach().reshape(-1, self.embedding_dim),
            reduction="sum",
        )

        loss = (
            commitment_loss * self.commitment_loss_factor
            + embedding_loss * self.quantization_loss_factor
        )
        
        quantized = z + (quantized - z).detach()
        
        quantized = quantized.permute(0, 3, 1, 2)

        quantized_indices=quantized_indices.unsqueeze(1)
        
        if self.training:
            self.update_codebook_usage_count(encoding_indices=quantized_indices)

        return quantized, quantized_indices, loss   
    
# Loss function for VQ-VAE
def vqvae_loss(x, x_recon, codebook_loss):
    # Reconstruction loss (e.g., MSE loss between input and reconstruction)
    recon_loss = F.mse_loss(
            x_recon.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)
    # Total loss
    return (recon_loss + codebook_loss).mean(dim=0)    