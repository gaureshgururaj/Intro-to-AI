#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from svlearn.auto_encoders.auto_encoder_util import convert
from svlearn.auto_encoders.resnet_vqvae_auto_encoder import ResNetVQVAEAutoEncoder
from svlearn.auto_encoders.resnet_vqvae_auto_encoder_modified import ResNetVQVAEAutoEncoderModified
from svlearn.auto_encoders.vanilla_auto_encoder import Autoencoder
from svlearn.auto_encoders.variational_auto_encoder import VariationalAutoencoder
from svlearn.auto_encoders.vqvae_embedding import vqvae_loss

def variational_ae_loss_function(recon_x, x, mu, logvar) -> torch.Tensor:
    """The VAE loss function

    Args:
        recon_x (torch.Tensor): The reconstructed output
        x (torch.Tensor): The input
        mu (torch.Tensor): The mean values arising from the  hidden representation
        logvar (torch.Tensor): The log variance values arising from the hidden representation

    Returns:
        torch.Tensor: The reconstruction loss + kl divergence loss
    """
    mse = 0.5 * F.mse_loss(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1) 
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return (mse + kld).mean(dim=0)  

# Custom Contractive Autoencoder Loss
def contractive_loss(inputs: torch.Tensor, hidden: torch.Tensor, c_factor: float) -> torch.Tensor :
    """Calculates the contractive term of the loss

    Args:
        inputs (torch.Tensor): The input vector that is fed to the encoder
        hidden (torch.Tensor): The hidden vector
        c_factor (float): The lambda factor

    Returns:
        torch.Tensor: The contractive loss term
    """
    
    # Compute the Jacobian matrix (gradient of hidden with respect to inputs)
    batch_size = inputs.size(0)
    
    # Create a tensor to store the contractive loss for each example
    contractive_term = 0.0
    
    for i in range(batch_size):
        # For each batch element, compute the Jacobian (partial derivative)
        latent = hidden[i]
        grad_hidden = torch.autograd.grad(
            latent, inputs[i], 
            grad_outputs=torch.ones_like(latent)
        )[0]
        
        # If grad_hidden is None (meaning hidden[i] did not depend on inputs[i]), skip
        if grad_hidden is not None:
            print('came here')
            # Compute the Frobenius norm of the Jacobian (sum of squared gradients)
            contractive_term += torch.sum(grad_hidden ** 2)
    
    # Scale the contractive term by the regularization parameter lambda
    contractive_term = c_factor * contractive_term / batch_size
    
    return contractive_term

def train_autoencoder(model: Autoencoder, 
                      train_loader: DataLoader, 
                      val_loader: DataLoader, 
                      num_epochs: int =10, 
                      learning_rate: float =1e-3, 
                      device: str ='cpu',
                      mode: str =None, 
                      noise_factor: float =0.2, 
                      mask_ratio: float=0.2,
                      lr_step_size: int =10,
                      lr_gamma: float =0.1,
                      weight_decay: float=1e-5,
                      lr_mode: str = "stepLR",
                      checkpoint_file: str =None,):
    """The training loop for the autoencoder

    Args:
        model (Autoencoder): The Autoencoder model
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader): Validation dataloader
        num_epochs (int, optional): Number of epochs. Defaults to 10.
        learning_rate (float, optional): Learning rate. Defaults to 1e-3.
        device (str, optional): Could be 'cuda' or 'cpu'. Defaults to 'cpu'.
        mode (str, optional): Can be None or 'denoising' or 'masked' or 'contractive'
        noise_factor (float, optional): Default is 0.2 (goes with denoising)
        mask_ratio (float, optional): Default is 0.2 (goes with masked)
        lr_step_size (int, optional): Default 10 epochs for the step scheduler
        lr_gamma (float, optional): Default 0.1 for multiplicative factor every lr_step_size epochs
        weight_decay (float, optional): Default 1e-5 for the Adam optimizer weight decay
        lr_mode (str, optional): Defaults to stepLR.  Other choice is reduceLROnPlateau
        checkpoint_file (str, optional): The path to save the model of the best run. Defaults to None. 
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    if lr_mode == "reduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
    if isinstance(model, VariationalAutoencoder):
        mode = 'variational'
    elif isinstance(model, ResNetVQVAEAutoEncoder) or isinstance(model, ResNetVQVAEAutoEncoderModified):
        mode = 'vqvae'
    
    print(mode)
    train_batch_size = train_loader.batch_size
    val_batch_size = val_loader.batch_size

    best_val_loss = float('inf')   

    # initialize results dictionary
    results = {"epoch": [],
               "train loss": [],
               "val loss": [],
               "lr": [],
               }
        
    print(f'The training batch size is {train_batch_size} and the validation batch size is {val_batch_size}')
    print(f'The length of training loader is {len(train_loader)} and the length of val loader is {len(val_loader)}')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        if isinstance(model, ResNetVQVAEAutoEncoder):
            model.vq_embedding.initialize_codebook_usage_to_zero()
        elif isinstance(model, ResNetVQVAEAutoEncoderModified):
            model.vq_embedding.initialize_codebook_usage_to_zero(latent_grid_size=(28,28))
        
        for images, _ in train_loader:  # Labels are not needed
            loss = train_func(model, device, mode, noise_factor, mask_ratio, criterion, optimizer, images, train_batch_size)
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Evaluate on validation set
        val_loss = validation_loss(model, val_loader, device, mode, criterion, val_batch_size)    
        
        avg_val_loss = val_loss / len(val_loader)

        if lr_mode == "stepLR":
            scheduler.step()    
        else :
            # Step the scheduler at the end of each epoch
            scheduler.step(avg_val_loss)   
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"""Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, \
              Val Loss: {avg_val_loss:.4f}, Current LR: {current_lr}""".replace("\n", ""))

        # save
        if checkpoint_file is not None:           
            results["epoch"].append(epoch + 1)     
            results["train loss"].append(avg_train_loss)  
            results["val loss"].append(avg_val_loss)
            results["lr"].append(current_lr)
            current_val_loss = avg_val_loss
            if best_val_loss > current_val_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'results' : results
                    }, checkpoint_file)
                if mode == 'vqvae':
                    folder_path = os.path.dirname(checkpoint_file)
                    if isinstance(model, ResNetVQVAEAutoEncoder):
                        codebook_file = 'vqvae_codebook_usage.npy'
                    else :
                        codebook_file = 'vqvae_codebook_usage_modified.npy'
                    np.save(f"{folder_path}/{codebook_file}", model.vq_embedding.codebook_usage_per_position)
                    
                best_val_loss = current_val_loss
                
def validation_loss(model, val_loader, device, mode, criterion, val_batch_size) -> float:
    """Computes the validation loss

    Args:
        model (nn.Module): The autoencoder
        val_loader (DataLoader): The validation data loader
        device (str): Could be cuda or cpu
        mode (str): If mode is 'variational' compute the variational loss, else just the MSE loss
        criterion (_Loss): Defines the loss criterion to be applied - used when mode is not 'variational'
        val_batch_size (int): The size of the batch of validation data loader

    Returns:
        float: The validation loss for the epoch - obtained as a total of all the average validation losses for each batch
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            if mode == 'variational':
                reconstructed, _, mu, logvar = model(images)
                loss = variational_ae_loss_function(reconstructed, images, mu, logvar)
            elif mode == 'vqvae':
                reconstructed, z, z_q, _, codebook_loss = model(images)
                loss = vqvae_loss(x=images, x_recon=reconstructed, codebook_loss=codebook_loss)
            else :
                reconstructed, _ = model(images)
                loss = criterion(reconstructed, images)
            val_loss += loss.item()
    return val_loss

def frobenius_norm_jacobian_approximation(model, x, noise_scale=1e-5, num_samples=10):
    """
    Approximate the Frobenius norm of the Jacobian of the encoder output
    with respect to the input using finite differences.

    Args:
        model: The autoencoder model with an encoder method.
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        noise_scale (float): Scale for the noise added to the inputs.
        num_samples (int): Number of samples for averaging the approximation.

    Returns:
        torch.Tensor: Approximated Frobenius norm of the Jacobian.
    """
    # Get the original encoder output for the unperturbed input
    output_original = model.encoder(x)

    frobenius_norm = 0.0

    for _ in range(num_samples):
        # Create random noise with the same shape as the input
        noise = noise_scale * torch.randn_like(x)  # Shape: (batch_size, channels, height, width)

        # Compute encoder outputs for the perturbed inputs
        output_perturbed = model.encoder(x + noise)

        # Compute the difference in outputs
        diff = output_perturbed - output_original  # Shape: (batch_size, hidden_dim)

        # Compute the Frobenius norm of the Jacobian approximation
        frobenius_norm += torch.linalg.norm(diff.view(-1)) ** 2

    # Average the Frobenius norm approximation
    frobenius_norm = frobenius_norm / num_samples

    return frobenius_norm

def train_func(model, device, mode, noise_factor, mask_ratio, criterion, optimizer, images, train_batch_size) -> torch.Tensor:
    """_summary_

    Args:
        model (nn.Module): The autoencoder
        device (str): Could be cpu or cuda
        mode (str): Could be None or 'masked' or 'denoising' or 'variational' or 'contractive'
        noise_factor (float): The noise factor added for the 'denoising' case.
        mask_ratio (float): The mask ratio applied for the 'masked' case.
        criterion (_Loss): The loss criterion to be applied for the non 'variational' case
        optimizer (Optimizer): The optimizer being used for gradient descent
        images (torch.Tensor): The input images
        train_batch_size (int): The training batch size 

    Returns:
        torch.Tensor: The training loss
    """
    images = images.to(device)
    optimizer.zero_grad()
    if mode == 'denoising':
                # Add noise to the images for denoising autoencoder
        noisy_images = model.add_noise(images, noise_factor=noise_factor)  
        reconstructed, _ = model(noisy_images)         
    elif mode == 'masked':
                # Mask patches of the image for masked autoencoder
        masked_images = model.generate_mask(images, mask_ratio=mask_ratio)  
        reconstructed, _ = model(masked_images) 
    elif mode == 'variational':
        reconstructed, _, mu, logvar = model(images)
    elif mode == 'vqvae':
        reconstructed, z, z_q, _, codebook_loss = model(images)
    else :
        reconstructed, _ = model(images)
            
    if mode == 'variational':
        loss = variational_ae_loss_function(reconstructed, images, mu, logvar)
    elif mode == 'vqvae':
        loss = vqvae_loss(x=images, x_recon=reconstructed, codebook_loss=codebook_loss)
    else:    
        loss = criterion(reconstructed, images)
        
    if mode == 'contractive':
        c_factor = 1e-4
        #contractive_penalty = contractive_loss(inputs=images, hidden=hidden, c_factor=c_factor)
        jac = torch.autograd.functional.jacobian(model.encoder, images)
        jac_norm = torch.norm(jac.view(train_batch_size, -1), dim=1) ** 2
        contraction_loss = torch.mean(jac_norm)
        contractive_penalty = c_factor * contraction_loss

        loss += contractive_penalty
                
    loss.backward()
    optimizer.step()
    return loss

def get_latent_representations(model: Autoencoder, 
                               data_loader: DataLoader, 
                               device: str='cpu') -> np.ndarray:
    """Gets the hidden vectors of the auto encoder (post training)

    Args:
        model (Autoencoder): Trained Autoencoder
        data_loader (DataLoader): The dataloader whose hidden representation we want
        device (str, optional): Could be 'cuda' or 'cpu'. Defaults to 'cpu'.

    Returns:
        np.ndarray: The hidden vectors corresponding to the input images of the dataloader
    """
    latent_representations = []
    model.eval()
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            if isinstance(model, VariationalAutoencoder):
                _, z, _, _ = model(images)
            else:
                _, z = model(images)
            z = z.view(z.size(0), -1)
            latent_representations.append(z.cpu().numpy())
    return np.vstack(latent_representations)

def visualize_reconstruction(model: Autoencoder, 
                             data_loader: DataLoader, 
                             device: str ='cpu'):
    """This visualizes the original and reconstructed images - 5 at a time

    Args:
        model (Autoencoder): The trained Autoencoder
        data_loader (DataLoader): The evaluation data loader from which we pick the first 5 samples
        device (str, optional): Could be 'cuda' or 'cpu'. Defaults to 'cpu'.
    """
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(data_loader))
        images = images.to(device)
        if isinstance(model, VariationalAutoencoder):
            reconstructed, _, _, _ = model(images)
        elif isinstance(model, ResNetVQVAEAutoEncoder) or isinstance(model, ResNetVQVAEAutoEncoderModified):
            reconstructed, _, _, _, _ = model(images)
        else:
            reconstructed, _ = model(images)
        
        # Display original and reconstructed images
        _, axs = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(5):
            axs[0, i].imshow(convert(images[i].cpu()))
            axs[1, i].imshow(convert(reconstructed[i].cpu()))
        plt.show()
        