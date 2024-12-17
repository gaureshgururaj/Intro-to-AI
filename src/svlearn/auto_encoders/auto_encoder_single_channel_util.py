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
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from svlearn.auto_encoders.auto_encoder_multi_channel_util import variational_ae_loss_function
from svlearn.auto_encoders.variational_auto_encoder_mnist import VariationalAutoencoderMnist
from svlearn.auto_encoders.vqvae_auto_encoder_mnist import VqvaeAutoencoderMnist
from svlearn.auto_encoders.vqvae_embedding import vqvae_loss

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
        for images in val_loader:
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

def train_autoencoder(model: nn.Module, 
                      train_loader: DataLoader, 
                      val_loader: DataLoader, 
                      num_epochs: int =10, 
                      learning_rate: float =1e-3, 
                      device: str ='cpu',
                      mode: str =None, 
                      noise_factor: float =0.2, 
                      mask_ratio: float=0.2,
                      l1_beta: float=1e-3,
                      lr_mode: str = "stepLR",
                      lr_step_size: int = 10,
                      lr_step_gamma: float = 0.1,
                      checkpoint_file: str = None,):
    """The training loop for the mnist autoencoder

    Args:
        model (Autoencoder): The Autoencoder model
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader): Validation dataloader
        num_epochs (int, optional): Number of epochs. Defaults to 10.
        learning_rate (float, optional): Learning rate. Defaults to 1e-3.
        device (str, optional): Could be 'cuda' or 'cpu'. Defaults to 'cpu'.
        mode (str, optional): Can be None or 'denoising' or 'masked' or 'contractive' or 'sparse' or 'l1_sparse'
        noise_factor (float, optional): Default is 0.2 (goes with denoising)
        mask_ratio (float, optional): Default is 0.2 (goes with masked)
        l1_beta (float, optional): Default is 1e-3 (goes with l1_sparse)
        lr_mode (str, optional): Defaults to stepLR, can be reduceLROnPlateau
        lr_step_size (int, optional): Defaults to 10 (goes with stepLR)
        lr_step_gamma (float, optional): Defaults to 0.1 (goes with stepLR)
        checkpoint_file(str, optional): model path to save model file
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')   
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_step_gamma)
    if lr_mode == "reduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
    # initialize results dictionary
    results = {"epoch": [],
               "train loss": [],
               "val loss": [],
               "lr": [],
               }    
    train_batch_size = train_loader.batch_size
    val_batch_size = val_loader.batch_size
    
    if isinstance(model, VariationalAutoencoderMnist):
        mode = 'variational'   
    elif isinstance(model, VqvaeAutoencoderMnist):
        mode = 'vqvae'
        
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        if mode == 'vqvae':
            model.vq_embedding.initialize_codebook_usage_to_zero(latent_grid_size=(1,1))
                    
        for images in train_loader:  # Labels are not needed
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
                reconstructed, z, z_q, indices, codebook_loss = model(images)          
            else :
                reconstructed, hidden = model(images)     
                        
            if mode == 'variational':
                loss = variational_ae_loss_function(reconstructed, images, mu, logvar)
            elif mode == 'vqvae':
                loss = vqvae_loss(x=images, x_recon=reconstructed, codebook_loss=codebook_loss)                
            else:    
                loss = criterion(reconstructed, images)
            
            if mode == 'contractive':
                c_factor = 1e-4
                # Calculate the Jacobian matrix numerically
                jac = torch.autograd.functional.jacobian(model.encoder, images)
                jac_norm = torch.norm(jac, dim=1) ** 2
                contraction_loss = torch.mean(jac_norm)
                loss += c_factor * contraction_loss      
            elif mode == 'sparse':
                sparsity_target=0.05
                beta=1e-3
                mean_activation = torch.mean(hidden, dim=0)
                sparsity_penalty = sparsity_loss(sparsity_target=sparsity_target, mean_activation=mean_activation)
                loss += beta * sparsity_penalty
            elif mode == 'l1_sparse':
                sparsity_penalty = l1_sparsity_loss(encoded=hidden, beta=l1_beta)
                loss += sparsity_penalty
                
            loss.backward()
            optimizer.step()
            
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Current LR: {current_lr}')
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
                    np.save(f"{folder_path}/vqvae_codebook_usage.npy", model.vq_embedding.codebook_usage_per_position)                
                best_val_loss = current_val_loss  
                
def l1_sparsity_loss(encoded: torch.Tensor, beta: float) -> torch.Tensor:
    """L1 sparsity loss

    Args:
        encoded (torch.Tensor): hidden vector
        beta (float): penalty factor

    Returns:
        torch.Tensor: sparsity loss
    """

    # Calculate the number of active neurons for each input (hard sparsity)
    l1_norm = torch.sum(torch.mean(torch.abs(encoded), dim=0)) 
    return beta * l1_norm

def sparsity_loss(sparsity_target: float, mean_activation: torch.Tensor) -> torch.Tensor:
    """The sparsity loss computed through the kl divergence

    Args:
        sparsity_target (float): _description_
        mean_activation (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    return torch.sum(kl_divergence(sparsity_target, mean_activation))   
        
# Sparsity penalty term (KL divergence)
def kl_divergence(p, p_hat):
    """kl div term between specified target p and the hidden vector mean p_hat

    Args:
        p (torch.Tensor): target sparsity
        p_hat (float): hidden vector mean

    Returns:
        torch.Tensor: kl div loss
    """
    return p * torch.log(p / p_hat) + (1 - p) * torch.log((1 - p) / (1 - p_hat))
    
def get_latent_representations(model: nn.Module, 
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
        for images in data_loader:
            images = images.to(device)
            _, z = model(images)
            latent_representations.append(z.cpu().numpy())
    return np.vstack(latent_representations)

def visualize_reconstruction(model: nn.Module, 
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
        images = next(iter(data_loader))
        images = images.to(device)
        reconstructed, _ = model(images)
        
        # Display original and reconstructed images
        _, axs = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(5):
            axs[0, i].imshow(images[i].cpu().squeeze(0), cmap='gray')
            axs[1, i].imshow(reconstructed[i].cpu().squeeze(0), cmap='gray')
        plt.show()
        