#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#
from typing import List
import torch
import torch.nn as nn
from torchvision.transforms import v2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from svlearn.auto_encoders.resnet_vqvae_auto_encoder import ResNetVQVAEAutoEncoder
from svlearn.auto_encoders.resnet_vqvae_auto_encoder_modified import ResNetVQVAEAutoEncoderModified
from svlearn.auto_encoders.variational_auto_encoder import VariationalAutoencoder
from svlearn.auto_encoders.variational_auto_encoder_mnist import VariationalAutoencoderMnist
from svlearn.auto_encoders.vqvae_auto_encoder_mnist import VqvaeAutoencoderMnist
from svlearn.train.visualization_utils import denormalize

# Generate samples using GMM
def sample_from_gmm(gmm: GaussianMixture, 
                    num_samples: int) -> np.ndarray :
    """Generate samples from the Gaussian Mixture Model

    Args:
        gmm (GaussianMixture): The passed in Gaussian mixture model
        num_samples (int): The number of samples to generate

    Returns:
        np.ndarray: The sample numpy vectors (as many as num_smaples)
    """
    latent_samples = gmm.sample(num_samples)[0]
    return latent_samples

def generate_images_from_latent(model: nn.Module, 
                                latent_samples: np.ndarray, 
                                device: str='cpu') -> torch.Tensor:
    """Generate images from the latent sample vectors

    Args:
        model (nn.Module): The trained Autoencoder - should have encoder and decoder defined
        latent_samples (np.ndarray): The sample hidden vectors
        device (str, optional): Could be 'cuda' or 'cpu'. Defaults to 'cpu'.

    Returns:
        torch.Tensor: The decoded images
    """
    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        latent_samples = torch.from_numpy(latent_samples).float().to(device)
        generated_images = model.decoder(latent_samples)  # Use the decoder part of the autoencoder
    return generated_images

def convert(image):
    """Converts the image back to the unnormalized version (required for the 3-color images)

    Args:
        image (_type_): normalized image

    Returns:
        _type_: un-normalized image
    """

    # Denormalize the image (use the same mean and std as used in transforms.Normalize)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    denorm_img = denormalize(image, mean, std)
    
    # Convert the tensor image back to PIL image for display
    return v2.ToPILImage()(denorm_img)
    
def interpolate(model: nn.Module, 
                image1: torch.Tensor, 
                image2: torch.Tensor, 
                num_steps: int =10, 
                device: str ='cpu') -> List[np.ndarray]:
    """Interpolate between two input images passed in as torch Tensors by returning
    a series of intermediate generated images from the trained autoencoder

    Args:
        model (nn.Module): The trained autoencoder should contain the decoder and encoder
        image1 (torch.Tensor): The first input image
        image2 (torch.Tensor): The second input image
        num_steps (int, optional): The number of intermediate images. Defaults to 10.
        device (str, optional): Could be 'cuda' or 'cpu'. Defaults to 'cpu'.

    Returns:
        List[np.ndarray]: The generated images starting from reconstructed image1 to reconstructed image2 by
        following a linear interpolation through the hidden vectors.
    """
    model.eval()
    with torch.no_grad():
        if isinstance(model, VariationalAutoencoder):
            _, z1, _, _ = model(image1.unsqueeze(0).to(device))
            _, z2, _, _ = model(image2.unsqueeze(0).to(device)) 
        elif isinstance(model, ResNetVQVAEAutoEncoder) or isinstance(model, ResNetVQVAEAutoEncoderModified):
            _, z1, _, _, _ = model(image1.unsqueeze(0).to(device))
            _, z2, _, _, _ = model(image2.unsqueeze(0).to(device))             
        else: 
            _, z1 = model(image1.unsqueeze(0).to(device))
            _, z2 = model(image2.unsqueeze(0).to(device))
        
        interpolation_steps = torch.linspace(0, 1, num_steps)
        interpolated_images = []
        
        for alpha in interpolation_steps:
            z_interpolated = (1 - alpha) * z1 + alpha * z2
            if isinstance(model, ResNetVQVAEAutoEncoder) or isinstance(model, ResNetVQVAEAutoEncoderModified):
                z_interpolated, _ = model.vq_embedding(z_interpolated)
                
            interpolated_image = model.decoder(z_interpolated)
            interpolated_images.append(interpolated_image.squeeze().cpu().numpy())
    
    return interpolated_images

def visualize_generated_images(generated_images: List[np.ndarray],
                               is_color: bool = False):
    """Visualize the list of generated images passed in

    Args:
        generated_images (List[np.ndarray]): A list of generated images
        is_color (bool): True if it is a 3-channel image, else False
    """
    _, axs = plt.subplots(1, len(generated_images), figsize=(15, 4))
    for i, img in enumerate(generated_images):
        if is_color:
            axs[i].imshow(convert(img.cpu()))
        else:
            axs[i].imshow(img.cpu().squeeze(0), cmap='gray')
        axs[i].axis('off')
    plt.show()

# Function to interpolate between two latent vectors
def interpolate_latents(z1: torch.Tensor, 
                        z2: torch.Tensor, 
                        num_steps: int=10) -> torch.Tensor:
    """Interpolates between z1 and z2

    Args:
        z1 (torch.Tensor): Starting hidden vector
        z2 (torch.Tensor): Ending hidden vector
        num_steps (int, optional): The number of intermediate linear interpolation steps. Defaults to 10.

    Returns:
        torch.Tensor: The stacked hidden vectors starting from first to last
    """
    alphas = np.linspace(0, 1, num_steps)
    z1 = z1.squeeze(0)  
    z2 = z2.squeeze(0) 
    interpolated_z = [(1 - alpha) * z1 + alpha * z2 for alpha in alphas]
    return torch.stack(interpolated_z).to(z1.device)

# Function to visualize interpolated images
def visualize_interpolations(model: nn.Module, 
                             image1: torch.Tensor, 
                             image2: torch.Tensor, 
                             num_steps: int =10, 
                             is_color: bool = False,
                             device: str ='cpu'):
    """Visualizes the interpolated images between a starting and ending image

    Args:
        model (nn.Module): The trained autoencoder should have encoder and decoder
        image1 (torch.Tensor): The first image
        image2 (torch.Tensor): The second image
        num_steps (int, optional): The number of intermediate steps. Defaults to 10.
        is_color (bool): True if 3-channel image, else False
        device (str, optional): Could be 'cuda' or 'cpu'. Defaults to 'cpu'.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Get latent representations of the two images
        if isinstance(model, VariationalAutoencoder) or isinstance(model, VariationalAutoencoderMnist):
            _, z1, _, _ = model(image1.unsqueeze(0).to(device))  # Single batch with 1st image
            _, z2, _, _ = model(image2.unsqueeze(0).to(device))  # Single batch with 2nd image      
        elif isinstance(model, VqvaeAutoencoderMnist) or isinstance(model, ResNetVQVAEAutoEncoder) or isinstance(model, ResNetVQVAEAutoEncoderModified):
            _, z1, _, _, _ = model(image1.unsqueeze(0).to(device))  # Single batch with 1st image
            _, z2, _, _, _ = model(image2.unsqueeze(0).to(device))  # Single batch with 2nd image                
        else :      
            _, z1 = model(image1.unsqueeze(0).to(device))  # Single batch with 1st image
            _, z2 = model(image2.unsqueeze(0).to(device))  # Single batch with 2nd image
        # Perform interpolation
        interpolated_latents = interpolate_latents(z1, z2, num_steps)
        if isinstance(model, VqvaeAutoencoderMnist):
            interpolated_latents, _, _ = model.vq_embedding(interpolated_latents)        
        # Decode interpolated latents into images
        generated_images = model.decoder(interpolated_latents)
    
    # Visualize the interpolation
    _, axs = plt.subplots(1, num_steps, figsize=(15, 3))
    for i in range(num_steps):
        if is_color:
            axs[i].imshow(convert(generated_images[i].cpu()))
        else:
            axs[i].imshow(generated_images[i].cpu().squeeze(0), cmap='gray')
        axs[i].axis('off')
    plt.show()

