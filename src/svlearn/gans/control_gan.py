#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import torch
from torch import nn

#  -------------------------------------------------------------------------------------------------
def control_generation(generator: nn.Module, 
                                        classifier: nn.Module, 
                                        z: torch.Tensor,
                                        target_class: int, 
                                        labels: torch.Tensor = None,
                                        alpha: float =0.1, 
                                        steps: int =10) -> torch.Tensor:
    """Use grad wrt input to move generation towards one of the classes of the classifier

    Args:
        generator (nn.Module): trained generator from GAN
        classifier (nn.Module): trained classifier 
        z (torch.Tensor): input noise
        target_class (int): target class towards which generation needs to move
        alpha (float, optional): factor to move z along gradient. Defaults to 0.1.
        steps (int, optional): number of steps by which to move z. Defaults to 10.

    Returns:
        torch.Tensor: generated image
    """
    z = z.clone().detach().requires_grad_(True)  # Clone and set requires_grad
    generated_images = []
    for _ in range(steps):
        
        generated_image: torch.Tensor = generator(z)  # Generate an image
        image = generated_image.clone().cpu().detach().squeeze().numpy()
        generated_images.append(image)
        # Get classifier logit for the target class
        logits = classifier(generated_image)[:, target_class]
        
        # Backpropagate to compute gradient with respect to z
        logits.sum().backward()
        with torch.no_grad():
            z += alpha * z.grad  # Update z to steer towards target class
            z.grad.zero_()       # Clear gradients for the next step

    return generated_images  # Return the modified generated image

#  -------------------------------------------------------------------------------------------------
