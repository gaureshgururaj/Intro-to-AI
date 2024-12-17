#  Copyright (c) 2024.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: SupportVectors AI Team
#

import torch

### Soft Clipping
def soft_clipping(x: torch.Tensor , a=1.0) -> torch.Tensor :
    """soft clipping function defined as 1/a * ln ([1+exp(a*x)]/[1-exp(a*(x-1))])

    Args:
        x (torch.Tensor): The x value(s) passed in
        a (float, optional): The soft clipping parameter. Defaults to 1.0.
    Returns:
        torch.Tensor: The y value(s) returned by the soft clipping function
    """
    return (1/a) * torch.log((1 + torch.exp(a * x))/(1 + torch.exp(a * (x - 1))))

### Soft Root Sign
def soft_root_sign(x: torch.Tensor , a=2.0, b=3.0) -> torch.Tensor :
    """soft root sign function defined as x / (x/a + exp(-x/b))
    Args:
        x (torch.Tensor): The x value(s) passed in
        a (float, optional): The a parameter defaults to 2.0.
        b (float, optional): The b parameter defaults to 3.0
    Returns:
        torch.Tensor: The y value(s) returned by the soft root sign function
    """    
    return (x / (x/a + torch.exp(-x/b)))

### Hexpo
def hexpo(x: torch.Tensor , a=1.0, b=1.0, c=1.0, d=1.0) -> torch.Tensor :
    """hexpo function defined as -a (exp(-x/b) -1), for x >= 0; 
                                  c (exp(-x/d) -1), for x < 0;
    Args:
        x (torch.Tensor): The x value(s) passed in
        a (float, optional): The a parameter defaults to 1.0.
        b (float, optional): The b parameter defaults to 1.0
        c (float, optional): The c parameter defaults to 1.0.
        d (float, optional): The d parameter defaults to 1.0        
    Returns:
        torch.Tensor: The y value(s) returned by the hexpo function
    """
    y = torch.where(x >= 0,
                 -a * (torch.exp(-x/b) - 1),
                 c * (torch.exp(-x/d) - 1))
    return y

### Softsign
def softsign(x: torch.Tensor ) -> torch.Tensor :
    """softsign function defined as x / (1+ |x|)
    Args:
        x (torch.Tensor): The x value(s) passed in      
    Returns:
        torch.Tensor: The y value(s) returned by the softsign function
    """
    return x / ( 1 + torch.abs(x))