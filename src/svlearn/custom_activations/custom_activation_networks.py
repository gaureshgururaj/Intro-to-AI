#  Copyright (c) 2024.  SupportVectors AI Lab
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: SupportVectors AI Team
#
import torch.nn as nn

from svlearn.custom_activations.custom_activation_functions import hexpo, soft_clipping, soft_root_sign, softsign

class SoftClipping(nn.Module):
    def __init__(self, a:float = 1.0):
        super(SoftClipping, self).__init__()
        self.a = a

    def forward(self, x):
        return soft_clipping(x, self.a)
    
    def __repr__(self):
        return f"SoftClipping(a={self.a!r})"    
    
class SoftRootSign(nn.Module):
    def __init__(self, a:float = 2.0, b:float = 3.0):
        super(SoftRootSign, self).__init__()
        self.a = a 
        self.b = b

    def forward(self, x):
        return soft_root_sign(x, self.a, self.b)

    def __repr__(self):
        return f"SoftRootSign(a={self.a!r}, b={self.b!r})"      
    
class Hexpo(nn.Module):
    def __init__(self, a:float = 1.0, b:float = 1.0, c:float = 1.0, d:float = 1.0):
        super(Hexpo, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def forward(self, x):
        return hexpo(x, self.a, self.b, self.c, self.d)

    def __repr__(self):
        return f"Hexpo(a={self.a!r}, b={self.b!r}, c={self.c!r}, d={self.d!r})"      
        
class SoftSign(nn.Module):
    def __init__(self):
        super(SoftSign, self).__init__()

    def forward(self, x):
        return softsign(x)
    
    def __repr__(self):
        return "SoftSign()"