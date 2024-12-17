#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

from torch import nn
from svlearn.gans import current_task, Task

#  -------------------------------------------------------------------------------------------------

def get_tree_classifier(num_classes: int = 2, ) -> nn.Module:
    """Get classifier module using CNN layers

    Args:
        num_classes (int, optional): Number of classes. Defaults to 2.

    Returns:
        nn.Module: The classifier model
    """
    model = nn.Sequential(
            # ----------------------------------------------------------------------------------------------------------------------------

            # Convolution Block 1
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0),      # ( B , 3 , 64 , 64 ) ->  ( B , 6 , 60 , 60 )
                nn.BatchNorm2d(num_features=6),                                     
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size=2, stride=2),                                   # ( B , 6 , 60 , 60 ) ->  ( B , 6 , 30 , 30 )

            # ----------------------------------------------------------------------------------------------------------------------------
            # Convolution Block 2
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0),     # ( B , 6 , 30 , 30 ) ->  ( B , 16 , 26 , 26 )
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),                                   # ( B , 16 , 26 , 26 ) ->  ( B , 16 , 13 , 13 )

            # ----------------------------------------------------------------------------------------------------------------------------
            # Convolution Block 3
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4),               # ( B , 16 , 13 , 13 ) ->  ( B , 32 , 10 , 10 )                           
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),                                   # ( B , 32 , 10 , 10 )   ->  ( B , 32 , 5 , 5 ) 

            # ----------------------------------------------------------------------------------------------------------------------------
                nn.Flatten(), # Change from 2D image to 1D tensor to be able to pass inputs to linear layer
            # ----------------------------------------------------------------------------------------------------------------------------
        
            # Linear Block 1
                nn.Linear(in_features=32 * 5 * 5, out_features=180),
                nn.ReLU(),

            # ----------------------------------------------------------------------------------------------------------------------------
            # Linear block 2
                nn.Linear(in_features=180, out_features=84),
                nn.ReLU(),

            # ----------------------------------------------------------------------------------------------------------------------------
                nn.Linear(in_features=84, out_features=num_classes)
            # ----------------------------------------------------------------------------------------------------------------------------
            )
    return model

def get_digits_classifier():
    model = nn.Sequential(
            # ----------------------------------------------------------------------------------------------------------------------------

            # Convolution Block 1
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=7, padding=3),     # ( B , 3 , 28 , 28 ) ->  ( B , 6 , 28 , 28 )
                nn.BatchNorm2d(num_features=6),                                     
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size=2, stride=2),                                  # ( B , 6 , 28 , 28 ) ->  ( B , 6 , 14 , 14 )

            # ----------------------------------------------------------------------------------------------------------------------------
            # Convolution Block 2
                nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2),    # ( B , 6 , 14 , 14 ) ->  ( B , 16 , 14 , 14 )
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),                                  # ( B , 16 , 14 , 14 ) ->  ( B , 16 , 7 , 7 )

            # ----------------------------------------------------------------------------------------------------------------------------
            # Convolution Block 3
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),   # ( B , 16 , 7 , 7 ) ->  ( B , 32 , 7 , 7 )                           
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),

            # ----------------------------------------------------------------------------------------------------------------------------
                nn.Flatten(), # Change from 2D image to 1D tensor to be able to pass inputs to linear layer
            # ----------------------------------------------------------------------------------------------------------------------------
        
            # Linear Block 1
                nn.Linear(in_features=32 * 7 * 7, out_features=180),
                nn.ReLU(),

            # ----------------------------------------------------------------------------------------------------------------------------
            # Linear block 2
                nn.Linear(in_features=180, out_features=84),
                nn.ReLU(),

            # ----------------------------------------------------------------------------------------------------------------------------
                nn.Linear(in_features=84, out_features=10)
            # ----------------------------------------------------------------------------------------------------------------------------
            )
    
    return model

# ----------------------------------------------------------------------------------------------------------------------------

def get_task_specific_classifier():
    if current_task == Task.TREE.value:
        return get_tree_classifier()
    
    elif current_task == Task.MNIST.value:
        return get_digits_classifier()

#  -------------------------------------------------------------------------------------------------