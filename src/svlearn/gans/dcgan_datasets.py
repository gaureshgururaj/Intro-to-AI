#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#  
#   Use is limited to the duration and purpose of the training at SupportVectors.
#  
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

from collections import defaultdict
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision.datasets as dset
import torchvision.transforms as transforms

from svlearn.gans import config, Task, current_task
from svlearn.trees.tree_dataset import TreeDataset
from svlearn.trees.preprocess import Preprocessor

#  -------------------------------------------------------------------------------------------------

random.seed(42)

class BalancedImageDataset(Dataset):
    def __init__(self, original_dataset):
        """
        :param original_dataset: Pre-created ImageFolder dataset with transforms already applied.
        """
        self.original_dataset = original_dataset

        # Separate indices by class
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(original_dataset.imgs):
            class_indices[label].append(idx)

        # Determine the target size to balance classes
        max_class_count = max(len(indices) for indices in class_indices.values())

        # Duplicate samples from minority classes to match the majority class count
        self.balanced_indices = []
        for label, indices in class_indices.items():
            duplicated_indices = indices
            while len(duplicated_indices) < max_class_count:
                duplicated_indices += random.sample(indices, min(len(indices), max_class_count - len(duplicated_indices)))
            self.balanced_indices.extend(duplicated_indices)
        
        # Shuffle indices to randomize order
        random.shuffle(self.balanced_indices)

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        # Access the pre-transformed image and label directly
        return self.original_dataset[self.balanced_indices[idx]]

#  -------------------------------------------------------------------------------------------------

def load_tree_dataset(balanced: bool = False):
    """loads the train dataset for trees classification

    Returns:
        Tuple[Dataset , Dataset]: a tuple of train dataset and test dataset 
    """
    dataroot = config[Task.TREE.value]['data']
    image_size = config[Task.TREE.value]['image_size']
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))  
    if balanced:
        dataset = BalancedImageDataset(dataset)

    return dataset

#  -------------------------------------------------------------------------------------------------

def load_mnist_dataset():
    """loads the train dataset for mnist digits classification

    Returns:
        Tuple[Dataset , Dataset]: a tuple of train dataset and test dataset 
    """
    dataroot = config[Task.MNIST.value]['data']
    transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5,), (0.5,))  # Normalize images to [-1, 1]
            ])
            # Load the training dataset
    return dset.MNIST(root=dataroot, train=True, download=True, transform=transform)

#  -------------------------------------------------------------------------------------------------

def load_tree_train_test_split() -> Tuple[Dataset , Dataset]:
    """loads the train and test dataset for tree classification

    Returns:
        Tuple[Dataset , Dataset]: a tuple of train dataset and test dataset 
    """
    data_dir = config[Task.TREE.value]['data']
    preprocessor = Preprocessor()
    train_df, val_df, _ = preprocessor.preprocess(data_dir)
    train_transform = v2.Compose([
        v2.ToImage(), 
        v2.RandomResizedCrop(64 , scale = (0.5, 1)), # Randomly crop and resize to 64 X 64
        v2.RandomHorizontalFlip(p=0.5),       # Randomly flip the image horizontally with a 50% chance
        v2.ColorJitter(brightness=0.4 , contrast=0.4, saturation=0.4), # randomly change the brightness , contrast and saturation of images
        v2.ToDtype(torch.float32, scale=True), # ensure the tensor is of float datatype
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize tensor 
        
    ])

    test_transform = v2.Compose([
        v2.ToImage(), 
        v2.Resize(size=(64 , 64)),  # resize all images to a standard size suitable for the cnn model
        v2.ToDtype(torch.float32, scale=True), # ensure te tensor is of float datatype
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize tensor 
    ])

    train_dataset = TreeDataset(train_df, transform=train_transform)
    test_dataset = TreeDataset(val_df, transform=test_transform)

    return train_dataset , test_dataset

#  -------------------------------------------------------------------------------------------------
def load_mnist_train_test_split() -> Tuple[Dataset , Dataset]:
    """loads the train and test dataset for mnist digits classification

    Returns:
        Tuple[Dataset , Dataset]: a tuple of train dataset and test dataset 
    """
    data_dir = config[Task.MNIST.value]['data']
    transform = v2.Compose([
    v2.ToTensor(),  # Convert images to PyTorch tensors
    v2.Normalize((0.5,), (0.5,))  # Normalize images to [-1, 1]
    ])
    train_dataset =  dset.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = dset.MNIST(root=data_dir, train=False, download=True, transform=transform)

    return train_dataset , test_dataset

#  -------------------------------------------------------------------------------------------------

def load_task_specific_dataset(balanced: bool = False, test: bool=False):
    if current_task == Task.TREE.value:
        return load_tree_dataset(balanced)
    elif current_task ==  Task.MNIST.value:
        return load_mnist_dataset()
    else:
        raise ValueError(f"The dataset cannot be loaded for the task provided - {current_task}.")

#  -------------------------------------------------------------------------------------------------

def load_task_specific_tain_test_split():
    """loads the train and test dataset depending on the current task provided.

    Args:
        current_task (str): tree-classification or mnist-classification
    """

    if current_task == Task.TREE.value:
        return load_tree_train_test_split()

    elif current_task ==  Task.MNIST.value:
        return load_mnist_train_test_split()
    else:
        raise ValueError(f"The dataset cannot be loaded for the task provided - {current_task}.")
#  -------------------------------------------------------------------------------------------------
