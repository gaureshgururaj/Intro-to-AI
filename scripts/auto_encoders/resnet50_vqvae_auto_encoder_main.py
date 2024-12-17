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
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from svlearn.config.configuration import ConfigurationMixin
from svlearn.trees.preprocess import Preprocessor
from svlearn.trees.tree_dataset import TreeDataset
from svlearn.auto_encoders.auto_encoder_multi_channel_util import train_autoencoder
from svlearn.auto_encoders.resnet_vqvae_auto_encoder import ResNetVQVAEAutoEncoder

if __name__ == "__main__":
    config = ConfigurationMixin().load_config()
    data_dir = config['tree-classification']['data']
    results_dir = config['tree-classification']['results']
    
    preprocessor = Preprocessor()
    train_df, val_df, label_encoder = preprocessor.preprocess(data_dir)
    
    train_transform = v2.Compose([
        v2.ToImage(), 
        v2.RandomResizedCrop(224 , scale = (0.5, 1)), # Randomly crop and resize to 224x224
        v2.RandomHorizontalFlip(p=0.5),       # Randomly flip the image horizontally with a 50% chance
        v2.ColorJitter(brightness=0.4 , contrast=0.4, saturation=0.4), # randomly change the brightness , contrast and saturation of images
        v2.ToDtype(torch.float32, scale=True), # ensure te tensor is of float datatype
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize tensor 
        
    ])

    test_transform = v2.Compose([
        v2.ToImage(), 
        v2.Resize(size=(224 , 224)),  # resize all images to a standard size suitable for the cnn model
        v2.ToDtype(torch.float32, scale=True), # ensure te tensor is of float datatype
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize tensor 
    ])
    
    train_dataset = TreeDataset(train_df, transform=train_transform)
    val_dataset = TreeDataset(val_df, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False) 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetVQVAEAutoEncoder().to(device)
    
    checkpoint_file = f'{results_dir}/trees_resnet50_vqvae_autoencoder.pt'
    
    starting_learning_rate = 1e-3
    
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        results = checkpoint['results']
        if 'lr' in results:
            starting_learning_rate = results['lr'][-1]
            
    train_autoencoder(model=model, 
                      train_loader=train_loader, 
                      val_loader=val_loader, 
                      num_epochs=50, 
                      device=device, 
                      lr_mode="reduceLROnPlateau", 
                      checkpoint_file=checkpoint_file,
                      learning_rate=starting_learning_rate)