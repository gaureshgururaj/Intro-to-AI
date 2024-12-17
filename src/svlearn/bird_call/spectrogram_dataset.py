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
from torch.utils.data import Dataset
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  -------------------------------------------------------------------------------------------------

# Custom Dataset Class for the spectrograms
class SpectrogramDataset(Dataset):
    def __init__(self, data_dir):
        """A custom torch dataset that retrieves spectrograms of audio sounds.

        Args:
            data_dir (_type_): root directory containing the index file
        """
        # load the metadata dataframe
        with open(f"{data_dir}/metadata.json" , "r") as f:
            index_dict = json.load(f)

        self.index_df = pd.DataFrame.from_dict(index_dict)
        # list of paths to spectrogram files
        self.X_paths = self.index_df['spectrogram_path'].to_list()
        # list of corresponding labels
        self.labels = self.index_df['label'].to_list()

        self.audio_paths = self.index_df['audio_path'].to_list()
            
    def __len__(self):
        return len(self.X_paths)


    def __getitem__(self, idx):
        sample_path = self.X_paths[idx]
        label = self.labels[idx]

        with open(sample_path , "rb") as f:
            spectrogram = np.load(f)

        return torch.tensor(spectrogram), torch.tensor(label, dtype=torch.long)
    
    def get_audio_path(self, idx: int) -> str:
        """Get the audio path of a sample

        Args:
            idx (int): sample index

        Returns:
            str: audio filepath
        """
        return self.audio_paths[idx]
    
    def plot_spectrogram(self, idx: int, title: str=None,) -> None:
        """plots the spectrogram of the audio at provided index of the dataset

        Args:
            idx (int): sample index
            title (str, optional): title of plot. Defaults to None.
        """
        spectrogram , _ = self.__getitem__(idx)
        fig, axs = plt.subplots(1, 1)
        axs.set_title(title or 'Spectrogram (db) ')
        axs.set_ylabel('freq_bin')
        axs.set_xlabel('frame')
        im = axs.imshow(spectrogram[0], origin='lower', aspect='auto')
        fig.colorbar(im, ax=axs)
        plt.show(block=False)

#  -------------------------------------------------------------------------------------------------