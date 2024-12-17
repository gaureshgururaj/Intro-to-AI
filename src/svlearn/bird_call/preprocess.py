#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import os
import librosa
import numpy as np 
from svlearn.common.utils import ensure_directory , directory_readable
from svlearn.config.configuration import ConfigurationMixin
import pandas as pd
from tqdm import tqdm_notebook
import joblib
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", message="Xing stream size off by more than 1%")

#  -------------------------------------------------------------------------------------------------

class Preprocessor():
    def __init__(self) -> None:
        """Preprocesses audio files for CNN training. 
        """
        self.duration = 10.0
        self.sample_rate = 22050
        self.target_length = self.duration * self.sample_rate

        self.n_fft=1024
        self.hop_length=512
        self.n_mels=128
        self.names_to_labels = {}
        self.labels_to_names = {}
        self.processed_file_name_length = 10

#  -------------------------------------------------------------------------------------------------

    def load_audio(self, audio_file_path: str ) -> np.ndarray:
        """loads audio file from specified path and 
        applies some transformations to bring all audio samples to a common duration and sample rate

        Args:
            audio_file_path (str): full path to audio file

        Returns:
            np.ndarray: audio as numpy array
        """
        try:
            audio, _ = librosa.load( audio_file_path , duration=self.duration, 
                                    mono=True, sr = self.sample_rate)

            original_length = len(audio)
    
            if original_length < self.target_length:
                # Padding if the audio is shorter than the target length
                padding = int(self.target_length - original_length)
                audio = np.pad(audio, (0, padding), mode='constant')
            
            return audio
        except Exception:
            print(f'error loading {audio_file_path}')
            raise

#  -------------------------------------------------------------------------------------------------

    def convert_to_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """converts audio to mel-spectrogram

        Args:
            audio (np.ndarray): audio as numpy array

        Returns:
            np.ndarray: melspectrogram
        """
        spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        # Convert to log scale (for better CNN input performance)
        mel_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # adds a dimension to denote the channel i.e. From (128 , 431) to (1, 128 , 431)
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)

        return mel_spectrogram

#  -------------------------------------------------------------------------------------------------

    def process_audio_file(self, audio_file: str, raw_class_dir: str , processed_class_dir: str,
                            label: int , idx: int , metadata: Dict) -> Tuple[Dict , int]:
        """processes an audio sample from raw mp3 / wav format to mel-spectrogram and saves in .npy
        format 

        Args:
            audio_file (str): audio file name
            raw_class_dir (str): path to parent directory of audio file
            processed_class_dir (str): path to directory where the spectrogram is to be saved
            label (int): label associated with the sample
            idx (int): index of audio sample
            metadata (Dict): dictionary that associates raw files, processed files and labels

        Returns:
            Tuple[Dict , int]: Updated metadata dictionary and index
        """

        # print(f'processing {audio_file}')
        if not (audio_file.endswith(".wav") or audio_file.endswith(".mp3")):
            return metadata , idx
        
        input_file_path = os.path.join(raw_class_dir , audio_file)
        output_file_path = f"{processed_class_dir}/{label}_{'0' * (self.processed_file_name_length - len(str(idx)))}{idx}.npy"

        audio = self.load_audio(input_file_path)
        mel_spectrogram = self.convert_to_mel_spectrogram(audio)

        with open(output_file_path, 'wb') as f:
            np.save(f, mel_spectrogram)

        metadata.append({"id": idx , "audio_path": input_file_path , "spectrogram_path": output_file_path , "label": label})
        
        return metadata , idx+1

#  -------------------------------------------------------------------------------------------------

    def transform_audio(self, audio_file_path: str , bird_name: str) -> Tuple[np.ndarray , int]:
        """transforms a single audio sample (specifically during inference)

        Args:
            audio_file_path (str): path to audiofile
            bird_name (str): one of the bird classes

        Raises:
            ValueError: if the bird name provided is not fitted

        Returns:
            Tuple[np.ndarray , int]: spectrgram as a numpy array along with the encoded label
        """
        

        if bird_name not in self.names_to_labels.keys():
            raise ValueError(f"Bird name should be one of {self.names_to_labels.keys()}")

        audio = self.load_audio(audio_file_path)
        mel_spectrogram = self.convert_to_mel_spectrogram(audio)

        return mel_spectrogram , self.names_to_labels[bird_name]

#  -------------------------------------------------------------------------------------------------

    def fit(self, data_dir: str) -> str:
        """encodes the class labels

        Args:
            data_dir (str): path to the root data directory

        Returns:
            str: path to raw directory
        """
        raw_dir = f"{data_dir}/raw"
        directory_readable(raw_dir)

        classes = os.listdir(raw_dir)

        for label , bird_class in enumerate(classes):
            self.names_to_labels[bird_class] = label
            self.labels_to_names[label] = bird_class
        
        return raw_dir

#  -------------------------------------------------------------------------------------------------

    def fit_transform(self , data_dir: str) -> None:
        """fits the data and transforms all raw audio files to mel spectrograms

        Args:
            data_dir (str): path to the root data directory
        """

        raw_dir = self.fit(data_dir)

        processed_dir = f"{data_dir}/preprocessed"
        ensure_directory(processed_dir)

        idx = 0
        data = []

        for bird_class , label in tqdm_notebook(self.names_to_labels.items(), desc='classes'):

            raw_class_dir = os.path.join(raw_dir, bird_class)
            directory_readable(raw_class_dir)
            processed_class_dir = os.path.join(processed_dir, bird_class)
            ensure_directory(processed_class_dir)

            for audio_file in tqdm_notebook(os.listdir(raw_class_dir), desc='sample'):
                data , idx = self.process_audio_file(audio_file, raw_class_dir, processed_class_dir, label, idx , data)

        # save metadata
        pd.DataFrame.from_records(data).to_json(f"{data_dir}/metadata.json")    

#  -------------------------------------------------------------------------------------------------

    def preprocess_for_vit(self , data_dir: str):
        raw_dir = self.fit(data_dir)
        data_dict = {'label': [], 'path': []}

        for bird_class , label in tqdm_notebook(self.names_to_labels.items(), desc='classes'):

            raw_class_dir = os.path.join(raw_dir, bird_class)
            directory_readable(raw_class_dir)

            for audio_file in tqdm_notebook(os.listdir(raw_class_dir), desc='sample'):
                data_dict['path'].append(f"{raw_class_dir}/{audio_file}")
                data_dict['label'].append(label)
        
        data_df = pd.DataFrame(data_dict)
        train_df , test_df = train_test_split(data_df, test_size=0.2)

        return train_df , test_df, self.names_to_labels


if __name__ == "__main__":
    config = ConfigurationMixin().load_config()
    data_dir = config['bird-call-classification']['data']
    results_dir = config['bird-call-classification']['results']

    preprocessor = Preprocessor()
    preprocessor.fit_transform(data_dir)
    joblib.dump(preprocessor, f"{results_dir}/preprocessor.joblib")


