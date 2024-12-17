import os
import librosa  # Library for audio processing
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from svlearn.config.configuration import ConfigurationMixin

# Step 1: Audio Preprocessing (Handling varying lengths by padding/truncating)
def load_audio_file(file_path, sample_rate=22050, duration=5):
    """
    Load an audio file, resample it to a given sample rate, and ensure a fixed duration.
    If the audio is shorter than the duration, pad it. If it's longer, truncate it.

    Args:
        file_path (str): Path to the audio file.
        sample_rate (int): Target sample rate for the audio.
        duration (int): Target duration in seconds.

    Returns:
        np.array: The processed audio signal.
    """
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    target_length = duration * sample_rate
    
    if len(audio) < target_length:
        # Padding if the audio is shorter than the target length
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    else:
        # Truncating if the audio is longer than the target length
        audio = audio[:target_length]
    
    return audio

# Step 2: Convert audio to a spectrogram
def audio_to_spectrogram(audio, sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128):
    """
    Convert an audio signal to a mel-spectrogram suitable for CNN input.

    Args:
        audio (np.array): The audio signal.
        sample_rate (int): Sample rate of the audio.
        n_fft (int): Number of FFT components.
        hop_length (int): Number of samples between successive frames.
        n_mels (int): Number of Mel bands to generate.

    Returns:
        np.array: The mel-spectrogram (2D array).
    """
    spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    # Convert to log scale (for better CNN input performance)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram

# Step 3: Custom Dataset Class
class BirdAudioDataset(Dataset):
    """
    Custom Dataset class for bird call classification. Loads audio files, converts them to spectrograms, 
    and returns the spectrogram along with the label.
    """
    def __init__(self, audio_dir, transform=None, sample_rate=22050, duration=5):
        """
        Args:
            audio_dir (str): Directory with bird call subfolders (one for each class).
            transform (callable, optional): Optional transform to be applied on a sample.
            sample_rate (int): Sample rate for audio processing.
            duration (int): Fixed duration for all audio samples (in seconds).
        """
        self.audio_dir = audio_dir
        self.transform = transform
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_paths = []
        self.labels = []
        self._load_audio_paths_and_labels()

    def _load_audio_paths_and_labels(self):
        """
        Helper function to load all audio paths and corresponding labels from directory structure.
        Assumes each subfolder in `audio_dir` corresponds to a separate class.
        """
        for label, bird_class in enumerate(os.listdir(self.audio_dir)):
            bird_folder = os.path.join(self.audio_dir, bird_class)
            if os.path.isdir(bird_folder):
                for audio_file in os.listdir(bird_folder):
                    if audio_file.endswith(".wav") or audio_file.endswith(".mp3"):
                        self.audio_paths.append(os.path.join(bird_folder, audio_file))
                        self.labels.append(label)  # 0 for first class, 1 for second

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (spectrogram, label) where spectrogram is the transformed audio sample and label is the class.
        """
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        # Load and process the audio file
        audio = load_audio_file(audio_path, sample_rate=self.sample_rate, duration=self.duration)
        spectrogram = audio_to_spectrogram(audio, sample_rate=self.sample_rate)

        # Optional transform (e.g., normalization or data augmentation)
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        # Add a channel dimension for CNN input (1 channel for grayscale)
        spectrogram = np.expand_dims(spectrogram, axis=0)

        return torch.tensor(spectrogram, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Sample usage (testing the dataset):
if __name__ == "__main__":
    config = ConfigurationMixin().load_config()
    audio_dir = config['bird-call-classification']['data']
    
    dataset = BirdAudioDataset(audio_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Plotting a sample spectrogram to visualize
    spectrogram, label = dataset[0]
    plt.imshow(spectrogram[0], cmap='hot', origin='lower')
    plt.title(f"Spectrogram for class: {label}")
    plt.colorbar()
    plt.show()
