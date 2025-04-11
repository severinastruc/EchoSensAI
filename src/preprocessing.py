import os

import librosa
import numpy as np


class AudioProcess:
    """
    A class for loading and preprocessing a single audio file (mono or multi channels).
    """

    def __init__(self, file_path, target_sample_rate=44100, target_channel = 2):
        """
        Initialize the Audio class.

        Args:
            file_path (str): Path to the audio file.
            target_sample_rate (int): The target sample rate for resampling.
            target_channel (int): The target number of channel (mono, stereo, 5.1) for rechanneling. 
        """
        self.file_path = file_path
        # Target properties
        self.target_sample_rate = target_sample_rate
        self.target_channel = target_channel
        # Signal properties
        self.sr = None
        self.channel = None
        # Signal value
        self.y = None
        self.y_processed = np.copy(self.y)

    def load(self):
        """
        Load the audio file and ensure the audio signal is in the shape (n, N),
        where n is the number of channels and N is the number of samples.
        """
        self.y, self.sr = librosa.load(self.file_path, sr=None, mono=False)

        # If the audio is mono (1D array), reshape it to (1, N)
        if self.y.ndim == 1:
            self.y = np.expand_dims(self.y, axis=0)

        # Initialize y_processed as a copy of y
        self.y_processed = np.copy(self.y)

        # Set the number of channels
        self.get_channel()

    def get_channel(self):
        if self.y.ndim == 1:  # Mono audio
            self.channel = 1
        else:  # Multi-channel audio
            self.channel = self.y.shape[0]

    def resample(self):
        """
        Resample the audio to the target sample rate, handling multi-channel audio.

        If the audio is multi-channel, each channel is resampled individually.
        """
        if self.sr != self.target_sample_rate:
            y_resampled = []
            for i, channel in enumerate(self.y_processed):
                y_resampled.append(librosa.resample(channel, orig_sr=self.sr, target_sr=self.target_sample_rate))
            self.y_processed = np.array(y_resampled)
            self.sr = self.target_sample_rate


