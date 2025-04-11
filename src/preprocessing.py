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
        
        # Set the number of channels
        self.channel = self.y.shape[0]
        
        # Initialize y_processed as a copy of y
        self.y_processed = np.copy(self.y)

    def resample(self):
        """
        Resample the audio to the target sample rate, handling multi-channel audio.

        If the audio is multi-channel, each channel is resampled individually.
        """
        if self.sr != self.target_sample_rate:
            y_resampled = []
            for channel in self.y_processed:
                y_resampled.append(librosa.resample(channel, orig_sr=self.sr, target_sr=self.target_sample_rate))
            self.y_processed = np.array(y_resampled)
            self.sr = self.target_sample_rate

    def pad_or_truncate(self, target_length_ms):
        """
        Pad or truncate the audio signal to a target length in milliseconds, keeping the center of the sound.
    
        Args:
            target_length_ms (int): Target length in milliseconds.
        """
        # Convert target length from milliseconds to samples
        target_length_samples = int((target_length_ms / 1000) * self.sr)
    
        # Get the current length of the audio signal
        current_length = len(self.y_processed[0])
    
        if current_length < target_length_samples:
            # Padding: Add symmetric padding to the left and right
            padding = target_length_samples - current_length
            left_padding = padding // 2
            right_padding = padding - left_padding
            y_padded = []
            for channel in self.y_processed:
                y_padded.append(np.pad(channel, (left_padding, right_padding), mode='constant'))
            self.y_processed = np.array(y_padded)
        else:
            # Truncation: Keep the center of the audio signal
            start = (current_length - target_length_samples) // 2
            end = start + target_length_samples
            y_truncated = []
            for channel in self.y_processed:
                y_truncated.append(channel[start:end])
            self.y_processed = np.array(y_truncated)

    def rechanneling(self):
        """
        Convert the audio signal to the target number of channels (self.channel_target).

        """
        if self.channel_target == 1 and self.channel > 1:  # Multi to mono
                # Downmix all channels to mono by averaging the channels
                self.y_processed = np.mean(self.y_processed, axis=0, keepdims=True)
        elif self.channel_target == 2:  # Convert to stereo
            if self.channel == 1:  # Mono to stereo
                # Duplicate the mono channel to create stereo
                self.y_processed = np.stack([self.y_processed[0], self.y_processed[0]], axis=0)

    
