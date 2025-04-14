import numpy as np
import librosa
import tensorflow as tf
from joblib import Parallel, delayed
import os

from src.logger import logger_prep  # Import the preprocessing logger


class AudioProcess:
    """
    A class for loading and preprocessing a single audio file (mono or multi channels).
    """

    def __init__(self, file_path, target_sample_rate=44100, target_channel = 2, target_length_ms=None):
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
        self.target_length_ms = target_length_ms
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
        logger_prep.info(f"Loading audio file: {self.file_path}")
        try:
            self.y, self.sr = librosa.load(self.file_path, sr=None, mono=False)
            logger_prep.debug(f"Loaded audio file with sample rate: {self.sr}, channels: {self.channel}")
        except Exception as e:
            logger_prep.error(f"Error loading file {self.file_path}: {e}")
            raise

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
            logger_prep.info(f"Resampling audio from {self.sr} Hz to {self.target_sample_rate} Hz")
            try:
                y_resampled = []
                for channel in self.y_processed:
                    y_resampled.append(librosa.resample(channel, orig_sr=self.sr, target_sr=self.target_sample_rate))
                self.y_processed = np.array(y_resampled)
                self.sr = self.target_sample_rate
                logger_prep.info("Resampling completed successfully")
            except Exception as e:
                logger_prep.error(f"Error during resampling: {e}")
                raise

    def pad_or_truncate(self, target_length_ms):
        """
        Pad or truncate the audio signal to a target length in milliseconds, keeping the center of the sound.
    
        Args:
            target_length_ms (int): Target length in milliseconds.
        """
        logger_prep.info(f"Padding or truncating audio to {target_length_ms} ms")
        try:
            # Convert target length from milliseconds to samples
            target_length_samples = int((target_length_ms / 1000) * self.sr)
        
            # Get the current length of the audio signal
            current_length = len(self.y_processed[0])
            logger_prep.debug(f"Current length:  {current_length}s")
            if current_length < target_length_samples:
                # Padding: Add symmetric padding to the left and right
                logger_prep.debug(f"Padding start")
                padding = target_length_samples - current_length
                left_padding = padding // 2
                right_padding = padding - left_padding
                y_padded = []
                for channel in self.y_processed:
                    y_padded.append(np.pad(channel, (left_padding, right_padding), mode='constant'))
                self.y_processed = np.array(y_padded)
                logger_prep.debug(f"Padding end")
            else:
                # Truncation: Keep the center of the audio signal
                logger_prep.debug(f"Truncation start")
                start = (current_length - target_length_samples) // 2
                end = start + target_length_samples
                y_truncated = []
                for channel in self.y_processed:
                    y_truncated.append(channel[start:end])
                self.y_processed = np.array(y_truncated)
                logger_prep.debug(f"Truncation end")
            logger_prep.info("Padding or truncation completed successfully")
        except Exception as e:
            logger_prep.error(f"Error during padding or truncation: {e}")
            raise

    def rechanneling(self):
        """
        Convert the audio signal to the target number of channels (self.target_channel).

        """
        logger_prep.info(f"Rechanneling audio to {self.target_channel} channels")
        try:
            if self.target_channel == 1 and self.channel > 1:  # Multi to mono
                    # Downmix all channels to mono by averaging the channels
                    self.y_processed = np.mean(self.y_processed, axis=0, keepdims=True)
            elif self.target_channel == 2:  # Convert to stereo
                if self.channel == 1:  # Mono to stereo
                    # Duplicate the mono channel to create stereo
                    self.y_processed = np.stack([self.y_processed[0], self.y_processed[0]], axis=0)
            logger_prep.info("Rechanneling completed successfully")
        except Exception as e:
            logger_prep.error(f"Error during rechanneling: {e}")
            raise

    def compute_mel_spectrogram(self, n_mels=128, n_fft=2048, hop_length=512):
        """
        Compute the mel-spectrogram of the audio.

        Args:
            y (numpy array): Audio time series.
            sr (int): Sample rate.

        Returns:
            numpy array: Mel-spectrogram.
        """
        logger_prep.debug(f"Computing Mel-Spectrogram")
        mel_spectrogram = librosa.feature.melspectrogram(y=self.y_processed, sr=self.sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db


    def compute_spectrogram(self, n_fft=2048, hop_length=512):
        """
        Compute the spectrogram of the audio.

        Returns:
            numpy array: Spectrogram.
        """
        logger_prep.debug(f"Computing Spectrogram")
        spectrogram = librosa.stft(self.y_processed, n_fft=n_fft, hop_length=hop_length)
        spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
        return spectrogram_db


    def normalize_spectrogram(self, spectrogram):
        """
        Normalize the spectrogram to the range [-1, 1].

        Args:
            spectrogram (numpy array): Spectrogram to normalize.

        Returns:
            numpy array: Normalized spectrogram.
        """
        try:
            logger_prep.debug(f"Computing Normalization")
            normalized = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())  # Normalize to [0, 1]
            return normalized
            #return 2 * normalized - 1  # Scale to [-1, 1]
        except Exception as e:
            logger_prep.error(f"Error normalizing spectrogram for file {self.file_path}: {e}")
            raise

    def preprocess(self, use_mel_spectrogram=False, n_mels=128, n_fft=2048, hop_length=512):
        """
        Run the full preprocessing pipeline: load, resample, convert to stereo,
        and transform to spectrogram or mel-spectrogram.

        Args:
            use_mel_spectrogram (bool): Whether to compute a mel-spectrogram instead of a standard spectrogram.
            n_mels (int): Number of mel bands (used only if use_mel_spectrogram is True).
            n_fft (int): FFT window size.
            hop_length (int): The number of samples between successive frames in the spectrogram computation.

        Returns:
            numpy array: Normalized spectrogram or mel-spectrogram.
        """
        logger_prep.info(f"Beginning audio preprocessing")
        self.load()
        self.resample()
        if self.target_length_ms != None:
            self.pad_or_truncate(self.target_length_ms)
        self.rechanneling()

        logger_prep.info(f"Computing spectrogram")
        if use_mel_spectrogram:
            spectrogram = self.compute_mel_spectrogram(n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        else:
            spectrogram = self.compute_spectrogram(n_fft=n_fft, hop_length=hop_length)
        normalized_spectrogram = self.normalize_spectrogram(spectrogram)
        logger_prep.info(f"Preprocessing audio done")
        return normalized_spectrogram


class BatchAudioProcessor:
    """
    A class to preprocess audio files in batches for machine learning model training.
    """

    def __init__(self, file_paths, labels, target_sample_rate=44100, target_channel=2, target_length_ms=1000):
        """
        Initialize the BatchAudioProcessor.

        Args:
            file_paths (list of str): List of paths to audio files.
            labels (list of int): List of labels corresponding to the audio files.
            target_sample_rate (int): Target sample rate for resampling.
            target_channel (int): Target number of channels (e.g., 1 for mono, 2 for stereo).
            target_length_ms (int): Target length of audio in milliseconds.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.target_sample_rate = target_sample_rate
        self.target_channel = target_channel
        self.target_length_ms = target_length_ms

    def _process_file(self, file_path, use_mel_spectrogram, n_mels, n_fft, hop_length):
        """
        Helper function to preprocess a single audio file.

        Args:
            file_path (str): Path to the audio file.
            use_mel_spectrogram (bool): Whether to compute mel-spectrograms instead of standard spectrograms.
            n_mels (int): Number of mel bands (used only if use_mel_spectrogram is True).
            n_fft (int): FFT window size.
            hop_length (int): Number of samples between successive frames in the spectrogram computation.

        Returns:
            numpy array: Preprocessed spectrogram (shape: [frequency_bins, time_frames, n_channel]).
        """
        audio_processor = AudioProcess(
            file_path=file_path,
            target_sample_rate=self.target_sample_rate,
            target_channel=self.target_channel,
            target_length_ms=self.target_length_ms
        )

        # Preprocess the audio file
        spectrogram = audio_processor.preprocess(
            use_mel_spectrogram=use_mel_spectrogram,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )  # Shape: (n_channel, frequency_bins, time_frames)

        # Transpose the spectrogram to match TensorFlow's expected input shape
        return np.transpose(spectrogram, (1, 2, 0))  # Shape: (frequency_bins, time_frames, n_channel)

    def preprocess_batch_serial(self, use_mel_spectrogram=False, n_mels=128, n_fft=2048, hop_length=512):
        """
        Preprocess all audio files in the batch using serial processing.

        Args:
            use_mel_spectrogram (bool): Whether to compute mel-spectrograms instead of standard spectrograms.
            n_mels (int): Number of mel bands (used only if use_mel_spectrogram is True).
            n_fft (int): FFT window size.
            hop_length (int): Number of samples between successive frames in the spectrogram computation.

        Returns:
            tuple: A tuple containing:
                - numpy array of preprocessed spectrograms (shape: [batch_size, frequency_bins, time_frames, n_channel]).
                - numpy array of one-hot encoded labels.
        """
        logger_prep.info("Starting serial batch preprocessing...")
        spectrograms = []
        for i, file_path in enumerate(self.file_paths):
            logger_prep.info(f"Processing file {i + 1}/{len(self.file_paths)}: {file_path}")
            try:
                spectrogram = self._process_file(file_path, use_mel_spectrogram, n_mels, n_fft, hop_length)
                spectrograms.append(spectrogram)
            except Exception as e:
                logger_prep.error(f"Error processing file {file_path}: {e}")
        logger_prep.info("Serial batch preprocessing completed.\n")

        # Convert the list of spectrograms to a numpy array
        spectrograms = np.array(spectrograms)  # Shape: (batch_size, frequency_bins, time_frames, n_channel)

        # One-hot encode the labels
        num_classes = len(set(self.labels)) + 1
        labels_one_hot = tf.one_hot(self.labels, num_classes)

        return spectrograms, labels_one_hot

    def preprocess_batch_parallel(self, use_mel_spectrogram=False, n_mels=128, n_fft=2048, hop_length=512, n_jobs=-1):
        """
        Preprocess all audio files in the batch using parallel processing.

        Args:
            use_mel_spectrogram (bool): Whether to compute mel-spectrograms instead of standard spectrograms.
            n_mels (int): Number of mel bands (used only if use_mel_spectrogram is True).
            n_fft (int): FFT window size.
            hop_length (int): Number of samples between successive frames in the spectrogram computation.
            n_jobs (int): Number of parallel jobs (-1 uses all available CPUs).

        Returns:
            tuple: A tuple containing:
                - numpy array of preprocessed spectrograms (shape: [batch_size, frequency_bins, time_frames, n_channel]).
                - numpy array of one-hot encoded labels.
        """
        logger_prep.info("Starting parallel batch preprocessing...")
        try:
            spectrograms = Parallel(n_jobs=n_jobs)(
                delayed(self._process_file)(
                    file_path, use_mel_spectrogram, n_mels, n_fft, hop_length
                ) for file_path in self.file_paths
            )
            logger_prep.info("Parallel batch preprocessing completed.")
        except Exception as e:
            logger_prep.error(f"Error during parallel batch preprocessing: {e}")
            raise

        # Convert the list of spectrograms to a numpy array
        spectrograms = np.array(spectrograms)  # Shape: (batch_size, frequency_bins, time_frames, n_channel)

        # One-hot encode the labels
        num_classes = len(set(self.labels)) + 1
        labels_one_hot = tf.one_hot(self.labels, num_classes)

        return spectrograms, labels_one_hot


def isPreprocessSaved(preprocessed_dir):
    """
    Check if the preprocessing has already been done by verifying the existence of saved spectrogram files.

    Args:
        preprocessed_dir (str): Path to the directory where preprocessed spectrograms are saved.

    Returns:
        bool: True if the directory contains saved spectrogram files, False otherwise.
    """
    # Check if the directory exists and contains any files
    if os.path.exists(preprocessed_dir) and len(os.listdir(preprocessed_dir)) > 0:
        return True
    return False


def load_saved_spectrograms(preprocessed_dir):
    """
    Load saved spectrograms and labels from the specified directory.

    Args:
        preprocessed_dir (str): Path to the directory where preprocessed spectrograms are saved.

    Returns:
        tuple: A tuple containing:
            - list of spectrograms.
            - list of labels.
    """
    spectrograms = []
    labels = []

    # Iterate over .npz files in the directory
    for idx, file_name in enumerate(os.listdir(preprocessed_dir)):
        if file_name.endswith(".npz"):
            logger_prep.info(f"Loading file {idx + 1}/{len(os.listdir(preprocessed_dir))}: {file_name}")
            file_path = os.path.join(preprocessed_dir, file_name)
            data = np.load(file_path)
            spectrograms.append(data["spectrogram"])
            labels.append(data["label"])

    return spectrograms, labels


def save_spectrogram(preprocessed_dir, file_name, spectrogram, label, logger):
    """
    Save a spectrogram and its label to a .npz file.

    Args:
        preprocessed_dir (str): Directory to save the .npz file.
        file_name (str): Name of the file (without extension).
        spectrogram (numpy array): The spectrogram to save.
        label (int): The label to save.
        logger (logging.Logger): Logger instance for logging.

    Returns:
        bool: True if the file was saved successfully, False otherwise.
    """
    try:
        # Ensure the directory exists
        os.makedirs(preprocessed_dir, exist_ok=True)

        # Construct the file path
        spectrogram_file = os.path.join(preprocessed_dir, file_name.replace('.wav', '.npz'))

        # Save the spectrogram and label
        np.savez_compressed(
            spectrogram_file,
            spectrogram=spectrogram,
            label=label
        )
        logger.info(f"Saved spectrogram: {spectrogram_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving spectrogram for {file_name}: {e}")
        return False


def add_spectrogram_to_df(df_dataset, file_name, spectrogram, label, logger):
    """
    Add a spectrogram and label to the df_dataset DataFrame.

    Args:
        df_dataset (pandas.DataFrame): The DataFrame to update.
        file_name (str): Name of the file (used as the index in df_dataset).
        spectrogram (numpy array): The spectrogram to add.
        label (int): The label to add.
        logger (logging.Logger): Logger instance for logging.

    Returns:
        bool: True if the spectrogram and label were added successfully, False otherwise.
    """
    try:
        # Assign spectrogram and label to the DataFrame
        df_dataset.at[file_name, 'spectrogram'] = spectrogram
        df_dataset.at[file_name, 'label'] = label
        logger.info(f"Added spectrogram and label to df_dataset for {file_name}")
        return True
    except Exception as e:
        logger.error(f"Error adding spectrogram and label to df_dataset for {file_name}: {e}")
        return False




