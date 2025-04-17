import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import AudioProcess

def compare_spectrograms(df_dataset, config_preprocess, num_samples=5):
    """
    Compare spectrograms stored in the DataFrame with spectrograms computed from the file paths
    using the AudioProcess class.

    Args:
        df_dataset (pd.DataFrame): DataFrame containing 'spectrogram' and 'path' columns.
        config_preprocess (dict): Preprocessing configuration (e.g., target sample rate, channels, etc.).
        num_samples (int): Number of samples to compare. Defaults to 5.
    """
    target_sample_rate = config_preprocess["target_sample_rate"]
    target_channel = config_preprocess["target_channel"]
    target_length_ms = config_preprocess.get("target_audio_length", None)
    use_mel_spectrogram = config_preprocess.get("use_mel_spectrogram", True)
    n_mels = config_preprocess.get("n_mels", 128)
    n_fft = config_preprocess.get("n_fft", 2048)
    hop_length = config_preprocess.get("hop_length", 512)

    for idx, row in df_dataset.sample(n=num_samples, random_state=42).iterrows():
        # Extract stored spectrogram and file path
        stored_spectrogram = row['spectrogram']
        file_path = row['path']
        label = row['class']
        class_val = row['class']

        # Compute spectrogram using the AudioProcess class
        try:
            audio_processor = AudioProcess(
                file_path=file_path,
                target_sample_rate=target_sample_rate,
                target_channel=target_channel,
                target_length_ms=target_length_ms
            )
            computed_spectrogram = audio_processor.preprocess(
                use_mel_spectrogram=use_mel_spectrogram,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length
            )
            computed_spectrogram = np.squeeze(computed_spectrogram)
            stored_spectrogram = np.squeeze(stored_spectrogram)
            # Plot side-by-side comparison
            plt.figure(figsize=(12, 6))

            # Plot stored spectrogram
            plt.subplot(1, 2, 1)
            if stored_spectrogram is not None:
                plt.imshow(stored_spectrogram, aspect='auto', origin='lower', cmap='hot')
                plt.title(f"Stored Spectrogram (Label: {label})")
                plt.colorbar()
            else:
                plt.title("Stored Spectrogram: None")

            # Plot computed spectrogram
            plt.subplot(1, 2, 2)
            plt.imshow(computed_spectrogram, aspect='auto', origin='lower', cmap='hot')
            plt.title(f"Computed Spectrogram (Label: {class_val})")
            plt.colorbar()

            plt.suptitle(f"File: {file_path}")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

def check_spectrogram_range(df_dataset, value_range=(0, 1)):
    """
    Check that all values in the spectrograms are within a specified range.

    Args:
        df_dataset (pd.DataFrame): DataFrame containing 'spectrogram' column.
        value_range (tuple): The range (min, max) to check. Defaults to (0, 1).

    Returns:
        bool: True if all spectrograms are within the range, False otherwise.
    """
    min_val, max_val = value_range
    all_within_range = True

    for idx, row in df_dataset.iterrows():
        spectrogram = row['spectrogram']
        if spectrogram is not None:
            spectrogram = np.squeeze(spectrogram)
            print(f"Spectrogram at index {idx}: min={spectrogram.min()}, max={spectrogram.max()}")
            if np.any(spectrogram < min_val) or np.any(spectrogram > max_val):
                print(f"Spectrogram at index {idx} has values out of range {value_range}.")
                all_within_range = False

    if all_within_range:
        print(f"All spectrograms are within the range {value_range}.")
    else:
        print(f"Some spectrograms have values outside the range {value_range}.")

    return all_within_range
