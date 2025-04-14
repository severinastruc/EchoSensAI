import src.data_loader as dl
from src.utils import load_config
import src.preprocessing as prep
from src.logger import logger_main  # Import the main logger
import os
import numpy as np

# Load configuration
CONFIG_PATH = "./config/config.json"
logger_main.info(f"Loading config file: {CONFIG_PATH}")
config = load_config(CONFIG_PATH)
config_preprocess = config["preprocess_constants"]

DATASET_PATH = config["dataset_path"]
PREPROCESSED_DIR = config["preproc_ds_path"]
MONO = config_preprocess["target_channel"]

BOOL_NMELS = config_preprocess["use_mel_spectrogram"]
N_MELS = config_preprocess["n_mels"]
N_FFT = config_preprocess["n_fft"]
HOP_LENGTH = config_preprocess["hop_length"]

# Load the audio dataset
logger_main.info(f"Loading Audio dataset: {DATASET_PATH}")
df_dataset = dl.get_audio_UrbanSound8K(DATASET_PATH)

df_dataset["spectrogram"] = None
df_dataset["label"] = None


if prep.isPreprocessSaved(PREPROCESSED_DIR):
    logger_main.info("Preprocessed data found. Loading saved spectrograms...")
    spectrograms, labels = prep.load_saved_spectrograms(PREPROCESSED_DIR)

    # Assign loaded spectrograms and labels to the correct rows in df_dataset
    for file_name, spectrogram, label in zip(df_dataset.index, spectrograms, labels):
        prep.add_spectrogram_to_df(df_dataset, file_name, spectrogram, label, logger_main)

else:
    logger_main.info("No preprocessed data found. Starting preprocessing...")
    processor = prep.BatchAudioProcessor(
        file_paths=df_dataset['path'].tolist(),
        labels=df_dataset['class'].tolist(),
        target_sample_rate=config_preprocess["target_sample_rate"],
        target_channel=config_preprocess["target_channel"],
        target_length_ms=config_preprocess["target_audio_length"]
    )
    spectrograms, labels = processor.preprocess_batch_serial(
        use_mel_spectrogram=True, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    # Save preprocessed data
    for file_name, spectrogram, label in zip(df_dataset.index, spectrograms, labels):
        # Save the spectrogram and label
        if prep.save_spectrogram(PREPROCESSED_DIR, file_name, spectrogram, label, logger_main):
            # Add to the DataFrame only if saving was successful
            prep.add_spectrogram_to_df(df_dataset, file_name, spectrogram, label, logger_main)

    logger_main.info("Preprocessed data saved.")


for fold_idx in range(10):
    logger_main.info(f"Starting fold {fold_idx + 1}/10...")
    train_df, test_df = dl.split_folds(df_dataset, test_fold=fold_idx)

    # Train model

    # Test model

    # End of cross validation
    logger_main.info(f"Fold {fold_idx + 1} completed.")

