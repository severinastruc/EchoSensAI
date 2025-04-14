import csv
from pathlib import Path
import os

import numpy as np
import pandas as pd
import h5py

from src.preprocessing import BatchAudioProcessor
from src.logger import logger_prep

def get_audio_UrbanSound8K(path: str):
    """
    Extracts the names, paths, classes, and fold numbers of audio files from the UrbanSound8K dataset.

    Args:
        path (str): The base path to the UrbanSound8K dataset.

    Returns:
        tuple:
            - name_list (list): List of the name of the audio files.             
            - paths_list (list): List of full paths to the audio files.
            - class_list (list): List of class IDs corresponding to each audio file.
            - fold_number (list): List of fold numbers corresponding to each audio file.

    Example:
        >>> paths_list, class_list, fold_number = get_audio("./path/to/UrbanSound8K/")
        >>> print(paths_list[1])
        ./path/to/UrbanSound8K/audio/fold1/101415-3-0-2.wav
    """
    metadata_path = path + "metadata/UrbanSound8K.csv"
    audio_path = path + "audio/"
    name_list, paths_list, class_list, fold_number = [], [], [], []

    with open(metadata_path, mode='r') as file:
        metadata_file = csv.DictReader(file)
        for line in metadata_file:
            class_list.append(int(line["classID"]))

            fold_number.append(int(line["fold"]))

            path = audio_path + "fold" + line["fold"] + "/" + line["slice_file_name"]
            paths_list.append(path)
            
            name = Path(path).stem
            name_list.append(name)


    return pd.DataFrame({'path': paths_list, 'class':class_list, 'fold':fold_number}, index=name_list)

def save_to_hdf5(file_path, spectrograms, labels, file_names):
    """
    Save spectrograms, labels, and file names to an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        spectrograms (numpy.ndarray): Array of spectrograms.
        labels (numpy.ndarray): Array of labels.
        file_names (list or np.ndarray): List or array of file names.
    """
    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("spectrograms", data=spectrograms, compression="gzip")
        hf.create_dataset("labels", data=labels)
        hf.create_dataset("file_names", data=np.array(file_names, dtype="S"))  # Save file names as strings
    logger_prep.info(f"Data saved to HDF5 file: {file_path}")


def load_from_hdf5(file_path):
    """
    Load spectrograms, labels, and file names from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        tuple: (spectrograms, labels, file_names)
    """
    with h5py.File(file_path, "r") as hf:
        spectrograms = hf["spectrograms"][:]
        labels = hf["labels"][:]
        file_names = hf["file_names"][:].astype(str)  # Convert byte strings to regular strings
    logger_prep.info(f"Data loaded from HDF5 file: {file_path}")
    return spectrograms, labels, file_names


def preprocess_and_save(df_dataset, config_preprocess, preprocessed_dir, subset_size=None):
    """
    Preprocess the dataset and save it to an HDF5 file.

    Args:
        df_dataset (pd.DataFrame): DataFrame containing the dataset.
        config_preprocess (dict): Preprocessing configuration.
        preprocessed_dir (str): Directory to save the preprocessed data.
        subset_size (int, optional): Number of rows to process. Defaults to None.

    Returns:
        tuple: (spectrograms, labels, file_names)
    """
    if subset_size:
        df_dataset = df_dataset.sample(n=subset_size, random_state=42)

    processor = BatchAudioProcessor(
        file_paths=df_dataset['path'].tolist(),
        labels=df_dataset['class'].tolist(),
        target_sample_rate=config_preprocess["target_sample_rate"],
        target_channel=config_preprocess["target_channel"],
        target_length_ms=config_preprocess["target_audio_length"]
    )

    spectrograms, labels = processor.preprocess_batch_serial(
        use_mel_spectrogram=True,
        n_mels=config_preprocess["n_mels"],
        n_fft=config_preprocess["n_fft"],
        hop_length=config_preprocess["hop_length"]
    )

    # Save to HDF5
    os.makedirs(preprocessed_dir, exist_ok=True)
    hdf5_file = os.path.join(preprocessed_dir, "preprocessed_data.h5")
    save_to_hdf5(hdf5_file, np.array(spectrograms), np.array(labels), df_dataset.index.tolist())

    return spectrograms, labels, df_dataset.index.tolist()


def load_or_preprocess(df_dataset, config_preprocess, preprocessed_dir, subset_size=None):
    """
    Load preprocessed data from HDF5 or preprocess and save if not available.

    Args:
        df_dataset (pd.DataFrame): DataFrame containing the dataset.
        config_preprocess (dict): Preprocessing configuration.
        preprocessed_dir (str): Directory to save/load the preprocessed data.
        subset_size (int, optional): Number of rows to process. Defaults to None.

    Returns:
        tuple: (spectrograms, labels, file_names)
    """
    hdf5_file = os.path.join(preprocessed_dir, "preprocessed_data.h5")

    if os.path.exists(hdf5_file):
        logger_prep.info("Preprocessed data found. Loading from HDF5...")
        return load_from_hdf5(hdf5_file)
    else:
        logger_prep.info("No preprocessed data found. Starting preprocessing...")
        return preprocess_and_save(df_dataset, config_preprocess, preprocessed_dir, subset_size)


def split_folds(dataframe, test_fold):
    """
    Split a DataFrame into two DataFrames based on the 'fold' column.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame containing a 'fold' column.
        test_fold (int): The fold number to use as the test set.

    Returns:
        train_df (pandas.DataFrame): DataFrame containing rows where 'fold' != test_fold.
        test_df (pandas.DataFrame): DataFrame containing rows where 'fold' == test_fold.
    """
    if 'fold' not in dataframe.columns:
        raise ValueError("The dataset does not contain a 'fold' column.")
    if test_fold not in dataframe['fold'].unique():
        raise ValueError(f"The dataset does not contain the fold number {test_fold}.")
    # Select rows where 'fold' equals the test_fold
    test_df = dataframe[dataframe['fold'] == test_fold]
    # Select rows where 'fold' does not equal the test_fold
    train_df = dataframe[dataframe['fold'] != test_fold]

    return train_df, test_df

def add_spectrograms_to_df(df_dataset, spectrograms, labels, file_names, logger):
    """
    Add spectrograms and labels to the df_dataset DataFrame.

    Args:
        df_dataset (pd.DataFrame): The DataFrame to update.
        spectrograms (list or np.ndarray): List or array of spectrograms.
        labels (list or np.ndarray): List or array of labels.
        file_names (list or np.ndarray): List or array of file names.
        logger (logging.Logger): Logger instance for logging.

    Returns:
        pd.DataFrame: Updated DataFrame with spectrograms and labels.
    """
    try:
        # Ensure the DataFrame has the necessary columns
        if 'spectrogram' not in df_dataset.columns:
            df_dataset['spectrogram'] = None
        if 'label' not in df_dataset.columns:
            df_dataset['label'] = None

        # Update the DataFrame
        for file_name, spectrogram, label in zip(file_names, spectrograms, labels):
            if file_name in df_dataset.index:
                df_dataset.at[file_name, 'spectrogram'] = spectrogram
                df_dataset.at[file_name, 'label'] = label
            else:
                logger.warning(f"File name {file_name} not found in df_dataset index.")

        logger.info("Spectrograms and labels successfully added to df_dataset.")
        return df_dataset
    except Exception as e:
        logger.error(f"Error adding spectrograms and labels to df_dataset: {e}")
        raise

