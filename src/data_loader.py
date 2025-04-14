import csv
from pathlib import Path
import os

import pandas as pd
import librosa

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

            fold_number.append(line["fold"])

            path = audio_path + "fold" + line["fold"] + "/" + line["slice_file_name"]
            paths_list.append(path)
            
            name = Path(path).stem
            name_list.append(name)


    return pd.DataFrame({'path': paths_list, 'class':class_list, 'fold':fold_number}, index=name_list)

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
    # Select rows where 'fold' equals the test_fold
    test_df = dataframe[dataframe['fold'] == test_fold]

    # Select rows where 'fold' does not equal the test_fold
    train_df = dataframe[dataframe['fold'] != test_fold]

    return train_df, test_df



