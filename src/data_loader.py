import csv
from pathlib import Path

import numpy as np
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
            class_list.append(line["classID"])

            fold_number.append(line["fold"])

            path = audio_path + "fold" + line["fold"] + "/" + line["slice_file_name"]
            paths_list.append(path)
            
            name = Path(path).stem
            name_list.append(name)


    return name_list, paths_list, class_list, fold_number


def get_audio_properties(path: str):
    """
    Extract audio properties from an audio file.
    Audio properties: length, number of channel, sample frequency.

    Args:
        path (str): Path of the audio file.
    
    Returns:
        tuple:
            - length (float): length in second of the audio file.
            - channel_number (int): number of channel (mono=1, stereo=2).
            - sr (float): sample frequency of the audio file in Hz.
    """
    y, sr = librosa.load(path, sr=None, mono=False)
    length = librosa.get_duration(y=y, sr=sr)
    if y.ndim == 1:  # Mono audio
        channel_number = 1
    else:  # Multi-channel audio
        channel_number = y.shape[0]
    return length, channel_number, sr

def get_audio_ds_properties(paths: list, index = None):
    """
    Extract all the audio properties from an audio dataset.
    Audio properties from get_audio_properties

    Args:
        paths (list): List of full paths to the audio files.

    Returns:
        properties_df (pandas.DataFrame): row = files, columns=properties, index = names
    """
    columns = ["length", "channel_nb", "sr",]
    properties = []
    for path in paths:
        length, channel_nb, sr = get_audio_properties(path)
        properties.append([length, channel_nb, sr])
    properties_df =  pd.DataFrame(properties, columns=columns)
    if index != None:
        properties_df.index = index
    return properties_df

def get_properties_stat(properties_df: pd.DataFrame):
    """
    Compute statistics for audio file properties in a dataset.

    Args:
        properties_df (pandas.DataFrame): A DataFrame where each row corresponds to an audio file 
                                          and columns represent properties such as "length", 
                                          "channel_nb", and "sr".

    Returns:
        tuple:
            - length_stats (list): [average length, max length, min length] of audio files in seconds.
            - channel_nb (list): Unique values of the number of channels used in the dataset.
            - sr_stat (list): Unique sample rates (in Hz) present in the dataset.
    """
    # Compute statistics for the "length" column
    length_stats = [
        properties_df["length"].mean(),  # Average length
        properties_df["length"].max(),   # Maximum length
        properties_df["length"].min()    # Minimum length
    ]

    # Get unique values for "channel_nb" and "sr" columns
    channel_nb = properties_df["channel_nb"].unique().tolist()
    sr_stat = properties_df["sr"].unique().tolist()
    channel_nb.sort()
    sr_stat.sort()
    return length_stats, channel_nb, sr_stat
