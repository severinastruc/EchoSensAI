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
            class_list.append(int(line["classID"]))

            fold_number.append(line["fold"])

            path = audio_path + "fold" + line["fold"] + "/" + line["slice_file_name"]
            paths_list.append(path)
            
            name = Path(path).stem
            name_list.append(name)


    return name_list, paths_list, class_list, fold_number

def load_audio_file(path: str, sr=None, mono=False):
    """
    Load the audio file as a list of value. 
    The loaded file has its original sample rate or a defined sample rate stated
    in the input. The output can have one (mono) or two channel (stereo).
 
    Args:
        path (str): Path of the audio file.
        sr (float, optional): Value of the audio file sample rate.
        mono (bool, optional): Number of channel (stereo or mono).
    
    Returns:
        y (list): List of the value of the audio file.
        sr (float): Sample rate of the audio file.     
    """
    y, sr = librosa.load(path, sr=sr, mono=mono)
    return y, sr



