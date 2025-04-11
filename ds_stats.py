import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa

import src.data_loader as dl
from src.utils import load_config

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
    y, sr = dl.load_audio_file(path, mono = False)
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

def plot_sr_histogram(properties_df: pd.DataFrame, output_path=None):
    """
    Plot a bar chart of the sample rates (sr) in the dataset, with each sample rate at equal distance,
    and display the value of each bar above it.

    Args:
        properties_df (pandas.DataFrame): DataFrame containing the audio properties, including "sr".
        output_path (str, optional): Path to save the bar chart as an image. If None, the plot is shown.
    """
    # Count the occurrences of each sample rate
    sr_counts = properties_df["sr"].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sr_counts.index.astype(str), sr_counts.values, color='skyblue', edgecolor='black')
    plt.title("Sample Rate Distribution", fontsize=16)
    plt.xlabel("Sample Rate (Hz)", fontsize=14)
    plt.ylabel("Number of Audio Files", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)  # Rotate x-axis labels for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values above the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)

    if output_path:
        plt.savefig(output_path)
        print(f"Sample rate bar chart saved to {output_path}")
    else:
        plt.show()


def plot_channel_nb_histogram(properties_df: pd.DataFrame, output_path=None):
    """
    Plot a histogram of the number of channels (channel_nb) in the dataset,
    and display the value of each bar above it.

    Args:
        properties_df (pandas.DataFrame): DataFrame containing the audio properties, including "channel_nb".
        output_path (str, optional): Path to save the histogram as an image. If None, the plot is shown.
    """
    unique_channels = sorted(properties_df["channel_nb"].unique())  # Get unique channel numbers
    channel_counts = properties_df["channel_nb"].value_counts().sort_index()

    plt.figure(figsize=(8, 6))
    bars = plt.bar(channel_counts.index.astype(str), channel_counts.values, color='lightgreen', edgecolor='black')
    plt.title("Channel Number Histogram", fontsize=16)
    plt.xlabel("Number of Channels", fontsize=14)
    plt.ylabel("Number of Audio Files", fontsize=14)
    plt.xticks(unique_channels)  # Ensure ticks align with unique channel numbers
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values above the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=10)

    if output_path:
        plt.savefig(output_path)
        print(f"Channel number histogram saved to {output_path}")
    else:
        plt.show()


def main():
    # Load configuration
    CONFIG_PATH = "./config/config.json"
    config = load_config(CONFIG_PATH)

    # Access dataset path
    DATASET_PATH = config["dataset_path"]
    STATS_PATH = config["results_path_stats"]

    # Get the audio properties
    name_list, paths_list, class_list, fold_number = dl.get_audio_UrbanSound8K(DATASET_PATH)
    properties_df = get_audio_ds_properties(paths_list, index=name_list)

    # Get statistics
    length_stats, channel_nb, sr_stat = get_properties_stat(properties_df)

    # Convert np.float64 to Python float
    length_stats = list(map(float, length_stats))
    print(f"Length Stats (avg, max, min): {np.around(length_stats,decimals=2)}")
    print("Unique Channel Numbers:", channel_nb)
    print("Unique Sample Rates:", sr_stat)

    # Plot, save, and display the histograms
    plot_sr_histogram(properties_df, output_path=STATS_PATH+"sr_histogram.png")
    plot_channel_nb_histogram(properties_df, output_path=STATS_PATH+"channel_nb_histogram.png")

if __name__ == "main":
    main()
