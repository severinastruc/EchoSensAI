import csv

def get_audio(path: str):
    """
    Extracts the paths, classes, and fold numbers of audio files from the UrbanSound8K dataset.

    Args:
        path (str): The base path to the UrbanSound8K dataset.

    Returns:
        tuple:
            - paths_list (list): List of full paths to the audio files.
            - class_list (list): List of class IDs corresponding to each audio file.
            - fold_number (list): List of fold numbers corresponding to each audio file.

    Example:
        >>> paths_list, class_list, fold_number = get_audio("./data/UrbanSound8K/")
        >>> print(paths_list[0])
        ./data/UrbanSound8K/audio/fold1/101415-3-0-2.wav
    """
    metadata_path = path + "metadata/UrbanSound8K.csv"
    audio_path = path + "audio/"
    paths_list, class_list, fold_number = [], [], []

    with open(metadata_path, mode='r') as file:
        metadata_file = csv.DictReader(file)
        for line in metadata_file:
            class_list.append(line["classID"])
            fold_number.append(line["fold"])
            paths_list.append(audio_path + "/fold" + line["fold"] + "/" + line["slice_file_name"])

    return paths_list, class_list, fold_number




