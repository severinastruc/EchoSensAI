from src.data_loader import get_audio_UrbanSound8K, get_audio_properties, get_audio_ds_properties
from src.utils import load_config

# Load configuration
CONFIG_PATH = "./config/config.json"
config = load_config(CONFIG_PATH)

# Access dataset path
DATASET_PATH = config["dataset_path"]

# Get the audio list
name_list, paths_list, class_list, fold_number = get_audio_UrbanSound8K(DATASET_PATH)

for i in range(4):
    print("Name: ", name_list[i])
    print("Path: ", paths_list[i])
    print("class: ", class_list[i])
    print("fold: ", fold_number[i])
    length, channel_number, sr = get_audio_properties(paths_list[i])
    print("length: ", length)
    print("channel_number: ", channel_number)
    print("sr: ", sr)

df = get_audio_ds_properties(paths_list[:4], index=name_list[:4])
print(df)
