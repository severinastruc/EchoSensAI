import src.data_loader as dl
from src.utils import load_config

# Load configuration
CONFIG_PATH = "./config/config.json"
config = load_config(CONFIG_PATH)

# Access dataset path
DATASET_PATH = config["dataset_path"]

# Get the audio list
name_list, paths_list, class_list, fold_number = dl.get_audio_UrbanSound8K(DATASET_PATH)

for i in range(4):
    print("Name: ", name_list[i])
    print("Path: ", paths_list[i])
    print("class: ", class_list[i])
    print("fold: ", fold_number[i])
    length, channel_number, sr = dl.get_audio_properties(paths_list[i])
    print("length: ", length)
    print("channel_number: ", channel_number)
    print("sr: ", sr)

properties_df = dl.get_audio_ds_properties(paths_list, index=name_list)


# Get statistics
length_stats, channel_nb, sr_stat = dl.get_properties_stat(properties_df)

# Convert np.float64 to Python float
length_stats = list(map(float, length_stats))
print("Length Stats (avg, max, min):", length_stats)
print("Unique Channel Numbers:", channel_nb)
print("Unique Sample Rates:", sr_stat)
