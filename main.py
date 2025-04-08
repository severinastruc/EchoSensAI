from src.data_loader import get_audio_UrbanSound8K
from src.utils import load_config

# Load configuration
CONFIG_PATH = "./config/config.json"
config = load_config(CONFIG_PATH)

# Access dataset path
DATASET_PATH = config["dataset_path"]

# Get the audio list
paths_list, class_list, fold_number = get_audio_UrbanSound8K(DATASET_PATH)

for i in range(10):
    print("Path: :", paths_list[i])
    print("class: :", class_list[i])
    print("fold: :", fold_number[i])
