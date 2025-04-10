import src.data_loader as dl
from src.utils import load_config

# Load configuration
CONFIG_PATH = "./config/config.json"
config = load_config(CONFIG_PATH)

# Access dataset path
DATASET_PATH = config["dataset_path"]

# Get the audio list
name_list, paths_list, class_list, fold_number = dl.get_audio_UrbanSound8K(DATASET_PATH)

