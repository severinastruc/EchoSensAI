import json
import os


def load_config(config_path: str):
    """
    Loads the JSON configuration file.

    Args:
        config_path (str): Path to the JSON configuration file.
    
    Returns:
        dict: Configuration data as a dictionary.    
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config_file = json.load(file)
    return config_file
