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
    try:
        with open(config_path, "r") as file:
            config_file = json.load(file)
        return config_file
    
    except FileNotFoundError:
        print("The file does not exist.")
        return None

    except json.JSONDecodeError:
        print("Invalid JSON format.")
        return None
    
