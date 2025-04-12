import logging

from src.utils import load_config

# Load configuration
CONFIG_PATH = "./config/config.json"
config = load_config(CONFIG_PATH)

LEVEL = config["logger_level"]
LOG_PATH = config["log_path"]

dic_level = {'debug': logging.DEBUG,'info': logging.INFO, 'warning': logging.WARNING}


# Configure the logger
logging.basicConfig(
    level=dic_level[LEVEL],  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    filename=LOG_PATH,
    filemode='w',
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    datefmt='%m/%d/%Y %I:%M:%S %p',
)

logger = logging.getLogger("PreprocessingPipeline") # Create a logger instance
