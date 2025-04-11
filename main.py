import src.data_loader as dl
from src.utils import load_config
from src.preprocessing import AudioProcess
from ds_stats import get_audio_properties

# Load configuration
CONFIG_PATH = "./config/config.json"
config = load_config(CONFIG_PATH)

DATASET_PATH = config["dataset_path"]
MONO = config["preprocess_constants"]["target_channel"]
SAMPLE_RATE = config["preprocess_constants"]["sample_rate_target"]


# Get the audio list
name_list, paths_list, class_list, fold_number = dl.get_audio_UrbanSound8K(DATASET_PATH)

"""
# Load audio
file_path = paths_list[2]
audio_processor = AudioProcess(file_path, target_sample_rate=48000)
y, sr = audio_processor.load_audio()

length, channel_number, sr = get_audio_properties(file_path)
print(f"Sample rate: {sr}")
print(f"Channel: {channel_number}")
print(f"Length: {length}")

# Resample audio
y_resampled = audio_processor.resample_audio(y, sr)

# Convert mono to stereo
y_stereo = audio_processor.mono_to_stereo(y_resampled)

# Compute spectrogram
spectrogram = audio_processor.compute_spectrogram(y_stereo, audio_processor.target_sample_rate)

# Plot spectrogram
audio_processor.plot_spectrogram(spectrogram, audio_processor.target_sample_rate)
"""
