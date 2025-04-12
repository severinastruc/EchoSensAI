import src.data_loader as dl
from src.utils import load_config
import src.preprocessing as prep



# Load configuration
CONFIG_PATH = "./config/config.json"
config = load_config(CONFIG_PATH)

DATASET_PATH = config["dataset_path"]
MONO = config["preprocess_constants"]["target_channel"]
SAMPLE_RATE = config["preprocess_constants"]["sample_rate_target"]


# Get the audio list
name_list, file_paths, class_list, fold_number = dl.get_audio_UrbanSound8K(DATASET_PATH)

# Initialize the BatchAudioProcessor
batch_processor = prep.BatchAudioProcessor(
    file_paths=file_paths[:20],
    labels=class_list[:20],
    target_sample_rate=44100,
    target_channel=2,
    target_length_ms=1000
)

# Serial processing
spectrograms_serial, labels_serial = batch_processor.preprocess_batch_serial(
    use_mel_spectrogram=True, n_mels=128, n_fft=2048, hop_length=512
)


# Parallel processing
spectrograms_parallel, labels_parallel = batch_processor.preprocess_batch_parallel(
    use_mel_spectrogram=True, n_mels=128, n_fft=2048, hop_length=512, n_jobs=4
)

print(f"Spectrograms_serial shape: {spectrograms_serial.shape}")
print(f"Spectrograms_parallel shape: {spectrograms_parallel.shape}")
