import src.data_loader as dl
from src.utils import load_config
import src.preprocessing as prep



# Load configuration
CONFIG_PATH = "./config/config.json"
config = load_config(CONFIG_PATH)

DATASET_PATH = config["dataset_path"]
MONO = config["preprocess_constants"]["target_channel"]

BOOL_NMELS = config["preprocess_constants"]["use_mel_spectrogram"]
N_MELS = config["preprocess_constants"]["n_mels"]
N_FFT = config["preprocess_constants"]["n_fft"]
HOP_LENGTH = config["preprocess_constants"]["hop_length"]


# Get the audio list
name_list, file_paths, class_list, fold_number = dl.get_audio_UrbanSound8K(DATASET_PATH)

# Initialize the BatchAudioProcessor
batch_processor = prep.BatchAudioProcessor(
    file_paths=file_paths[:20],
    labels=class_list[:20],
    target_sample_rate=config["preprocess_constants"]["target_sample_rate"],
    target_channel=config["preprocess_constants"]["target_channel"],
    target_length_ms=config["preprocess_constants"]["target_audio_length"]
)

# Serial processing
spectrograms_serial, labels_serial = batch_processor.preprocess_batch_serial(
    use_mel_spectrogram=True, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
)


"""# Parallel processing
spectrograms_parallel, labels_parallel = batch_processor.preprocess_batch_parallel(
    use_mel_spectrogram=True, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, n_jobs=config["n_job"]
)"""

print(f"Spectrograms_serial shape: {spectrograms_serial.shape}")
#print(f"Spectrograms_parallel shape: {spectrograms_parallel.shape}")
