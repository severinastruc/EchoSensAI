import numpy as np
from tensorflow import keras
from keras._tf_keras.keras.callbacks import EarlyStopping

from src.logger import logger_main
from src.model import create_crnn_model
from src.data_loader import get_audio_UrbanSound8K, load_or_preprocess, split_folds, add_spectrograms_to_df
from src.utils import load_config

from tests.test_data_loader import check_spectrogram_range, compare_spectrograms

# Load configuration
CONFIG_PATH = "./config/config.json"
logger_main.info(f"Loading config file: {CONFIG_PATH}")
config = load_config(CONFIG_PATH)
config_preprocess = config["preprocess_constants"]

DATASET_PATH = config["dataset_path"]
PREPROCESSED_DIR = config["preproc_ds_path"]

# Load the audio dataset
logger_main.info(f"Loading Audio dataset: {DATASET_PATH}")
df_dataset = get_audio_UrbanSound8K(DATASET_PATH)

# Load or preprocess data
subset_size = None #config.get("subset_size", None)  # Use subset for faster testing
spectrograms, labels, file_names = load_or_preprocess(df_dataset, config_preprocess, PREPROCESSED_DIR, subset_size)

# Add spectrograms and labels to df_dataset
df_dataset = add_spectrograms_to_df(df_dataset, spectrograms, labels, file_names, logger_main)

"""test_bool = check_spectrogram_range(df_dataset)
print(test_bool)
compare_spectrograms(df_dataset=df_dataset, config_preprocess=config_preprocess)
print(df_dataset['class'].value_counts())"""

# Cross-validation
test_losses, test_accuracies = [], []
for fold_idx in range(1, 10):
    logger_main.info(f"Starting fold {fold_idx}/10...")

    # Split the dataset into training and testing sets
    train_df, test_df = split_folds(df_dataset, test_fold=fold_idx)

    # Extract spectrograms and labels
    X_train = np.array(train_df['spectrogram'].tolist(), dtype=np.float32)
    y_train = np.array(train_df['class'].tolist(), dtype=np.int32)
    X_test = np.array(test_df['spectrogram'].tolist(), dtype=np.float32)
    y_test = np.array(test_df['class'].tolist(), dtype=np.int32)

    # Create the CRNN model
    input_shape = X_train.shape[1:]
    logger_main.info(f"Model input shape: {input_shape}")
    num_classes = len(np.unique(y_train))
    logger_main.info(f"Model number of classes: {num_classes}")
    model = create_crnn_model(input_shape, num_classes)

    # Train the model
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    logger_main.info(f"Fold {fold_idx} - Test Accuracy: {test_accuracy:.4f}")

# Log final results
logger_main.info(f"Test Losses: Count={len(test_losses)}, Mean={np.mean(test_losses):.4f}, Std={np.std(test_losses):.4f}")
logger_main.info(f"Test Accuracies: Count={len(test_accuracies)}, Mean={np.mean(test_accuracies):.4f}, Std={np.std(test_accuracies):.4f}")

