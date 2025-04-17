from tensorflow import keras
from keras._tf_keras.keras import layers, models

def create_crnn_model(input_shape, num_classes):
    """
    Create a CRNN model for sound classification.

    Args:
        input_shape (tuple): Shape of the input spectrogram (frequency_bins, time_frames, channels).
        num_classes (int): Number of output classes.

    Returns:
        tensorflow.keras.Model: Compiled CRNN model.
    """
    model = models.Sequential()

    # Convolutional layers for feature extraction
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Reshape for recurrent layers
    # (batch_size, time_steps, features)
    model.add(layers.Reshape((-1, model.output_shape[-1])))

    # Recurrent layers for temporal modeling
    model.add(layers.GRU(64, return_sequences=True, activation='relu'))
    model.add(layers.GRU(64, activation='relu'))

    # Fully connected layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))  # Dropout for regularization
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def create_cnn_model(input_shape, num_classes):
    """
    Create a CNN model for sound classification.

    Args:
        input_shape (tuple): Shape of the input spectrogram (frequency_bins, time_frames, channels).
        num_classes (int): Number of output classes.

    Returns:
        tensorflow.keras.Model: Compiled CNN model.
    """
    model = models.Sequential()

    # Convolutional Block 1
    model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Convolutional Block 2
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Convolutional Block 3
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

