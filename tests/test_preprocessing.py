import pytest
import numpy as np
from src.preprocessing import AudioProcess

@pytest.fixture
def audio_processor(request):
    """
    Fixture to initialize the AudioProcess object with dummy data.
    """
    file_path = "./tests/data/6988-5-0-1.wav"
    target_sample_rate = 44100
    target_channel = 2
    processor = AudioProcess(file_path, target_sample_rate, target_channel)

    # Create dummy audio data based on the number of channels
    num_channels = getattr(request, "param", 1)  # Default to mono
    if num_channels == 1:
        processor.y = np.random.rand(44100)  # 1 second of mono audio
        processor.y = np.expand_dims(processor.y, axis=0)
    else:
        processor.y = np.random.rand(num_channels, 44100)  # Multi-channel audio
    processor.channel = processor.y.shape[0]
    processor.sr = 44100
    processor.y_processed = np.copy(processor.y)
    return processor

@pytest.mark.parametrize("audio_processor", [1, 2], indirect=True)
def test_load(audio_processor):
    """
    Test the load method for mono and stereo audio.
    """
    audio_processor.load()
    assert audio_processor.y is not None
    assert audio_processor.sr == 44100
    assert audio_processor.y.ndim == 2  # Ensure the shape is (n, N)

@pytest.mark.parametrize("audio_processor", [1, 2], indirect=True)
def test_resample(audio_processor):
    """
    Test the resample method for mono and stereo audio.
    """
    audio_processor.target_sample_rate = 22050  # New target sample rate
    audio_processor.resample()
    assert audio_processor.sr == audio_processor.target_sample_rate
    # t = N1/sr1 = N2/sr2 with N number of sample, sr sample rate, t length in s
    expected_length = int(audio_processor.y.shape[1] * (22050 / 44100)) # N2
    assert audio_processor.y_processed.shape[1] == expected_length  # Ensure resampled length matches target

@pytest.mark.parametrize("audio_processor", [1, 2], indirect=True)
def test_pad_or_truncate(audio_processor):
    """
    Test the pad_or_truncate method for mono and stereo audio.
    """
    # Test truncation
    audio_processor.y_processed = np.random.rand(audio_processor.y_processed.shape[0], 88200)  # 2 seconds of audio
    center_original = audio_processor.y_processed[:, 22050:66150]  # Center 1 second of original
    audio_processor.pad_or_truncate(1000)  # Target length: 1 second
    center_truncated = audio_processor.y_processed    
    assert audio_processor.y_processed.shape[1] == 44100  # Ensure the length is truncated to 1 second
    assert np.all(center_original[0] == center_truncated[0]) # Ensure the center of the audio is preserved

    # Test padding
    audio_processor.y_processed = np.random.rand(audio_processor.y_processed.shape[0], 22050)  # 0.5 seconds of audio
    audio_processor.pad_or_truncate(1000)  # Target length: 1 second
    assert audio_processor.y_processed.shape[1] == 44100  # Ensure the length is padded to 1 second
    # Ensure padding is symmetric
    left_padding = audio_processor.y_processed[:, :11025]
    right_padding = audio_processor.y_processed[:, -11025:]
    assert np.all(left_padding == 0), "Left padding is not zero"
    assert np.all(right_padding == 0), "Right padding is not zero"

@pytest.mark.parametrize("audio_processor, target_channels, expected_shape", [
    (1, 2, (2, 44100)),  # Mono to stereo (duplicate)
    (1, 1, (1, 44100)),  # Mono to mono (same)
    (2, 1, (1, 44100)),  # Stereo to mono (mean)
    (2, 2, (2, 44100)),  # Stereo to stereo (same)
], indirect=["audio_processor"])
def test_rechanneling(audio_processor, target_channels, expected_shape):
    """
    Test the rechanneling method for various input and target channel configurations.
    """
    audio_processor.target_channel = target_channels
    audio_processor.rechanneling()
    # Assert the output shape matches the expected shape
    assert audio_processor.y_processed.shape == expected_shape

    # Additional checks for specific cases
    if audio_processor.channel == 1 and target_channels == 2:  # Mono to stereo
        assert np.all(audio_processor.y_processed[0] == audio_processor.y_processed[1])
    elif audio_processor.channel == 2 and target_channels == 1:  # Stereo to mono
        expected_mono = np.mean(audio_processor.y, axis=0, keepdims=True)
        assert np.all(audio_processor.y_processed == expected_mono)

@pytest.mark.parametrize("audio_processor", [1, 2], indirect=True)
def test_normalize_spectrogram(audio_processor):
    """
    Test the normalize_spectrogram method for a spectrogram.
    """
    # Generate a dummy spectrogram
    spectrogram = np.random.rand(128, 128) * 100  # Random spectrogram with large values

    # Normalize the spectrogram
    normalized = audio_processor.normalize_spectrogram(spectrogram)

    # Assert the normalized spectrogram is within the range [-1, 1]
    assert np.all(normalized >= -1) and np.all(normalized <= 1), "Normalized spectrogram is not in range [-1, 1]"

@pytest.mark.parametrize("audio_processor, use_mel_spectrogram, target_channels, expected_shape", [
    (1, False, 2, (2, 1025, 87)),  # Mono to stereo, standard spectrogram 
    (1, True, 1, (1, 128, 87)),    # Mono to mono, mel-spectrogram
    (2, True, 1, (1, 128, 87)),    # Stereo to mono, mel-spectrogram 
    (2, False, 2, (2, 1025, 87)),  # Stereo to stereo, standard spectrogram
], indirect=["audio_processor"])
def test_preprocess(audio_processor, use_mel_spectrogram, target_channels, expected_shape):
    """
    Test the preprocess method for various configurations with a final audio length of 1000 ms.
    expected shape = (number of channel, frequency bins, time frames)
    frequency bins = n_fft // 2 + 1 = 1025 for spectrogram, n_mel=128 for mel-spectrogram
    time frames = (N + librosa padding - n_fft) / hop) + 1 = 87
        with padding = hop_length - (N mod hop_length) 
    """
    audio_processor.target_length_ms = 1000
    audio_processor.target_channel = target_channels
    spectrogram = audio_processor.preprocess(use_mel_spectrogram=use_mel_spectrogram, n_mels=128, n_fft=2048, hop_length=512)

    # Assert the output is a numpy array
    assert isinstance(spectrogram, np.ndarray), "Output is not a numpy array"

    # Assert the spectrogram is not empty
    assert spectrogram.size > 0, "Spectrogram is empty"

    # Assert the spectrogram values are within the range [-1, 1]
    assert np.all(spectrogram >= -1) and np.all(spectrogram <= 1), "Spectrogram values are not in range [-1, 1]"

    # Assert the shape of the spectrogram matches the expected shape
    assert spectrogram.shape == expected_shape, f"Unexpected spectrogram shape: {spectrogram.shape}"
