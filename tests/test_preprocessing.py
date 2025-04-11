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



