"""
Configuration constants for UrbanSoundAI.
Paths, target classes, and spectrogram/hyperparameters.
"""
import os

# -----------------------------------------------------------------------------
# Paths (relative to project root)
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "UrbanSound8K")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
METADATA_CSV = os.path.join(DATA_DIR, "metadata", "UrbanSound8K.csv")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")

# Default model filename (used by train.py and predict.py)
DEFAULT_MODEL_NAME = "urbansound_cnn.pt"

# -----------------------------------------------------------------------------
# Target classes (first version: 4 classes only)
# -----------------------------------------------------------------------------
TARGET_CLASSES = [
    "siren",
    "dog_bark",
    "drilling",
    "engine_idling",
]
NUM_CLASSES = len(TARGET_CLASSES)

# -----------------------------------------------------------------------------
# Audio / spectrogram settings (librosa)
# -----------------------------------------------------------------------------
SR = 22050  # Sample rate (UrbanSound8K is 22.05 kHz)
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
# Fixed spectrogram size: (n_mels, time_steps). Pad or trim to this.
SPEC_TIME_STEPS = 173  # ~4 sec at 22050 Hz with hop 512: (22050*4)/512 ≈ 172

# -----------------------------------------------------------------------------
# Training hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2  # Fraction of data for testing
RANDOM_STATE = 42
