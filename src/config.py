import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Path to the seismic data file
SEGFAST_FILE_PATH = os.path.join(DATA_DIR, 'sgy', 'seismic.sgy')

# Size of the generated image
# TARGET_IMAGE_SIZE = (256, 256)

# Batch size for training
# BATCH_SIZE = 36

# Number of training epochs
# NUM_EPOCHS = 1000

# # Learning rate for the optimizer
# LEARNING_RATE = 1e-2

# Directory to save trained models
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Path to save/load the trained model
MODEL_PATH = os.path.join(MODELS_DIR, 'diffusion_model.pth')

# Ensure necessary directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Verify the existence of the seismic data file
if not os.path.exists(SEGFAST_FILE_PATH):
    raise FileNotFoundError(f"Seismic data file not found at {SEGFAST_FILE_PATH}")
