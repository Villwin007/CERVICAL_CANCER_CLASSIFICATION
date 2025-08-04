import os

BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 5
EPOCHS = 30
LR = 1e-4
DEVICE = 'cuda'  # use 'cpu' if no GPU

# Paths
DATA_DIR = "data/raw/cervicalCancer"
MODEL_DIR = "outputs/models"
RESULTS_DIR = "outputs/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
