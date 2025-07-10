import os

import torch

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# DATA
DATA_DIR = "data"
X_TRAIN = os.path.join(DATA_DIR, "X_train.zip")
Y_TRAIN = os.path.join(DATA_DIR, "y_train.csv")
X_TEST = os.path.join(DATA_DIR, "X_test.zip")
TARGET_NAMES = ["Normal", "AF", "Other", "Noisy"]

# OUTPUTS DIR
OUTPUTS = "outputs/"
SAVE = True

# PREDICTIONS ON TEST DATASET
BEST_MODEL_PATH = ""
PREDICTION_BASE_FILE = "base.csv"
PREDICTION_AUGMENT_FILE = "augment.csv"
PREDICTION_REDUCTION_FILE = "reduced.csv"

# DATA AUGMENTATION
DO_ADD_NOISE = True
DO_AMPLITUDE_SCALING = True
DO_TIME_SHIFT = True
DO_FREQUENCY_AUGMENT = True
NOISE_LEVEL = 0.01
SCALE_RANGE = (0.8, 1.2)
SHIFT_RANGE = (-10, 10)
FREQ_NOISE_LEVEL = 0.05


# Training Parameters
MODEL_NAME = "SimplifiedECGClassifier"  # "ECGClassifier" or "SimplifiedECGClassifier"
TEST_SIZE = 0.2
RANDOM_SEED = 42
NUM_CLASSES = 4
MAX_SEQ_LEN = 3000
MIN_SEQ_LEN = 500

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 1
NUM_WORKERS = 0

N_FFT = 256
HOP_LENGTH = N_FFT // 4

CONV1_OUT_CHANNELS = 16
CONV1_KERNEL_SIZE = (5, 3)
CONV1_POOL_KERNEL_SIZE = (2, 2)
CONV1_POOL_STRIDE = (2, 2)

CONV2_OUT_CHANNELS = 32
CONV2_KERNEL_SIZE = (5, 3)
CONV2_POOL_KERNEL_SIZE = (2, 2)
CONV2_POOL_STRIDE = (2, 2)

RNN_HIDDEN_SIZE = 64
RNN_NUM_LAYERS = 1

NUM_TRAIN_SAMPLES = 64
NUM_VAL_SAMPLES = 32
