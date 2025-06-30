import torch
import zipfile
import struct
import pandas as pd
import os
from torch.utils.data import Dataset


def read_zip_binary(zip_path):
    """Read binary data from a zip file."""
    ragged_array = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        base_name = os.path.basename(zip_path)
        inner_path = os.path.splitext(base_name)[0] + ".bin"

        with zf.open(inner_path, "r") as r:
            while True:
                size_bytes = r.read(4)
                if not size_bytes:
                    break
                sub_array_size = struct.unpack("i", size_bytes)[0]
                sub_array = list(
                    struct.unpack(f"{sub_array_size}h", r.read(sub_array_size * 2))
                )
                ragged_array.append(sub_array)
    return ragged_array


class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels
        self.lengths = [len(signal) for signal in signals]

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float)
        length = self.lengths[idx]
        label = self.labels[idx]

        std = signal.std()
        mean = signal.mean()
        if std > 0:
            signal = (signal - mean) / std
        else:
            signal = signal - mean
        return signal, length, label


def load_data():
    """Load and split ECG data using code from data.ipynb"""
    # Load signals
    base_dir = "../data"
    zip_data_path = os.path.join(base_dir, "X_train.zip")
    print("Reading ECG signals from:", zip_data_path)
    ecg_signals = read_zip_binary(zip_data_path)
    print(f"Loaded {len(ecg_signals)} ECG signals.")

    # Load labels
    labels_path = os.path.join(base_dir, "y_train.csv")
    print("Reading labels from:", labels_path)
    labels_df = pd.read_csv(labels_path, header=None, names=["label"])
    print(f"Loaded {len(labels_df)} labels.")
    labels = labels_df["label"].values

    return ecg_signals, labels
