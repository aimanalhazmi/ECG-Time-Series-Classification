import torch
import zipfile
import struct
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def read_zip_binary(zip_path):
    """Read binary data from a zip file."""
    ragged_array = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        base_name = os.path.basename(zip_path)
        inner_path = os.path.splitext(base_name)[0] + ".bin"

        with zf.open(inner_path, 'r') as r:
            while True:
                size_bytes = r.read(4)
                if not size_bytes:
                    break
                sub_array_size = struct.unpack('i', size_bytes)[0]
                sub_array = list(struct.unpack(f'{sub_array_size}h', r.read(sub_array_size * 2)))
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
        return signal, length, label


def load_data(test_size=0.2, random_state=42):
    """Load and split ECG data using code from data.ipynb"""
    # Load signals
    zip_data_path = "..\data\X_train.zip"
    ecg_signals = read_zip_binary(zip_data_path)

    # Load labels
    labels_path = "..\data\y_train.csv"
    labels_df = pd.read_csv(labels_path, header=None, names=["label"])
    labels = labels_df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        ecg_signals,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)

    return train_dataset, val_dataset