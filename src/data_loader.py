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


def normalize(signal):
    std = signal.std()
    mean = signal.mean()
    if std > 0:
        return (signal - mean) / std
    return signal - mean


class ECGDataset(Dataset):
    def __init__(self, signals, labels, transform=None):
        self.signals = signals
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        if self.transform:
            signal = self.transform(signal)

        length = len(signal)
        signal = torch.tensor(signal, dtype=torch.float)
        signal = normalize(signal)

        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None

        return signal, length, label


def load(cfg, train_data: bool):
    """
    Load ECG data from a zip file.
    """

    if train_data:
        # Load signals
        print("Reading Train ECG signals from:", cfg.X_TRAIN)
        ecg_signals = read_zip_binary(cfg.X_TRAIN)
        print(f"Loaded {len(ecg_signals)} Train ECG signals.")

        # Load labels
        print("Reading Training labels from:", cfg.Y_TRAIN)
        labels_df = pd.read_csv(cfg.Y_TRAIN, header=None, names=["label"])
        print(f"Loaded {len(labels_df)} labels.")
        labels = labels_df["label"].values
        return ecg_signals, labels
    else:
        print("Reading Test ECG signals from:", cfg.X_TEST)
        test_ecg_signals = read_zip_binary(cfg.X_TEST)
        print(f"Loaded {len(test_ecg_signals)} Test ECG signals.")
        return test_ecg_signals, None