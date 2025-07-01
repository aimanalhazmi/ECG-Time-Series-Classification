import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os
from datetime import datetime


def collate_fn(batch):
    data = [item[0] for item in batch]
    lengths = torch.tensor([item[1] for item in batch], dtype=torch.long)
    if batch[0][2] is not None:
        labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    else:
        labels = None
    padded_data = pad_sequence(data, batch_first=True, padding_value=0.0)
    packed_data = pack_padded_sequence(
        padded_data, lengths, batch_first=True, enforce_sorted=False
    )
    return packed_data, lengths, labels


def latest_train_dir(output_dir):
    # Find the most recent train_results_1* directory
    train_dirs = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("train_results") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not train_dirs:
        raise FileNotFoundError("No training result directories found.")

    latest_dir = max(
        train_dirs,
        key=lambda name: os.path.getmtime(os.path.join(output_dir, name)),
    )
    return latest_dir


def create_dir(output_dir, sub_dir, use_timestamp=True):
    output = os.path.abspath(output_dir)
    os.makedirs(output, exist_ok=True)

    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sub_dir_name = f"{sub_dir}_{timestamp}"
    else:
        sub_dir_name = sub_dir
    save_to = os.path.join(output, sub_dir_name)
    os.makedirs(save_to, exist_ok=True)
    return save_to
