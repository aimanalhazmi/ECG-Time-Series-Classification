import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import os
from datetime import datetime
from pathlib import Path
import config as cfg


def get_saved_model(output_dir):
    model_path = cfg.BEST_MODEL_PATH
    if os.path.exists(model_path):
        print(f"Loading best model from {model_path}")
        model = torch.load(model_path)
        parent_name = Path(model_path).parent.name
        model_dir = parent_name.split("_")[-1] if "_" in parent_name else parent_name
    else:
        print(f"No best model found at {model_path}")
        latest_dir = latest_train_dir(output_dir)
        fallback_model_path = os.path.join(output_dir, latest_dir, "best_model.pt")

        if not os.path.exists(fallback_model_path):
            raise FileNotFoundError(
                f"'best_model.pt' not found in latest directory: {fallback_model_path}"
            )
        print(f"Loading best model from {fallback_model_path}")
        model = torch.load(str(fallback_model_path))
        dir_name = Path(latest_dir).name
        model_dir = "_".join(dir_name.split("_")[-2:]) if "_" in dir_name else dir_name

    return model, model_dir

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
    # Find the most recent train_results_baseline* directory
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
