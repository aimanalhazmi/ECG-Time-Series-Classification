import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import os
from datetime import datetime
from pathlib import Path
import inspect

def get_saved_model(output_dir, model_path=""):
    """
    Finds and loads a saved model state dictionary, and infers the model name from the path.

    Args:
        output_dir (str): The base directory where training results are saved.
        model_path (str, optional): A direct path to a 'final_model_augmentation_task.pt' file.

    Returns:
        tuple: A tuple containing:
            - model_state_dict (dict): The loaded model's state dictionary.
            - model_name (str): The inferred name of the model architecture.
    """

    def _get_model_name_from_path(path):
        dir_name = Path(path).parent.name
        parts = dir_name.split("_")
        # Expects dir name like: train_results_MODELNAME_TIMESTAMP
        if len(parts) >= 3 and parts[0] == "train" and parts[1] == "results":
            return parts[2]
        print(
            f"Could not infer model name from path '{dir_name}'. "
            f"Using directory name as identifier."
        )
        return dir_name

    final_model_path = ""
    if os.path.exists(model_path):
        print(f"Loading best model from {model_path}")
        final_model_path = model_path
    else:
        print(f"No best model found at {model_path}, searching latest in {output_dir}")
        latest_dir = latest_train_dir(output_dir)
        fallback_model_path = os.path.join(output_dir, latest_dir, "final_model_augmentation_task.pt")

        if not os.path.exists(fallback_model_path):
            raise FileNotFoundError(
                f"'final_model_augmentation_task.pt' not found in latest directory: {fallback_model_path}"
            )
        print(f"Loading best model from {fallback_model_path}")
        final_model_path = fallback_model_path

    model_state_dict = torch.load(str(final_model_path))
    model_name = _get_model_name_from_path(final_model_path)

    return model_state_dict, model_name


def collate_fn(batch):
    data = [item[0] for item in batch]
    lengths = torch.tensor([len(d) for d in data], dtype=torch.long)
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
    # Find the most recent train_results* directory
    train_dirs = [
        d
        for d in os.listdir(output_dir)
        if d.startswith("train_results") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not train_dirs:
        raise FileNotFoundError(f"No training result directories found in {output_dir}.")

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

def save_config_to_txt(cfg, save_to):
    config_items = {k: v for k, v in cfg.__dict__.items() if not k.startswith("__") and not inspect.ismodule(v)}
    with open(os.path.join(save_to, "run_config.txt"), "w") as f:
        for k, v in sorted(config_items.items()):
            f.write(f"{k}: {v}\n")