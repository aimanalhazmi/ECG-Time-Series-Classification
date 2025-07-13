import argparse
import config
from src.data_loader import load, ECGDataset
from src.transforms import ECGTransform
from src.train import train_evaluate, evaluate_best_model
from src.test import predict
from src.utils import *

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from src.model import ModelSelector
from src.metrics import (
    plot_evaluation_metric,
    plot_loss)
from sklearn.model_selection import train_test_split

def train_pipeline(augmented: bool = False, reduced: bool = False):
    cfg = config

    if reduced:
        task_prefix = "train_results_reduced"
        print("Starting pipeline for REDUCTION task...")
    elif augmented:
        task_prefix = "train_results_augmented"
        print("Starting pipeline for AUGMENTED MODELING task...")
    else:
        task_prefix = "train_results_baseline"
        print("Starting pipeline for BASELINE MODELING task...")

    save_to = create_dir(cfg.OUTPUTS, f"{task_prefix}_{cfg.MODEL_NAME}")
    save_config_to_txt(cfg, save_to)

    ecg_signals, labels = load(cfg, train_data=True)

    print("Splitting Train ECG signals into training and validation sets.")
    X_train, X_val, y_train, y_val = train_test_split(
        ecg_signals,
        labels,
        test_size=cfg.TEST_SIZE,
        stratify=labels,
        random_state=cfg.RANDOM_SEED,
    )
    print(f"Split completed: {len(X_train)} training samples and {len(X_val)} validation samples.")

    transform = None
    if augmented:
        print("Applying data augmentation to the training set...")
        transform = ECGTransform(cfg)

    if reduced:
        # TODO: Implement reduction logic here on X_train and X_val if needed
        raise NotImplementedError("This Task is not implemented yet.")


    train_dataset = ECGDataset(X_train, y_train, transform=transform)
    val_dataset = ECGDataset(X_val, y_val, transform=None)  # No augmentation on validation set

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
    )

    print(f"Initializing model: {cfg.MODEL_NAME}")
    model = ModelSelector.get_model(cfg.MODEL_NAME, cfg)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    best_model_dict, train_metrics = train_evaluate(
        model, train_loader, val_loader, criterion, optimizer, cfg, save_model_to=save_to,
    )

    model.load_state_dict(best_model_dict)

    print("Training completed successfully. Plotting evaluation metrics...")
    plot_evaluation_metric(
        train_metrics["acc"]["train"], train_metrics["acc"]["val"],
        metric="Accuracy", save=cfg.SAVE, save_to=save_to,
    )
    plot_evaluation_metric(
        train_metrics["f1"]["train"], train_metrics["f1"]["val"],
        metric="F1-Score", save=cfg.SAVE, save_to=save_to,
    )
    plot_loss(
        train_metrics["loss"]["train"], train_metrics["loss"]["val"],
        save=cfg.SAVE, save_to=save_to,
    )
    evaluate_best_model(model=model, dataloader=val_loader, cfg=cfg, save_to=save_to)
    print(f"Saved results & best model to '/{'/'.join(save_to.strip('/').split('/')[-2:])}' ...")


def test_pipeline(prediction_file, model_path):
    cfg = config
    output_dir = cfg.OUTPUTS

    ecg_signals, _ = load(cfg, train_data=False)
    test_dataset = ECGDataset(ecg_signals, None)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.NUM_WORKERS,
    )

    state_dict, model_name = get_saved_model(output_dir, device=cfg.DEVICE, model_path=model_path)
    print(f"Inferred model architecture: {model_name}")

    model = ModelSelector.get_model(model_name, cfg)
    model.load_state_dict(state_dict)

    save_to = create_dir(output_dir, f"test_results_{prediction_file.split(".")[0]}_{model_name}", use_timestamp=False)
    predict(
        model=model,
        dataloader=test_loader,
        device=cfg.DEVICE,
        prediction_file=os.path.join(save_to, prediction_file),
    )


def choose_train_task(task):
    if task == "modeling":
        train_pipeline(augmented=False, reduced=False)
    elif task == "modeling_augmented":
        train_pipeline(augmented=True, reduced=False)
    elif task == "reduction":
        train_pipeline(augmented=True, reduced=True)
    else:
        raise ValueError(f"Unknown task: {task}")

def choose_test_pipeline(task, model_path=""):
    if task == "modeling":
        test_pipeline(config.PREDICTION_BASE_FILE, model_path)
    elif task == "modeling_augmented":
        test_pipeline(config.PREDICTION_AUGMENT_FILE, model_path)
    elif task == "reduction":
        test_pipeline(config.PREDICTION_REDUCTION_FILE, model_path)
    else:
        raise ValueError(f"Unknown task: {task}")


def main():
    parser = argparse.ArgumentParser(
        description="Train or predict for a specific task."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        required=True,
        help="Choose 'train' to train the model or 'predict' to make predictions."
    )
    parser.add_argument(
        "--task",
        choices=["modeling", "modeling_augmented", "reduction"],
        required=True,
        help="Specify the task: 'modeling', 'modeling_augmented', or 'reduction'."
    )
    parser.add_argument(
        "--model",
        help="Path to the trained model (.pt file) to use for prediction."
    )

    args = parser.parse_args()

    if args.mode == "train":
        choose_train_task(task=args.task)
    elif args.mode == "predict":
        if args.model:
            choose_test_pipeline(task=args.task, model_path=args.model)
        else:
            parser.error("--model is required when --mode is 'predict' (provide path to trained .pt model)")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()