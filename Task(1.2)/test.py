import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import load, ECGDataset
from utils import *
from model import ECGClassifier
import config
import os
from pathlib import Path


def predict(model, dataloader, device, prediction_base_file):
    model.to(device)
    model.eval()
    y_pred = []
    print("Starting prediction ...")
    with torch.no_grad():
        for packed_inputs, lengths, _ in tqdm(dataloader):
            outputs = model(packed_inputs.to(device), lengths.to(device))
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
    pd.DataFrame(y_pred, columns=["y_pred"]).to_csv(
        str(prediction_base_file), index=False, sep=";"
    )
    print(
        f"Prediction completed successfully. Output saved to '{prediction_base_file}'."
    )


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


if __name__ == "__main__":
    cfg = config
    output_dir = cfg.OUTPUTS

    ecg_signals, _ = load(cfg, train_data=False)

    test_dataset = ECGDataset(ecg_signals, None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.NUM_WORKERS,
    )

    saved_model, model_dir = get_saved_model(output_dir)

    model = ECGClassifier(cfg)
    model.load_state_dict(saved_model)

    save_to = create_dir(output_dir, f"test_results_{model_dir}", use_timestamp=False)
    predict(
        model=model,
        dataloader=test_loader,
        device=cfg.DEVICE,
        prediction_base_file=os.path.join(save_to, cfg.PREDICTION_BASE_FILE),
    )
