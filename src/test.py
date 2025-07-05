import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_loader import load, ECGDataset
from src.utils import *
from src.model import ECGClassifier
import config
import os
from pathlib import Path


def predict(model, dataloader, device, prediction_file):
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
        str(prediction_file), index=False, sep=";"
    )
    print(
        f"Prediction completed successfully. Output saved to '/{'/'.join(prediction_file.strip('/').split('/')[-3:])}'.")



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

    saved_model, model_dir = get_saved_model(output_dir, model_path=cfg.BEST_MODEL_PATH)

    model = ECGClassifier(cfg)
    model.load_state_dict(saved_model)

    save_to = create_dir(output_dir, f"test_results_{model_dir}", use_timestamp=False)
    predict(
        model=model,
        dataloader=test_loader,
        device=cfg.DEVICE,
        prediction_file=os.path.join(save_to, cfg.PREDICTION_BASE_FILE),
    )
