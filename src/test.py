import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_loader import load, ECGDataset
from src.utils import *
from src.model import ModelSelector
import config
import os


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

    # Ensure parent directory exists
    Path(prediction_file).parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(y_pred, columns=["y_pred"]).to_csv(
        str(prediction_file), index=False, sep=";"
    )
    print(
        f"Prediction completed successfully. Output saved to '/{'/'.join(str(prediction_file).strip('/').split('/')[-3:])}'.")


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

    # Get saved model state and infer the model's architecture name from the path
    saved_state_dict, model_name = get_saved_model(output_dir, device=cfg.DEVICE, model_path=cfg.BEST_MODEL_PATH)

    print(f"Inferred model architecture: {model_name}")

    # Instantiate the correct model architecture
    model = ModelSelector.get_model(model_name, cfg)

    # Load the state dictionary
    model.load_state_dict(saved_state_dict)

    # Create a results directory named after the model being tested
    save_to = create_dir(output_dir, f"test_results_{model_name}", use_timestamp=False)

    predict(
        model=model,
        dataloader=test_loader,
        device=cfg.DEVICE,
        prediction_file=os.path.join(save_to, cfg.PREDICTION_BASE_FILE),
    )