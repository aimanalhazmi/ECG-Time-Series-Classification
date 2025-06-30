import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import config
from model import ECGClassifier
from metrics import (
    plot_evaluation_metric,
    plot_loss,
    plot_confusion_matrix,
    plot_classification_report,
)
from data import load_data, ECGDataset
from tqdm import tqdm
from time import time


def collate_fn(batch):
    data = [item[0] for item in batch]
    lengths = torch.tensor([item[1] for item in batch], dtype=torch.long)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    padded_data = pad_sequence(data, batch_first=True, padding_value=0.0)
    packed_data = pack_padded_sequence(
        padded_data, lengths, batch_first=True, enforce_sorted=False
    )
    return packed_data, lengths, labels


def train(model, train_loader, criterion, optimizer, device):
    print("Training...")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    y_pred = []
    y_true = []

    for packed_inputs, lengths, labels in tqdm(train_loader):
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(packed_inputs.to(device), lengths.to(device))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    f1 = f1_score(y_true, y_pred, average="weighted")
    return avg_loss, accuracy, (f1 * 100)


def evaluate(model, val_loader, criterion, device):
    print("Evaluating...")
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    y_pred = []
    y_true = []

    with torch.no_grad():
        for packed_inputs, lengths, labels in tqdm(val_loader):
            labels = labels.to(device)
            outputs = model(packed_inputs.to(device), lengths.to(device))
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    f1 = f1_score(y_true, y_pred, average="weighted")
    return avg_loss, accuracy, (f1 * 100)


def train_evaluate(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device
):
    print(f"Training will run on: {device}")
    model.to(device)

    best_val_loss = float("inf")
    best_model = None

    train_loss_list = []
    val_loss_list = []

    train_metrics = {}
    train_acc_list = []
    val_acc_list = []
    train_f1_list = []
    val_f1_list = []
    start_time = time()
    for epoch in range(num_epochs):
        print(f"{'-'*25} Epoch {epoch + 1}/{num_epochs} {'-'*25}")
        train_loss, train_acc, train_f1_score = train(
            model, train_loader, criterion, optimizer, device
        )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        train_f1_list.append(train_f1_score)

        val_loss, val_acc, val_f1_score = evaluate(model, val_loader, criterion, device)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        val_f1_list.append(val_f1_score)
        print(f"Train Loss: {train_loss:.4f}, Train F1-Score: {train_f1_score:.2f}%")
        print(
            f"Validation Loss: {val_loss:.4f}, Validation F1-Score: {val_f1_score:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            # torch.save(best_model, "best_model.pt")
            print("Saving new best model..")
        print(f"{'-'*25} End of Epoch {epoch+1} {'-'*25}")

    end_time = time()
    print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")
    train_metrics["acc"] = {"train": train_acc_list, "val": val_acc_list}
    train_metrics["f1"] = {"train": train_f1_list, "val": val_f1_list}
    train_metrics["loss"] = {"train": train_loss_list, "val": val_loss_list}
    return best_model, train_metrics


def evaluate_best_model(model, dataloader, device, target_names=None):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for packed_inputs, lengths, labels in dataloader:
            labels = labels.to(device)
            outputs = model(packed_inputs.to(device), lengths.to(device))
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    plot_confusion_matrix(y_true=y_true, y_pred=y_pred, target_names=target_names)
    plot_classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
    print(f"Evaluation of best saved model completed — F1-Score: {f1 * 100:.2f}")


def plot_samples(train_loader, target_names):
    for batch in train_loader:
        packed, lengths, labels = batch
        padded, _ = pad_packed_sequence(packed, batch_first=True)
        for i in range(4):
            plt.plot(padded[i][: lengths[i]].cpu())
            plt.title(f"Label: {target_names[labels[i].item()]}")
            plt.show()
            plt.close()
        break


if __name__ == "__main__":
    cfg = config
    ecg_signals, labels = load_data()

    print("Splitting ECG signals into training and test sets.")
    X_train, X_val, y_train, y_val = train_test_split(
        ecg_signals,
        labels,
        test_size=cfg.TEST_SIZE,
        stratify=labels,
        random_state=cfg.RANDOM_SEED,
    )
    print(
        f"Split completed: {len(X_train)} training samples and {len(X_val)} validation samples."
    )
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(cfg.DEVICE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.NUM_WORKERS,
    )

    target_names = ["Normal", "AF", "Other", "Noisy"]
    # plot_samples(train_loader, target_names=target_names)

    model = ECGClassifier(cfg)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    best_model_dict, train_metrics = train_evaluate(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        cfg.NUM_EPOCHS,
        cfg.DEVICE,
    )

    best_model = ECGClassifier(cfg)
    best_model.load_state_dict(best_model_dict)

    # Metrics
    plot_evaluation_metric(
        train_metrics["acc"]["train"], train_metrics["acc"]["val"], metric="Accuracy"
    )
    plot_evaluation_metric(
        train_metrics["f1"]["train"], train_metrics["f1"]["val"], metric="F1-Score"
    )
    plot_loss(train_metrics["loss"]["train"], train_metrics["loss"]["val"])
    evaluate_best_model(
        model=best_model,
        dataloader=val_loader,
        device=cfg.DEVICE,
        target_names=target_names,
    )
