import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import config
from model import ECGClassifier
from metrics import plot_acc, plot_loss
from data import load_data


def collate_fn(batch):
    data = [item[0] for item in batch]
    lengths = torch.tensor([item[1] for item in batch], dtype=torch.long)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    padded_data = pad_sequence(data, batch_first=True, padding_value=0.0)
    packed_data = pack_padded_sequence(padded_data, lengths,
                                       batch_first=True, enforce_sorted=False)
    return packed_data, lengths, labels


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    print(f"Training on {device}")
    model.to(device)

    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, (packed_inputs, lengths, labels) in enumerate(train_loader):
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(packed_inputs.to(device), lengths.to(device))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc_train = 100 * correct_train / total_train
        train_loss_list.append(epoch_loss)
        train_acc_list.append(epoch_acc_train)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc_train:.2f}%")

        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        with torch.no_grad():
            for packed_inputs, lengths, labels in val_loader:
                labels = labels.to(device)
                outputs = model(packed_inputs.to(device), lengths.to(device))
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_acc_val = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(epoch_acc_val)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Acc: {epoch_acc_val:.2f}%")

    print("Finished Training")
    return train_acc_list, val_acc_list, train_loss_list, val_loss_list

def plot_confusion_matrix(model, dataloader, device, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for packed_inputs, lengths, labels in dataloader:
            labels = labels.to(device)
            outputs = model(packed_inputs.to(device), lengths.to(device))
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    cfg = config
    train_dataset, val_dataset = load_data(test_size=0.2, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = ECGClassifier(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    train_acc_list, val_acc_list, train_loss_list, val_loss_list = train_model(model, train_loader, val_loader, criterion, optimizer, cfg.NUM_EPOCHS, cfg.DEVICE)
    # torch.save(model.state_dict(), "ecg_classifier_model.pth")
    # print("Model saved to ecg_classifier_model.pth")

    # Metrics
    plot_acc(train_acc_list, val_acc_list)
    plot_loss(train_loss_list, val_loss_list)
    class_names = ["Normal", "AF", "Other", "Noisy"]
    plot_confusion_matrix(model, val_loader, cfg.DEVICE, class_names)