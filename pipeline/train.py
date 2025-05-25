import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np

import config
from model import ECGClassifier


class DummyECGDataset(Dataset):
    def __init__(self, num_samples, min_len, max_len, num_classes):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.num_classes = num_classes
        self.data = []
        self.labels = []
        self.lengths = []
        for _ in range(num_samples):
            seq_len = np.random.randint(min_len, max_len + 1)
            ecg_signal = torch.randn(seq_len)
            self.data.append(ecg_signal)
            self.labels.append(torch.randint(0, num_classes, (1,)).item())
            self.lengths.append(seq_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.lengths[idx], self.labels[idx]


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
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Acc: {epoch_acc_val:.2f}%")

    print("Finished Training")


if __name__ == "__main__":
    cfg = config
    train_dataset = DummyECGDataset(cfg.NUM_TRAIN_SAMPLES, cfg.MIN_SEQ_LEN, cfg.MAX_SEQ_LEN, cfg.NUM_CLASSES)
    val_dataset = DummyECGDataset(cfg.NUM_VAL_SAMPLES, cfg.MIN_SEQ_LEN, cfg.MAX_SEQ_LEN, cfg.NUM_CLASSES)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = ECGClassifier(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    train_model(model, train_loader, val_loader, criterion, optimizer, cfg.NUM_EPOCHS, cfg.DEVICE)
    # torch.save(model.state_dict(), "ecg_classifier_model.pth")
    # print("Model saved to ecg_classifier_model.pth")