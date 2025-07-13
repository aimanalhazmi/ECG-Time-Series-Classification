import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def plot_evaluation_metric(
    train_metric, val_metric, metric="Accuracy", save=True, save_to=""
):
    plt.figure(figsize=(10, 5))
    epochs = list(range(1, len(train_metric) + 1))
    plt.plot(epochs, train_metric, label="Train", marker="o")
    plt.plot(epochs, val_metric, label="Validation", marker="s")
    plt.title(f"{metric} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(f"{metric} (%)")
    plt.xticks(range(1, len(train_metric) + 1))
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0, len(epochs) + 1, 5))
    plt.tight_layout()
    if save:
        plt.savefig(f"{save_to}/{metric}.png")
    else:
        plt.show()
    plt.close()


def plot_loss(train_loss, val_loss, save=True, save_to=""):
    plt.figure(figsize=(10, 5))
    epochs = list(range(1, len(train_loss) + 1))
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, len(train_loss) + 1))
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0, len(epochs) + 1, 5))
    plt.tight_layout()
    if save:
        plt.savefig(f"{save_to}/loss.png")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, target_names, save=True, save_to=""):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save:
        plt.savefig(f"{save_to}/confusion_matrix.png")
    else:
        plt.show()
    plt.close()


def plot_classification_report(y_true, y_pred, target_names, save=True, save_to=""):
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    rows = []
    supports = []
    row_labels = []

    for label in target_names:
        row_labels.append(label)
        rows.append(
            [
                report[label]["precision"],
                report[label]["recall"],
                report[label]["f1-score"],
            ]
        )
        supports.append(report[label]["support"])
    rows.append(
        [
            report["macro avg"]["precision"],
            report["macro avg"]["recall"],
            report["macro avg"]["f1-score"],
        ]
    )
    supports.append(report["macro avg"]["support"])

    rows.append(
        [
            report["weighted avg"]["precision"],
            report["weighted avg"]["recall"],
            report["weighted avg"]["f1-score"],
        ]
    )
    supports.append(report["weighted avg"]["support"])

    report_array = np.array(rows)
    row_labels += ["Macro Avg", "Weighted Avg"]

    report_array = np.hstack([report_array, np.array(supports).reshape(-1, 1)])

    plt.figure(figsize=(9, 4))
    sns.heatmap(
        report_array,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        xticklabels=["Precision", "Recall", "F1-score", "Support"],
        yticklabels=row_labels,
    )
    plt.title("Classification Report")
    plt.tight_layout()
    if save:
        plt.savefig(f"{save_to}/classification_report.png")
    else:
        plt.show()
    plt.close()

    # Save text version
    with open(f"{save_to}/classification_report.txt", "w") as f:
        f.write(
            classification_report(
                y_true, y_pred, target_names=target_names, zero_division=0
            )
        )
