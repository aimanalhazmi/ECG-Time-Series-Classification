import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def plot_evaluation_metric(train_metric, val_metric, metric="Accuracy"):
    epochs = list(range(1, len(train_metric) + 1))
    plt.plot(epochs, train_metric, label="Train", marker="o")
    plt.plot(epochs, val_metric, label="Validation", marker="s")
    plt.title(f"{metric} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(f"{metric} (%)")
    plt.xticks(range(1, len(train_metric) + 1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{metric}.png")
    # plt.show()
    plt.close()


def plot_loss(train_loss, val_loss):
    epochs = list(range(1, len(train_loss) + 1))
    plt.plot(epochs, train_loss, label="Train", marker="o")
    plt.plot(epochs, val_loss, label="Validation", marker="s")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, len(train_loss) + 1))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/loss.png")
    # plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, target_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    # plt.show()
    plt.close()


def plot_classification_report(y_true, y_pred, target_names):
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    rows = []
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
    rows.append(
        [
            report["macro avg"]["precision"],
            report["macro avg"]["recall"],
            report["macro avg"]["f1-score"],
        ]
    )
    rows.append(
        [
            report["weighted avg"]["precision"],
            report["weighted avg"]["recall"],
            report["weighted avg"]["f1-score"],
        ]
    )

    report_array = np.array(rows)
    row_labels += ["Macro Avg", "Weighted Avg"]
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        report_array,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        xticklabels=["Precision", "Recall", "F1-score"],
        yticklabels=row_labels,
    )
    plt.title("Classification Report")
    plt.tight_layout()
    plt.savefig("results/classification_report.png")
    # plt.show()
    plt.close()
