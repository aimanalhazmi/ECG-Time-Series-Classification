import matplotlib.pyplot as plt

def plot_acc(train_acc, val_acc):
    epochs = list(range(1, len(train_acc) + 1))
    plt.plot(epochs, train_acc, label="Train", marker="o")
    plt.plot(epochs, val_acc, label="Validation", marker="s")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/accuracy.png")
    plt.show()

def plot_loss(train_loss, val_loss):
    epochs = list(range(1, len(train_loss) + 1))
    plt.plot(epochs, train_loss, label="Train", marker="o")
    plt.plot(epochs, val_loss, label="Validation", marker="s")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/loss.png")
    plt.show()
