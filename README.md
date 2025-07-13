# AMLS SoSe 2025 – ECG Time Series Classification

## Project Overview

The goal of this project is to develop and evaluate machine learning pipelines for classifying univariate ECG time series data into four rhythm categories:

- **0** – Normal  
- **1** – AF (Atrial Fibrillation)  
- **2** – Other rhythms  
- **3** – Noisy (unclassifiable)

The ECG signals are sampled at 300 Hz and provided in binary format. Labels are only available for the training data. We address the following tasks:

1. **Dataset Exploration**: Analyze signal statistics, class distribution, and define a validation split that reflects the overall dataset.
2. **Modeling and Tuning**: Train at least two different model architectures, evaluate performance on train and validation sets, and tune hyperparameters.
3. **Data Augmentation & Feature Engineering**: Enhance model robustness with time/frequency domain augmentations and optional feature extraction.
4. **Data Reduction**: Reduce dataset size using sampling, compression, or embeddings, and evaluate model performance at different reduction levels.

The final deliverables include a report, runnable code for all tasks, and test prediction files.

## Structure
```
amls-ecg-time-series-classification/
├── data/                        # Raw or processed datasets
├── outputs/                     # Model outputs, logs, predictions, plots
├── src/                         # Source code modules (e.g., model, train, test, utils, etc.)
├── data_exploration.ipynb       # Jupyter notebook for initial data analysis
├── config.py                    # Configuration variables and constants
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Setup

1.  **Clone and Navigate to the Project Folder**
    ```bash
    cd amls-ecg-time-series-classification
    ```
2.  **Create and Activate a Virtual Environment**

    (This only works for linux and macOS systems since there is no support for makefile on windows. 
On windows please install the requirements manually.)
    -   **`make` or `make install`**: Creates a `.venv` and installs Python dependencies.
    -   **`make activate`**: Prints the activation command for your shell.
    -   **`make clean`**: Deletes the virtual environment.

    To set up and activate the environment:
    ```bash
    make
    source .venv/bin/activate
    ```

## Model Architectures

This project includes multiple model architectures, which can be selected in the `config.py` file.

-   **`ECGClassifier`**: The primary model, which uses a combination of Short-Time Fourier Transform (STFT) to create a spectrogram, followed by 2D Convolutional layers and an LSTM to classify the sequence.
-   **`SimplifiedECGClassifier`**: A simpler baseline model that uses a bidirectional LSTM directly on the raw, normalized time-series data without any frequency-domain transformation.

## How to Run

This project uses a script named main.py to handle training and prediction for different tasks.

### 1. Configure the Model

Open `config.py` and set the `MODEL_NAME` variable to the architecture you want to train or evaluate.

```python
# In config.py
MODEL_NAME = "ECGClassifier"  # or "SimplifiedECGClassifier"
```

### 2. Train a Model

Run the training script from the root directory of the project. It will use the model specified in `config.py`.

```bash
python3 main.py --mode train --task modeling
```

### Available Modes
- train – Train the model
- predict – Run predictions using a trained model

### Available Tasks
| Task Name            | Description                                             |
|----------------------|---------------------------------------------------------|
| `modeling`           | Standard model training and evaluation                  |
| `modeling_augmented` | Modeling using augmented data                           |
| `reduction`          | Dimensionality reduction + modeling with augmented data |

-   The script will train the selected model, split the data, and evaluate on a validation set.
-   Results, including performance plots and the best model weights (`best_model.pt`), will be saved to a uniquely named directory that includes the model name and a timestamp, for example: `outputs/train_results_baseline_ECGClassifier_20250710_112836/`.

### 3. Run Predictions

To generate predictions on the test set, run the testing script.

e.g. 
```bash
python3 main.py --mode predict --task modeling_augmented --model outputs/train_results_augmented_ECGClassifier_20250710_173927/final_model_augmentation_task.pt
```

-   **Automatic Model Detection**: By default, the script automatically finds the most recently trained model in the `outputs/` directory. It intelligently infers the model architecture (e.g., `ECGClassifier`) from the directory name and loads the correct model.
-   **Specifying a Model**: To use a specific model file, update the `BEST_MODEL_PATH` variable in `config.py` with the direct path to your `best_model.pt` file.
    ```python
    # In config.py, to specify a model for prediction
    BEST_MODEL_PATH = "outputs/train_results_SimplifiedECGClassifier_20250708_160000/final_model_augmentation_task.pt"
    ```
-   Prediction results will be saved in a new directory named after the model being tested, such as `outputs/test_results_SimplifiedECGClassifier/`.

## Team

-   Aiman Al-Hazmi
-   Friedrich Hagedorn