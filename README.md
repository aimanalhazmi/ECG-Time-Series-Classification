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

The final deliverables include a report, runnable code for all tasks, and three test prediction files (`base.csv`, `augment.csv`, `reduced.csv`).


## Repository Structure



## How to Run

1. **Clone and Navigate to the Project Folder**
   ```bash
   cd amls-ecg-time-series-classification
   ```
2. Create and Activate a Virtual Environment

- **make or make install:**  Create .venv and install Python dependencies

- **make activate:**  Prints the activation command

- **make clean:**  Deletes the virtual environment

To set up and activate the environment:
   ```bash
    make         
    source .venv/bin/activate
  ```
      
## Team
- Aiman Al-Hazmi
- Pia Droop
- Friedrich Hagedorn