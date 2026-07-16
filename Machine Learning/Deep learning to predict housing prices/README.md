
# 🏠 California Housing Price Prediction using Deep Neural Networks

A production-style machine learning project that predicts median house prices using a TensorFlow/Keras neural network.  

The project demonstrates a complete ML lifecycle:

- Data loading
- Data preprocessing
- Model development
- Training pipeline
- Model evaluation
- Hyperparameter tuning
- Model persistence
- Batch inference

The project is structured like a real-world ML engineering repository rather than a single Jupyter notebook.

---

# Project Architecture

housing-ml/

│
├── data/
│ └── housing.csv
│
├── sample_data/
│ └── sample.csv
│
├── models/
│ ├── model.keras
│ ├── scaler.pkl
│ └── history.pkl
│
├── src/
│ ├── config.py
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── model.py
│ ├── trainer.py
│ ├── evaluate.py
│ └── tuner.py
│
├── train.py
├── predict.py
├── tune.py
├── requirements.txt
└── README.md


---

# Problem Statement

Given California housing data, predict: median_house_value

using geographical, demographic, and housing features.

This is a supervised regression problem.

---

# Dataset

The project uses the California Housing dataset.

Features:

| Feature | Description |
|---|---|
| longitude | Geographic longitude |
| latitude | Geographic latitude |
| housing_median_age | Median age of houses |
| total_rooms | Total rooms in area |
| population | Population |
| households | Number of households |
| median_income | Median income |

Target:


median_house_value


---

# Machine Learning Pipeline

         housing.csv
              |
              v
      Data Loading
              |
              v
      Train/Test Split
              |
              v
      MinMaxScaler
              |
              v
    Neural Network Model
              |
              v
    Training + Callbacks
              |
              v
      Model Evaluation
              |
              v
      Saved Model Artifact
              |
              v
         Prediction


         ---

# Model Architecture

The model uses a fully connected neural network.

Example architecture:

Input Layer
|
Dense(32, ReLU)
|
Dense(32, ReLU)
|
Dense(32, ReLU)
|
Output Layer



Optimizer:


Adam


Loss:


Mean Squared Error


Metrics:


RMSE
MAE

# Installation

Clone repository:

```bash
git clone <repository-url>

cd neural_nets

Install dependencies:

pip install -r requirements.txt

Training

Run:

python train.py

The pipeline will:

Load the dataset
Split data
Fit preprocessing pipeline
Train neural network
Save model artifacts
Evaluate performance

Generated files:

models/

model.keras
scaler.pkl
history.pkl

Hyperparameter Tuning

The project uses KerasTuner Hyperband optimization.

Run:

python tune.py

Parameters tuned:

Number of layers
Number of neurons
Dropout rate
Learning rate

Example search:

Dense layers:
16 - 128 neurons

Learning rate:
0.001
0.0005
0.0001

Dropout:
0 - 30%
Prediction

Prediction uses saved production artifacts.

Example:

python predict.py --input sample_data/sample.csv

Output:

Predictions:

$413810.31

The inference pipeline:

Input CSV
    |
    v
Saved Scaler
    |
    v
Saved Neural Network
    |
    v
Prediction
Model Evaluation

Metrics used:

RMSE

Measures prediction error magnitude.

Lower is better.

MAE

Average absolute prediction error.

R² Score

Measures explained variance.
