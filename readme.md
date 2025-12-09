# Explainable Machine Learning for Robot Telemetry: Detecting DoS Attacks and Malfunctions


## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)

## Prerequisites

- Python 3.12 or higher
- pip (Python package installer)
- Jupyter Notebook

## Installation

### 1. Create and Activate Virtual Environment

**Linux / Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

## Dataset Setup

### Install  the Dataset

1. Dataset are in the githut root folder
just run the cell and load them in 
their respetive dataframe


## Running the Project

Execute the notebook sections in order:

1. **Data Loading** - Load dataset and encode target variable (y) using LabelEncoder.
2. **Feature Scaling** -Normalize features with StandardScaler → X_scaled
4. **Train-Test Split** - Splits data for training and testing
5. **Model Creation & Hyperparameter Tuning** 
-  SVM: Tune via GridSearchCV

- LSTM: Tune via KerasTuner on X_seq

- VAE + RF: Use unsupervised VAE to extract latent features, then train RandomForest
6. **Training** - Train final selected models best SVC, tuned LSTM, VAE → RF

7. **Evaluation & Comparison** - Evaluation & Comparison: Compute metrics (Accuracy, F1-Score) and generate model comparison table.
7. **Explainable AI (XAI):** - Perform SHAP analysis to interpret feature importance and non-linear relationships.

## Project Structure

```
.
├── .venv/                  # Virtual environment
├── requirements.txt        # Python dependencies
├── notebook.ipynb          # Main Jupyter notebook
├── README.md              
├── Dos.csv.md              
├── Malfunction.csv              
├── Normal.csv              
└── saved_models /                # Dataset  
```
## Troubleshooting

### Issue: Module not found
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```


### Issue: Virtual environment not activating
**Solution**: Check Python installation and ensure `.venv` folder exists

## Dependencies

-numpy
-pandas
-scikit-learn
-matplotlib
-seaborn
-plotly
-tensorflow / keras
-shap
-xgboost
-jupyter / ipython / ipykernel
-tqdm

