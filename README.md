# MSE446 — Financial ML: XLE Next-Day Direction Prediction

## How to Run the Code

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset

From the project root directory:

```bash
python data/data.py
```

This downloads market data via Yahoo Finance, constructs features (macro indicators + individual energy stock log returns), scales them, and saves train/val/test splits as CSVs into `data/`.

### 3. Run the Models

* LSTM: Open and run `lstm_model/main.ipynb` in Jupyter.
* SVM: Open and run `model/svm_model.ipynb` in Jupyter.
* Gradient Boosting (XGBoost): Run `python GradientBoosting/train_gb.py`
* Random Forest: Open and run `model/random_forest.ipynb` in Jupyter
* Logistic Regression: 

## Dependencies

Listed in `requirements.txt`:

## Folder Structure

```
MSE446-FinML/
├── README.md
├── requirements.txt
├── data/
│   ├── data.py                    # Dataset pipeline (download, clean, scale, split)
│   ├── check.py                   # Data inspection and extraction utilities
│   ├── energy_tickers.txt         # List of energy sector tickers
│   ├── X_train(2016-2023).csv     # Training features
│   ├── X_val(2024).csv            # Validation features
│   ├── X_test(2025).csv           # Test features
│   ├── y_train1(2016-2023).csv    # Training labels
│   ├── y_val(2024).csv            # Validation labels
│   └── y_test(2025).csv           # Test labels
├── GradientBoosting/              
│   └── train_gb.py                # Gradient Boosting(XGBoost) model training and evaluation
├── lstm_model/
│   ├── config.ipynb               # configuration file for LSTM model
│   ├── data.ipynb                 # data loading for LSTM model
│   ├── main.ipynb                 # main pipeline for LSTM model training and evaluation
│   └──  modules.ipynb             # LSTM classifier
├── model/
│    └── svm_model.ipynb           # SVM model training and evaluation
│    └── random_forest.ipynb       # Random forest model training and evaluation
└── LR_model.ipynb                 # Logistic regression model training, and evaluation
```

## Reproducibility Instructions

1. **Python version**: Use Python 3.9+.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Regenerate the dataset** (optional — pre-built CSVs are included):
   ```bash
   python data/data.py
   ```
   Note: Re-running `data.py` fetches live data from Yahoo Finance. Market data updates daily, so results may differ slightly if new trading days have elapsed since the CSVs were generated.
4. **Run the models:

    Logistic Regression: Open and run LR_model.ipynb in Jupyter
    SVM: Open and run model/svm_model.ipynb in Jupyter
    Random forest: Open and run model/random_forest.ipynb in Jupyter
    LSTM: Open and run lstm_model/main.ipynb in Jupyter
    
    Gradient Boosting (XGBoost):
    ```bash
    python GradientBoosting/train_gb.py
    ```
    Execute all notebook cells in order. Notebooks use fixed random states where applicable for reproducibility.

5. **Expected outputs:
    Each model produces:
    
    Baseline and tuned performance metrics (Accuracy, F1-score, ROC-AUC)
    Confusion matrices
    ROC curves
    Sensitivity analysis across different classification thresholds
    
    These outputs allow for consistent comparison of model performance across different algorithms.
