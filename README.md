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

### 3. Run the Model

Open and run `model/svm_model.ipynb` in Jupyter.

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
└── model/
    └── svm_model.ipynb            # SVM model training and evaluation
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
4. **Run the notebook**:
   ```bash
   jupyter notebook model/svm_model.ipynb
   ```
   Execute all cells in order. The notebook uses `random_state=42` throughout for deterministic results given the same input data.
5. **Expected outputs**: The notebook produces baseline and tuned SVM accuracy/ROC-AUC metrics, confusion matrices, ROC curves, and a sensitivity analysis across decision thresholds.
