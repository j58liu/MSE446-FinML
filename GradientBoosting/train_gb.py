# ============================================================
# MSE446 — Financial ML: XLE Direction Prediction
# Gradient Boosting (XGBoost)
# ============================================================

import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
   accuracy_score,
   f1_score,
   precision_score,
   recall_score,
   roc_auc_score,
   confusion_matrix,
   roc_curve,
   auc
)
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import UndefinedMetricWarning
import warnings


# ------------------------
# Suppress warnings and verbose output
# ------------------------
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ------------------------
# 1. Load Data
# ------------------------
X_train = pd.read_csv('data/X_train(2016-2023).csv', index_col=0)
y_train = pd.read_csv('data/y_train1(2016-2023).csv', index_col=0).values.ravel()
X_val = pd.read_csv('data/X_val(2024).csv', index_col=0)
y_val = pd.read_csv('data/y_val(2024).csv', index_col=0).values.ravel()
X_test = pd.read_csv('data/X_test(2025).csv', index_col=0)
y_test = pd.read_csv('data/y_test(2025).csv', index_col=0).values.ravel()


# Optional: Load scaler
scaler_path = 'data/scaler.pkl'
if os.path.exists(scaler_path):
   scaler = joblib.load(scaler_path)


# Create folder for results
os.makedirs('GradientBoostingResults', exist_ok=True)


# Dictionary to store results
results = {}


# ------------------------
# 2. Baseline Model
# ------------------------
print("\n=== TRAINING BASELINE MODEL ===")
baseline_params = {
   "objective": "binary:logistic",
   "learning_rate": 0.1,
   "max_depth": 3,
   "subsample": 1.0,
   "colsample_bytree": 1.0,
   "eval_metric": "logloss",
   "seed": 42
}


dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)


start_time = time.time()
baseline_model = xgb.train(
   baseline_params,
   dtrain,
   num_boost_round=100,
   evals=[(dtrain, "train"), (dval, "val")],
   early_stopping_rounds=20,
   verbose_eval=False
)
baseline_time = time.time() - start_time


y_pred_proba = baseline_model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)


results['baseline'] = {
   'model': baseline_model,
   'y_pred': y_pred,
   'y_pred_proba': y_pred_proba,
   'accuracy': accuracy_score(y_test, y_pred),
   'f1': f1_score(y_test, y_pred),
   'precision': precision_score(y_test, y_pred),
   'recall': recall_score(y_test, y_pred),
   'roc_auc': roc_auc_score(y_test, y_pred_proba),
   'confusion_matrix': confusion_matrix(y_test, y_pred),
   'runtime_sec': baseline_time
}


baseline_model.save_model('GradientBoostingResults/xgb_baseline.json')
print("Baseline model saved at GradientBoostingResults/xgb_baseline.json")


# ------------------------
# 3. Manual Tuned Model
# ------------------------
print("\n=== TRAINING MANUAL_TUNED MODEL ===")
manual_params = {
   "objective": "binary:logistic",
   "learning_rate": 0.05,
   "max_depth": 5,
   "subsample": 0.8,
   "colsample_bytree": 0.8,
   "eval_metric": "logloss",
   "seed": 42
}


start_time = time.time()
manual_model = xgb.train(
   manual_params,
   dtrain,
   num_boost_round=500,
   evals=[(dtrain, "train"), (dval, "val")],
   early_stopping_rounds=50,
   verbose_eval=False
)
manual_time = time.time() - start_time


y_pred_proba = manual_model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)


results['manual_tuned'] = {
   'model': manual_model,
   'y_pred': y_pred,
   'y_pred_proba': y_pred_proba,
   'accuracy': accuracy_score(y_test, y_pred),
   'f1': f1_score(y_test, y_pred),
   'precision': precision_score(y_test, y_pred),
   'recall': recall_score(y_test, y_pred),
   'roc_auc': roc_auc_score(y_test, y_pred_proba),
   'confusion_matrix': confusion_matrix(y_test, y_pred),
   'runtime_sec': manual_time
}


manual_model.save_model('GradientBoostingResults/xgb_manual_tuned.json')
print("Manual tuned model saved at GradientBoostingResults/xgb_manual_tuned.json")


# ------------------------
# 4. Hyperparameter Tuned Model
# ------------------------
print("\n=== TRAINING HYPERTUNED MODEL ===")
xgb_clf = xgb.XGBClassifier(
   objective="binary:logistic",
   use_label_encoder=False,
   eval_metric='logloss',
   n_jobs=-1,
   seed=42
)


param_grid = {
   'n_estimators': [100, 150, 200],
   'learning_rate': [0.01, 0.05, 0.1],
   'max_depth': [2, 3, 5],
   'subsample': [0.7, 0.8, 1.0],
   'colsample_bytree': [0.7, 0.8, 1.0]
}


start_time = time.time()
grid_search = GridSearchCV(
   xgb_clf,
   param_grid,
   scoring='f1',
   cv=2,
   verbose=0,
   n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
hypertuned_time = time.time() - start_time


y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)


results['hypertuned'] = {
   'model': best_model,
   'y_pred': y_pred,
   'y_pred_proba': y_pred_proba,
   'accuracy': accuracy_score(y_test, y_pred),
   'f1': f1_score(y_test, y_pred),
   'precision': precision_score(y_test, y_pred),
   'recall': recall_score(y_test, y_pred),
   'roc_auc': roc_auc_score(y_test, y_pred_proba),
   'confusion_matrix': confusion_matrix(y_test, y_pred),
   'runtime_sec': hypertuned_time
}


best_model.save_model('GradientBoostingResults/xgb_hypertuned.json')
print("Hypertuned model saved at GradientBoostingResults/xgb_hypertuned.json")
print(f"Best hypertuned params: {grid_search.best_params_}")

# ------------------------
# 5. Save all metrics to CSV
# ------------------------
metrics_df = pd.DataFrame({
   m: [results[model][m] for model in ['baseline', 'manual_tuned', 'hypertuned']]
   for m in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
}, index=['baseline', 'manual_tuned', 'hypertuned'])
metrics_df.to_csv('GradientBoostingResults/metrics_comparison.csv')
print("All metrics saved at GradientBoostingResults/metrics_comparison.csv")

# ------------------------
# 6. Feature Importance for Manual Tuned
# ------------------------
plt.figure(figsize=(10,6))
xgb_importance = manual_model.get_score(importance_type="weight")
xgb_importance = dict(sorted(xgb_importance.items(), key=lambda item: item[1], reverse=True))
plt.bar(range(len(xgb_importance)), list(xgb_importance.values()))
plt.xticks(range(len(xgb_importance)), list(xgb_importance.keys()), rotation=90)
plt.title("Feature Importance (Manual Tuned)")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig('GradientBoostingResults/feature_importance_manual_tuned.png')
plt.close()

# ------------------------
# 7. Confusion Matrices
# ------------------------
fig, axes = plt.subplots(1, 3, figsize=(18,5))
for i, name in enumerate(['baseline', 'manual_tuned', 'hypertuned']):
    cm = results[name]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'{name.upper()} Confusion Matrix')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('GradientBoostingResults/confusion_matrices_comparison.png')
plt.close()

# ------------------------
# 8. Bar Chart of Metrics
# ------------------------
metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall']
metrics_plot_df = pd.DataFrame({
    metric: [results[m][metric] for m in ['baseline', 'manual_tuned', 'hypertuned']]
    for metric in metrics_to_plot
}, index=['baseline', 'manual_tuned', 'hypertuned'])
metrics_plot_df.plot(kind='bar', figsize=(10,6))
plt.title('Comparison of Key Metrics Across Models')
plt.ylabel('Score')
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('GradientBoostingResults/metrics_comparison_plot.png')
plt.close()

# ------------------------
# 9. Sensitivity Analysis (F1, Precision, Recall, Accuracy vs Threshold)
# ------------------------
y_proba = results['hypertuned']['y_pred_proba']
thresholds = np.linspace(0,1,101)
f1_scores, precision_scores, recall_scores, acc_scores = [], [], [], []

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_t, zero_division=0))
    precision_scores.append(precision_score(y_test, y_pred_t, zero_division=0))
    recall_scores.append(recall_score(y_test, y_pred_t))
    acc_scores.append(accuracy_score(y_test, y_pred_t))

plt.figure(figsize=(10,6))
plt.plot(thresholds, f1_scores, label='F1 Score')
plt.plot(thresholds, precision_scores, label='Precision')
plt.plot(thresholds, recall_scores, label='Recall')
plt.plot(thresholds, acc_scores, label='Accuracy')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Sensitivity Analysis (Hypertuned Model)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('GradientBoostingResults/sensitivity_analysis.png')
plt.close()

# ------------------------
# 10. ROC Curve for all models
# ------------------------
plt.figure(figsize=(10,6))
for name in ['baseline', 'manual_tuned', 'hypertuned']:
    y_pred_proba = results[name]['y_pred_proba']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name.upper()} (AUC = {roc_auc_val:.3f})')

plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('GradientBoostingResults/roc_curve_comparison.png')
plt.close()

print("\nAll plots saved in 'GradientBoostingResults/' folder.")

# ------------------------
# 11. Print Final Results in Console (with Runtime)
# ------------------------
print("\n================ FINAL MODEL PERFORMANCE ================\n")
for name in ['baseline', 'manual_tuned', 'hypertuned']:
    print(f"\n=== {name.upper()} TEST RESULTS ===")
    acc = results[name]['accuracy']
    f1 = results[name]['f1']
    precision = results[name]['precision']
    recall = results[name]['recall']
    roc_auc = results[name]['roc_auc']
    cm = results[name]['confusion_matrix']
    runtime = results[name]['runtime_sec']
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Runtime (seconds): {runtime:.2f}")
    print("Confusion Matrix:")
    print(cm)