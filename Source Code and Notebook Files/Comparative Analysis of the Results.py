import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, roc_auc_score, f1_score, recall_score, precision_score
)
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Loading the data


def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Class'])
    y = df['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Model predictions and probabilities


def model_predictions(model, X, model_type='sklearn'):
    if model_type == 'keras':
        preds = model.predict(X).ravel()
        return (preds > 0.5).astype(int), preds
    elif model_type == 'xgboost':
        preds = model.predict(xgb.DMatrix(X))
        return (preds > 0.5).astype(int), preds
    else:
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X)[:, 1]
        else:
            preds = model.decision_function(X)
        return model.predict(X), preds

# Loading the models


def load_models():
    models = {
        'Random Forest': joblib.load('random_forest_model.pkl'),
        'KNN': joblib.load('knn_model.pkl'),
        'MLP': tf.keras.models.load_model('mlp_model.h5'),
        'CNN': tf.keras.models.load_model('cnn_model3.h5'),
        'Logistic Regression': joblib.load('logistic_regression_model.pkl'),
        'SVM': joblib.load('svm_model.pkl'),
        'XGBoost': joblib.load('xgboost_model.pkl')
    }
    return models

# Evaluating all models and plot comparison


def evaluate_models(models, X, y):
    results = []

    sns.set(style="whitegrid")

    # PRC
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        if name in ['MLP', 'CNN']:
            preds, pred_probs = model_predictions(model, X, model_type='keras')
        elif name == 'XGBoost':
            preds, pred_probs = model_predictions(
                model, X, model_type='xgboost')
        else:
            preds, pred_probs = model_predictions(model, X)

        precision_curve, recall_curve, _ = precision_recall_curve(
            y, pred_probs)
        pr_auc = auc(recall_curve, precision_curve)
        plt.plot(recall_curve, precision_curve,
                 label=f'{name} (AUC = {pr_auc:.2f})')

    plt.title('Precision-Recall Curve Comparison')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # ROC-AUC
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        if name in ['MLP', 'CNN']:
            preds, pred_probs = model_predictions(model, X, model_type='keras')
        elif name == 'XGBoost':
            preds, pred_probs = model_predictions(
                model, X, model_type='xgboost')
        else:
            preds, pred_probs = model_predictions(model, X)

        fpr, tpr, _ = roc_curve(y, pred_probs)
        roc_auc = roc_auc_score(y, pred_probs)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
    plt.title('ROC Curve Comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Precision, Recall, and F1 Score Comparison (Bar Plot)
    for name, model in models.items():
        if name in ['MLP', 'CNN']:
            preds, _ = model_predictions(model, X, model_type='keras')
        elif name == 'XGBoost':
            preds, _ = model_predictions(model, X, model_type='xgboost')
        else:
            preds, _ = model_predictions(model, X)

        precision = precision_score(y, preds)
        recall = recall_score(y, preds)
        f1 = f1_score(y, preds)

        results.append({
            'Model': name,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

    df_results = pd.DataFrame(results).set_index('Model')
    plt.figure(figsize=(10, 6))
    df_results.plot(kind='bar')
    plt.title('Precision, Recall, and F1 Score Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # calling load_models() function to load the models
    X, y = load_data('sample creditcard files.csv')
    models = load_models()

    # calling the evaluate_model() function for visualisations 
    evaluate_models(models, X, y)
